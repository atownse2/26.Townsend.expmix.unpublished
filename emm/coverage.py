import os
import pickle
from typing import TypedDict, cast

import ROOT
import numpy as np

import matplotlib.pyplot as plt

from .models import evaluate_pdf
from .fitting import FitResult, fit_random_restarts

from tools import storage
from tools import scale_out as so

coverage_cache = storage.ensure_cache("coverage")


class CoverageFitResult(FitResult):
    predictions: np.ndarray


def get_coverage_cache_path(toy_model, seed, n_bootstraps, n_events):
    return f"{coverage_cache}/{toy_model.name}_seed{seed}_{n_bootstraps}bootstraps_{n_events}events.pkl"

def run_coverage_fits(
    x, toy_model, model_primitives,
    n_bootstraps, n_events, seed,
    test_points
    ):

    # Set the random seed for reproducibility
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))
    np.random.seed(int(seed))

    data = toy_model.pdf.generate(ROOT.RooArgSet(x), n_events)
    data_arr = np.array([data.get(i).getRealValue("x") for i in range(n_events)])

    # bootstrapped_param_list = []
    model_names = [mp.name for mp in model_primitives]
    test_val_dict = {model_name: {} for model_name in model_names}
    for i_boot in range(n_bootstraps):
        bootstrapped_data = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrapped_dataset = ROOT.RooDataSet("bootstrapped_data", "bootstrapped_data", ROOT.RooArgSet(x))
        for val in bootstrapped_data:
            x.setVal(val)
            bootstrapped_dataset.add(ROOT.RooArgSet(x))

        for model_primitive in model_primitives:
            model = model_primitive(x)
            fit_result = fit_random_restarts(
                x, bootstrapped_dataset, model_primitive,
                seed, n_restarts=10, n_retries=42,
                save=False,
            )
            if fit_result is None:
                continue

            # Get predictions at test points
            model = model_primitive(x) # Re-initialize the model to ensure it's in a clean state
            model.set_params(fit_result["final_pars"])
            coverage_fit_result = cast(CoverageFitResult, fit_result)
            coverage_fit_result["predictions"] = evaluate_pdf(x, model, test_points)

            test_val_dict[model.name][i_boot] = coverage_fit_result

    # Cache results
    output_file = get_coverage_cache_path(toy_model, seed, n_bootstraps, n_events)
    with open(output_file, "wb") as f:
        pickle.dump(test_val_dict, f)

def run_coverage_tasks(
    x, toy_models, model_primitives,
    seeds, n_bootstraps,
    n_events, test_points,
    use_condor=False,
    remake = False,
    ):
    tasks = []
    for toy_model in toy_models:
        for seed in seeds:
            output_file = get_coverage_cache_path(toy_model, seed, n_bootstraps, n_events)
            if os.path.exists(output_file) and remake:
                os.remove(output_file)
            elif os.path.exists(output_file) and not remake:
                print(f"Coverage results for {toy_model.name} seed {seed} already exist. Skipping.")
                continue

            task = so.Task(
                run_coverage_fits,
                x,
                toy_model,
                model_primitives,
                n_bootstraps,
                n_events,
                seed,
                test_points,
            )
            tasks.append(task)

    return so.run_tasks(
        tasks,
        use_condor=use_condor,
        condor_job_name=f"coverage_{n_bootstraps}b_{n_events}e",
        env_wrapper=so.run_in_mamba
    )

def load_and_format_coverage_result(toy_model_name, true_vals, cache_file, alpha=0.05, model_selection=True, n_events=None):
    
    if model_selection and n_events is None:
        raise ValueError("n_events must be provided when model_selection is True.")
    assert n_events is not None
    
    with open(cache_file, "rb") as f:
        result = pickle.load(f)
    
    in_CI_dict = {}
    for model_name, bootstrapped_fit_results in result.items():
        if model_selection and "ExponentialMixture" in model_name:
            continue
        bootstrapped_predictions = []
        for i_boot, fit_result in bootstrapped_fit_results.items():
            bootstrapped_predictions.append(fit_result["predictions"])
        bootstrapped_predictions = np.array(bootstrapped_predictions)
        lower, upper = np.percentile(
            bootstrapped_predictions,
            [100 * (alpha / 2), 100 * (1 - alpha / 2)],
            axis=0
            )
        in_ci = (true_vals >= lower) & (true_vals <= upper)
        in_CI_dict[model_name] = in_ci
    
    if model_selection:
        # Also compute for the ExponentialMixtureModel with model selection
        bootstrapped_AICs = {}
        bootstrapped_BICs = {}
        bootstrapped_predictions_AIC = {}
        bootstrapped_predictions_BIC = {}
        for model_name, bootstrapped_fit_results in result.items():
            if "ExponentialMixture" not in model_name:
                continue
            for i_boot, fit_result in bootstrapped_fit_results.items():
                if i_boot not in bootstrapped_AICs:
                    bootstrapped_AICs[i_boot] = np.inf
                if i_boot not in bootstrapped_BICs:
                    bootstrapped_BICs[i_boot] = np.inf

                # Compute AIC and BIC
                k = int(model_name.split("-")[-1])
                n_params = 2*k-1
                aic = 2*n_params + 2*fit_result["nll"]
                bic = n_params * np.log(n_events) + 2*fit_result["nll"]
                if aic < bootstrapped_AICs[i_boot]:
                    bootstrapped_AICs[i_boot] = aic
                    bootstrapped_predictions_AIC[i_boot] = fit_result["predictions"]
                if bic < bootstrapped_BICs[i_boot]:
                    bootstrapped_BICs[i_boot] = bic
                    bootstrapped_predictions_BIC[i_boot] = fit_result["predictions"]
        
        # Compute in_CI for AIC
        names = ["Exponential Mixture (AIC)", "Exponential Mixture (BIC)"]
        predictions = [
            np.array(list(bootstrapped_predictions_AIC.values())),
            np.array(list(bootstrapped_predictions_BIC.values()))
        ]
        for name, preds in zip(names, predictions):
            lower, upper = np.percentile(
                preds,
                [100 * (alpha / 2), 100 * (1 - alpha / 2)],
                axis=0
                )
            in_ci = (true_vals >= lower) & (true_vals <= upper)
            in_CI_dict[name] = in_ci

    return in_CI_dict, toy_model_name

def get_coverage_results(
    toy_models,
    seeds,
    n_bootstraps,
    n_events,
    true_pdf_vals,
    model_selection=True,
    alpha=0.05,
    ):
    
    tasks = []
    for toy_model in toy_models:
        for seed in seeds:
            cache_file = get_coverage_cache_path(toy_model, seed, n_bootstraps, n_events)
            if not os.path.exists(cache_file):
                continue
            task = so.Task(
                load_and_format_coverage_result,
                toy_model.name,
                true_pdf_vals[toy_model.name],
                cache_file,
                alpha,
                model_selection=model_selection,
                n_events=n_events,)
            tasks.append(task)
    
    results = so.run_tasks(tasks)
    in_CI_results = {tm.name: {} for tm in toy_models}
    for in_CI_dict, toy_model_name in results:
        for model_name, in_CI in in_CI_dict.items():
            if model_name not in in_CI_results[toy_model_name]:
                in_CI_results[toy_model_name][model_name] = []
            in_CI_results[toy_model_name][model_name].append(in_CI)
    
    coverages = {tm.name: {} for tm in toy_models}
    for toy_model_name, test_model_dict in in_CI_results.items():
        for test_model_name, in_CI_list in test_model_dict.items():
            in_CI_array = np.array(in_CI_list)
            coverage = np.mean(in_CI_array, axis=0)
            coverages[toy_model_name][test_model_name] = coverage

    return coverages

def plot_coverages(
        coverages, test_points,
        alpha=0.05,
        y_min=0.4,
        models_to_skip=None,
        range: tuple | None = None,
        skip_every: int = 1,
        linewidth=3.5,
        labelsize=16,
        fontsize=18,
    ):

    if range is not None:
        mask = (test_points >= range[0]) & (test_points <= range[1])
    else:
        mask = np.ones_like(test_points, dtype=bool)

    if skip_every > 1:
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[np.arange(0, len(mask), skip_every)] = True
        mask = mask & new_mask

    fig, axs = plt.subplots(
        len(coverages), 1,
        figsize=(15, 4 * len(coverages)),
        sharex=True, gridspec_kw={'hspace': 0})

    colors = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#984ea3', '#a65628',
                '#999999', '#e41a1c', '#dede00']
    
    line_styles = ["solid", "dashed", "dotted", "dashdot"]

    for i, (toy_model_name, test_model_dict) in enumerate(coverages.items()):
        ax = axs[i] if len(coverages) > 1 else axs

        # Print the true model name in the top left corner of the plot
        ax.text(
            0.01, 0.97,
            f"Truth Model: ${toy_model_name}$",
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='top',
            color=colors[i % len(colors)],
            alpha=1,
        )

        for j, (test_model_name, coverage) in enumerate(test_model_dict.items()):
            if models_to_skip is not None and test_model_name in models_to_skip:
                continue
            alpha_line = 0.3
            line_style=line_styles[0]
            label = test_model_name
            if "_" in label:
                label = f"${label}$"
            if test_model_name == toy_model_name:
                # label += " (true model)"
                alpha_line = 1.0
            if "Exponential Mixture" in test_model_name:
                alpha_line = 1.0
                if "AIC" in test_model_name:
                    line_style = line_styles[1]
                else:
                    line_style = line_styles[2]
            ax.plot(
                test_points[mask],
                100*coverage[mask],
                label=label,
                color=colors[j % len(colors)],
                linestyle=line_style,
                alpha=alpha_line,
                linewidth=linewidth,
                )
        ax.axhline(
            100*(1 - alpha),
            color='k',
            linestyle='--',
            alpha=0.75,
            linewidth=linewidth
        )
        # ax.set_ylabel(f"Coverage for {toy_model_name} truth")
        ax.set_ylabel("Coverage (C) [%]", fontsize=fontsize)
        # Make the y-tick labels larger
        ax.tick_params(axis='y', labelsize=labelsize)

        ax.set_ylim(100*y_min, 100)
        if i == 0:
            # Customize legend: organize f_1-f_4 in 2x2 grid, then exponential mixtures
            handles, labels = ax.get_legend_handles_labels()
            f_models = [h for h, l in zip(handles, labels) if l.startswith('$f_')]
            exp_models = [h for h, l in zip(handles, labels) if 'Exponential' in l]
            
            # Create custom legend with f_1-f_4 in 2x2 grid, then exponential mixtures
            all_handles = f_models + exp_models
            all_labels = [l for h, l in zip(handles, labels) if l.startswith('$f_') or 'Exponential' in l]
            
            legend = ax.legend(
                all_handles, all_labels,
                framealpha=0, 
                loc=(0.2, 0),
                fontsize=fontsize,
                ncol=2  # Use 2 columns for compact layout
            )
            # Set legend line alpha to 1 (full opacity) for legend display only
            for line in legend.get_lines():
                line.set_alpha(1.0)

        # Remove 1.0 from yticks so it doesn't overlap with the other plot
        yticks = ax.get_yticks().tolist()
        yticks = yticks[1:-1]
        ax.set_yticks(yticks)
    
    axs[-1].set_xlabel(r"$m_{\gamma\gamma}$ [GeV]", fontsize=fontsize)
    axs[-1].tick_params(axis='x', labelsize=labelsize)