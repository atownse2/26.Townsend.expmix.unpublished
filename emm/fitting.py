import time
import random
import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict, Union

import numpy as np

import ROOT # type: ignore
from array import array

from tools import storage
from tools import scale_out as so

random_string = lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))

default_fit_options = [
    ROOT.RooFit.PrintLevel(-1),
    # ROOT.RooFit.IntegrateBins(0.0001),
    # ROOT.RooFit.Offset(True),
    # # ROOT.RooFit.Strategy(2),
    # ROOT.RooFit.Save(),
    # # ROOT.RooFit.Range("fit_range")
]


class FitResult(TypedDict):
    status: int
    nll: float
    initial_pars: Dict[str, float]
    final_pars: Dict[str, float]
    n_retries: int


def fit(model, data, fit_args=None, print_level=0):
    t1 = time.time()
    if print_level > 0:
        print(f"Fitting model {model.name} to data {data.GetName()} with {len(data)} entries")

    fit_args = list(fit_args) if fit_args is not None else []
    if not any(option.GetName() == "Save" for option in fit_args):
        fit_args.append(ROOT.RooFit.Save(True))

    fit_result = model.pdf.fitTo(
        data,
        *fit_args
    )

    if print_level > 0:
        t2 = time.time()
        print(f"Fitted in {t2 - t1:.2f} seconds")
        print(f"Fit status: {fit_result.status()}, covQual: {fit_result.covQual()}")
        print(f"NLL: {fit_result.minNll()}")

    return fit_result

def fit_n_retries(
    model, data, n_retries,
    fit_options=None,
    print_level=0,
    ) -> Optional[FitResult]:
    """
    Fit a model to data, retrying up to n_retries times if the fit fails
    (fit status > 2). Returns the fit result if successful, or None if all attempts fail.
    """
    import ROOT
    # Copy so we don't mutate the caller's list.
    fit_options = list(default_fit_options if fit_options is None else fit_options)

    # Add Save(True) to the fit options if not already present, so we can access the fit result.
    if not any(option.GetName() == "Save" for option in fit_options):
        fit_options.append(ROOT.RooFit.Save(True))

    initial_pars = {p.GetName(): p.getVal() for p in model.params()}

    for i_retry in range(n_retries):
        fit_result = model.pdf.fitTo(
            data,
            *fit_options,
        )
        try:
            fit_status = fit_result.status()
        except ReferenceError:
            if print_level > 2:
                print(f"Fit attempt {i_retry + 1} returned no result.")
            continue

        if (fit_status <= 2):
            if i_retry > 0 and print_level > 2:
                print(f"Fit succeeded after {i_retry} retries")

            final_pars = {p.GetName(): p.getVal() for p in model.params()}

            result: FitResult = {
                "status": fit_status,
                "nll": fit_result.minNll(),
                "initial_pars": initial_pars,
                "final_pars": final_pars,
                "n_retries": i_retry,
            }
            return result
        
    if print_level > 1:
        print(f"Fit failed after {n_retries} attempts, returning None" )
    return None

# Random restarts:
fit_cache = storage.ensure_cache("fits")
def random_restarts_filename(
        model_name,
        data_name,
        n_samples,
        seed,
        ):
    tags = f"{model_name}_{data_name}_nrestarts{n_samples}_seed{seed}.pkl"
    f = f"{fit_cache}/{tags}"
    return f

def fit_random_restart(
    x, data,
    model_primitive,
    i_seed,
    n_retries=5,
    fit_options=None,
    print_level=0,
) -> Optional[FitResult]:

    rng = np.random.default_rng(seed=i_seed)

    model = model_primitive(x)
    model.randomize_params(rng=rng)

    fit_result = fit_n_retries(
        model, data, n_retries,
        fit_options=fit_options,
        print_level=print_level,
    )

    if fit_result is None:
        if print_level > 1:
            print(f"Random Restart {i_seed+1}: Fit failed after {n_retries} attempts.")
    else:
        if print_level > 3:
            print(f"Random Restart {i_seed+1}: Fit succeeded with NLL={fit_result['nll']:.4f} after {fit_result['n_retries']} retries.")
    
    del model  # Free memory
    return fit_result


def fit_random_restarts(
        x, data,
        model_primitive,
        seed, n_restarts,
        n_retries=5,
        save=False,
        save_as=None,
        fit_options=None,
        print_level=0,
        return_all_results=False,
        use_multiprocessing=False,
    ) -> Union[Optional[FitResult], List[FitResult]]:

    rng = np.random.default_rng(seed=seed)
    restart_seeds = rng.integers(0, 2**32 - 1, size=n_restarts)

    tasks = []
    for i in range(n_restarts):
        task = so.Task(
            fit_random_restart,
            x, data, model_primitive, restart_seeds[i],
            n_retries=n_retries,
            print_level=print_level,
            fit_options=fit_options,
        )
        tasks.append(task)

    if use_multiprocessing:
        fit_results = so.run_tasks(tasks)
    else:
        fit_results = [task.run() for task in tasks]

    fit_results = [r for r in fit_results if r is not None]

    if len(fit_results) == 0:
        if print_level > 0:
            print(f"All {n_restarts} random restarts failed for model {model_primitive.name}.")
        return [] if return_all_results else None
    
    if save or save_as is not None:
        model = model_primitive(x)
        fout = save_as if save_as is not None else random_restarts_filename(model.name, data.GetName(), n_restarts, seed)
        with open(fout, "wb") as f:
            pickle.dump(fit_results, f)

    if return_all_results:
        return fit_results

    # Return the best fit result
    best_fit_result = min(fit_results, key=lambda r: r["nll"])
    if print_level > 2:
        print(f"Random Restarts: {len(fit_results)} successful fits out of {n_restarts} attempts.")
        print(f"Best NLL: {best_fit_result['nll']}")

    return best_fit_result


def fit_random_restarts_until_converged(
        x, data,
        model_primitive,
        seed,
        n_near_minimum=5,
        nll_threshold=0.01,
        max_restarts=100,
        n_retries=5,
        save=False,
        fit_options=None,
        print_level=0,
    ) -> Optional[FitResult]:
    """Fit random restarts until enough results lie near the current minimum NLL.

    Stops when ``n_near_minimum`` successful fits have NLL values within
    ``nll_threshold`` of the best successful fit, or after ``max_restarts``
    attempts. Returns the best fit, or ``None`` if every attempt fails.
    """
    if n_near_minimum < 1:
        raise ValueError("n_near_minimum must be at least 1")
    if nll_threshold < 0:
        raise ValueError("nll_threshold must be non-negative")
    if max_restarts < 1:
        raise ValueError("max_restarts must be at least 1")

    fit_results: List[FitResult] = []
    n_attempts = 0
    for i_restart in range(max_restarts):
        n_attempts += 1
        fit_result = fit_random_restart(
            x,
            data,
            model_primitive,
            seed + i_restart,
            n_retries=n_retries,
            fit_options=fit_options,
            print_level=print_level,
        )
        if fit_result is None:
            continue

        fit_results.append(fit_result)
        best_fit_result = min(fit_results, key=lambda result: result["nll"])
        n_near_best = sum(
            result["nll"] <= best_fit_result["nll"] + nll_threshold
            for result in fit_results
        )
        if n_near_best >= n_near_minimum:
            break

    if not fit_results:
        print("No successful fits were found.")
        return None

    best_fit_result = min(fit_results, key=lambda result: result["nll"])
    if save:
        model = model_primitive(x)
        fout = random_restarts_filename(
            model.name, data.GetName(), len(fit_results), seed
        )
        with open(fout, "wb") as f:
            pickle.dump(fit_results, f)

    if print_level > 0:
        print(
            f"Random Restarts: {len(fit_results)} successful fits after "
            f"{n_attempts} attempts. Best NLL: {best_fit_result['nll']}"
        )

    return best_fit_result


# Goodness-of-fit metrics
def compute_information_criteria(nll, n_params, n_observations):
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")

    return {
        "AIC": 2 * n_params + 2 * nll,
        "BIC": n_params * np.log(n_observations) + 2 * nll,
    }


def format_paper_comparison_table(fit_results: Sequence[Mapping[str, Any]]) -> str:
    """Format fit summaries as the LaTeX comparison table used in the paper.

    Each result must provide ``label``, ``chi2``, ``ndf``, ``chi2_ndf``, ``aic``,
    and ``bic``. The formatter does not inspect model types, so it supports
    exponential mixtures and arbitrary analytic comparison functions alike.
    """
    if not fit_results:
        raise ValueError("fit_results must contain at least one result")

    aic_min = min(float(result["aic"]) for result in fit_results)
    bic_min = min(float(result["bic"]) for result in fit_results)
    lines = [
        r"\begin{tabular}{lccccc}",
        r"Model & $\chi^2$ & NDF & $\chi^2_\nu$ & $\Delta$AIC & $\Delta$BIC \\",
        r"\hline",
    ]
    for result in fit_results:
        lines.append(
            f"{result['label']} & {float(result['chi2']):.1f} & {int(result['ndf'])} & "
            f"{float(result['chi2_ndf']):.2f} & "
            f"{float(result['aic']) - aic_min:.1f} & "
            f"{float(result['bic']) - bic_min:.1f} \\\\"
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines)

def rebin_for_low_stats(hist, min_events=30):
    """
    Dynamically merges adjacent bins of a TH1 histogram until every bin 
    contains at least `min_events`. Returns a new variable-bin-width TH1.
    """
    edges = [hist.GetBinLowEdge(1)]
    current_events = 0
    
    for b in range(1, hist.GetNbinsX() + 1):
        current_events += hist.GetBinContent(b)
        # If the accumulated events hit the threshold, seal the bin edge
        if current_events >= min_events:
            edges.append(hist.GetBinLowEdge(b) + hist.GetBinWidth(b))
            current_events = 0
            
    # Handle leftovers: merge remaining events into the final bin
    if current_events > 0 and len(edges) > 1:
        edges[-1] = hist.GetBinLowEdge(hist.GetNbinsX()) + hist.GetBinWidth(hist.GetNbinsX())
    elif len(edges) == 1:
        # Fallback if the entire histogram has fewer than min_events
        edges.append(hist.GetBinLowEdge(hist.GetNbinsX()) + hist.GetBinWidth(hist.GetNbinsX()))
        
    # Convert list to a C-style double array for ROOT
    edges_arr = array('d', edges)
    
    # TH1::Rebin handles the recalculation of contents and SumW2 errors automatically
    rebinned_hist = hist.Rebin(len(edges_arr) - 1, f"{hist.GetName()}_rebinned", edges_arr)
    return rebinned_hist

def chi2(x, hist, model, min_events=5, print_chi2=False, integrate_bins_precision=1e-3):
    """
    Compute chi2 for a 1D RooDataHist after merging adjacent bins
    until each combined bin contains at least min_events observed entries.
    """
    if min_events < 0:
        raise ValueError("min_events must be non-negative")

    if integrate_bins_precision is not None and integrate_bins_precision <= 0:
        raise ValueError("integrate_bins_precision must be positive or None")

    if not hasattr(model, "pdf") or not hasattr(model, "params"):
        raise TypeError("model must provide pdf and params()")

    pdf = model.pdf

    # Rebin the histogram to ensure at least min_events per bin
    rebinned_hist = rebin_for_low_stats(hist, min_events=min_events)
    n_combined_bins = rebinned_hist.GetNbinsX()
    ndf = n_combined_bins - len(model.params())

    rebinned_data = ROOT.RooDataHist(
        f"{rebinned_hist.GetName()}_rebinned",
        f"{rebinned_hist.GetTitle()} (rebinned)",
        x,
        rebinned_hist
    )

    chi2_args = []
    if integrate_bins_precision is not None:
        # Integrate the PDF across each rebinned bin. Point sampling at the bin
        # center can strongly overestimate chi2 for steeply falling models.
        chi2_args.append(ROOT.RooFit.IntegrateBins(integrate_bins_precision))

    chi2_var = ROOT.RooChi2Var(
        f"chi2_var_{random_string()}",
        "chi2",
        pdf,
        rebinned_data,
        *chi2_args,
    )
    raw_chi2 = float(chi2_var.getVal())
    result = {
        "chi2": raw_chi2,
        "ndf": ndf,
        "chi2_ndf": raw_chi2 / ndf,
        "n_combined_bins": int(n_combined_bins),
        "min_events": min_events,
        "integrate_bins_precision": integrate_bins_precision,
        "binned_data": rebinned_data,
        "hist": rebinned_hist,
        "bin_edges": rebinned_hist.GetXaxis().GetXbins(),
    }

    if print_chi2:
        print(
            f"chi2={result['chi2']:.2f}, ndf={result['ndf']}, "
            f"chi2/ndf={result['chi2_ndf']:.3f}, "
        )

    return result

# Cross-validation utilities
def train_test_split(x, data, n_folds, seed=1234):
    """Split a RooDataSet into train/test subsets for cross-validation.
    Returns lists of RooDataSets for training and testing, one per fold.
    If n_folds < 1, returns the leave-one-out split where each test set is a single entry.
    """

    import numpy as np

    n = data.numEntries()
    data_arr = np.array([data.get(i).getRealValue(x.GetName()) for i in range(n)])

    if n_folds < 1:
        n_folds = n
    else:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(data_arr)
    data_arr_folds = np.array_split(data_arr, n_folds)

    train_datasets = []
    test_datasets = []
    for i in range(n_folds):
        name = f"{data.GetName()}_{n_folds}folds_{i}"
        test_data_arr = data_arr_folds[i]
        train_data_arr = np.concatenate([data_arr_folds[j] for j in range(n_folds) if j != i])

        test_dataset = ROOT.RooDataSet(name+"_test", name+"_test", ROOT.RooArgSet(x))
        for val in test_data_arr:
            x.setVal(val)
            test_dataset.add(ROOT.RooArgSet(x))
        test_datasets.append(test_dataset)

        train_dataset = ROOT.RooDataSet(name+"_train", name+"_train", ROOT.RooArgSet(x))
        for val in train_data_arr:
            x.setVal(val)
            train_dataset.add(ROOT.RooArgSet(x))
        train_datasets.append(train_dataset)

    return train_datasets, test_datasets