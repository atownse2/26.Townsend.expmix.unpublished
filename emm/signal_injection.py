import gc
import os
import pickle
from typing import Dict, TypedDict, cast

import ROOT
import numpy as np

import matplotlib.pyplot as plt

from .models import (
    evaluate_pdf,
    ModelPrimitive,
    GaussianSignalModel,
    SignalPlusBackgroundModel,
)
from .fitting import FitResult, fit_random_restarts, fit_n_retries, train_test_split

from tools import storage
from tools import scale_out as so

# Spurious signal tests
signal_injection_cache = storage.ensure_cache("signal_injection")
def get_signal_injection_cache_path(
        toy_bkg_model, signal_point, c, seed, n_toys):

    tags = [
        f"{toy_bkg_model.name}",
        f"m{signal_point[0]}",
        f"w{signal_point[1]}",
        f"c{c}",
        f"seed{seed}",
        f"{n_toys}toys",
        "shared_toy_signed_fraction_v4",
    ]
    
    return os.path.join(signal_injection_cache, "_".join(tags) + "_injection_fits.pkl")


def run_signal_injection_fits(
        x_orig,
        toy_bkg_model,
        model_primitives,
        sigma_ref,
        c,
        seed,
        n_toys,
        n,
        signal_point,
        n_restarts, n_retries,
        fit_options: list = [],
        print_level: int = 0,
        save_as="default"
        ) -> list[dict]:

    n_inj = c * sigma_ref

    # Set seed
    ROOT.RooRandom.randomGenerator().SetSeed(int(seed))

    # s+b fits require additional fit options to ensure stability and robustness.
    sb_fit_options = fit_options.copy()

    # Added robustness because we allow for negative signal strengths in the fit,
    # which can lead to undefined regions in the likelihood.
    if ROOT.RooFit.RecoverFromUndefinedRegions(1.0) not in fit_options:
        sb_fit_options.append(ROOT.RooFit.RecoverFromUndefinedRegions(1.0))
    sb_fit_options.append(ROOT.RooFit.Extended(False))

    sig_mean, sig_width = signal_point

    all_fit_results = []
    for itoy in range(n_toys):
        x = x_orig.clone("x")

        # Fit every candidate background model to the same injected toy.
        toy_sig_model = GaussianSignalModel(x, sig_mean, sig_width)
        toy_model = SignalPlusBackgroundModel(toy_sig_model, toy_bkg_model, max_sig=n_inj)
        toy_model.set_param("n_sig", n_inj, constant=True)
        toy_data = toy_model.pdf.generate(ROOT.RooArgSet(x), n)

        for model_primitive in model_primitives:
            # Fit the background model first to stabilize the background fit
            bkg_fit_result = fit_random_restarts(
                x, toy_data, model_primitive, seed,
                n_restarts=n_restarts, n_retries=n_retries,
                save=False, fit_options=fit_options,
                print_level=print_level
            )
            if bkg_fit_result is None:
                if print_level > 0:
                    print(f"Background fit failed for toy {itoy}, model {model_primitive.name}. Skipping.")
                continue
        
            # Bkg-only fit
            bkg_model = model_primitive(x)
            bkg_model.set_params(bkg_fit_result["final_pars"])
            sig_model = GaussianSignalModel(x, sig_mean, sig_width)
            bkg_only_model = SignalPlusBackgroundModel(sig_model, bkg_model, max_sig=2*n_inj)
            bkg_only_model.set_param("n_sig", 0, constant=True)  # Set signal strength to zero for bkg-only fit
            bkg_only_fit_result = fit_n_retries(
                bkg_only_model, toy_data, n_retries=n_retries,
                fit_options=sb_fit_options,
                print_level=print_level
            )
            if bkg_only_fit_result is None:
                if print_level > 0:
                    print(f"Background-only fit failed for toy {itoy}, model {model_primitive.name}. Skipping.")
                continue

            # For hypothesis testing, we need to fit both the null (background-only) and alternative (signal+background) models to the same toy data.
            # Fit the null model
            bkg_model = model_primitive(x)
            bkg_model.set_params(bkg_fit_result["final_pars"])
            sig_model = GaussianSignalModel(x, sig_mean, sig_width)
            null_model = SignalPlusBackgroundModel(sig_model, bkg_model, max_sig=2*n_inj)
            null_model.set_param("n_sig", n_inj, constant=True)  # Set signal strength to injected value for null fit

            null_fit_result = fit_n_retries(
                null_model, toy_data, n_retries=n_retries,
                fit_options=sb_fit_options,
                print_level=print_level
            )
            if null_fit_result is None:
                if print_level > 0:
                    print(f"Background-only fit failed for toy {itoy}, model {model_primitive.name}. Skipping.")
                continue

            # Now float the signal strength to fit the alternative model

            # Scale limits on signal strength based on the number of 
            # background events in the signal region. 
            # This improves the fit stability
            x.setRange("sig_range", sig_mean - sig_width, sig_mean + sig_width)
            subset = toy_data.reduce(CutRange="sig_range")
            n_evt_in_sig_region = subset.sumEntries()
            if n_evt_in_sig_region == 0:
                max_sig = 10
            else:
                max_sig = 10*np.sqrt(n_evt_in_sig_region)
            max_sig = max(max_sig, 2*n_inj)  

            # Initialize the signal + background model
            bkg_model = model_primitive(x)
            bkg_model.set_params(bkg_fit_result["final_pars"])
            sig_model = GaussianSignalModel(x, sig_mean, sig_width)
            alt_model = SignalPlusBackgroundModel(sig_model, bkg_model, max_sig=max_sig)
            alt_model.set_param("n_sig", n_inj, constant=False)  # Float the signal strength for the alternative fit

            alt_fit_result = fit_n_retries(
                alt_model, toy_data, n_retries=n_retries,
                fit_options=sb_fit_options,
                print_level=print_level,
            )

            if alt_fit_result is None:
                if print_level > 0:
                    print(f"Fit failed for toy {itoy}, signal point {signal_point}, model {model_primitive.name}. Skipping.")
                continue

            # Save relevant info
            fit_result = {
                "signal_mean" : sig_mean,
                "signal_width" : sig_width,
                "c": c,
                "n_inj": n_inj,
                "seed": seed,
                "toy_index": itoy,
                "bkg_model_name": bkg_model.name,
                "bkg_random_restart_min_nll": bkg_fit_result["nll"],
                "bkg_only_fit_status": bkg_only_fit_result["status"],
                "bkg_only_nll": bkg_only_fit_result["nll"],
                "null_fit_status": null_fit_result["status"],
                "null_nll": null_fit_result["nll"],
                "alt_fit_status": alt_fit_result["status"],
                "alt_nll": alt_fit_result["nll"],
                "alt_n_sig": alt_model.get_param("n_sig").getVal(),
            }

            all_fit_results.append(fit_result)

            # --- EXPLICIT CLEANUP (Inner Loop) ---
            del subset
            del fit_result
            del sig_model
            del bkg_model
            del alt_model

        # --- EXPLICIT CLEANUP (Outer Loop) ---
        del toy_data
        del toy_model
        del toy_sig_model
        del x
        
        # Periodically force Python to collect garbage to ensure C++ destructors fire
        if itoy % 10 == 0:
            gc.collect()

    # Save result to cache
    if save_as is not None:
        if save_as == "default":
            cache_file = get_signal_injection_cache_path(toy_bkg_model, signal_point, c, seed, n_toys)
        else:
            cache_file = save_as
        with open(cache_file, "wb") as f:
            pickle.dump(all_fit_results, f)

    return all_fit_results

def load_signal_injection_fit_results(
        toy_model, signal_point, c, seeds, n_toys_per_seed):
    results = []
    for seed in seeds:
        cache_file = get_signal_injection_cache_path(toy_model, signal_point, c, seed, n_toys_per_seed)
        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} does not exist. Skipping.")
            continue
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        results.extend(result)
    return results





# Signal injection tests
signal_injection_cache = storage.ensure_cache("signal_injection")

    # def get_signal_injection_fits_cache_path(toy_model, signal_point, c, seed, n_toys):
    #     sig_mean, sig_width = signal_point
    #     return (
    #         f"{signal_injection_cache}/{toy_model.name}_m{sig_mean}_w{sig_width}"
    #         f"_c{c}_seed{seed}_{n_toys}toys_injection_fits.pkl"
    #     )
