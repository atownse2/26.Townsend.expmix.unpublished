"""Canonical helpers for likelihood profiling and global scan workflows.

This module is the stable entrypoint for profile scans used by notebooks and
analysis scripts. It wraps the core scan/plot functionality from the split
code modules while
providing a cleaner, narrower interface.
"""

import os
import random
from typing import Dict, List, Optional, cast

import pandas as pd
import numpy as np

from .models import ExponentialMixtureModel
from .plotting import plot_pair_profiles
from tools import storage
from tools import scale_out as so

profile_cache = storage.ensure_cache("profiles")

def build_exponential_model(x, data, n_components: int, data_mean: Optional[float] = None, **kwargs):
    """Create an ExponentialMixtureModel with sensible defaults for profiling."""
    if data_mean is None:
        data_mean = data.mean(x)
    return ExponentialMixtureModel(x, n_components, data_mean=data_mean, **kwargs)


def run_profile_scan(
    model,
    data,
    pset: Dict[str, tuple],
    constant: bool = False,
    n_batches: int = 64,
    use_condor: bool = False,
    cache_name: Optional[str] = None,
    remake_cache: bool = False,
):
    """Run a profile scan over a parameter set and return a dataframe of NLL values."""
    return scan_parameters(
        model=model,
        data=data,
        pset=pset,
        constant=constant,
        n_batches=n_batches,
        use_condor=use_condor,
        cache_name=cache_name,
        remake_cache=remake_cache,
    )


def plot_profile_pairs(df, pset: Dict[str, tuple], **kwargs):
    """Plot pairwise profile projections for scan results."""
    return plot_pair_profiles(df=df, pset=pset, **kwargs)


def run_exponential_profile(
    x,
    data,
    n_components: int,
    pset: Dict[str, tuple],
    data_mean: Optional[float] = None,
    **scan_kwargs,
):
    """Convenience wrapper that builds an exponential model and profiles it."""
    model = build_exponential_model(
        x=x,
        data=data,
        n_components=n_components,
        data_mean=data_mean,
    )
    return run_profile_scan(model=model, data=data, pset=pset, **scan_kwargs)


def fit_points(ws_cache, points: List[Dict], profile: bool):
    import ROOT
    import pandas as pd

    # Load workspace from cache
    fin = ROOT.TFile(ws_cache, "READ")
    ws = fin.Get("ws")
    data = ws.data("data")
    pdf = ws.pdf("pdf")
    fin.Close()

    defaults = {par.GetName(): par.getVal() for par in pdf.getParameters(data)}
    
    nll = pdf.createNLL(data)

    new_points = []
    for i, point in enumerate(points):
        for par_name, par_value in point.items():
            par = ws.var(par_name)
            par.setVal(par_value)
            if profile:
                par.setConstant(True)

        fit_result = pdf.fitTo(data, Save=True, PrintLevel=-1)
        if fit_result.status() > 2:
            raise RuntimeError(
                f"Profile fit failed at point {point} with status {fit_result.status()}"
            )
        new_point = point.copy()
        for par_name, par_value in point.items():
            par = ws.var(par_name)
            new_point[f"{par_name}_final"] = par.getVal()
        # new_point["nll"] = fit(model, data, penalty=penalty, quiet=True, return_nll=True)
        new_point["nll"] = nll.getVal()
        new_points.append(new_point)
    
        # Reset parameters to defaults
        for par_name, par_value in defaults.items():
            par = ws.var(par_name)
            par.setVal(par_value)
            par.setConstant(False)

    return pd.DataFrame(new_points)

def concatenate_dfs(dfs: List[pd.DataFrame]):
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def scan_parameters(
    model, data, pset,
    constant=True, n_batches=12, use_condor=False,
    cache_name=None, remake_cache=False
    ):

    # Cache the ROOT objects in a workspace
    ws_cache_name = "tmp_workspace"
    if cache_name is not None:
        ws_cache_name = cache_name
    ws_cache = os.path.join(profile_cache, f"{ws_cache_name}.root")

    import ROOT
    ws = ROOT.RooWorkspace("ws", "ws")
    model.pdf.SetName("pdf")
    data.SetName("data")
    getattr(ws, 'import')(data)
    getattr(ws, 'import')(model.pdf)

    fout = ROOT.TFile(ws_cache, "RECREATE")
    ws.Write()
    fout.Close()

    if use_condor:
        assert cache_name is not None, "cache_file must be provided when use_condor is True"
    
    cache_file = None
    if cache_name is not None:
        cache_file = os.path.join(profile_cache, f"{cache_name}.csv")
        
        if os.path.exists(cache_file) and not remake_cache:
            df = pd.read_csv(cache_file)
            return df

    # Profile each pair of parameters
    pset_linspaces = {
        k: np.round(np.linspace(*v),3) for k, v in pset.items()
    }
    from itertools import product
    points = [dict(zip(pset_linspaces.keys(), vals)) for vals in product(*pset_linspaces.values())]
    # Randomize order
    random.shuffle(points)
    # Split into batches
    point_batches = [points[index::n_batches] for index in range(n_batches)]

    tasks = []
    for batch in point_batches:
        tasks.append(so.Task(fit_points, ws_cache, batch, constant))

    df = so.run_tasks(
        tasks,
        n_cores=n_batches,
        merge_results_fn=concatenate_dfs,
        use_condor=use_condor,
        cache_results=True if use_condor else False,
        condor_job_name=cache_name or ws_cache_name,
        env_wrapper=so.run_in_mamba,
        clear_logs=True,
    )

    # Remove nan
    df = cast(pd.DataFrame, df).dropna()

    if cache_file is not None:
        df.to_csv(cache_file, index=False)

    return df

def get_initial(n_components, worst_case=False):
    scan_file = os.path.join(profile_cache, f"{n_components}D_grid_restarts.csv")
    if os.path.exists(scan_file):
        df = pd.read_csv(scan_file)
        if worst_case:
            row = df.loc[df['nll'].idxmax()]
        else:
            row = df.loc[df['nll'].idxmin()]
        rates = { f"raw_rate_{i}": row[f"raw_rate_{i}"] for i in range(n_components) }
        return rates
    else:
        print(f"Warning: No scan file found for {n_components} components. Using default initial values.")
        return {}