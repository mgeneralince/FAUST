from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from fair_unaware.utils import compute_group_priors


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


@contextmanager
def temp_sys_path(path: Path):
    """Temporarily prepend a path to sys.path (and restore on exit)."""
    path = Path(path).expanduser().resolve()
    old = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = old


def import_fairreg_modules(fairreg_root: str | Path | None = None) -> Dict[str, Any]:
    """
    Import FairReg baseline modules.

    If fairreg_root is provided, it should be the folder that contains:
      data_prep.py, FairReg.py, evaluation_measures.py, and the data/ folder.
    """
    if fairreg_root is None:
        data_prep = importlib.import_module("data_prep")
        eval_measures = importlib.import_module("evaluation_measures")
        fairreg_mod = importlib.import_module("FairReg")
        inferred_root = Path(data_prep.__file__).resolve().parent
    else:
        with temp_sys_path(Path(fairreg_root)):
            data_prep = importlib.import_module("data_prep")
            eval_measures = importlib.import_module("evaluation_measures")
            fairreg_mod = importlib.import_module("FairReg")
            inferred_root = Path(data_prep.__file__).resolve().parent

    return {
        "ROOT": inferred_root,
        "get_lawschool_data": data_prep.get_lawschool_data,
        "get_adult_data": data_prep.get_adult_data,
        "get_communities_data": data_prep.get_communities_data,
        "get_frequencies": data_prep.get_frequencies,
        "DP_unfairness": eval_measures.DP_unfairness,
        "FairReg": fairreg_mod.FairReg,
    }


def load_dataset(
    dataset: str,
    *,
    fairreg_root: str | Path | None = None,
    data_scaling: bool = True,
    full_data: bool = False,
) -> Tuple[Any, Any, Any, Any, int, int, Any, dict]:
    """
    Load one of {adult, lawschool, communities} using the FairReg repo loaders.

    Returns: X, S, y, scaler, B, K, p, priors
    """
    fairreg = import_fairreg_modules(fairreg_root)

    get_lawschool_data = fairreg["get_lawschool_data"]
    get_adult_data = fairreg["get_adult_data"]
    get_communities_data = fairreg["get_communities_data"]
    get_frequencies = fairreg["get_frequencies"]
    root = Path(fairreg_root).expanduser().resolve() if fairreg_root is not None else fairreg["ROOT"]

    with pushd(root):
        if dataset == "lawschool":
            X, S, y = get_lawschool_data()
            y = y / 4.0
        elif dataset == "adult":
            X, S, y = get_adult_data()
            y = y / 100.0
        elif dataset == "communities":
            X, S, y = get_communities_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    scaler = None
    if data_scaling:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = scaler.fit_transform(pd.Series(y).values.reshape(-1, 1))
        y = pd.Series(y_scaled.flatten(), index=pd.Series(y).index if hasattr(y, "index") else None)

    #subsample to 2000 for adult/lawschool unless full_data (matches FairReg evaluation protocol even thoug i still don't understand why they do this 
    # probably for the Agarwal method they compare to which must be very slow )
    if (dataset in {"adult", "lawschool"}) and (not full_data):
        sample_size = 2000
        X, _, S, _, y, _ = train_test_split(
            X, S, y,
            test_size=1 - sample_size / len(X),
            random_state=42,
            stratify=S,
        )

    B = 1
    S_arr = np.asarray(S)
    K = int(np.unique(S_arr).size)
    p = get_frequencies(S)
    priors = compute_group_priors(S)


    return X, S, y, scaler, B, K, p, priors


def get_fairreg_objects(fairreg_root: str | Path | None = None):
    """Convenience: return (FairRegClass, DP_unfairness_fn)."""
    fairreg = import_fairreg_modules(fairreg_root)
    return fairreg["FairReg"], fairreg["DP_unfairness"]
