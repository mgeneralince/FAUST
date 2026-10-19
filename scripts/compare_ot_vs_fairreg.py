import os
import argparse
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm   
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import MinMaxScaler


from scipy.stats import ks_2samp    
import sys

                
from fair_unaware.plots.visualization import plot_fairreg_convergence

from fair_unaware import UnawareFairRegressor
from fair_unaware.baselines import AwareWassersteinBarycenterDPRegressor
from fair_unaware.baselines.frappe import FrappeRegressor
from fair_unaware.diagnostics import plot_basic_diagnostics
from data.synthetic import make_synthetic_dataset
from fair_unaware.plots.compare import (
    plot_fairreg_history,
    plot_pred_histograms_by_group,
    plot_pred_kde_by_group,
    plot_true_vs_pred,
    plot_tradeoff_curve,
)

from fair_unaware.baselines.fairreg_adapter import load_dataset, get_fairreg_objects

from fair_unaware.utils import (
    _parse_grid, tune_fairreg, _common_bins, plot_risk_unfairness_curve_annotated,
    compute_group_priors, write_runs_tables_tex, generate_custom_summary_tex
)
from fair_unaware.metrics import w2_gap_between_groups, tv_gap

import time 


warnings.filterwarnings("once", category=RuntimeWarning)
import logging
import warnings
from pathlib import Path

def redirect_warnings_to_file(warnings_path: Path):
    warnings_path = Path(warnings_path)
    warnings_path.parent.mkdir(parents=True, exist_ok=True)

    logging.captureWarnings(True)

    logger = logging.getLogger("py.warnings")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    logger.propagate = False  

    fh = logging.FileHandler(warnings_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    warnings.simplefilter("default")

def dbg_summary(name, A):
    A = np.asarray(A)
    print(
        f"{name}: shape={A.shape} finite={np.isfinite(A).mean():.2f} "
        f"maxabs={np.nanmax(np.abs(A)):.3e}"
    )

def check_finite(name, arr, *, print_ok=False):
    arr = np.asarray(arr)
    ok = np.isfinite(arr).all()
    if (not ok) or print_ok:
        mx = np.nanmax(np.abs(arr))
        print(f"[check] {name}: finite={ok} maxabs={mx:.3e}")
        if not ok:
            bad = np.where(~np.isfinite(arr))[0][:10]
            print(f"        first bad idx: {bad}")
    return ok


def make_run_dirs(exp_root: Path, dataset: str, exp_tag: str, cont_tag: str, run_id: int,
                  *, need_dirs: bool):
    """Create and return (run_dir, plots_dir, diag_dir, fairreg_dir)."""
    run_dir = exp_root / dataset / exp_tag / cont_tag / f"run_{run_id:02d}"
    plots_dir = run_dir / "plots"
    diag_dir = run_dir / "ot_diagnostics"
    fairreg_dir = run_dir / "fairreg"

    if need_dirs:
        for d in (plots_dir, diag_dir, fairreg_dir):
            d.mkdir(parents=True, exist_ok=True)

    return run_dir, plots_dir, diag_dir, fairreg_dir


def make_splits(X, S, y, *, train_size: float, test_size: float, seed: int):
    """
    EXACT same split logic you currently have.
    Returns a dict with all split arrays.
    """
    X_tmp, X_test, S_tmp, S_test, y_tmp, y_test = train_test_split(
        X, S, y,
        test_size=test_size,
        stratify=S,
        random_state=seed,
    )

    unlab_frac = 1.0 - train_size - test_size
    unlab_within = unlab_frac / (1.0 - test_size)

    X_train_full, X_unlab, S_train_full, S_unlab, y_train_full, y_unlab = train_test_split(
        X_tmp, S_tmp, y_tmp,
        test_size=unlab_within,
        stratify=S_tmp,
        random_state=seed,
    )


    return {
        "X_train_full": X_train_full,
        "S_train_full": S_train_full,
        "y_train_full": y_train_full,
        "X_unlab": X_unlab,
        "S_unlab": S_unlab,
        "y_unlab": y_unlab,
        "X_test": X_test,
        "S_test": S_test,
        "y_test": y_test,
    }

def _row_concat(A, B):
    """Concatenate rows for numpy arrays or pandas DataFrames/Series."""
    import numpy as np
    import pandas as pd

    # X: DataFrame
    if isinstance(A, pd.DataFrame) or isinstance(B, pd.DataFrame):
        return pd.concat([A, B], axis=0, ignore_index=True)

    # Series -> ndarray
    if isinstance(A, pd.Series):
        A = A.to_numpy()
    if isinstance(B, pd.Series):
        B = B.to_numpy()

    return np.concatenate([A, B], axis=0)


def maybe_make_tuning_split(split: dict, *, tune: bool, val_frac_train: float, dp_nbins: int, seed: int):
    """
    If tuning, split labeled TRAIN into fit/val and build dp_bins on val labels.
    Else return the full labeled train as fit and dp_bins=None.
    """
    X_train_full = split["X_train_full"]
    S_train_full = split["S_train_full"]
    y_train_full = split["y_train_full"]

    if tune:
        X_train_fit, X_train_val, S_train_fit, S_train_val, y_train_fit, y_train_val = train_test_split(
            X_train_full, S_train_full, y_train_full,
            test_size=val_frac_train,
            stratify=S_train_full,
            random_state=seed,
        )
        y_train_val_eval = np.asarray(y_train_val)
        dp_bins = _common_bins(y_train_val_eval, nbins=dp_nbins)
    else:
        X_train_fit, S_train_fit, y_train_fit = X_train_full, S_train_full, y_train_full
        X_train_val = S_train_val = y_train_val = None
        y_train_val_eval = None
        dp_bins = None

    return {
        "X_train_fit": X_train_fit,
        "S_train_fit": S_train_fit,
        "y_train_fit": y_train_fit,
        "X_train_val": X_train_val,
        "S_train_val": S_train_val,
        "y_train_val": y_train_val,
        "y_train_val_eval": y_train_val_eval,
        "dp_bins": dp_bins,
    }


def build_fairreg_candidate_grids(X_unlab, *, L_grid: str, beta_grid: str):
    """Your current L_auto/beta_auto + _parse_grid + unique logic."""
    N_unlab = len(X_unlab)
    L_auto = int(np.sqrt(N_unlab))
    beta_auto = 0.5 * np.sqrt(N_unlab) * np.log2(max(N_unlab, 2))

    L_list = _parse_grid(L_grid, L_auto, cast=float)
    L_list = [int(max(1, round(L))) for L in L_list]
    beta_list = _parse_grid(beta_grid, beta_auto, cast=float)
    beta_list = [float(b) for b in beta_list]

    L_list = list(dict.fromkeys(L_list))
    beta_list = list(dict.fromkeys(beta_list))
    return L_list, beta_list, L_auto, beta_auto


def fit_base_models(X_train_fit, S_train_fit, y_train_fit):
    """Fit reg+clf."""
    reg = LinearRegression(fit_intercept=True)
    clf = LogisticRegression(max_iter=2000)
    reg.fit(X_train_fit, y_train_fit)
    clf.fit(X_train_fit, S_train_fit)
    return reg, clf


def fairreg_group_params_from_clf(clf, S_train_full, *, eps: float):
    """
    Build (p_vec, K, eps_vec) exactly like your script.
    IMPORTANT: uses clf.classes_ ordering.
    """
    classes = clf.classes_
    p_vec = np.array([(np.asarray(S_train_full) == c).mean() for c in classes], dtype=float)
    p_vec = np.clip(p_vec, 1e-6, 1 - 1e-6)
    K = len(classes)
    eps_vec = [eps] * K
    return p_vec, K, eps_vec


def select_fairreg_hparams_or_auto(
    *,
    tune: bool,
    reg,
    clf,
    B: int,
    K: int,
    p_vec,
    eps_vec,
    T: int,
    X_unlab,
    tuning: dict,
    L_list,
    beta_list,
    FairReg,
    DP_unfairness,
    tune_objective: str,
    penalty_lambda: float,
    fairreg_dir: Path,
    dataset: str,
    run_id: int,
    L_arg,
    beta_arg,
):
    """
    Returns (best_L, best_beta, tuning_rows or None, row_sel dict).
    Also writes tuning CSV + curve if tune=True (exactly like before).
    """
    if tune:
        (best_L, best_beta), tuning_rows = tune_fairreg(
            reg, clf, B, K, p_vec, eps_vec, T,
            X_unlab_fit=X_unlab,
            X_val=tuning["X_train_val"],
            S_val=tuning["S_train_val"],
            y_val_eval=tuning["y_train_val_eval"],
            dp_bins=tuning["dp_bins"],
            L_list=L_list,
            beta_list=beta_list,
            FairRegClass=FairReg,
            dp_unfairness_fn=DP_unfairness,
            tune_objective=tune_objective,
            penalty_lambda=penalty_lambda,
        )

        pd.DataFrame(tuning_rows).to_csv(fairreg_dir / "fairreg_tuning.csv", index=False)

        plot_risk_unfairness_curve_annotated(
            tuning_rows,
            title=f"{dataset} run {run_id:02d} — FairReg tuning on train-val",
            savepath=str(fairreg_dir / "fairreg_tuning_curve_L_beta.png")
        )
    else:
        tuning_rows = None
        best_L = int(np.sqrt(len(X_unlab))) if str(L_arg).lower() == "auto" else int(float(L_arg))
        best_beta = (0.5 * np.sqrt(len(X_unlab)) * np.log2(max(len(X_unlab), 2))) if str(beta_arg).lower() == "auto" else float(beta_arg)

    row_sel = {"run": run_id, "best_L": int(best_L), "best_beta": float(best_beta)}

    if tune and tuning_rows is not None:
        best_val_risk = None
        best_val_unfair = None
        for r in tuning_rows:
            if r["L"] == int(best_L) and np.isclose(r["beta"], float(best_beta)):
                best_val_risk = float(r["risk"])
                best_val_unfair = float(r["unfair"])
                break
        row_sel["val_risk"] = best_val_risk
        row_sel["val_unfair"] = best_val_unfair

    return int(best_L), float(best_beta), tuning_rows, row_sel


def fit_fairreg_final(*, FairReg, reg, clf, B, K, p_vec, eps_vec, T, X_unlab, best_L, best_beta, keep_history: bool):
    fair_reg = FairReg(reg, clf, B=B, K=K, p=p_vec, eps=eps_vec, T=T, keep_history=keep_history)
    fair_reg.fit(X_unlab, beta=best_beta, L=best_L, alg={"base": "SGD3", "method": "ACSA2"})
    return fair_reg

def fit_frappe_model(*, X_train, S_train, y_train, X_unlab, S_unlab):
    frappe_model = FrappeRegressor(
        base_regressor=LinearRegression(), 
        hidden_dims=(64,), 
        lam=100.0, #high penalt
        epochs=100,
        lr=0.01
    )
    frappe_model.fit(X_train, S_train, y_train, X_unlab=X_unlab, S_unlab=S_unlab)
    return frappe_model

def fit_ot_model(*, X_train, S_train, y_train, X_unlab, priors, args, seed: int):
    ot_model = UnawareFairRegressor(
        base_regressor=LinearRegression(fit_intercept=True),
        base_classifier=LogisticRegression(max_iter=2000),
        fair_regressor=None,
        continuous_map=args.cont_map,
        continuous_epsilon=args.cont_epsilon,
        continuous_knn=args.cont_knn,
        sinkhorn_reg=args.sinkhorn_reg,
        delta_tol=args.delta_tol,
        random_state=seed,
        verbose=True,
        store_diagnostics=args.make_plots,
    )
    ot_model.fit(X_train, S_train, y_train, X_unlab=X_unlab, group_priors=priors)
    return ot_model


def fit_aware_model(*, X_train_full, S_train_full, y_train_full, X_unlab, S_unlab, priors, seed: int, estimate_S: bool = False, s_estimator=None):
    aware = AwareWassersteinBarycenterDPRegressor(
        base_regressor=LinearRegression(fit_intercept=True),
        random_state=seed,
        estimate_S=estimate_S,
        s_estimator=s_estimator,
    )
    aware.fit(
        X_train_full, S_train_full, y_train_full,
        X_unlab=X_unlab, S_unlab=S_unlab,
        group_priors=priors,
    )
    return aware

def aware_predict_with_shat(aware_model, X, s_estimator, *, mode="soft"):
    """
    Produce Aware-DP predictions when S is unknown, using an estimator of S|X.

    mode="hard": y = aware.predict(X, S_hat)
    mode="soft": y = sum_s P(S=s|X) * aware.predict(X, s)   (mixture of group maps)
    """
    X = np.asarray(X)
    n = X.shape[0]

    if mode == "hard":
        S_hat = s_estimator.predict(X)
        return aware_model.predict(X, S_hat)

    # soft mode
    proba = s_estimator.predict_proba(X)
    classes = s_estimator.classes_

    y = np.zeros(n, dtype=float)
    for j, c in enumerate(classes):
        S_c = np.full(n, c)
        y_c = aware_model.predict(X, S_c)
        y += proba[:, j] * y_c
    return y


def predict_inverse_scale_and_grid(
    *, reg_base, fair_reg, ot_model, aware, split: dict, scaler, dp_nbins: int,
    reg_base_full=None,           # NEW
    fair_reg_auto=None, frappe_model=None, aware_hat=None, s_hat_estimator=None, shat_mode="soft"
):
    X_test = split["X_test"]
    S_test = split["S_test"]
    y_test = split["y_test"]

    y_pred_base = reg_base.predict(X_test)
    y_pred_base_full = reg_base_full.predict(X_test) if reg_base_full is not None else None

    y_pred_fair = fair_reg.predict(X_test)
    y_pred_ot   = ot_model.predict(X_test)
    y_pred_aware= aware.predict(X_test, S_test)
    y_pred_frappe = frappe_model.predict(X_test) if frappe_model is not None else None

    if scaler is not None:
        y_test_eval = scaler.inverse_transform(y_test.values.reshape(-1, 1))[:, 0]
        y_pred_base_eval = scaler.inverse_transform(y_pred_base.reshape(-1, 1))[:, 0]
        y_pred_fair_eval = scaler.inverse_transform(y_pred_fair.reshape(-1, 1))[:, 0]
        y_pred_ot_eval   = scaler.inverse_transform(y_pred_ot.reshape(-1, 1))[:, 0]
        y_pred_aware_eval= scaler.inverse_transform(y_pred_aware.reshape(-1, 1))[:, 0]
        y_pred_base_full_eval = scaler.inverse_transform(y_pred_base_full.reshape(-1, 1))[:, 0] if y_pred_base_full is not None else None
        y_pred_frappe_eval = scaler.inverse_transform(y_pred_frappe.reshape(-1, 1))[:, 0] if y_pred_frappe is not None else None
    else:
        y_test_eval = np.asarray(y_test)
        y_pred_base_eval = y_pred_base
        y_pred_fair_eval = y_pred_fair
        y_pred_ot_eval   = y_pred_ot
        y_pred_aware_eval= y_pred_aware
        y_pred_base_full_eval = y_pred_base_full
        y_pred_frappe_eval = y_pred_frappe

    preds = {
        "ERM Unfair": y_pred_base_eval,          
        "FairReg": y_pred_fair_eval,
        "OT Unaware": y_pred_ot_eval,
        "OT Aware": y_pred_aware_eval,
    }
    if y_pred_base_full_eval is not None:
        preds["ERM Unfair (train+unlab)"] = y_pred_base_full_eval
        preds["Base (train+unlab)"] = y_pred_base_full_eval 

    if fair_reg_auto is not None:
        y_pred_fair_auto = fair_reg_auto.predict(X_test)
        y_pred_fair_auto_eval = scaler.inverse_transform(y_pred_fair_auto.reshape(-1,1))[:,0] if scaler else y_pred_fair_auto
        preds["FairReg (auto)"] = y_pred_fair_auto_eval
    
    if frappe_model is not None:
        preds["Frappe"] = y_pred_frappe_eval
        
    if aware_hat is not None and s_hat_estimator is not None:
        y_hat_hard = aware_predict_with_shat(aware_hat, X_test, s_hat_estimator, mode="hard")
        y_hat_hard = scaler.inverse_transform(y_hat_hard.reshape(-1, 1))[:, 0] if scaler else y_hat_hard
        preds["Aware DP (S_hat hard)"] = y_hat_hard

        y_hat_soft = aware_predict_with_shat(aware_hat, X_test, s_hat_estimator, mode="soft")
        y_hat_soft = scaler.inverse_transform(y_hat_soft.reshape(-1, 1))[:, 0] if scaler else y_hat_soft
        preds["Aware DP (S_hat soft)"] = y_hat_soft

    grid_data = [y_test_eval, y_pred_base_eval, y_pred_fair_eval, y_pred_ot_eval, y_pred_aware_eval]
    if "Base (train+unlab)" in preds: grid_data.append(preds["Base (train+unlab)"])
    if "FairReg (auto)" in preds: grid_data.append(preds["FairReg (auto)"])
    if "Frappe" in preds: grid_data.append(preds["Frappe"])
    if "Aware DP (S_hat hard)" in preds: grid_data.append(preds["Aware DP (S_hat hard)"])
    if "Aware DP (S_hat soft)" in preds: grid_data.append(preds["Aware DP (S_hat soft)"])

    grid = _common_bins(np.concatenate(grid_data), nbins=dp_nbins)
    return y_test_eval, preds, grid

def predict_unlab_inverse_scale_and_grid(*, reg, fair_reg, ot_model, aware, split: dict, scaler, dp_nbins: int, fair_reg_auto=None, frappe_model=None, aware_hat=None, s_hat_estimator=None, shat_mode="soft"):
    X_unlab = split["X_unlab"]
    S_unlab = split["S_unlab"]
    y_unlab = split["y_unlab"]

    y_pred_base = reg.predict(X_unlab)
    y_pred_fair = fair_reg.predict(X_unlab)
    y_pred_ot = ot_model.predict(X_unlab)
    y_pred_aware = aware.predict(X_unlab, S_unlab)
    y_pred_frappe = frappe_model.predict(X_unlab) if frappe_model is not None else None

    if scaler is not None:
        y_unlab_eval = scaler.inverse_transform(np.asarray(y_unlab).reshape(-1, 1))[:, 0]
        y_pred_base_eval = scaler.inverse_transform(y_pred_base.reshape(-1, 1))[:, 0]
        y_pred_fair_eval = scaler.inverse_transform(y_pred_fair.reshape(-1, 1))[:, 0]
        y_pred_ot_eval = scaler.inverse_transform(y_pred_ot.reshape(-1, 1))[:, 0]
        y_pred_aware_eval = scaler.inverse_transform(y_pred_aware.reshape(-1, 1))[:, 0]
        y_pred_frappe_eval = scaler.inverse_transform(y_pred_frappe.reshape(-1, 1))[:, 0] if y_pred_frappe is not None else None
    else:
        y_unlab_eval = np.asarray(y_unlab)
        y_pred_base_eval = y_pred_base
        y_pred_fair_eval = y_pred_fair
        y_pred_ot_eval = y_pred_ot
        y_pred_aware_eval = y_pred_aware
        y_pred_frappe_eval = y_pred_frappe

    preds_unlab = {
        "ERM Unfair": y_pred_base_eval,
        "FairReg": y_pred_fair_eval,
        "OT Unaware": y_pred_ot_eval,
        "OT Aware": y_pred_aware_eval,
    }

    if fair_reg_auto is not None:
        y_pred_fair_auto = fair_reg_auto.predict(X_unlab)
        y_pred_fair_auto_eval = scaler.inverse_transform(y_pred_fair_auto.reshape(-1,1))[:,0] if scaler else y_pred_fair_auto
        preds_unlab["FairReg (auto)"] = y_pred_fair_auto_eval
        
    if frappe_model is not None:
        preds_unlab["Frappe"] = y_pred_frappe_eval

    if aware_hat is not None and s_hat_estimator is not None:
        y_hat_hard = aware_predict_with_shat(aware_hat, X_unlab, s_hat_estimator, mode="hard")
        y_hat_hard = scaler.inverse_transform(y_hat_hard.reshape(-1, 1))[:, 0] if scaler else y_hat_hard
        preds_unlab["Aware DP (S_hat hard)"] = y_hat_hard

        y_hat_soft = aware_predict_with_shat(aware_hat, X_unlab, s_hat_estimator, mode="soft")
        y_hat_soft = scaler.inverse_transform(y_hat_soft.reshape(-1, 1))[:, 0] if scaler else y_hat_soft
        preds_unlab["Aware DP (S_hat soft)"] = y_hat_soft

    grid_data = [y_unlab_eval, y_pred_base_eval, y_pred_fair_eval, y_pred_ot_eval, y_pred_aware_eval]
    if "FairReg (auto)" in preds_unlab: grid_data.append(preds_unlab["FairReg (auto)"])
    if "Frappe" in preds_unlab: grid_data.append(preds_unlab["Frappe"])
    if "Aware DP (S_hat hard)" in preds_unlab: grid_data.append(preds_unlab["Aware DP (S_hat hard)"])
    if "Aware DP (S_hat soft)" in preds_unlab: grid_data.append(preds_unlab["Aware DP (S_hat soft)"])

    grid_unlab = _common_bins(np.concatenate(grid_data), nbins=dp_nbins)
    return y_unlab_eval, preds_unlab, grid_unlab

def compute_dp_ks_only(*, preds: dict, S, grid, DP_unfairness, prefix: str):
    S_arr = np.asarray(S).ravel()
    groups = np.unique(S_arr)
    g0, g1 = groups[0], groups[1]

    def ks_gap(yhat):
        yhat = np.asarray(yhat).ravel()
        return float(ks_2samp(yhat[S_arr == g0], yhat[S_arr == g1]).statistic)

    methods = {
            "base": "ERM Unfair",
            "fairreg": "FairReg",
            "ot": "OT Unaware",
            "aware": "OT Aware",
        }
    if "FairReg (auto)" in preds: methods["fairreg_auto"] = "FairReg (auto)"
    if "Frappe" in preds: methods["frappe"] = "Frappe"
    if "Aware DP (S_hat hard)" in preds: methods["aware_hat_hard"] = "Aware DP (S_hat hard)"
    if "Aware DP (S_hat soft)" in preds: methods["aware_hat_soft"] = "Aware DP (S_hat soft)"

    out = {}
    for out_name, pred_key in methods.items():
            dp = DP_unfairness(preds[pred_key], S, bins=grid)
            out[f"dp_{out_name}_{prefix}_max"] = float(max(dp.values()))
            out[f"w2_{out_name}_{prefix}"] = float(w2_gap_between_groups(preds[pred_key], S_arr, squared=False))
            out[f"ks_{out_name}_{prefix}"] = float(ks_gap(preds[pred_key]))
            out[f"tv_{out_name}_{prefix}"] = float(tv_gap(preds[pred_key], S_arr, bins=grid))

    return out

def compute_metrics(*, y_test_eval, preds: dict, S_test, grid, DP_unfairness):
    out = {}

    out["mse_base"] = mse(y_test_eval, preds["ERM Unfair"])
    if "ERM Unfair (train+unlab)" in preds: out["mse_base_full"] = mse(y_test_eval, preds["ERM Unfair (train+unlab)"])
    out["mse_fairreg"] = mse(y_test_eval, preds["FairReg"])
    out["mse_ot"] = mse(y_test_eval, preds["OT Unaware"])
    out["mse_aware"] = mse(y_test_eval, preds["OT Aware"])
    if "Frappe" in preds: out["mse_frappe"] = mse(y_test_eval, preds["Frappe"])

    dp_base = DP_unfairness(preds["ERM Unfair"], S_test, bins=grid)
    out["dp_base_max"] = float(max(dp_base.values()))
    if "ERM Unfair (train+unlab)" in preds:
        dp_base_full = DP_unfairness(preds["ERM Unfair (train+unlab)"], S_test, bins=grid)
        out["dp_base_full_max"] = float(max(dp_base_full.values()))
        
    dp_fair = DP_unfairness(preds["FairReg"], S_test, bins=grid)
    dp_ot   = DP_unfairness(preds["OT Unaware"], S_test, bins=grid)
    dp_aware= DP_unfairness(preds["OT Aware"], S_test, bins=grid)

    out["dp_fairreg_max"] = float(max(dp_fair.values()))
    out["dp_ot_max"] = float(max(dp_ot.values()))
    out["dp_aware_max"] = float(max(dp_aware.values()))
    if "Frappe" in preds: out["dp_frappe_max"] = float(max(DP_unfairness(preds["Frappe"], S_test, bins=grid).values()))

    S_test_arr = np.asarray(S_test).ravel()
    groups = np.unique(S_test_arr)
    g0, g1 = groups[0], groups[1]

    def ks_gap(yhat):
        yhat = np.asarray(yhat).ravel()
        return float(ks_2samp(yhat[S_test_arr == g0], yhat[S_test_arr == g1]).statistic)

    out["ks_base"] = ks_gap(preds["ERM Unfair"])
    if "ERM Unfair (train+unlab)" in preds: out["ks_base_full"] = ks_gap(preds["ERM Unfair (train+unlab)"])
    out["ks_fairreg"] = ks_gap(preds["FairReg"])
    out["ks_ot"] = ks_gap(preds["OT Unaware"])
    out["ks_aware"] = ks_gap(preds["OT Aware"])
    if "Frappe" in preds: out["ks_frappe"] = ks_gap(preds["Frappe"])

    out["w2_base"] = float(w2_gap_between_groups(preds["ERM Unfair"], S_test_arr, squared=False))
    out["tv_base"] = float(tv_gap(preds["ERM Unfair"], S_test_arr, bins=grid))
    if "ERM Unfair (train+unlab)" in preds:
        out["w2_base_full"] = float(w2_gap_between_groups(preds["ERM Unfair (train+unlab)"], S_test_arr, squared=False))
        out["tv_base_full"] = float(tv_gap(preds["ERM Unfair (train+unlab)"], S_test_arr, bins=grid))
    out["w2_fairreg"] = float(w2_gap_between_groups(preds["FairReg"], S_test_arr, squared=False))
    out["tv_fairreg"] = float(tv_gap(preds["FairReg"], S_test_arr, bins=grid))
    out["w2_ot"] = float(w2_gap_between_groups(preds["OT Unaware"], S_test_arr, squared=False))
    out["tv_ot"] = float(tv_gap(preds["OT Unaware"], S_test_arr, bins=grid))
    out["w2_aware"] = float(w2_gap_between_groups(preds["OT Aware"], S_test_arr, squared=False))
    out["tv_aware"] = float(tv_gap(preds["OT Aware"], S_test_arr, bins=grid))
    
    if "Frappe" in preds:
        out["w2_frappe"] = float(w2_gap_between_groups(preds["Frappe"], S_test_arr, squared=False))
        out["tv_frappe"] = float(tv_gap(preds["Frappe"], S_test_arr, bins=grid))

    if "FairReg (auto)" in preds:
        out["mse_fairreg_auto"] = float(mse(y_test_eval, preds["FairReg (auto)"]))
        dp_fair_auto = DP_unfairness(preds["FairReg (auto)"], S_test, bins=grid)
        out["dp_fairreg_auto_max"] = float(max(dp_fair_auto.values()))
        out["ks_fairreg_auto"] = float(ks_gap(preds["FairReg (auto)"]))
        out["w2_fairreg_auto"] = float(w2_gap_between_groups(preds["FairReg (auto)"], S_test_arr, squared=False))
        out["tv_fairreg_auto"] = float(tv_gap(preds["FairReg (auto)"], S_test_arr, bins=grid))

    if "Aware DP (S_hat hard)" in preds:
        out["mse_aware_hat_hard"] = mse(y_test_eval, preds["Aware DP (S_hat hard)"])
        out["dp_aware_hat_hard_max"] = float(max(DP_unfairness(preds["Aware DP (S_hat hard)"], S_test, bins=grid).values()))
        out["ks_aware_hat_hard"] = float(ks_gap(preds["Aware DP (S_hat hard)"]))
        out["w2_aware_hat_hard"] = float(w2_gap_between_groups(preds["Aware DP (S_hat hard)"], S_test_arr, squared=False))
        out["tv_aware_hat_hard"] = float(tv_gap(preds["Aware DP (S_hat hard)"], S_test_arr, bins=grid))

    if "Aware DP (S_hat soft)" in preds:
        out["mse_aware_hat_soft"] = mse(y_test_eval, preds["Aware DP (S_hat soft)"])
        out["dp_aware_hat_soft_max"] = float(max(DP_unfairness(preds["Aware DP (S_hat soft)"], S_test, bins=grid).values()))
        out["ks_aware_hat_soft"] = float(ks_gap(preds["Aware DP (S_hat soft)"]))
        out["w2_aware_hat_soft"] = float(w2_gap_between_groups(preds["Aware DP (S_hat soft)"], S_test_arr, squared=False))
        out["tv_aware_hat_soft"] = float(tv_gap(preds["Aware DP (S_hat soft)"], S_test_arr, bins=grid))

    return out

def maybe_make_plots_and_diags(*, args, preds, y_test_eval, split, grid, plots_dir, fairreg_dir, diag_dir,
                               fair_reg, ot_model, scaler):
    if not args.make_plots:
        return

    S_test = split["S_test"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    plot_pred_histograms_by_group(
        preds, S_test, bins=grid,
        title=f"{args.dataset} — predictions by group",
        savepath=str(plots_dir / "pred_hist_by_group.png"),
    )

    plot_pred_kde_by_group(
        preds, S_test,
        title=f"{args.dataset} — KDE by group",
        savepath=str(plots_dir / "pred_kde_by_group.png"),
    )

    plot_true_vs_pred(
        y_test_eval, preds,
        title=f"{args.dataset} — y_true vs y_pred",
        savepath=str(plots_dir / "true_vs_pred.png"),
    )

    plot_fairreg_history(
        fair_reg, X_test, S_test, y_test,
        data_scaling=(scaler is not None),
        scaler=scaler,
        moving_risk=100,
        moving_unf=50,
        title_prefix=f"{args.dataset} — ",
        savepath=str(fairreg_dir / "fairreg_history.png"),
    )

    # OT diagnostics on unlabeled 
    diag_prefix = diag_dir / "ot"
    plot_basic_diagnostics(
        ot_model,
        split["X_unlab"],
        split["S_unlab"],
        split["y_unlab"],
        show=False,
        save_prefix=str(diag_prefix),
        save_suffix="pdf",
        with_ot_matrices=True,
        save_transport=True,
    )


def maybe_make_tradeoff(*, args, FairReg, reg, clf, B, K, p_vec, best_beta, best_L, X_unlab, X_test,
                        y_test_eval, S_test, grid, y_pred_base_eval, y_pred_ot_eval, plots_dir):
    if not (args.make_tradeoff and args.make_plots):
        return

    eps_list = [float(x) for x in args.eps_list.split(",")]
    lambda_list = [float(x) for x in args.lambda_list.split(",")]

    fairreg_points = []
    for eps0 in eps_list:
        eps_vec = [eps0, eps0]
        fr = FairReg(reg, clf, B=B, K=K, p=p_vec, eps=eps_vec, T=args.T, keep_history=args.make_plots)
        fr.fit(X_unlab, beta=best_beta, L=best_L, alg={"base": "SGD3", "method": "ACSA2"})
        y_fr = fr.predict(X_test)
        # NOTE (be careful): y_fr is already on a corrected scale if scaler=None; if we want someting scaled, it must be scaled BEFORE this function.
        dp = DP_unfairness(y_fr, S_test, bins=grid)
        unfair = max(dp.values())
        risk = float(np.mean((y_test_eval - y_fr) ** 2))
        fairreg_points.append((unfair, risk))

    ot_points = []
    for lam in lambda_list:
        y_mix = (1 - lam) * y_pred_base_eval + lam * y_pred_ot_eval
        dp = DP_unfairness(y_mix, S_test, bins=grid)
        unfair = max(dp.values())
        risk = float(np.mean((y_test_eval - y_mix) ** 2))
        ot_points.append((unfair, risk))

    plot_tradeoff_curve(
        {"FairReg (eps sweep)": fairreg_points, "OT (lambda sweep)": ot_points},
        title=f"{args.dataset} — MSE vs unfairness tradeoff",
        savepath=str(plots_dir / "tradeoff_curve.png"),
    )

def main(args):
    FairReg, DP_unfairness = get_fairreg_objects(args.fairreg_root)

    fig_root = Path(args.out_dir)
    exp_root = fig_root if fig_root.name == "compare_ot_vs_fairreg" else fig_root / "compare_ot_vs_fairreg"


    if args.dataset == "synthetic":
        exp_tag = f"synthetic_n{args.synthetic_n}"
        X, S, y = make_synthetic_dataset(
            n=args.synthetic_n,
            random_state=args.synthetic_seed,
        )
        
        # SCLAING for fairreg 
        # scale Y between -1 et 1 so that beta_auto and the softmax do not crash 
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        y = pd.Series(y_scaled)

        # fix B to 1 accordin ti the scaling 
        B = 1.0 

        uniq = np.unique(S)
        K = int(len(uniq))
        p = np.array([(S == u).mean() for u in uniq], dtype=float)
        priors = {int(u): float((S == u).mean()) for u in uniq}

    else:
        exp_tag = "full" if args.full_data else "subsample_2000"
        X, S, y, scaler, B, K, p, priors = load_dataset(
            args.dataset,
            fairreg_root=args.fairreg_root,
            data_scaling=not args.no_scaling,
            full_data=args.full_data,
        )
    if not hasattr(S, "unique"):
        S = pd.Series(np.asarray(S).ravel(), name="S")

    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    # UNLAB = (1 - TRAIN - TEST) (implicitly done when splitting the daa)
    print(f"Using dataset={args.dataset} with train_size={TRAIN_SIZE}, test_size={TEST_SIZE}, full_data={args.full_data}")



    cont_tag = str(args.cont_map)

    summary_dir = exp_root / args.dataset / exp_tag / cont_tag
    summary_dir.mkdir(parents=True, exist_ok=True)

    redirect_warnings_to_file(summary_dir / "warnings.log")
    need_dirs = args.make_plots or args.tune_fairreg or args.make_tradeoff

    if args.load_results is not None and Path(args.load_results).exists():
            print(f"Loading existing results from {args.load_results}...")
            df = pd.read_csv(args.load_results)
            fairreg_selected = []


            if args.make_plots:
                run_dirs = sorted(Path(args.load_results).parent.glob("run_*"))
                
                if args.plot_only_run is not None:
                    target = f"run_{args.plot_only_run:02d}"
                    run_dirs = [d for d in run_dirs if d.name == target]
                    print(f"Skipping other runs. Only regenerating plots for: {target}")

                if not run_dirs:
                    print("Aucun dossier run_XX trouvé pour régénérer les graphes.")
                    
                for run_path in tqdm(run_dirs, desc="Regenerating Plots"):
                    plots_dir = run_path / "plots"
                    fairreg_dir = run_path / "fairreg"
                    diag_dir = run_path / "ot_diagnostics"
                    
                    # 1. Diagnostics & KDE
                    plot_data_file = run_path / "plot_data.joblib"
                    if plot_data_file.exists():
                        data = joblib.load(plot_data_file)

                        rename_map = {
                            "Base (train)": "ERM Unfair",
                            "Base (train+unlab)": "ERM Unfair (train+unlab)",
                            "OT": "OT Unaware",
                            "Aware DP": "OT Aware",
                            "Aware DP (S_hat hard)": r"OT $\hat{S}$ hard",
                            "Aware DP (S_hat soft)": r"OT $\hat{S}$ soft"
                        }
                        
                        anciennes_preds = data["preds"]
                        nouvelles_preds = {}
                        for k, v in anciennes_preds.items():
                            nouveau_nom = rename_map.get(k, k) 
                            nouvelles_preds[nouveau_nom] = v
                        data["preds"] = nouvelles_preds


                        df_preds = pd.DataFrame(data["preds"])
                        df_preds["y_true"] = data["y_test_eval"]

                        S_test = data["split"]["S_test"]
                        df_preds["S"] = S_test.values if hasattr(S_test, "values") else np.asarray(S_test)
                        
                        csv_pred_path = plots_dir / "predictions.csv"
                        df_preds.to_csv(csv_pred_path, index=False)


                        maybe_make_plots_and_diags(
                            args=args,
                            preds=data["preds"],
                            y_test_eval=data["y_test_eval"],
                            split=data["split"],
                            grid=data["grid"],
                            plots_dir=plots_dir,
                            fairreg_dir=fairreg_dir,
                            diag_dir=diag_dir,
                            fair_reg=data["fair_reg"],
                            ot_model=data["ot_model"],
                            scaler=data["scaler"]
                        )
            
                    # 2. Tradeoffs 
                    tradeoff_file = run_path / "tradeoff_data.joblib"
                    if args.make_tradeoff and tradeoff_file.exists():
                        td = joblib.load(tradeoff_file)
                        plot_tradeoff_curve(
                            {"FairReg (eps sweep)": td["fairreg_points"], "OT (lambda sweep)": td["ot_points"]},
                            title=f"{args.dataset} — MSE vs unfairness tradeoff",
                            savepath=str(plots_dir / "tradeoff_curve.png"),
                        )
    else: 
        results = []
        fairreg_selected = [] 

        for i in tqdm(range(1, args.num + 1), desc="Runs"):
            split = make_splits(X, S, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=i)

            # convenience 
            X_train_full = split["X_train_full"]
            S_train_full = split["S_train_full"]
            y_train_full = split["y_train_full"]
            X_unlab = split["X_unlab"]
            S_unlab = split["S_unlab"]
            y_unlab = split["y_unlab"]
            X_test = split["X_test"]
            S_test = split["S_test"]
            y_test = split["y_test"]


            priors_train = compute_group_priors(S_train_full)  


            # directories
            _, plots_dir, diag_dir, fairreg_dir = make_run_dirs(
                exp_root, args.dataset, exp_tag, cont_tag, i, need_dirs=need_dirs
            )

            # Tuning split 
            tuning = maybe_make_tuning_split(
                split,
                tune=args.tune_fairreg,
                val_frac_train=args.val_frac_train,
                dp_nbins=args.dp_nbins,
                seed=i,
            )

            #candidate gridds (may be upgraded but i don't know yet how)
            L_list, beta_list, L_auto, beta_auto = build_fairreg_candidate_grids(
                X_unlab, L_grid=args.L_grid, beta_grid=args.beta_grid
            )

            #base models for FairReg tuning
            reg_fr, clf_fr = fit_base_models(
                tuning["X_train_fit"], tuning["S_train_fit"], tuning["y_train_fit"]
            )


            if args.debug:
                check_finite("X_train_full", X_train_full)
                check_finite("X_unlab", X_unlab)
                check_finite("X_test", X_test)
                check_finite("y_train_full", y_train_full)
                check_finite("y_unlab", y_unlab)
                check_finite("y_test", y_test)
                dbg_summary("reg.coef_", reg_fr.coef_)
                dbg_summary("clf.coef_", clf_fr.coef_)


            #FairReg group params
            p_vec, K, eps_vec = fairreg_group_params_from_clf(
        clf_fr, S_train_full, eps=args.eps
    )
            
            # dynamical T turns 60 epochs 
            if args.T is not None:
                T_final = args.T
            else:
                T_final = 60 * len(X_unlab)
                if i == 1:
                    print(f"--- Info: T non spécifié. Utilisation de T={T_final} (60 époques) ---")

            #-------------------------------
            # SWEEP FAIRREG T (CONVERGENCE STUDY)
            if args.sweep_fairreg_T is not None or args.load_sweep_csv is not None:
                print(f"\n--- FairReg Convergence Study ---")
                
                

                if args.load_sweep_csv is not None:
                    print(f"Loading existing data from {args.load_sweep_csv}...")
                    df_conv = pd.read_csv(args.load_sweep_csv)
                    conv_csv_path = args.load_sweep_csv
                    

                else:
                    if args.sweep_fairreg_T.lower() == "auto":
                        epochs_list = list(range(1, 61)) # 1 à 60 époques
                        T_values = [e * len(X_unlab) for e in epochs_list]
                        test_beta = float(beta_auto)
                        test_L = int(L_auto)
                    else:
                        T_values = [int(t.strip()) for t in args.sweep_fairreg_T.split(",")]
                        test_beta = float(best_beta)
                        test_L = int(best_L)

                    convergence_results = []

                    y_base_test = reg_fr.predict(X_test)
                    if scaler is not None:
                        y_base_test_eval = scaler.inverse_transform(y_base_test.reshape(-1, 1))[:, 0]
                        y_test_eval_local = scaler.inverse_transform(y_test.values.reshape(-1, 1))[:, 0]
                    else:
                        y_base_test_eval = y_base_test
                        y_test_eval_local = np.asarray(y_test)
                        
                    global_grid = _common_bins(np.concatenate([y_test_eval_local, y_base_test_eval]), nbins=args.dp_nbins)

                    mse_val_0 = float(np.mean((y_test_eval_local - y_base_test_eval) ** 2))
                    dp_dict_0 = DP_unfairness(y_base_test_eval, S_test, bins=global_grid) 
                    dp_max_0 = float(max(dp_dict_0.values()))
                    S_test_arr = np.asarray(S_test).ravel()
                    w2_val_0 = float(w2_gap_between_groups(y_base_test_eval, S_test_arr, squared=False))
                    
                    convergence_results.append({
                        "Epoch": 0.0, "T": 0, "MSE": mse_val_0, "DP_max": dp_max_0, "W2": w2_val_0, "Time_sec": 0.0
                    })
                    print(f"  -> Result: Epoch=0, T=0, MSE={mse_val_0:.3f}, DP={dp_max_0:.3f}, W2={w2_val_0:.3f}, Time=0.0s")
                    
                    for current_T in T_values:
                        current_epoch = current_T / len(X_unlab)
                        print(f"Training FairReg with T = {current_T} (~{current_epoch:.0f} epochs)...")
                        start_t = time.time()
                        
                        # --- refit models for each T 
                        local_reg, local_clf = fit_base_models(tuning["X_train_fit"], tuning["S_train_fit"], tuning["y_train_fit"])
                        local_p_vec, local_K, local_eps_vec = fairreg_group_params_from_clf(local_clf, S_train_full, eps=args.eps)
                        
                        fr_sweep = FairReg(local_reg, local_clf, B=B, K=local_K, p=local_p_vec, eps=local_eps_vec, T=current_T, keep_history=False)
                        fr_sweep.fit(X_unlab, beta=test_beta, L=test_L, alg={"base": "SGD3", "method": "ACSA2"})
                        fit_time = time.time() - start_t
                        

                        y_pred_test = fr_sweep.predict(X_test)
                        

                        if scaler is not None:
                            y_pred_test_eval = scaler.inverse_transform(y_pred_test.reshape(-1, 1))[:, 0]
                        else:
                            y_pred_test_eval = y_pred_test

                        # 1. MSE
                        mse_val = float(np.mean((y_test_eval_local - y_pred_test_eval) ** 2))
                        
                        # 2. KS global grid 
                        dp_dict = DP_unfairness(y_pred_test_eval, S_test, bins=global_grid) 
                        dp_max = float(max(dp_dict.values()))
                        
                        # 3. W2
                        S_test_arr = np.asarray(S_test).ravel()
                        w2_val = float(w2_gap_between_groups(y_pred_test_eval, S_test_arr, squared=False))
                        
                        convergence_results.append({
                            "Epoch": current_epoch, "T": current_T, "MSE": mse_val, "DP_max": dp_max, "W2": w2_val, "Time_sec": fit_time
                        })
                        print(f"  -> Result: Epoch={current_epoch:.0f}, T={current_T}, MSE={mse_val:.3f}, DP={dp_max:.3f}, W2={w2_val:.3f}, Time={fit_time:.1f}s")
                        

                    df_conv = pd.DataFrame(convergence_results)
                    df_conv["run"] = i  
                    conv_csv_path = summary_dir / f"fairreg_convergence_{args.dataset}.csv"
                    
                    df_conv.to_csv(conv_csv_path, mode='w' if i == 1 else 'a', header=(i == 1), index=False)
                    print(f"Saved run {i} data to {conv_csv_path}")

                if i == args.num:
                    plot_path = summary_dir / f"fairreg_convergence_{args.dataset}.pdf"
                    plot_fairreg_convergence(pd.read_csv(conv_csv_path), args.dataset, plot_path)
                    print(f"Convergence study finished with {args.num} runs! Plot saved to {plot_path}")
                    sys.exit(0)
                else:
                    continue


            # 7)select FairReg hyperparams 
            best_L, best_beta, tuning_rows, row_sel = select_fairreg_hparams_or_auto(
                tune=args.tune_fairreg,
                reg=reg_fr,
                clf=clf_fr,
                B=B,
                K=K,
                p_vec=p_vec,
                eps_vec=eps_vec,
                T=T_final,
                X_unlab=X_unlab,
                tuning=tuning,
                L_list=L_list,
                beta_list=beta_list,
                FairReg=FairReg,
                DP_unfairness=DP_unfairness,
                tune_objective=args.tune_objective,
                penalty_lambda=args.penalty_lambda,
                fairreg_dir=fairreg_dir,
                dataset=args.dataset,
                run_id=i,
                L_arg=args.L,
                beta_arg=args.beta,
            )
            fairreg_selected.append(row_sel)

            #to keep the auto values too
            row_sel["auto_L"] = int(L_auto)
            row_sel["auto_beta"] = float(beta_auto)


            #refit on full TRAIN after tuning (exactly like before)
            t_fr_base = time.time()
            reg_fr.fit(X_train_full, y_train_full)
            clf_fr.fit(X_train_full, S_train_full)
            p_vec, K, eps_vec = fairreg_group_params_from_clf(
                clf_fr, S_train_full, eps=args.eps
            )
            time_fairreg_base = time.time() - t_fr_base

            #### fit non fairreg models to the whole train+unlab 
            if args.merge_train_unlab_for_non_fairreg:
                X_train_nf = _row_concat(X_train_full, X_unlab)
                S_train_nf = _row_concat(S_train_full, S_unlab)
                y_train_nf = _row_concat(y_train_full, y_unlab)

                priors_nf = compute_group_priors(S_train_nf)

                # For OT/Aware 
                X_unlab_nf = X_train_nf
                S_unlab_nf = S_train_nf
            else:
                X_train_nf, S_train_nf, y_train_nf = X_train_full, S_train_full, y_train_full
                priors_nf = compute_group_priors(S_train_full)
                X_unlab_nf = X_unlab
                S_unlab_nf = S_unlab

            if args.merge_train_unlab_for_non_fairreg:
                reg_nf, clf_nf = fit_base_models(X_train_nf, S_train_nf, y_train_nf)
            else:
                # backward compatibilty for the non refit version (but with the same train+unlab data if flag on) may be removed laterr 
                reg_nf, clf_nf = reg_fr, clf_fr
            ####    


            # 9) Fit FairReg / OT / Aware
            t0 = time.time()
            fair_reg = fit_fairreg_final(
                FairReg=FairReg, reg=reg_fr, clf=clf_fr, B=B, K=K, p_vec=p_vec,
                eps_vec=eps_vec, T=T_final, X_unlab=X_unlab, best_L=best_L,
                best_beta=best_beta, keep_history=args.make_plots,
            )
            time_fairreg = time_fairreg_base + (time.time() - t0)

            # accuracy of S-hat classifier used for Aware(S_hat) 
            S_test_arr  = np.asarray(S_test).ravel()
            S_unlab_arr = np.asarray(S_unlab).ravel()

            acc_shat_test  = accuracy_score(S_test_arr,  clf_fr.predict(X_test))
            acc_shat_unlab = accuracy_score(S_unlab_arr, clf_fr.predict(X_unlab))

            #fairreg tuning 
            fair_reg_auto = None
            if args.tune_fairreg:
                fair_reg_auto = fit_fairreg_final(
                    FairReg=FairReg,
                    reg=reg_fr, clf=clf_fr, B=B, K=K, p_vec=p_vec, eps_vec=eps_vec, 
                    T=T_final,
                    X_unlab=X_unlab,
                    best_L=L_auto,
                    best_beta=beta_auto,
                    keep_history=False,  
                )

            t0 = time.time()
            frappe_model = fit_frappe_model(
                X_train=X_train_nf, S_train=S_train_nf, y_train=y_train_nf,
                X_unlab=X_unlab_nf, S_unlab=S_unlab_nf
            )
            time_frappe = time.time() - t0


            t0 = time.time()
            ot_model = fit_ot_model(
            X_train=X_train_nf,
            S_train=S_train_nf,
            y_train=y_train_nf,
            X_unlab=X_unlab_nf,          
            priors=priors_nf,
            args=args,
            seed=i,
        )
            time_ot = time.time() - t0


            # accuracy of OT's internal classifier (used for delta / P(S \vert X) inside OT) 
            ot_clf = getattr(ot_model, "base_classifier_", None)
            if ot_clf is None:
                acc_shat_ot_test  = np.nan
                acc_shat_ot_unlab = np.nan
            else:
                acc_shat_ot_test  = accuracy_score(S_test_arr,  ot_clf.predict(X_test))
                acc_shat_ot_unlab = accuracy_score(S_unlab_arr, ot_clf.predict(X_unlab))

            t0 = time.time()

            aware = fit_aware_model(
            X_train_full=X_train_nf,
            S_train_full=S_train_nf,
            y_train_full=y_train_nf,
            X_unlab=X_unlab_nf,          
            S_unlab=S_unlab_nf,
            priors=priors_nf,
            seed=i,
        )
            time_aware = time.time() - t0

            t0 = time.time()

            aware_hat = fit_aware_model(
                X_train_full=X_train_nf,
                S_train_full=S_train_nf,
                y_train_full=y_train_nf,
                X_unlab=X_unlab,
                S_unlab=None,
                priors=None,
                seed=i,
                estimate_S=True,
                s_estimator=clf_nf,
            )
            time_aware_hat = time.time() - t0

            reg_base_full = reg_nf if args.merge_train_unlab_for_non_fairreg else None

            y_test_eval, preds, grid = predict_inverse_scale_and_grid(
                reg_base=reg_fr,                 
                reg_base_full=reg_base_full,     
                fair_reg=fair_reg,
                fair_reg_auto=fair_reg_auto,
                frappe_model=frappe_model,
                ot_model=ot_model,
                aware=aware,
                aware_hat=aware_hat,
                s_hat_estimator=clf_nf,
                shat_mode="soft",
                split=split,
                scaler=scaler,
                dp_nbins=args.dp_nbins,
            )

            # UNLAB fairness metrics (only when we keep a separate unlabeled pool) may be removed latedr since not relevant to our analyss
            if not args.merge_train_unlab_for_non_fairreg:
                y_unlab_eval, preds_unlab, grid_unlab = predict_unlab_inverse_scale_and_grid(
                    reg=reg_nf,
                    fair_reg=fair_reg,
                    fair_reg_auto=fair_reg_auto,
                    frappe_model=frappe_model,
                    ot_model=ot_model,
                    aware=aware,
                    aware_hat=aware_hat,
                    s_hat_estimator=clf_nf,
                    shat_mode="soft",
                    split=split,
                    scaler=scaler,
                    dp_nbins=args.dp_nbins,
                )

                unlab_fair = compute_dp_ks_only(
                    preds=preds_unlab,
                    S=S_unlab,
                    grid=grid_unlab,
                    DP_unfairness=DP_unfairness,
                    prefix="unlab",
                )
            else:
                unlab_fair = None

#============ SAVE ======
            run_dir = plots_dir.parent
            
            joblib.dump({
                "split": split,
                "fair_reg": fair_reg,
                "ot_model": ot_model,
                "frappe_model": frappe_model,
                "scaler": scaler,
                "preds": preds,
                "y_test_eval": y_test_eval,
                "grid": grid
            }, run_dir / "plot_data.joblib")
            

            if args.make_tradeoff and args.make_plots:
                joblib.dump({
                    "fairreg_points": fairreg_points,
                    "ot_points": ot_points
                }, run_dir / "tradeoff_data.joblib")



            #Plots + OT diags
            maybe_make_plots_and_diags(
                args=args,
                preds=preds,
                y_test_eval=y_test_eval,
                split=split,
                grid=grid,
                plots_dir=plots_dir,
                fairreg_dir=fairreg_dir,
                diag_dir=diag_dir,
                fair_reg=fair_reg,
                ot_model=ot_model,
                scaler=scaler,
            )

            #Tradeoff plots
            if args.make_tradeoff and args.make_plots:
                eps_list = [float(x) for x in args.eps_list.split(",")]
                lambda_list = [float(x) for x in args.lambda_list.split(",")]

                fairreg_points = []
                for eps0 in eps_list:
                    eps_vec2 = [eps0, eps0]
                    fr = FairReg(reg_fr, clf_fr, B=B, K=K, p=p_vec, eps=eps_vec2, T=T_final, keep_history=args.make_plots)
                    fr.fit(X_unlab, beta=best_beta, L=best_L, alg={"base": "SGD3", "method": "ACSA2"})
                    y_fr = fr.predict(X_test)

                    if scaler is not None:
                        y_fr = scaler.inverse_transform(y_fr.reshape(-1, 1))[:, 0]

                    dp = DP_unfairness(y_fr, S_test, bins=grid)
                    unfair = max(dp.values())
                    risk = float(np.mean((y_test_eval - y_fr) ** 2))
                    fairreg_points.append((unfair, risk))

                y_pred_base_eval = np.asarray(preds["Base (eta)"])
                y_pred_ot_eval = np.asarray(preds["OT"])

                ot_points = []
                for lam in lambda_list:
                    y_mix = (1 - lam) * y_pred_base_eval + lam * y_pred_ot_eval
                    dp = DP_unfairness(y_mix, S_test, bins=grid)
                    unfair = max(dp.values())
                    risk = float(np.mean((y_test_eval - y_mix) ** 2))
                    ot_points.append((unfair, risk))

                plot_tradeoff_curve(
                    {"FairReg (eps sweep)": fairreg_points, "OT (lambda sweep)": ot_points},
                    title=f"{args.dataset} — MSE vs unfairness tradeoff",
                    savepath=str(plots_dir / "tradeoff_curve.png"),
                )

            #metrics 
            metrics_row = compute_metrics(
                y_test_eval=y_test_eval,
                preds=preds,
                S_test=S_test,
                grid=grid,
                DP_unfairness=DP_unfairness,
            )
            if unlab_fair is not None:
                metrics_row.update(unlab_fair)

            metrics_row["time_fairreg"] = time_fairreg
            metrics_row["time_ot"] = time_ot
            metrics_row["time_frappe"] = time_frappe
            metrics_row["time_aware"] = time_aware
            metrics_row["time_aware_hat"] = time_aware_hat
            metrics_row["T_used"] = T_final

            metrics_row["acc_shat_test"] = float(acc_shat_test)
            metrics_row["acc_shat_unlab"] = float(acc_shat_unlab)

            metrics_row["acc_shat_ot_test"] = float(acc_shat_ot_test)
            metrics_row["acc_shat_ot_unlab"] = float(acc_shat_ot_unlab)



            metrics_row["run"] = i
            results.append(metrics_row)

            print(
                f"[run {i:02d}] "
                f"MSE base={metrics_row['mse_base']:.2f} fairreg={metrics_row['mse_fairreg']:.2f} "
                f"fairreg_auto={metrics_row.get('mse_fairreg_auto', float('nan')):.2f} "
                f"frappe={metrics_row.get('mse_frappe', float('nan')):.2f} "
                f"ot={metrics_row['mse_ot']:.2f} aware={metrics_row['mse_aware']:.2f} | "
                f"aware_hat_hard={metrics_row.get('mse_aware_hat_hard', float('nan')):.2f} "
                f"aware_hat_soft={metrics_row.get('mse_aware_hat_soft', float('nan')):.2f} "

                f"DP(max) base={metrics_row['dp_base_max']:.2f} fairreg={metrics_row['dp_fairreg_max']:.2f} "
                f"fairreg_auto={metrics_row.get('dp_fairreg_auto_max', float('nan')):.2f} "
                f"ot={metrics_row['dp_ot_max']:.2f} aware={metrics_row['dp_aware_max']:.2f} | "
                f"aware_hat_hard={metrics_row.get('dp_aware_hat_hard_max', float('nan')):.2f} "
                f"aware_hat_soft={metrics_row.get('dp_aware_hat_soft_max', float('nan')):.2f} "

                f"KS base={metrics_row['ks_base']:.2f} fairreg={metrics_row['ks_fairreg']:.2f} "
                f"fairreg_auto={metrics_row.get('ks_fairreg_auto', float('nan')):.2f} "
                f"ot={metrics_row['ks_ot']:.2f} aware={metrics_row['ks_aware']:.2f} "
                f"aware_hat_hard={metrics_row.get('ks_aware_hat_hard', float('nan')):.2f} "
                f"aware_hat_soft={metrics_row.get('ks_aware_hat_soft', float('nan')):.2f} "

                f"w2 base={metrics_row['w2_base']:.2f} fairreg={metrics_row['w2_fairreg']:.2f} "
                f"ot={metrics_row['w2_ot']:.2f} aware={metrics_row['w2_aware']:.2f} "
                f"aware_hat_hard={metrics_row.get('w2_aware_hat_hard', float('nan')):.2f} "
                f"aware_hat_soft={metrics_row.get('w2_aware_hat_soft', float('nan')):.2f} "

                f" | acc_shat(test)={metrics_row['acc_shat_test']:.2f} "
                f"acc_shat_ot(test)={metrics_row['acc_shat_ot_test']:.2f}"

            )

        df = pd.DataFrame(results)
        csv_out_path = summary_dir / "runs.csv" if not args.merge_train_unlab_for_non_fairreg else summary_dir / "runs_merge_train_unlab.csv"
        df.to_csv(csv_out_path, index=False)
        print(f"Saved computed results to {csv_out_path}")
    has_base_full = "mse_base_full" in df.columns
    has_auto = "mse_fairreg_auto" in df.columns
    has_aware_hat_hard = "mse_aware_hat_hard" in df.columns
    has_aware_hat_soft = "mse_aware_hat_soft" in df.columns
    has_frappe = "mse_frappe" in df.columns
    has_aware_hat = has_aware_hat_hard or has_aware_hat_soft


    cols_test = [
        "mse_base", "mse_fairreg",
        * (["mse_fairreg_auto"] if has_auto else []),
        * (["mse_frappe"] if has_frappe else []),
        "mse_ot", "mse_aware",
        * (["mse_aware_hat_hard"] if has_aware_hat_hard else []),
        * (["mse_aware_hat_soft"] if has_aware_hat_soft else []),

        "w2_base", "w2_fairreg",
        * (["w2_fairreg_auto"] if has_auto else []),
        * (["w2_frappe"] if has_frappe else []),
        "w2_ot", "w2_aware",
        * (["w2_aware_hat_hard"] if has_aware_hat_hard else []),
        * (["w2_aware_hat_soft"] if has_aware_hat_soft else []),

        "tv_base", "tv_fairreg",
        * (["tv_fairreg_auto"] if has_auto else []),
        * (["tv_frappe"] if has_frappe else []),
        "tv_ot", "tv_aware",
        * (["tv_aware_hat_hard"] if has_aware_hat_hard else []),
        * (["tv_aware_hat_soft"] if has_aware_hat_soft else []),

        "ks_base", "ks_fairreg",
        * (["ks_fairreg_auto"] if has_auto else []),
        * (["ks_frappe"] if has_frappe else []),
        "ks_ot", "ks_aware",
        * (["ks_aware_hat_hard"] if has_aware_hat_hard else []),
        * (["ks_aware_hat_soft"] if has_aware_hat_soft else []),

        "dp_base_max", "dp_fairreg_max",
        * (["dp_fairreg_auto_max"] if has_auto else []),
        * (["dp_frappe_max"] if has_frappe else []),
        "dp_ot_max", "dp_aware_max",
        * (["dp_aware_hat_hard_max"] if has_aware_hat_hard else []),
        * (["dp_aware_hat_soft_max"] if has_aware_hat_soft else []),
    ]

    if has_base_full:
        cols_test.insert(cols_test.index("mse_base") + 1, "mse_base_full")
        cols_test.insert(cols_test.index("w2_base") + 1, "w2_base_full")
        cols_test.insert(cols_test.index("tv_base") + 1, "tv_base_full")
        cols_test.insert(cols_test.index("ks_base") + 1, "ks_base_full")
        cols_test.insert(cols_test.index("dp_base_max") + 1, "dp_base_full_max")

    if args.merge_train_unlab_for_non_fairreg:
        cols_unlab = []
    else:
        cols_unlab = [
            "w2_base_unlab", "w2_fairreg_unlab",
            * (["w2_fairreg_auto_unlab"] if has_auto else []),
            "w2_ot_unlab", "w2_aware_unlab",
            * (["w2_aware_hat_hard_unlab"] if has_aware_hat_hard else []),
            * (["w2_aware_hat_soft_unlab"] if has_aware_hat_soft else []),

            "tv_base_unlab", "tv_fairreg_unlab",
            * (["tv_fairreg_auto_unlab"] if has_auto else []),
            "tv_ot_unlab", "tv_aware_unlab",
            * (["tv_aware_hat_hard_unlab"] if has_aware_hat_hard else []),
            * (["tv_aware_hat_soft_unlab"] if has_aware_hat_soft else []),

            "ks_base_unlab", "ks_fairreg_unlab",
            * (["ks_fairreg_auto_unlab"] if has_auto else []),
            "ks_ot_unlab", "ks_aware_unlab",
            * (["ks_aware_hat_hard_unlab"] if has_aware_hat_hard else []),
            * (["ks_aware_hat_soft_unlab"] if has_aware_hat_soft else []),

            "dp_base_unlab_max", "dp_fairreg_unlab_max",
            * (["dp_fairreg_auto_unlab_max"] if has_auto else []),
            "dp_ot_unlab_max", "dp_aware_unlab_max",
            * (["dp_aware_hat_hard_unlab_max"] if has_aware_hat_hard else []),
            * (["dp_aware_hat_soft_unlab_max"] if has_aware_hat_soft else []),
        ]

    # summary mse and unfairness 
    lines = []
    def write_summary_block(title: str, cols_block: list[str]):
        lines.append(title)
        name_w = max(len(c) for c in cols_block) if cols_block else 0
        for col in cols_block:
            mean = df[col].mean()
            std = df[col].std()
            line = f"{col:<{name_w}}: {mean:9.2f} ± {std:9.2f}"
            print(line)
            lines.append(line)

    lines.append("")
    write_summary_block("=== TEST metrics (mean ± std over runs) ===", cols_test)

    if not args.merge_train_unlab_for_non_fairreg:
        lines.append("")
        write_summary_block("=== UNLAB fairness metrics (mean ± std over runs) ===", cols_unlab)


    summary_path = summary_dir / "summary.txt" if not args.merge_train_unlab_for_non_fairreg else summary_dir / "summary_merge_train_unlab.txt"
    print("summary saved to:", summary_path)
    df.round(3).to_csv(summary_dir / "runs.csv", index=False)

    rename = {
            "mse_base": "MSE ERM Unfair", 
            "mse_fairreg": "MSE FairReg", 
            "mse_ot": "MSE OT unaware", 
            "mse_aware": "MSE OT aware", 
            "mse_frappe": "MSE FRAPPE",

            "w2_base": "$W_2$ test ERM",
            "w2_fairreg": "$W_2$ test FairReg",
            "w2_ot": "$W_2$ test OT unaware",
            "w2_frappe": "$W_2$ test FRAPPE",
            "w2_aware": "$W_2$ test OT aware",
            "w2_base_unlab": "$W_2$ unlab ERM",
            "w2_fairreg_unlab": "$W_2$ unlab FairReg",
            "w2_ot_unlab": "$W_2$ unlab OT unaware",
            "w2_aware_unlab": "$W_2$ unlab OT aware",
            "w2_frappe_unlab": "$W_2$ unlab FRAPPE",

            "tv_base": "TV test ERM",
            "tv_fairreg": "TV test FairReg",
            "tv_ot": "TV test OT unaware",
            "tv_aware": "TV test OT aware",
            "tv_base_unlab": "TV unlab ERM",
            "tv_fairreg_unlab": "TV unlab FairReg",
            "tv_ot_unlab": "TV unlab OT unaware",
            "tv_aware_unlab": "TV unlab OT aware",

            "ks_base": "KS test ERM", 
            "ks_fairreg": "KS test FairReg", 
            "ks_ot": "KS test OT unaware", 
            "ks_aware": "KS test OT aware", 
            "ks_base_unlab": "KS unlab ERM", 
            "ks_fairreg_unlab": "KS unlab FairReg", 
            "ks_ot_unlab": "KS unlab OT unaware", 
            "ks_aware_unlab": "KS unlab OT aware", 

            "dp_base_max": r"KS$^{\mathcal{T}}$ test ERM", 
            "dp_fairreg_max": r"KS$^{\mathcal{T}}$ test FairReg", 
            "dp_ot_max": r"KS$^{\mathcal{T}}$ test OT unaware", 
            "dp_aware_max": r"KS$^{\mathcal{T}}$ test OT aware", 
            "dp_base_unlab_max": r"KS$^{\mathcal{T}}$ unlab ERM", 
            "dp_fairreg_unlab_max": r"KS$^{\mathcal{T}}$ unlab FairReg", 
            "dp_ot_unlab_max": r"KS$^{\mathcal{T}}$ unlab OT unaware", 
            "dp_aware_unlab_max": r"KS$^{\mathcal{T}}$ unlab OT aware", 
    }

    if has_auto:
        rename.update({
            "mse_fairreg_auto": "MSE FairReg auto",
            "w2_fairreg_auto": "$W_2$ test FairReg auto",
            "tv_fairreg_auto": "TV test FairReg auto",
            "ks_fairreg_auto": "KS test FairReg auto",
            "dp_fairreg_auto_max": r"KS$^{\mathcal{T}}$ test FairReg auto",
            "w2_fairreg_auto_unlab": "$W_2$ unlab FairReg auto",
            "tv_fairreg_auto_unlab": "TV unlab FairReg auto",
            "ks_fairreg_auto_unlab": "KS unlab FairReg auto",
            "dp_fairreg_auto_unlab_max": r"KS$^{\mathcal{T}}$ unlab FairReg auto",
        })
    
    if has_aware_hat:
        rename.update({
            "mse_aware_hat_hard": "MSE OT aware (hard)",
            "w2_aware_hat_hard": "$W_2$ test OT aware (hard)",
            "tv_aware_hat_hard": "TV test OT aware (hard)",
            "ks_aware_hat_hard": "KS test OT aware (hard)",
            "dp_aware_hat_hard_max": r"KS$^{\mathcal{T}}$ test OT aware (hard)",
            "w2_aware_hat_hard_unlab": "$W_2$ unlab OT aware (hard)",
            "tv_aware_hat_hard_unlab": "TV unlab OT aware (hard)",
            "ks_aware_hat_hard_unlab": "KS unlab OT aware (hard)",
            "dp_aware_hat_hard_unlab_max": r"KS$^{\mathcal{T}}$ unlab OT aware (hard)",

            "mse_aware_hat_soft": "MSE OT aware (soft)",
            "w2_aware_hat_soft": "$W_2$ test OT aware (soft)",
            "tv_aware_hat_soft": "TV test OT aware (soft)",
            "ks_aware_hat_soft": "KS test OT aware (soft)",
            "dp_aware_hat_soft_max": r"KS$^{\mathcal{T}}$ test OT aware (soft)",
            "w2_aware_hat_soft_unlab": "$W_2$ unlab OT aware (soft)",
            "tv_aware_hat_soft_unlab": "TV unlab OT aware (soft)",
            "ks_aware_hat_soft_unlab": "KS unlab OT aware (soft)",
            "dp_aware_hat_soft_unlab_max": r"KS$^{\mathcal{T}}$ unlab OT aware (soft)",
        })

    if has_base_full:
        rename.update({
            "mse_base_full": "MSE ERM Unfair (train+unlab)",
            "w2_base_full": "$W_2$ test ERM (train+unlab)",
            "tv_base_full": "TV test ERM (train+unlab)",
            "ks_base_full": "KS test ERM (train+unlab)",
            "dp_base_full_max": r"KS$^{\mathcal{T}}$ test ERM (train+unlab)",
        })

    cols_mse_runs = ["run", "mse_base", "mse_fairreg"]
    if has_base_full: cols_mse_runs += ["mse_base_full"]
    if has_auto: cols_mse_runs += ["mse_fairreg_auto"]
    cols_mse_runs += ["mse_frappe", "mse_ot", "mse_aware"]
    if has_aware_hat_hard: cols_mse_runs += ["mse_aware_hat_hard"]
    if has_aware_hat_soft: cols_mse_runs += ["mse_aware_hat_soft"]

    # UNLAB table: W2/TV/KS/KS^T
    cols_unlab_runs = ["run"]
    if not args.merge_train_unlab_for_non_fairreg:
        metrics_unlab = ["w2", "tv", "ks", "dp"]
        suffixes_unlab = ["_base_unlab", "_fairreg_unlab"]
        if has_auto: suffixes_unlab += ["_fairreg_auto_unlab"]
        suffixes_unlab += ["_frappe_unlab", "_ot_unlab", "_aware_unlab"]
        if has_aware_hat_hard: suffixes_unlab += ["_aware_hat_hard_unlab"]
        if has_aware_hat_soft: suffixes_unlab += ["_aware_hat_soft_unlab"]

        for m in metrics_unlab:
            for s in suffixes_unlab:
                col_name = f"{m}{s}" + ("_max" if m == "dp" else "")
                cols_unlab_runs.append(col_name)

    # REST table: TEST W2/TV/KS/KS^T
    cols_rest_runs = ["run"]
    metrics_test = ["w2", "tv", "ks", "dp"]
    
    suffixes_test = ["_base", "_fairreg"]
    if has_base_full: suffixes_test.insert(1, "_base_full")
    if has_auto: suffixes_test += ["_fairreg_auto"]
    suffixes_test += ["_frappe", "_ot", "_aware"]
    if has_aware_hat_hard: suffixes_test += ["_aware_hat_hard"]
    if has_aware_hat_soft: suffixes_test += ["_aware_hat_soft"]

    for m in metrics_test:
        for s in suffixes_test:
            col_name = f"{m}{s}" + ("_max" if m == "dp" else "")
            cols_rest_runs.append(col_name)

    runs_tables_path = write_runs_tables_tex(
        df,
        out_path=summary_dir / "runs.tex" if not args.merge_train_unlab_for_non_fairreg else summary_dir / "runs_merge_train_unlab.tex",
        cols_mse=cols_mse_runs,
        cols_unlab=([] if args.merge_train_unlab_for_non_fairreg else cols_unlab_runs),
        cols_rest=cols_rest_runs,
        precision=2,
        rotate_rest=True,
        rename=rename,
        relative_to_base=False,
    )
    

    print("runs tables saved to:", runs_tables_path)



    #FairReg hyperparameter selection summary (per run + aggregate) 
    if len(fairreg_selected) > 0:
        df_sel = pd.DataFrame(fairreg_selected)

        #chosen parameters per run 
        sel_csv_path = os.path.join(summary_dir, "fairreg_selected_params.csv") if not args.merge_train_unlab_for_non_fairreg else os.path.join(summary_dir, "fairreg_selected_params_merge_train_unlab.csv")
        df_sel.to_csv(sel_csv_path, index=False)

        lines.append("")
        lines.append("=== FairReg selected hyperparameters (per run) ===")
        for _, r in df_sel.iterrows():
            msg = f"run {int(r['run']):02d}: L={int(r['best_L'])}, beta={float(r['best_beta']):.2f}"
            if "val_risk" in df_sel.columns:
                msg += f" | val_risk={r['val_risk']:.6g} val_unfair={r['val_unfair']:.6g}"
            lines.append(msg)

        lines.append("")
        lines.append("=== FairReg selected hyperparameters (aggregate) ===")
        lines.append(f"best_L mean±std   : {df_sel['best_L'].mean():.3f} ± {df_sel['best_L'].std():.3f}")
        lines.append(f"best_beta mean±std: {df_sel['best_beta'].mean():.6g} ± {df_sel['best_beta'].std():.6g}")

        #most frequent values (mode)
        lines.append(f"mode(L)           : {int(df_sel['best_L'].value_counts().idxmax())}")
        lines.append(f"mode(beta)        : {float(df_sel['best_beta'].value_counts().idxmax()):.6g}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    print(summary_path)

    # GENERATE TABLES
    out_abs = summary_dir / "summary.tex" if not args.merge_train_unlab_for_non_fairreg else summary_dir / "summary_merge_train_unlab.tex"
    out_rel = summary_dir / "summary_relative.tex" if not args.merge_train_unlab_for_non_fairreg else summary_dir / "summary_relative_merge_train_unlab.tex"

    generate_custom_summary_tex(df, out_abs, out_rel, args.dataset)

    print(f"Summary tex saved to:\n- {out_abs}\n- {out_rel}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "lawschool", "communities", "synthetic"], default="lawschool")
    parser.add_argument("--fairreg_root", default=None,
                        help="Path to the FairReg baseline repo (unaware-fair-reg). Only needed if FairReg is not installed as a package.")
     
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--T", type=int, default=None, help="Si None, calculé dynamiquement pour 25 époques.")
    parser.add_argument("--load_results", type=str, default=None, help="Chemin vers un runs.csv pour sauter l'entraînement.")
    parser.add_argument("--eps", type=float, default=0.00001)

    # FairReg hyperparams ('auto' is the sae as in thir evaluation.py)
    parser.add_argument("--beta", default="auto")
    parser.add_argument("--L", default="auto")

    # splits (defaults as evluation.py)
    parser.add_argument("--train_size", type=float, default=0.4)
    parser.add_argument("--test_size", type=float, default=0.2)

    # OT hypermarams 
    parser.add_argument("--sinkhorn_reg", type=float, default=None)
    parser.add_argument("--delta_tol", type=float, default=1e-6)
    parser.add_argument("--cont_map", choices=["rf", "entropic"], default="rf")
    parser.add_argument("--cont_epsilon", type=float, default=0.05)
    parser.add_argument("--cont_knn", type=int, default=None)

    parser.add_argument(
    "--merge_train_unlab_for_non_fairreg",
    action="store_true",
    help="If set: baseline/OT/Aware are trained on (train + unlab) as labeled data. "
         "FairReg stays unchanged (trained on train, uses unlab as unlabeled)."
)

    parser.add_argument("--sweep_fairreg_T", type=str, default=None, 
                        help="Comma-separated list of T (epochs) to test (e.g., '100,1000,10000'). Triggers early exit.")
    parser.add_argument("--load_sweep_csv", type=str, default=None, 
                        help="Path to a saved convergence CSV to replot without retraining.")

    parser.add_argument("--no_scaling", action="store_true")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--out_dir", default="notes/figs/compare_ot_vs_fairreg")

    parser.add_argument("--make_tradeoff", action="store_true")
    parser.add_argument("--eps_list", default="0, 0, 0, 0")  
    parser.add_argument("--lambda_list", default="0,0.25,0.5,0.75,1.0") 
    parser.add_argument(
    "--full_data",
    action="store_true",
    help="Use the full dataset (skip the 2000-point subsample for adult/lawschool).", default=True
)
    parser.add_argument("--synthetic_n", type=int, default=10000, help="Number of points for synthetic dataset.")
    parser.add_argument("--synthetic_gamma", type=float, default=1.0, help="Gamma parameter for synthetic dataset.")
    parser.add_argument("--synthetic_beta", type=float, default=1.0, help="Beta parameter for synthetic dataset.")
    parser.add_argument("--synthetic_seed", type=int, default=42, help="Random seed for synthetic dataset generation.")
    
    
    ## for debugging
    parser.add_argument("--debug", action="store_true", help="Print numeric sanity checks (finite/overflow diagnostics).")


  ## FairReg hyperparameter tuning
    parser.add_argument("--tune_fairreg", action="store_true",
                    help="Tune FairReg hyperparameters (L,beta) on a held-out subset of the TRAIN split.")
    parser.add_argument("--val_frac_train", type=float, default=0.25,
                        help="Fraction of X_train used as validation for tuning (rest used to fit reg+clf).")
    
    parser.add_argument("--L_grid", default="auto,0.5x,1x,2x",
                        help="Candidate L values. Tokens: 'auto' or multipliers like '0.5x,1x,2x' or raw numbers.")
    parser.add_argument("--beta_grid", default="auto,0.25x,0.5x,1x,2x,4x",
                        help="Candidate beta values. Tokens: 'auto' or multipliers like '0.5x,1x,2x' or raw numbers.")

    parser.add_argument("--dp_nbins", type=int, default=41,
                        help="Common number of bins for DP unfairness during tuning (to compare across different L).")
    parser.add_argument("--tune_objective", choices=["feasible_min_risk", "penalized"], default="feasible_min_risk")
    parser.add_argument("--penalty_lambda", type=float, default=100.0)
    parser.add_argument("--plot_only_run", type=int, default=None, 
                        help="If set with --load_results, only regenerate plots for this specific run number (e.g., 10).")

    args = parser.parse_args()
    main(args)
