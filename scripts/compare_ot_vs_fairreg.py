import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm   

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp    

from fair_unaware import UnawareFairRegressor
from fair_unaware.baselines import AwareWassersteinBarycenterDPRegressor
from fair_unaware.diagnostics import plot_basic_diagnostics
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
    compute_group_priors,  
)


warnings.filterwarnings("once", category=RuntimeWarning)


### ATTENTION j'ai enlevé les warnings qui rendaients fous dans le termoinal parce que j'ai pas réussi à résoudre le pb 
# mais c'est provisoire
import logging
import warnings
from pathlib import Path

def redirect_warnings_to_file(warnings_path: Path):
    warnings_path = Path(warnings_path)
    warnings_path.parent.mkdir(parents=True, exist_ok=True)

    # Route Python warnings -> logging ("py.warnings")
    logging.captureWarnings(True)

    logger = logging.getLogger("py.warnings")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # IMPORTANT: prevents printing to terminal

    fh = logging.FileHandler(warnings_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

    # Choose how many warnings to record:
    # "default" = once per location; "always" = every time; "once" = once overall
    warnings.simplefilter("default")

def dbg_summary(name, A):
    A = np.asarray(A)
    print(
        f"{name}: shape={A.shape} finite={np.isfinite(A).mean():.4f} "
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
    """Fit reg+clf exactly like your script."""
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


def fit_aware_model(*, X_train_full, S_train_full, y_train_full, X_unlab, S_unlab, priors, seed: int):
    aware = AwareWassersteinBarycenterDPRegressor(
        base_regressor=LinearRegression(fit_intercept=True),
        random_state=seed,
    )
    aware.fit(
        X_train_full, S_train_full, y_train_full,
        X_unlab=X_unlab, S_unlab=S_unlab,
        group_priors=priors,
    )
    return aware


def predict_inverse_scale_and_grid(*, reg, fair_reg, ot_model, aware, split: dict, scaler, dp_nbins: int):
    X_test = split["X_test"]
    S_test = split["S_test"]
    y_test = split["y_test"]

    y_pred_base = reg.predict(X_test)
    y_pred_fair = fair_reg.predict(X_test)
    y_pred_ot = ot_model.predict(X_test)
    y_pred_aware = aware.predict(X_test, S_test)

    if scaler is not None:
        # y_test is a pandas Series in your loaders
        y_test_eval = scaler.inverse_transform(y_test.values.reshape(-1, 1))[:, 0]
        y_pred_base_eval = scaler.inverse_transform(y_pred_base.reshape(-1, 1))[:, 0]
        y_pred_fair_eval = scaler.inverse_transform(y_pred_fair.reshape(-1, 1))[:, 0]
        y_pred_ot_eval = scaler.inverse_transform(y_pred_ot.reshape(-1, 1))[:, 0]
        y_pred_aware_eval = scaler.inverse_transform(y_pred_aware.reshape(-1, 1))[:, 0]
    else:
        y_test_eval = np.asarray(y_test)
        y_pred_base_eval = y_pred_base
        y_pred_fair_eval = y_pred_fair
        y_pred_ot_eval = y_pred_ot
        y_pred_aware_eval = y_pred_aware

    preds = {
        "Base (eta)": y_pred_base_eval,
        "FairReg": y_pred_fair_eval,
        "OT": y_pred_ot_eval,
        "Aware DP": y_pred_aware_eval,
    }

    grid = _common_bins(
        np.concatenate([y_test_eval, y_pred_base_eval, y_pred_fair_eval, y_pred_ot_eval, y_pred_aware_eval]),
        nbins=dp_nbins,
    )

    return y_test_eval, preds, grid

def predict_unlab_inverse_scale_and_grid(*, reg, fair_reg, ot_model, aware, split: dict, scaler, dp_nbins: int):
    X_unlab = split["X_unlab"]
    S_unlab = split["S_unlab"]
    y_unlab = split["y_unlab"]

    y_pred_base = reg.predict(X_unlab)
    y_pred_fair = fair_reg.predict(X_unlab)
    y_pred_ot = ot_model.predict(X_unlab)
    y_pred_aware = aware.predict(X_unlab, S_unlab)

    if scaler is not None:
        y_unlab_eval = scaler.inverse_transform(np.asarray(y_unlab).reshape(-1, 1))[:, 0]
        y_pred_base_eval = scaler.inverse_transform(y_pred_base.reshape(-1, 1))[:, 0]
        y_pred_fair_eval = scaler.inverse_transform(y_pred_fair.reshape(-1, 1))[:, 0]
        y_pred_ot_eval = scaler.inverse_transform(y_pred_ot.reshape(-1, 1))[:, 0]
        y_pred_aware_eval = scaler.inverse_transform(y_pred_aware.reshape(-1, 1))[:, 0]
    else:
        y_unlab_eval = np.asarray(y_unlab)
        y_pred_base_eval = y_pred_base
        y_pred_fair_eval = y_pred_fair
        y_pred_ot_eval = y_pred_ot
        y_pred_aware_eval = y_pred_aware

    preds_unlab = {
        "Base (eta)": y_pred_base_eval,
        "FairReg": y_pred_fair_eval,
        "OT": y_pred_ot_eval,
        "Aware DP": y_pred_aware_eval,
    }

    grid_unlab = _common_bins(
        np.concatenate([y_unlab_eval, y_pred_base_eval, y_pred_fair_eval, y_pred_ot_eval, y_pred_aware_eval]),
        nbins=dp_nbins,
    )
    return y_unlab_eval, preds_unlab, grid_unlab


def compute_dp_ks_only(*, preds: dict, S, grid, DP_unfairness, prefix: str):
    S_arr = np.asarray(S).ravel()
    groups = np.unique(S_arr)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {groups}")
    g0, g1 = groups[0], groups[1]

    def ks_gap(yhat):
        yhat = np.asarray(yhat).ravel()
        return float(ks_2samp(yhat[S_arr == g0], yhat[S_arr == g1]).statistic)

    dp_base = DP_unfairness(preds["Base (eta)"], S, bins=grid)
    dp_fair = DP_unfairness(preds["FairReg"], S, bins=grid)
    dp_ot = DP_unfairness(preds["OT"], S, bins=grid)
    dp_aware = DP_unfairness(preds["Aware DP"], S, bins=grid)

    return {
        f"dp_base_{prefix}_max": float(max(dp_base.values())),
        f"dp_fairreg_{prefix}_max": float(max(dp_fair.values())),
        f"dp_ot_{prefix}_max": float(max(dp_ot.values())),
        f"dp_aware_{prefix}_max": float(max(dp_aware.values())),
        f"ks_base_{prefix}": ks_gap(preds["Base (eta)"]),
        f"ks_fairreg_{prefix}": ks_gap(preds["FairReg"]),
        f"ks_ot_{prefix}": ks_gap(preds["OT"]),
        f"ks_aware_{prefix}": ks_gap(preds["Aware DP"]),
    }


def compute_metrics(*, y_test_eval, preds: dict, S_test, grid, DP_unfairness):
    mse_base = mse(y_test_eval, preds["Base (eta)"])
    mse_fair = mse(y_test_eval, preds["FairReg"])
    mse_ot = mse(y_test_eval, preds["OT"])
    mse_aware = mse(y_test_eval, preds["Aware DP"])

    dp_base = DP_unfairness(preds["Base (eta)"], S_test, bins=grid)
    dp_fair = DP_unfairness(preds["FairReg"], S_test, bins=grid)
    dp_ot = DP_unfairness(preds["OT"], S_test, bins=grid)
    dp_aware = DP_unfairness(preds["Aware DP"], S_test, bins=grid)

    S_test_arr = np.asarray(S_test).ravel()
    groups = np.unique(S_test_arr)
    if len(groups) != 2:
        raise ValueError(f"KS expects 2 groups, got {groups}")

    g0, g1 = groups[0], groups[1]

    def ks_gap(yhat):
        yhat = np.asarray(yhat).ravel()
        y0 = yhat[S_test_arr == g0]
        y1 = yhat[S_test_arr == g1]
        return float(ks_2samp(y0, y1).statistic)

    ks_base = ks_gap(preds["Base (eta)"])
    ks_fair = ks_gap(preds["FairReg"])
    ks_ot = ks_gap(preds["OT"])
    ks_aware = ks_gap(preds["Aware DP"])

    return {
        "mse_base": mse_base,
        "mse_fairreg": mse_fair,
        "mse_ot": mse_ot,
        "mse_aware": mse_aware,
        "dp_base_max": max(dp_base.values()),
        "dp_fairreg_max": max(dp_fair.values()),
        "dp_ot_max": max(dp_ot.values()),
        "dp_aware_max": max(dp_aware.values()),
        "ks_base": ks_base,
        "ks_fairreg": ks_fair,
        "ks_ot": ks_ot,
        "ks_aware": ks_aware,
    }


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
        # NOTE (faire attention): y_fr deja sur un scale corrigé si  scaler=None; si on veut un truc scale il fat scale AVANT cette fonction.
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


    # X, S, y, scaler, B, K, p, priors = load_dataset(args.dataset, data_scaling=not args.no_scaling)
    exp_tag = "full" if args.full_data else "subsample_2000"
    X, S, y, scaler, B, K, p, priors = load_dataset(
        args.dataset,
        fairreg_root=args.fairreg_root,
        data_scaling=not args.no_scaling,
        full_data=args.full_data,
    )

    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    # UNLAB = (1 - TRAIN - TEST) (but implicitly it does this)
    print(f"Using dataset={args.dataset} with train_size={TRAIN_SIZE}, test_size={TEST_SIZE}, full_data={args.full_data}")

    results = []
    fairreg_selected = []  # per-run chosen (L, beta) +++ val metrics

    cont_tag = str(args.cont_map)

    summary_dir = exp_root / args.dataset / exp_tag / cont_tag
    summary_dir.mkdir(parents=True, exist_ok=True)

        # pour cacher les warning sous le tapis 
    redirect_warnings_to_file(summary_dir / "warnings.log")
    need_dirs = args.make_plots or args.tune_fairreg or args.make_tradeoff

    for i in tqdm(range(1, args.num + 1), desc="Runs"):

        # 1)make the spplit
        split = make_splits(X, S, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=i)

        # convenience (lazyness)
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


        # 2)directories
        _, plots_dir, diag_dir, fairreg_dir = make_run_dirs(
            exp_root, args.dataset, exp_tag, cont_tag, i, need_dirs=need_dirs
        )

        # 3) Tuning split 
        tuning = maybe_make_tuning_split(
            split,
            tune=args.tune_fairreg,
            val_frac_train=args.val_frac_train,
            dp_nbins=args.dp_nbins,
            seed=i,
        )

        # 4) candidate gridds (may be upgraded but i don't know yet how)
        L_list, beta_list, _, _ = build_fairreg_candidate_grids(
            X_unlab, L_grid=args.L_grid, beta_grid=args.beta_grid
        )


        # 5)fit base models on X_train_fit
        reg, clf = fit_base_models(
            tuning["X_train_fit"], tuning["S_train_fit"], tuning["y_train_fit"]
        )

        if args.debug:
            check_finite("X_train_full", X_train_full)
            check_finite("X_unlab", X_unlab)
            check_finite("X_test", X_test)
            check_finite("y_train_full", y_train_full)
            check_finite("y_unlab", y_unlab)
            check_finite("y_test", y_test)
            dbg_summary("reg.coef_", reg.coef_)
            dbg_summary("clf.coef_", clf.coef_)


        # 6)FairReg group params
        p_vec, K, eps_vec = fairreg_group_params_from_clf(
            clf, S_train_full, eps=args.eps
        )


        # 7)select FairReg hyperparams 
        best_L, best_beta, tuning_rows, row_sel = select_fairreg_hparams_or_auto(
            tune=args.tune_fairreg,
            reg=reg,
            clf=clf,
            B=B,
            K=K,
            p_vec=p_vec,
            eps_vec=eps_vec,
            T=args.T,
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


        # 8) refit on full TRAIN after tuning (exactly like before)
        if args.tune_fairreg:
            reg.fit(X_train_full, y_train_full)
            clf.fit(X_train_full, S_train_full)
            p_vec, K, eps_vec = fairreg_group_params_from_clf(
                clf, S_train_full, eps=args.eps
            )

        # 9) Fit FairReg / OT / Aware
        fair_reg = fit_fairreg_final(
            FairReg=FairReg,
            reg=reg,
            clf=clf,
            B=B,
            K=K,
            p_vec=p_vec,
            eps_vec=eps_vec,
            T=args.T,
            X_unlab=X_unlab,
            best_L=best_L,
            best_beta=best_beta,
            keep_history=args.make_plots,
        )

        ot_model = fit_ot_model(
            X_train=X_train_full,
            S_train=S_train_full,
            y_train=y_train_full,
            X_unlab=X_unlab,
            priors=priors_train,
            args=args,
            seed=i,
        )

        aware = fit_aware_model(
            X_train_full=X_train_full,
            S_train_full=S_train_full,
            y_train_full=y_train_full,
            X_unlab=X_unlab,
            S_unlab=S_unlab,
            priors=priors_train,
            seed=i,
        )

        # 10) predict + inverse scale + grid + preds dict
        y_test_eval, preds, grid = predict_inverse_scale_and_grid(
            reg=reg,
            fair_reg=fair_reg,
            ot_model=ot_model,
            aware=aware,
            split=split,
            scaler=scaler,
            dp_nbins=args.dp_nbins,
        )

        # fairness metrics on UNLAB (the support used by post-processing) 
        y_unlab_eval, preds_unlab, grid_unlab = predict_unlab_inverse_scale_and_grid(
            reg=reg,
            fair_reg=fair_reg,
            ot_model=ot_model,
            aware=aware,
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


        # 11) Plots + OT diags
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

        # 12) Tradeoff plots
        if args.make_tradeoff and args.make_plots:
            eps_list = [float(x) for x in args.eps_list.split(",")]
            lambda_list = [float(x) for x in args.lambda_list.split(",")]

            fairreg_points = []
            for eps0 in eps_list:
                eps_vec2 = [eps0, eps0]
                fr = FairReg(reg, clf, B=B, K=K, p=p_vec, eps=eps_vec2, T=args.T, keep_history=args.make_plots)
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

        # 13)metrics 
        metrics_row = compute_metrics(
            y_test_eval=y_test_eval,
            preds=preds,
            S_test=S_test,
            grid=grid,
            DP_unfairness=DP_unfairness,
        )

        metrics_row.update(unlab_fair)

        metrics_row["run"] = i
        results.append(metrics_row)

        print(
            f"[run {i:02d}] "
            f"MSE base={metrics_row['mse_base']:.4f} fairreg={metrics_row['mse_fairreg']:.4f} "
            f"ot={metrics_row['mse_ot']:.4f} aware={metrics_row['mse_aware']:.4f} | "
            f"DP(max) base={metrics_row['dp_base_max']:.4f} fairreg={metrics_row['dp_fairreg_max']:.4f} "
            f"ot={metrics_row['dp_ot_max']:.4f} aware={metrics_row['dp_aware_max']:.4f} | "
            f"KS base={metrics_row['ks_base']:.4f} fairreg={metrics_row['ks_fairreg']:.4f} "
            f"ot={metrics_row['ks_ot']:.4f} aware={metrics_row['ks_aware']:.4f}"
        )

    df = pd.DataFrame(results)

    cols_test = [
        "mse_base", "mse_fairreg", "mse_ot", "mse_aware",
        "dp_base_max", "dp_fairreg_max", "dp_ot_max", "dp_aware_max",
        "ks_base", "ks_fairreg", "ks_ot", "ks_aware",
    ]

    cols_unlab = [
        "dp_base_unlab_max", "dp_fairreg_unlab_max", "dp_ot_unlab_max", "dp_aware_unlab_max",
        "ks_base_unlab", "ks_fairreg_unlab", "ks_ot_unlab", "ks_aware_unlab",
    ]



    #summary mse and unfairness 
    lines = []
    def write_summary_block(title: str, cols_block: list[str]):
        lines.append(title)
        name_w = max(len(c) for c in cols_block)
        for col in cols_block:
            mean = df[col].mean()
            std = df[col].std()
            line = f"{col:<{name_w}}: {mean:9.4f} ± {std:9.4f}"
            print(line)
            lines.append(line)

    lines.append("")
    write_summary_block("=== TEST metrics (mean ± std over runs) ===", cols_test)
    lines.append("")
    write_summary_block("=== UNLAB fairness metrics (mean ± std over runs) ===", cols_unlab)


    summary_path = summary_dir / "summary.txt"
    print("summary saved to:", summary_path)
    df.round(3).to_csv(summary_dir / "runs.csv", index=False)

    # FairReg hyperparameter selection summary (per run + aggregate) 
    if len(fairreg_selected) > 0:
        df_sel = pd.DataFrame(fairreg_selected)

        #chosen parameters per run 
        sel_csv_path = os.path.join(summary_dir, "fairreg_selected_params.csv")
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "lawschool", "communities"], default="lawschool")
    parser.add_argument("--fairreg_root", default=None,
                        help="Path to the FairReg baseline repo (unaware-fair-reg). Only needed if FairReg is not installed as a package.")
     
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=0.01)

    # FairReg hyperparams ('auto' is the sae as in thir evaluation.py)
    parser.add_argument("--beta", default="auto")
    parser.add_argument("--L", default="auto")

    # splits (defaults as evluation.py)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--test_size", type=float, default=0.1)

    # OT hypermarams 
    parser.add_argument("--sinkhorn_reg", type=float, default=None)
    parser.add_argument("--delta_tol", type=float, default=1e-6)
    parser.add_argument("--cont_map", choices=["rf", "entropic"], default="rf")
    parser.add_argument("--cont_epsilon", type=float, default=0.05)
    parser.add_argument("--cont_knn", type=int, default=None)


    parser.add_argument("--no_scaling", action="store_true")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--out_dir", default="notes/figs/compare_ot_vs_fairreg")

    parser.add_argument("--make_tradeoff", action="store_true")
    parser.add_argument("--eps_list", default="0.00390625,0.0078125,0.015625,0.03125")  # the list of psilon for the fairreg 
    parser.add_argument("--lambda_list", default="0,0.25,0.5,0.75,1.0") # the list of lambda for the OT mix 
    parser.add_argument(
    "--full_data",
    action="store_true",
    help="Use the full dataset (skip the 2000-point subsample for adult/lawschool).", default=True
)
    
    
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
    ## End FairReg hyperparameter tuning

    args = parser.parse_args()
    main(args)
