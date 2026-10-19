import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.metrics import mean_squared_error as mse

from fair_unaware import UnawareFairRegressor
from fair_unaware.baselines.frappe import FrappeRegressor
from fair_unaware.baselines import AwareWassersteinBarycenterDPRegressor
from fair_unaware.baselines.fairreg_adapter import load_dataset, get_fairreg_objects
from fair_unaware.utils import (
    compute_group_priors,
    write_summary_relative_tex,
    _common_bins
)
from scripts.compare_ot_vs_fairreg import (
    make_splits, make_run_dirs, redirect_warnings_to_file,
    fit_fairreg_final, predict_inverse_scale_and_grid,
    fairreg_group_params_from_clf, build_fairreg_candidate_grids,
    maybe_make_tuning_split, select_fairreg_hparams_or_auto,
    check_finite,
    _row_concat
)
from fair_unaware.metrics import dp_cdf_max_gap, w2_gap_between_groups

from data.synthetic import make_synthetic_dataset
from scripts.compare_ot_vs_fairreg import (
    make_splits, redirect_warnings_to_file,
    fit_fairreg_final, predict_inverse_scale_and_grid,
    fairreg_group_params_from_clf
)


def generate_latex_table(df_table):
    """
    Takes the structured DataFrame and generates the LaTeX code
    with multirow formatting for the Base Models.
    """
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Base Model} & \\textbf{Method} & \\textbf{MSE} $\\downarrow$ & \\textbf{$W_2$} $\\downarrow$ & \\textbf{KS$^{\\mathcal{T}}$} $\\downarrow$ \\\\",
        "\\midrule"
    ]
    
    current_base = None
    for _, row in df_table.iterrows():
        base_col = row["_base_model"]
        method_str = row["Method"]
        
        if current_base != base_col:
            if current_base is not None:
                lines.append("\\midrule")
            current_base = base_col
            lines.append(f"\\multirow{{5}}{{*}}{{\\textbf{{{current_base}}}}}")
            

        if method_str in ["FairReg", "FAUST-A"]:
                    lines.append(" \\cmidrule{2-5}")
        mse_str = str(row["MSE"]).replace("±", "$\\pm$")
        dp_str = str(row["KS$^{\\mathcal{T}}$"]).replace("±", "$\\pm$")
        w2_str = str(row["$W_2$"]).replace("±", "$\\pm$")
            
        lines.append(f" & {method_str} & {mse_str} & {w2_str} & {dp_str} \\\\")
        
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{adjustbox}",
        "\\caption{Mean and standard deviation of fairness and accuracy metrics across 10 runs. Statistically insignificant differences with the best unaware method are highlighted in blue.}",
        "\\label{tab:benchmark_results}",
        "\\end{table}"
    ])
    return "\n".join(lines)


def get_base_model_factory(base_type: str, seed: int):
    """
    Returns fresh instances of (regressor, classifier) based on type.
    """
    if base_type == "linear":
        reg = LinearRegression(fit_intercept=True)
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    
    elif base_type == "rf":
        reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=seed)
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=seed)
        
    elif base_type == "gbm":
        reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
        
    else:
        raise ValueError(f"Unknown base_type: {base_type}")
        
    return reg, clf

def main(args):
    # 1. Setup
    FairReg, DP_unfairness = get_fairreg_objects(args.fairreg_root)
    fig_root = Path(args.out_dir)
    exp_root = fig_root / "benchmark_base_models"
    
    if args.dataset == "synthetic":
        X, S, y = make_synthetic_dataset(n=args.synthetic_n, random_state=42)
        scaler, B, K, p, priors = None, np.max(np.abs(y)), 2, None, compute_group_priors(S)
    else:
        X, S, y, scaler, B, K, p, priors = load_dataset(
            args.dataset, fairreg_root=args.fairreg_root, 
            data_scaling=not args.no_scaling, full_data=args.full_data
        )
    
    if not hasattr(S, "unique"):
        S = pd.Series(np.asarray(S).ravel(), name="S")

    # 2. Main Loop over Base Models
    if args.load_csv:
        print(f"Loading existing data from {args.load_csv}...")
        df_all = pd.read_csv(args.load_csv)
    else:
        base_types = args.base_types.split(",")
        all_results = []

        for base_type in base_types:
            print(f"\n\n{'='*40}\nBenchmarking Base Model: {base_type.upper()}\n{'='*40}")
            

            type_dir = exp_root / args.dataset / base_type
            type_dir.mkdir(parents=True, exist_ok=True)
            redirect_warnings_to_file(type_dir / "warnings.log")

            for i in tqdm(range(1, args.num + 1), desc=f"{base_type} Runs"):
                split = make_splits(X, S, y, train_size=args.train_size, test_size=args.test_size, seed=i)
                
                X_train, S_train, y_train = split["X_train_full"], split["S_train_full"], split["y_train_full"]
                # X_unlab, S_unlab = split["X_unlab"], split["S_unlab"]
                X_unlab, S_unlab, y_unlab = split["X_unlab"], split["S_unlab"], split["y_unlab"]
                X_test, S_test, y_test = split["X_test"], split["S_test"], split["y_test"]

                tuning = maybe_make_tuning_split(
                    split,
                    tune=args.tune_fairreg,
                    val_frac_train=args.val_frac_train,
                    dp_nbins=args.dp_nbins,
                    seed=i,
                )


                L_list, beta_list, L_auto, beta_auto = build_fairreg_candidate_grids(
                    X_unlab, L_grid=args.L_grid, beta_grid=args.beta_grid
                )

                if args.merge_train_unlab_for_non_fairreg:
                    X_train_nf = _row_concat(X_train, X_unlab)
                    S_train_nf = _row_concat(S_train, S_unlab)
                    y_train_nf = _row_concat(y_train, y_unlab)
                    priors_nf = compute_group_priors(S_train_nf)
                    X_unlab_nf = X_train_nf
                    S_unlab_nf = S_train_nf
                else:
                    X_train_nf, S_train_nf, y_train_nf = X_train, S_train, y_train
                    priors_nf = priors
                    X_unlab_nf = X_unlab
                    S_unlab_nf = S_unlab

                reg_base, clf_base = get_base_model_factory(base_type, seed=i)
                reg_base.fit(tuning["X_train_fit"], tuning["y_train_fit"])
                clf_base.fit(tuning["X_train_fit"], tuning["S_train_fit"])


                
                    # --- 3. FairReg (Baseline) ---
                p_vec, K_fr, eps_vec = fairreg_group_params_from_clf(clf_base, S_train, eps=args.eps)
                
                # Dynamic T
                T_fr = int(30 * len(X_unlab)) if args.T <= 0 else args.T


                best_L, best_beta, tuning_rows, row_sel = select_fairreg_hparams_or_auto(
                    tune=args.tune_fairreg,
                    reg=reg_base,
                    clf=clf_base,
                    B=B,
                    K=K_fr,
                    p_vec=p_vec,
                    eps_vec=eps_vec,
                    T=T_fr,
                    X_unlab=X_unlab,
                    tuning=tuning,
                    L_list=L_list,
                    beta_list=beta_list,
                    FairReg=FairReg,
                    DP_unfairness=DP_unfairness,
                    tune_objective=args.tune_objective,
                    penalty_lambda=args.penalty_lambda,
                    fairreg_dir=type_dir,  
                    dataset=args.dataset,
                    run_id=i,
                    L_arg=args.L,
                    beta_arg=args.beta,
                )

                if args.tune_fairreg:
                    reg_base.fit(X_train, y_train)
                    clf_base.fit(X_train, S_train)
                    p_vec, K_fr, eps_vec = fairreg_group_params_from_clf(clf_base, S_train, eps=args.eps)

                fair_reg_model = fit_fairreg_final(
                    FairReg=FairReg, reg=reg_base, clf=clf_base, 
                    B=B, K=K_fr, p_vec=p_vec, eps_vec=eps_vec, T=T_fr, 
                    X_unlab=X_unlab, best_L=best_L, best_beta=best_beta, keep_history=False
                )

                # --- 1. Unaware OT ---
                reg_ot, clf_ot = get_base_model_factory(base_type, seed=i)
                
                ot_model = UnawareFairRegressor(
                    base_regressor=reg_ot,
                    base_classifier=clf_ot,
                    continuous_map=args.cont_map,
                    continuous_epsilon=args.cont_epsilon,
                    sinkhorn_reg=args.sinkhorn_reg,
                    delta_tol=args.delta_tol,
                    random_state=i,
                    verbose=False
                )


                # aware_model.fit(X_train, S_train, y_train, X_unlab=X_fair, S_unlab=S_fair)
                ot_model.fit(X_train_nf, S_train_nf, y_train_nf, X_unlab=X_unlab_nf, group_priors=priors_nf)

                # --- 2. Aware OT (Baseline) ---
                reg_aware = clone(reg_base)
                aware_model = AwareWassersteinBarycenterDPRegressor(
                    base_regressor=reg_aware,
                    random_state=i
                )

                aware_model.fit(X_train_nf, S_train_nf, y_train_nf, X_unlab=X_unlab_nf, S_unlab=S_unlab_nf)



                y_test_eval, preds, grid = predict_inverse_scale_and_grid(
                    reg_base=reg_base,
                    fair_reg=fair_reg_model,
                    ot_model=ot_model,
                    aware=aware_model,
                    split=split,
                    scaler=scaler,
                    dp_nbins=args.dp_nbins
                )

                # --- 3. FRAPPÉ (Baseline) ---
                frappe_model = FrappeRegressor(
                    base_regressor=clone(reg_base), 
                    hidden_dims=(64,), 
                    lam=100.0, 
                    epochs=100,
                    lr=0.01
                )
                frappe_model.fit(X_train_nf, S_train_nf, y_train_nf, X_unlab=X_unlab_nf, S_unlab=S_unlab_nf)

                y_test_eval, preds, grid = predict_inverse_scale_and_grid(
                    reg_base=reg_base,
                    fair_reg=fair_reg_model,
                    ot_model=ot_model,
                    aware=aware_model,
                    frappe_model=frappe_model,
                    split=split,
                    scaler=scaler,
                    dp_nbins=args.dp_nbins
                )

                row = {
                    "base_type": base_type,
                    "run": i,
                    
                    # MSE
                    "mse_base": mse(y_test_eval, preds["ERM Unfair"]),
                    "mse_fairreg": mse(y_test_eval, preds["FairReg"]),
                    "mse_frappe": mse(y_test_eval, preds["Frappe"]),
                    "mse_ot": mse(y_test_eval, preds["OT Unaware"]),
                    "mse_aware": mse(y_test_eval, preds["OT Aware"]),
                    
                    # KS 
                    "dp_base": dp_cdf_max_gap(preds["ERM Unfair"], S_test, bins=grid),
                    "dp_fairreg": dp_cdf_max_gap(preds["FairReg"], S_test, bins=grid),
                    "dp_frappe": dp_cdf_max_gap(preds["Frappe"], S_test, bins=grid),
                    "dp_ot": dp_cdf_max_gap(preds["OT Unaware"], S_test, bins=grid),
                    "dp_aware": dp_cdf_max_gap(preds["OT Aware"], S_test, bins=grid),
                    
                    # W2 Gap
                    "w2_base": w2_gap_between_groups(preds["ERM Unfair"], S_test),
                    "w2_fairreg": w2_gap_between_groups(preds["FairReg"], S_test),
                    "w2_frappe": w2_gap_between_groups(preds["Frappe"], S_test),
                    "w2_ot": w2_gap_between_groups(preds["OT Unaware"], S_test),
                    "w2_aware": w2_gap_between_groups(preds["OT Aware"], S_test),
                }

                all_results.append(row)


            df_type = pd.DataFrame([r for r in all_results if r["base_type"] == base_type])
            df_type.to_csv(type_dir / "runs.csv", index=False)
            

            write_summary_relative_tex(
                df_type, 
                out_path=type_dir / "summary_relative.tex",
                dataset_name=f"{args.dataset} ({base_type})",
                percent=False
            )
            print(f"Saved results for {base_type} in {type_dir}")


    dataset_dir = exp_root / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not args.load_csv:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(dataset_dir / "all_runs_consolidated.csv", index=False)
        

    grouped = df_all.groupby("base_type")
    means = grouped.mean()
    stds = grouped.std()

    methods = {
        "ERM Unfair": ("mse_base", "dp_base", "w2_base"),
        "FairReg": ("mse_fairreg", "dp_fairreg", "w2_fairreg"),
        "FRAPPÉ": ("mse_frappe", "dp_frappe", "w2_frappe"),
        "FAUST-U": ("mse_ot", "dp_ot", "w2_ot"),
        "FAUST-A": ("mse_aware", "dp_aware", "w2_aware"),
    }
    
    base_order = ["linear", "rf", "gbm"]
    base_order = [b for b in base_order if b in means.index]

    table_rows = []
    from scipy.stats import wilcoxon

    for base in base_order:
        df_base = df_all[df_all["base_type"] == base]
        m_base = means.loc[base]
        
        unaware_methods = ["FAUST-U", "FairReg", "FRAPPÉ"]
        
        best_mse = min(m_base[methods[m][0]] for m in unaware_methods)
        best_dp  = min(m_base[methods[m][1]] for m in unaware_methods)
        best_w2  = min(m_base[methods[m][2]] for m in unaware_methods)
        
        best_method_mse = [m for m in unaware_methods if m_base[methods[m][0]] == best_mse][0]
        best_method_dp  = [m for m in unaware_methods if m_base[methods[m][1]] == best_dp][0]
        best_method_w2  = [m for m in unaware_methods if m_base[methods[m][2]] == best_w2][0]

        best_raw_mse = df_base[methods[best_method_mse][0]]
        best_raw_dp  = df_base[methods[best_method_dp][1]]
        best_raw_w2  = df_base[methods[best_method_w2][2]]
        
        for i, (method_name, cols) in enumerate(methods.items()):
            mse_col, dp_col, w2_col = cols
            
            m_mse, s_mse = m_base[mse_col], stds.loc[base, mse_col]
            m_dp, s_dp   = m_base[dp_col], stds.loc[base, dp_col]
            m_w2, s_w2   = m_base[w2_col], stds.loc[base, w2_col]
            
            raw_mse, raw_dp, raw_w2 = df_base[mse_col], df_base[dp_col], df_base[w2_col]
            

            def format_val(val_mean, val_std, best_mean, raw_data, best_raw_data, current_method):
                base_str = f"{val_mean:.3f} ± {val_std:.3f}"
                if current_method not in ["FAUST-U", "FairReg", "FRAPPÉ"]:
                    return base_str
                
                if abs(val_mean - best_mean) < 1e-6:
                    return f"\\textbf{{{base_str}}}"
                else:
                    try:
                        diff = np.asarray(raw_data) - np.asarray(best_raw_data)
                        if not np.all(diff == 0): 
                            _, pval = wilcoxon(raw_data, best_raw_data)
                            if pval > 0.05:
                                return f"\\textcolor{{blue}}{{{base_str}}}"
                    except Exception:
                        pass
                return base_str

            mse_val = format_val(m_mse, s_mse, best_mse, raw_mse, best_raw_mse, method_name)
            dp_val  = format_val(m_dp, s_dp, best_dp, raw_dp, best_raw_dp, method_name)
            w2_val  = format_val(m_w2, s_w2, best_w2, raw_w2, best_raw_w2, method_name)
            
            display_base = base.upper() if i == 0 else ""
            
            row = {
                "Base Model": display_base,
                "_base_model": base.upper(), 
                "Method": method_name,
                "MSE": mse_val,
                "$W_2$": w2_val,
                "KS$^{\\mathcal{T}}$": dp_val,
            }
            table_rows.append(row)
            
    df_table = pd.DataFrame(table_rows)
    
    latex_str = generate_latex_table(df_table)
    with open(dataset_dir / "formatted_summary.tex", "w") as f:
        f.write(latex_str)



    csv_display = df_table.drop(columns=["_base_model"]).copy()
    for col in ["MSE", "$W_2$", "KS$^{\\mathcal{T}}$"]:
        csv_display[col] = csv_display[col].apply(lambda x: re.sub(r'\\textbf\{([^}]+)\}', r'\1', str(x)))
        csv_display[col] = csv_display[col].apply(lambda x: re.sub(r'\\textcolor\{blue\}\{([^}]+)\}', r'\1', str(x)))
    

    csv_display.to_csv(dataset_dir / "formatted_summary.csv", index=False)
        

    print("\n" + "="*70)
    print(f" FINAL RESULTS FOR: {args.dataset.upper()}")
    print("="*70)
    print(csv_display.to_string(index=False))
    print(f"\nSaved CSV: {dataset_dir / 'formatted_summary.csv'}")
    print(f"Saved LaTeX: {dataset_dir / 'formatted_summary.tex'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "lawschool", "communities", "synthetic"], default="lawschool")
    parser.add_argument("--base_types", type=str, default="linear,rf,gbm", help="Comma-separated list of base models to test")
    parser.add_argument("--num", type=int, default=10, help="Number of random splits per model type")
    

    parser.add_argument("--fairreg_root", default=None)
    parser.add_argument("--out_dir", default="notes/figs")
    parser.add_argument("--train_size", type=float, default=0.4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--full_data", action="store_true", default=True)
    parser.add_argument("--no_scaling", action="store_true")
    
    parser.add_argument("--merge_train_unlab_for_non_fairreg", action="store_true",
                        help="If set, OT and Aware models will use the union of Train and Unlabeled data for fairness calibration.")
    

    parser.add_argument("--cont_map", default="rf")
    parser.add_argument("--cont_epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn_reg", type=float, default=None)
    parser.add_argument("--delta_tol", type=float, default=1e-6)
    

    parser.add_argument("--dp_nbins", type=int, default=41)
    parser.add_argument("--T", type=int, default=-1, help="FairReg iterations. Set to -1 for dynamical T (2.5 * len(X_unlab))")
    parser.add_argument("--eps", type=float, default=0.00001) 
    parser.add_argument("--tune_fairreg", action="store_true", help="Enable grid search tuning for L and beta on a validation split.")
    parser.add_argument("--val_frac_train", type=float, default=0.25)
    parser.add_argument("--L_grid", default="auto,0.5x,1x,2x")
    parser.add_argument("--beta_grid", default="auto,0.25x,0.5x,1x,2x,4x")
    parser.add_argument("--tune_objective", choices=["feasible_min_risk", "penalized"], default="feasible_min_risk")
    parser.add_argument("--penalty_lambda", type=float, default=100.0)
    parser.add_argument("--L", default="auto")
    parser.add_argument("--beta", default="auto")

    parser.add_argument("--load_csv", type=str, default=None, help="Path to a consolidated CSV to skip training.")

    # Synthetic params
    parser.add_argument("--synthetic_n", type=int, default=10000)

    args = parser.parse_args()
    main(args)
