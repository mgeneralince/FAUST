import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.synthetic import make_synthetic_dataset
from fair_unaware.experiments.confounded import run_one_all_methods
from fair_unaware.plots.visualization import (
    plot_gamma_sweep_multi,        
    scatter_XY_colored_by_S,   
    plot_eta_delta_scatter,  
    plot_gamma_sweep_relative_curves,
    plot_mse_vs_w2_tradeoff  
)
from fair_unaware.utils import _latex_escape


import warnings 
from joblib import Parallel, delayed
import contextlib
from joblib import parallel

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback

log_file = open("model_warnings.log", "w")
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    log_file.write(warnings.formatwarning(message, category, filename, lineno, line))
    log_file.flush()
warnings.showwarning = custom_warning_handler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def worker(g, run_idx, args):
    """Worker function to run a single experiment configuration."""
    import warnings
    

    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):

        with open("model_warnings.log", "a") as f:
            f.write(warnings.formatwarning(message, category, filename, lineno, line))
            
    warnings.showwarning = custom_warning_handler


    current_seed = args.seed + run_idx 
    
    X, S, Y = make_synthetic_dataset(
        n=args.n,
        gamma=float(g),
        #beta=float(args.beta),
        random_state=current_seed,
    )

    res = run_one_all_methods(
        args.method, X, S, Y,
        gamma=float(g),
        beta=float(args.beta),
        use_oracle=args.use_oracle,
        seed=current_seed,
        return_artifacts=True,
        cont_map=args.cont_map,
        cont_epsilon=args.cont_epsilon,
        cont_knn=args.cont_knn,
        sinkhorn_reg=args.sinkhorn_reg,
        delta_tol=args.delta_tol,
        fairreg_root=args.fairreg_root,
        fairreg_L=args.fairreg_L,
        fairreg_beta=args.fairreg_beta,
        fairreg_T=args.fairreg_T,
        fairreg_B=args.fairreg_B,
        fairreg_eps_list=args.fairreg_eps_list,
    )

    metrics = res[0] 
    metrics["gamma"] = float(g)
    metrics["run"] = run_idx
    
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["OT", "Sinkhorn"], default="OT")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--num_runs", type=int, default=10)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma_min", type=float, default=0.01)
    p.add_argument("--gamma_max", type=float, default=0.99)
    p.add_argument("--use_oracle", action="store_true", help="Whether to use oracle models (which know the true data-generating process) instead of fitted models. This is for sanity-checking purposes: we should see the same trends across gamma, but with better absolute performance.")
    p.add_argument("--gamma_steps", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="notes/figs/")
    p.add_argument("--cont_map", choices=["rf", "entropic", "knn"], default="rf")
    p.add_argument("--cont_epsilon", type=float, default=0.05)
    p.add_argument("--cont_knn", type=int, default=None)
    p.add_argument(
        "--sinkhorn_reg",
        type=float,
        default=0.01,
        help="Used only if --method=Sinkhorn. Typical values: 0.005–0.05.",
    )
    p.add_argument("--delta_tol", type=float, default=1e-6)
    p.add_argument(
        "--load_results", 
        type=str, 
        default=None, 
        help="Path to a raw_results CSV file to skip computation."
    )

    # FairReg baseline 
    p.add_argument(
        "--fairreg_root",
        type=str,
        default=None,
        help="Path to unaware-fair-reg repo if FairReg is not installed as a package.",
    )
    p.add_argument("--fairreg_L", type=str, default="auto")
    p.add_argument("--fairreg_beta", type=str, default="auto")
    p.add_argument("--fairreg_T", type=int, default=None, help="If none, computed to have 60 epochs.")
    p.add_argument("--fairreg_B", type=float, default=1.0, help="1 because data are sclaed between -1 and 1")
    p.add_argument(
        "--fairreg_eps_list",
        type=str,
        default="0, 0",
    )

    args = p.parse_args()

    fig_root = Path(args.out_dir)

    if args.use_oracle:
        oracle_folder = "oracle_bayes"
    else:
        oracle_folder = "learned_models"

    exp_root = fig_root / "confounded" / oracle_folder / f"{args.cont_map}"
    xy_dir = exp_root / "scatter_XY_by_gamma"
    ed_dir = exp_root / "eta_delta_by_gamma"
    tables_dir = exp_root / "tables"

    for d in [xy_dir, ed_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    #logscale
    gammas = np.linspace(args.gamma_min, args.gamma_max, args.gamma_steps)
    results_csv_path = exp_root / f"raw_results_{args.method}_n{args.n}_beta{args.beta}.csv"

    if args.load_results is not None:
        print(f"Loading existing results from: {args.load_results}")
        df_rows_all = pd.read_csv(args.load_results)
        rows = df_rows_all.to_dict('records')
        
    else:

        #parallelize across gamma values and runs
        tasks = [(g, run_idx) for g in gammas for run_idx in range(args.num_runs)]
        
        print(f"Launching {len(tasks)} tasks in parallel...")

        with tqdm_joblib(tqdm(desc="Experiment Progress", total=len(tasks))):
            rows = Parallel(n_jobs=-1)(
                delayed(worker)(g, run_idx, args) for g, run_idx in tasks
            )
        
        df_rows_all = pd.DataFrame(rows)
        df_rows_all.to_csv(results_csv_path, index=False)
        print(f"Saved raw data to: {results_csv_path}")


    df_mean = df_rows_all.groupby("gamma").mean().reset_index()
    df_std = df_rows_all.groupby("gamma").std().reset_index().fillna(0.0)

    df_rows = pd.DataFrame({"gamma": df_mean["gamma"]})
    needed = ["mse_base", "mse_ot", "mse_aware", "mse_fairreg", "mse_ablation", "mse_shat", "mse_frappe",
              "w2_base",  "w2_ot",  "w2_aware",  "w2_fairreg",  "w2_ablation",  "w2_shat", "w2_frappe",
              "dpmax_base", "dpmax_ot", "dpmax_aware", "dpmax_fairreg", "dpmax_ablation", "dpmax_shat", "dpmax_frappe",
              "var_base", "var_ot", "var_aware", "var_fairreg", "var_ablation", "var_shat", "var_frappe"]
    for col in needed:
        df_rows[f"{col}_mean"] = df_mean.get(col, np.nan)
        df_rows[f"{col}_std"] = df_std.get(col, np.nan)

    df_rows = df_rows.sort_values("gamma").reset_index(drop=True)

    indices = np.linspace(0, len(df_rows) - 1, 5, dtype=int)
    df_rows_filtered = df_rows.iloc[indices].copy()

    method_specs = [
                ("base",     "ERM Unfair"),    
                ("ot",       "FAUST-U"), 
                ("ablation", r"FAUST-U ($\tilde{\eta}$)"), 
                ("shat",     r"Plug-In W2"),   
                ("aware",    "FAUST-A"),      
                ("fairreg",  "FairReg"),  
                ("frappe",   "FRAPPÉ"),
            ]

    def build_block(df_subset: pd.DataFrame) -> pd.DataFrame:
        recs = []
        for _, r in df_subset.iterrows():
            g = float(r["gamma"])
            for key, method_name in method_specs:
                recs.append({
                    r"$\gamma$": g,  
                    "Method": method_name,
                    "MSE": r.get(f"mse_{key}_mean", np.nan),
                    "MSE_std": r.get(f"mse_{key}_std", np.nan),
                    r"DP$_{\max}$": r.get(f"dpmax_{key}_mean", np.nan),
                    r"DP$_{\max}$_std": r.get(f"dpmax_{key}_std", np.nan),
                    r"$W_2$": r.get(f"w2_{key}_mean", np.nan),
                    r"$W_2$_std": r.get(f"w2_{key}_std", np.nan),
                })
        tab = pd.DataFrame(recs)

        metric_cols = ["MSE", r"DP$_{\max}$", r"$W_2$"]
        method_order = ["ERM Unfair", "FairReg", "FRAPPÉ", "FAUST-U", r"FAUST-U ($\tilde{\eta}$)", r"Plug-In $W_2$", "FAUST-A"]
        tab["Method"] = pd.Categorical(tab["Method"], categories=method_order, ordered=True)
        tab = tab.sort_values([r"$\gamma$", "Method"], kind="stable")


        for g_val, g_block in tab.groupby(r"$\gamma$", sort=False):
            for c in metric_cols:
                vals = g_block[c].to_numpy(dtype=float)
                if not np.isfinite(vals).any():
                    for idx in g_block.index:
                        tab.loc[idx, c] = "--"
                    continue

                best = np.nanmin(vals)
                mask = np.isfinite(g_block[c]) & (g_block[c] == best)

                for idx in g_block.index:
                    mean_val = tab.loc[idx, c]
                    std_val = tab.loc[idx, f"{c}_std"]
                    
                    if not np.isfinite(mean_val):
                        tab.loc[idx, c] = "--"
                    elif idx in g_block.index[mask]:
                        tab.loc[idx, c] = rf"\textbf{{{mean_val:.2f}}} $\pm$ {std_val:.2f}"
                    else:
                        tab.loc[idx, c] = rf"{mean_val:.2f} $\pm$ {std_val:.2f}"

        tab = tab.drop(columns=[f"{c}_std" for c in metric_cols])
        tab[r"$\gamma$"] = tab[r"$\gamma$"].map(lambda x: f"{float(x):.2f}")
        tab = tab.rename(columns={
            "MSE":  r"$\mathrm{MSE}\,\downarrow$",
            r"DP$_{\max}$": r"$\mathrm{DP}_{\max}\,\downarrow$",
            r"$W_2$":   r"$W_2\,\downarrow$",
        })
        tab = tab.set_index([r"$\gamma$", "Method"])
        return tab

    tab_full = build_block(df_rows_filtered)

    latex_tabular = tab_full.to_latex(
        escape=False,
        float_format="%.2f",
        na_rep="--",
        multirow=True,
        column_format="llrrr",
    )

    tex_path = tables_dir / f"sweep_{args.method}_n{args.n}_beta{args.beta}.tex"
    caption = _latex_escape(rf"Confounded sweep ({args.method}, n={args.n}, $\beta$={args.beta}, continuous map={args.cont_map}).")
    label = f"tab:confounded-{args.method}-n{args.n}-beta{args.beta}-{args.cont_map}"

    latex_full = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"{latex_tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    tex_path.write_text(latex_full, encoding="utf-8")

    print("Saved:", tex_path)

    pretty_labels = {
        "base": "ERM",
        "ot": r"FAUST-U $\mathcal{W}_2$",
        "aware": r"FAUST-A $\mathcal{W}_2$",
        "fairreg": "FairReg",
        "frappe": "FRAPPÉ",
        "ablation": r"FAUST-U ($\tilde{\eta}$)",
        "shat": r"Plug-In $\mathcal{W}_2$"
    }
    pretty_colors = {
            "base": "black",
            "ot": "tab:blue",
            "aware": "tab:green",
            "fairreg": "tab:purple",
            "frappe": "tab:pink",
            "ablation": "salmon",
            "shat": "tab:cyan"
        }
    
    pretty_linestyles = {
        "base": "-",        
        "ot": "-",          
        "aware": "-",       
        "fairreg": "-",     
        "frappe": "-",
        "ablation": "--",   
        "shat": "-"         
    }
    # RELATIVE CURVE 
    fig_path = exp_root / f"sweep_{args.method}_n{args.n}_beta{args.beta}_{args.gamma_steps}g.pdf"
    
    combined_path_rel = plot_gamma_sweep_multi(
        rows,
        save_path=fig_path,
        dp_keys=("w2_base", "w2_ot", "w2_aware", "w2_fairreg", "w2_frappe", "w2_shat"),
        mse_keys=("mse_base", "mse_ot", "mse_aware", "mse_fairreg", "mse_frappe", "mse_shat"),
        relative=True,
        relative_mode="ratio",   
        marker=None,           
        dp_ylabel=r"$W_2$",
        labels_dict=pretty_labels,
        colors_dict=pretty_colors,
        linestyles_dict=pretty_linestyles
    )
    print("Saved Relative sweep:", combined_path_rel)

    # ABSOLUTE CURVE 
    fig_path_abs = exp_root / f"sweep_abs_{args.method}_n{args.n}_beta{args.beta}_{args.gamma_steps}g.pdf"
    
    combined_path_abs = plot_gamma_sweep_multi(
        rows,
        save_path=fig_path_abs,
        dp_keys=("w2_base", "w2_ot", "w2_aware", "w2_fairreg", "w2_frappe", "w2_shat"),    
        mse_keys=("mse_base", "mse_ot", "mse_aware", "mse_fairreg", "mse_frappe", "mse_shat"),
        relative=False,         
        marker=None,            
        dp_ylabel=r"$W_2$",
        labels_dict=pretty_labels,
        colors_dict=pretty_colors  , 
        linestyles_dict=pretty_linestyles
    )
    print("Saved Absolute sweep:", combined_path_abs)

    # TRADEOFF CURVE (MSE vs W2)
    tradeoff_path = exp_root / f"tradeoff_mse_w2_{args.method}_n{args.n}_beta{args.beta}_{args.gamma_steps}g.pdf"
    
    plot_mse_vs_w2_tradeoff(
        rows,
        save_path=tradeoff_path,
        w2_keys=("w2_base", "w2_ot", "w2_aware", "w2_fairreg", "w2_frappe", "w2_shat"),
        mse_keys=("mse_base", "mse_ot", "mse_aware", "mse_fairreg", "mse_frappe", "mse_shat"),
        labels_dict=pretty_labels,
    )
    print("Saved Trade-off sweep:", tradeoff_path)

    # VARIANCE SWEEP 
    fig_path_var = exp_root / f"sweep_abs_var_{args.method}_n{args.n}_beta{args.beta}_{args.gamma_steps}g.pdf"
    
    combined_path_var = plot_gamma_sweep_multi(
        rows,
        save_path=fig_path_var,
        dp_keys=("var_base", "var_ot", "var_aware", "var_fairreg", "var_frappe", "var_shat"),
        mse_keys=("mse_base", "mse_ot", "mse_aware", "mse_fairreg", "mse_frappe", "mse_shat"),
        relative=False,         
        marker=None,            
        title_prefix="Within-Group Variance",
        dp_ylabel=r"Variance ($\sigma^2$)",
        log_x=True,
        log_y=True, 
        labels_dict=pretty_labels,
        colors_dict=pretty_colors ,
        linestyles_dict=pretty_linestyles
    )
    print("Saved Variance sweep:", combined_path_var)

if __name__ == "__main__":
    main()
