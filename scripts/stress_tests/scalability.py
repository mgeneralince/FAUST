import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import types
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.synthetic import make_synthetic_dataset
from fair_unaware import UnawareFairRegressor
import logging
from pathlib import Path
import warnings
from tqdm import tqdm 


def generate_latex_table(df, out_path="notes/figs/stress_tests/scalability/finite_sample_results/scalability_table.tex"):
    """
    Groups the benchmark dataframe by N_unlab and generates a LaTeX table
    matching the exact formatting of the paper's appendix.
    """
    agg = df.groupby("N_unlab").agg(["mean", "std"])
    
    lines = [
        r"\begin{table}[ht!]",
        r"    \centering",
        r"    \begin{tabular}{r S[table-format=4.2] @{\,\(\pm\)\,} S[table-format=3.2] S[table-format=4.2] @{\,\(\pm\)\,} S[table-format=3.2]}",
        r"    \toprule",
        r"    \textbf{Unlabeled Samples ($N$)} & \multicolumn{2}{c}{\textbf{Exact OT Step (sec) $\downarrow$}} & \multicolumn{2}{c}{\textbf{Total Pipeline Time (sec) $\downarrow$}} \\",
        r"    \midrule"
    ]
    
    for N in agg.index:
        ot_mean = agg.loc[N, ("ot_time_emd", "mean")]
        ot_std = agg.loc[N, ("ot_time_emd", "std")]
        tot_mean = agg.loc[N, ("total_time_emd", "mean")]
        tot_std = agg.loc[N, ("total_time_emd", "std")]
        
        row = f"    {int(N):,} & {ot_mean:.2f} & {ot_std:.2f} & {tot_mean:.2f} & {tot_std:.2f} \\\\"
        lines.append(row)
        
    lines.extend([
        r"    \bottomrule",
        r"    \end{tabular}",
        r"    \vspace{0.5em}",
        r"    \caption{\textbf{Empirical scalability of the exact discrete OT mapping.} Runtimes evaluated across increasing dataset sizes.}",
        r"    \label{tab:scalability_n}",
        r"\end{table}"
    ])
    
    latex_str = "\n".join(lines)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
        
    print(f"\n=== LaTeX Table Saved to {out_path} ===")
    print(latex_str)

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


def run_scalability_benchmark(sample_sizes=[1000, 2000, 5000, 10000, 20000, 50000], n_runs=5):
    redirect_warnings_to_file(Path("notes/figs/stress_tests/scalability/finite_sample_results/scalability_warnings.log"))
    results = []

    for N in tqdm(sample_sizes, desc="Sample Sizes"):
        print(f"\n--- Testing N_unlab = {N} ---")
        
        for run in tqdm(range(n_runs), desc="Runs", leave=False):
            # Generate synthetic data
            X, S, Y = make_synthetic_dataset(n=N, random_state=42 + run)
            
            # exact EMD 
            ot_exact = UnawareFairRegressor(
                base_regressor=LinearRegression(),
                base_classifier=LogisticRegression(max_iter=2000),
                sinkhorn_reg=None,  
                relax_lambda=0.5,   
                random_state=42 + run
            )
            
            # fit time 
            t0 = time.perf_counter()
            ot_exact.fit(X, S, Y, X_unlab=X)
            t_total_exact = time.perf_counter() - t0
            t_ot_exact = ot_exact.timings_.get("ot_total", np.nan)

            # Sinkhorn Approximation
            ot_sinkhorn = UnawareFairRegressor(
                base_regressor=LinearRegression(),
                base_classifier=LogisticRegression(max_iter=2000),
                sinkhorn_reg=0.01, 
                relax_lambda=0.5,
                random_state=42 + run
            )
            
            t0 = time.perf_counter()
            ot_sinkhorn.fit(X, S, Y, X_unlab=X)
            t_total_sinkhorn = time.perf_counter() - t0
            t_ot_sinkhorn = ot_sinkhorn.timings_.get("ot_total", np.nan)

            results.append({
                "N_unlab": N,
                "run": run,
                "ot_time_emd": t_ot_exact,
                "total_time_emd": t_total_exact,
                "ot_time_sinkhorn": t_ot_sinkhorn,
                "total_time_sinkhorn": t_total_sinkhorn,
            })

    df = pd.DataFrame(results)
    summary = df.groupby("N_unlab").agg(["mean", "std"])
    print("\n=== Benchmark Summary (Seconds) ===")
    print(summary)
    
    csv_path = Path("notes/figs/stress_tests/scalability/scalability_benchmark.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    generate_latex_table(df, out_path="notes/figs/stress_tests/scalability/scalability_table.tex")
    
    return df

if __name__ == "__main__":
    run_scalability_benchmark()