import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.synthetic import make_synthetic_dataset
from fair_unaware.metrics import w2_gap_between_groups
from fair_unaware.ot.barycenter import compute_cost_matrix, solve_ot_plan, compute_barycentric_predictions_from_plan
from sklearn.linear_model import LinearRegression, LogisticRegression
import logging
import warnings
from tqdm import tqdm


plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "stix",
    "font.family": "serif"
})



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


def solve_minibatch_barycenter(h_plus, d_plus, h_minus, d_minus, a, b, batch_size=2000):
    """
    Minibatch OT paradigm. Randomly partitions the measures into blocks of size `batch_size`.
    """
    idx_plus = np.random.permutation(len(h_plus))
    idx_minus = np.random.permutation(len(h_minus))
    
    K_batches = max(1, len(h_plus) // batch_size)
    
    splits_plus = np.array_split(idx_plus, K_batches)
    splits_minus = np.array_split(idx_minus, K_batches)
    
    f_plus = np.zeros(len(h_plus))
    f_minus = np.zeros(len(h_minus))
    
    for sp, sm in zip(splits_plus, splits_minus):
        hp_b, dp_b, a_b = h_plus[sp], d_plus[sp], a[sp]
        hm_b, dm_b, b_b = h_minus[sm], d_minus[sm], b[sm]
        
        a_b = a_b / (a_b.sum() + 1e-12)
        b_b = b_b / (b_b.sum() + 1e-12)
        
        C_b = compute_cost_matrix(hp_b, dp_b, hm_b, dm_b)
        pi_b = solve_ot_plan(a_b, b_b, C_b, sinkhorn_reg=None)
        
        fp_b, fm_b = compute_barycentric_predictions_from_plan(
            pi_b, hp_b, dp_b, hm_b, dm_b, a_b, b_b
        )
        
        f_plus[sp] = fp_b
        f_minus[sm] = fm_b
        
    return f_plus, f_minus, K_batches

def main():

    #PARAMs
    N = 200_000 
    batch_sizes = [50, 100, 500, 1000, 2000, 5000, 10000]
    out_dir = Path("notes/figs/stress_tests/minibatch_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    redirect_warnings_to_file(out_dir / "minibatch_scalability_warnings.log")
    
    print(f"--- Generating massive dataset (N={N}) ---")
    X, S, Y = make_synthetic_dataset(n=N, random_state=42)
    
    # BASE ESTIMATORS
    print("Training base estimators...")
    reg = LinearRegression().fit(X, Y)
    eta = reg.predict(X)
    
    clf = LogisticRegression(max_iter=500).fit(X, S)
    proba = clf.predict_proba(X)
    
    p0 = (S == clf.classes_[0]).mean()
    p1 = (S == clf.classes_[1]).mean()
    delta = proba[:, 0] / p0 - proba[:, 1] / p1
    
    # Unfairness
    w2_base = w2_gap_between_groups(eta, S, squared=False)
    mse_base = np.mean((Y - eta)**2) # <--- ADD THIS
    print(f"Base Unconstrained W2: {w2_base:.4f} | MSE: {mse_base:.4f}")
    
    #OT code as in the initial code 
    plus_mask = delta > 1e-6
    minus_mask = delta < -1e-6
    
    h_plus, d_plus = eta[plus_mask], delta[plus_mask]
    h_minus, d_minus = eta[minus_mask], delta[minus_mask]
    
    a = d_plus / d_plus.sum()
    b = -d_minus / (-d_minus.sum())
    
    # moving batch sizes 
    results = []
    print("\n" + "="*70)
    print(f"{'Batch Size (m)':<15} | {'Num Batches (K)':<15} | {'Time (sec)':<15} | {'W2 Unfairness':<15}")
    print("="*70)
    
    for m in tqdm(batch_sizes, desc="Minibatch Sizes"):
        t0 = time.time()
        f_plus_mb, f_minus_mb, K_batches = solve_minibatch_barycenter(
            h_plus, d_plus, h_minus, d_minus, a, b, batch_size=m
        )
        t_mb = time.time() - t0
        
        y_fair_mb = eta.copy()
        y_fair_mb[plus_mask] = f_plus_mb
        y_fair_mb[minus_mask] = f_minus_mb
        
        w2_mb = w2_gap_between_groups(y_fair_mb, S, squared=False)
        mse_mb = np.mean((Y - y_fair_mb)**2) 
        
        print(f"{m:<15} | {K_batches:<15} | {t_mb:<15.2f} | {w2_mb:<15.4f} | {mse_mb:<15.4f}")
        
        results.append({
            "Batch_Size_m": m,
            "Num_Batches_K": K_batches,
            "Time_sec": t_mb,
            "W2_Unfairness": w2_mb,
            "MSE": mse_mb 
        })
        
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "minibatch_ablation_results.csv", index=False)
    
    #LATEX TABLE
    def generate_latex_table(df_res, out_path):
        lines = [
            r"\begin{table}[ht!]",
            r"    \centering",
            r"    \begin{tabular}{r r S[table-format=4.2] S[table-format=1.3] S[table-format=1.3]}",
            r"    \toprule",
            r"    \textbf{Minibatch Size ($m$)} & \textbf{Num Batches ($K$)} & \textbf{OT Time (sec) $\downarrow$} & \textbf{MSE $\downarrow$} & \textbf{$\mathcal{W}_2 \downarrow$} \\",
            r"    \midrule"
        ]
        for _, row in df_res.iterrows():
            m = int(row['Batch_Size_m'])
            k = int(row['Num_Batches_K'])
            t = row['Time_sec']
            mse = row['MSE']
            w2 = row['W2_Unfairness']
            lines.append(f"    {m:,} & {k:,} & {t:.2f} & {mse:.3f} & {w2:.3f} \\\\")
        
        lines.extend([
            r"    \bottomrule",
            r"    \end{tabular}",
            r"    \vspace{0.5em}",
            r"    \caption{\textbf{Minibatch Optimal Transport on $N=200,000$.} Averaging the continuous maps across minibatches reduces the computation time to a fraction of a second while maintaining strict statistical parity.}",
            r"    \label{tab:minibatch}",
            r"\end{table}"
        ])
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved LaTeX table to {out_path}")

    generate_latex_table(df, out_dir / "minibatch_table.tex")
    
    #PLOT 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    
    #  W2 vs Batch Size
    ax1.plot(df["Batch_Size_m"], df["W2_Unfairness"], marker='o', color='tab:blue', lw=2)
    ax1.axhline(w2_base, color='gray', linestyle='--', label="Unconstrained Base")
    ax1.set_xscale('log')
    ax1.set_xlabel('Minibatch Size ($m$)')
    ax1.set_ylabel('Unfairness ($\mathcal{W}_2$ gap)')
    ax1.set_title('Fairness vs. Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #  MSE vs Batch Size
    ax2.plot(df["Batch_Size_m"], df["MSE"], marker='s', color='tab:green', lw=2)
    ax2.axhline(mse_base, color='gray', linestyle='--', label="Unconstrained Base")
    ax2.set_xscale('log')
    ax2.set_xlabel('Minibatch Size ($m$)')
    ax2.set_ylabel('MSE')
    ax2.set_title('Accuracy vs. Batch Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Time vs Batch Size
    ax3.plot(df["Batch_Size_m"], df["Time_sec"], marker='x', color='tab:red', lw=2)
    ax3.set_xscale('log')
    ax3.set_xlabel('Minibatch Size ($m$)')
    ax3.set_ylabel('Time (sec)')
    ax3.set_title('Computation Time')
    ax3.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(out_dir / "minibatch_ablation.pdf", transparent=True)
    print(f"Saved Plot to {out_dir / 'minibatch_ablation.pdf'}")

if __name__ == "__main__":
    main()