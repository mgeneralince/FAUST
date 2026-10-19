import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import argparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from fair_unaware.esimator import UnawareFairRegressor
from fair_unaware.baselines.fairreg_adapter import load_dataset
from fair_unaware.metrics import w2_gap_between_groups


plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["Times New Roman"] 
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


def main(args):
    redirect_warnings_to_file(Path("notes/figs/stress_tests/finite_sample_results/finite_sample_warnings.log"))
    dataset = args.dataset
    fairreg_root = "../unaware-fair-reg" 
    out_dir = Path("notes/figs/stress_tests/finite_sample_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #nb runs 
    n_runs = 10 
    
    # sample size to test (N)
    sample_sizes = [50, 100, 500, 1000, 5000, 10000] 
    
    #lambda for the tradoff
    lambda_grid = [0.0, 1e-4, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
    
    csv_path = out_dir / f"finite_sample_data_{dataset}.csv"
    
    print(f"Loading {dataset} dataset...")
    X, S, y, scaler, _, _, _, _ = load_dataset(dataset, fairreg_root=fairreg_root, data_scaling=True, full_data=True)
    
    # fix test set for all methods fair comparison
    X_train_full, X_test, S_train_full, S_test, y_train_full, y_test = train_test_split(
        X, S, y, test_size=0.2, stratify=S, random_state=42
    )
    
    results = []
    
    if csv_path.exists():
        print(f"\n--- Found existing data for {dataset}! Loading... ---")
        df_agg = pd.read_csv(csv_path)
    else:
        print(f"\n--- Running Finite Sample Stress Test ({n_runs} runs per point) ---")
        
        for run_idx in tqdm(range(n_runs), desc="Random Seeds"):
            for N in tqdm(sample_sizes, desc="Sample Sizes (N)", leave=False):
                
                #chek we ahve enough sampoe in the dataset
                actual_N = min(N, len(X_train_full))
                
                #subsample
                if actual_N == len(X_train_full):
                    X_train_N, S_train_N, y_train_N = X_train_full, S_train_full, y_train_full
                else:
                    X_train_N, _, S_train_N, _, y_train_N, _ = train_test_split(
                        X_train_full, S_train_full, y_train_full, 
                        train_size=actual_N, stratify=S_train_full, random_state=run_idx
                    )
                
                for lam in lambda_grid:
                    #base estimators
                    base_lr = LinearRegression(fit_intercept=True)
                    base_clf = LogisticRegression(max_iter=2000)
                    
                    #OT trianing
                    ot_model = UnawareFairRegressor(
                        base_regressor=base_lr, base_classifier=base_clf,
                        relax_lambda=lam, continuous_map="rf", verbose=False
                    )
                    
                    try:
                        ot_model.fit(X_train_N, S_train_N, y_train_N)
                        y_pred_scaled = ot_model.predict(X_test)
                        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[:, 0]
                        y_test_eval = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))[:, 0]
                        
                        mse_val = mean_squared_error(y_test_eval, y_pred)
                        w2_val = w2_gap_between_groups(y_pred, S_test, squared=False)
                    except Exception as e:
                        #if the model crashes and misses a class
                        mse_val = np.nan
                        w2_val = np.nan
                        
                    results.append({
                        "Run": run_idx,
                        "Sample_Size_N": actual_N,
                        "Lambda": lam,
                        "MSE": mse_val,
                        "W2": w2_val
                    })

        df = pd.DataFrame(results).dropna()
        df_agg = df.groupby(["Sample_Size_N", "Lambda"]).agg(
            MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
            W2_mean=("W2", "mean"), W2_std=("W2", "std")
        ).reset_index()
        df_agg.to_csv(csv_path, index=False)

    # PLOT
    df_agg = df_agg[df_agg["Sample_Size_N"] > 99]


    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    lam_unc = df_agg["Lambda"].min()
    lam_str = df_agg["Lambda"].max()
    
    df_unc = df_agg[df_agg["Lambda"] == lam_unc].sort_values("Sample_Size_N")
    df_str = df_agg[df_agg["Lambda"] == lam_str].sort_values("Sample_Size_N")
    
    available_N = sorted(df_agg["Sample_Size_N"].unique())
    
    # MSE vs N
    
    axes[0].plot(df_str["Sample_Size_N"], df_str["MSE_mean"], marker='s', color="tab:blue", lw=2)
    axes[0].fill_between(df_str["Sample_Size_N"], df_str["MSE_mean"] - df_str["MSE_std"], 
                         df_str["MSE_mean"] + df_str["MSE_std"], color="tab:blue", alpha=0.15)
    
    axes[0].set_xscale("log")
    axes[0].set_xticks(available_N)
    axes[0].set_xticklabels([str(n) for n in available_N])
    axes[0].set_xlabel("Sample Size (N)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE vs. N")
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    axes[0].grid(True, alpha=0.3)
    # axes[0].legend()
    
    # W2 vs N 
    
    axes[1].plot(df_str["Sample_Size_N"], df_str["W2_mean"], marker='s', color="tab:blue", lw=2)
    axes[1].fill_between(df_str["Sample_Size_N"], df_str["W2_mean"] - df_str["W2_std"], 
                         df_str["W2_mean"] + df_str["W2_std"], color="tab:blue", alpha=0.15)
    
    axes[1].set_xscale("log")
    axes[1].set_xticks(available_N)
    axes[1].set_xticklabels([str(n) for n in available_N])
    axes[1].set_xlabel("Sample Size (N)")
    axes[1].set_ylabel(r"$W_2$")
    axes[1].set_title(r"Unfairness vs. N")
    axes[1].grid(True, alpha=0.3)
    # axes[1].legend()

    # Pareto frontier 
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(available_N)))
    
    for idx, N in enumerate(available_N):
        sub = df_agg[df_agg["Sample_Size_N"] == N].sort_values(by="W2_mean")
        
        axes[2].plot(sub["W2_mean"], sub["MSE_mean"], marker='o', 
                color=colors[idx], label=f"N = {N}", lw=2, markersize=6)
        
        axes[2].fill_between(sub["W2_mean"], 
                        sub["MSE_mean"] - sub["MSE_std"], 
                        sub["MSE_mean"] + sub["MSE_std"], 
                        color=colors[idx], alpha=0.15)
        
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    axes[2].set_xlabel(r"$W_2$")
    axes[2].set_ylabel("MSE")
    axes[2].set_title("Accuracy-Fairness Tradeoff")
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    axes[2].legend(title="Training Size", fontsize=10, loc= 'upper right')
    # axes[2].legend(title="Training Size", fontsize=10, bbox_to_anchor=(0.05, 1), loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_path = out_dir / f"finite_sample_combined_{dataset}.pdf"
    fig.savefig(plot_path, bbox_inches='tight', transparent=True)
    print(f"\nSaved Combined Plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run finite sample stress test for OT Unaware.")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="communities", 
        choices=["adult", "lawschool", "communities", "synthetic"],
        help="The name of the dataset to run the test on."
    )
    
    args = parser.parse_args()
    main(args)

