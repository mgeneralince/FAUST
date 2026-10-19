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
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error

from fair_unaware.esimator import UnawareFairRegressor
from fair_unaware.baselines.fairreg_adapter import load_dataset
from fair_unaware.metrics import w2_gap_between_groups
from fair_unaware.utils import _common_bins


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

# NOISE INJECTION 
class NoisyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_reg, noise_std=0.0, seed=None):
        self.base_reg = base_reg
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        
    def fit(self, X, y):
        self.base_reg.fit(X, y)
        return self
        
    def predict(self, X):
        eta = self.base_reg.predict(X)
        if self.noise_std > 0:
            eta += self.rng.normal(0, self.noise_std, size=eta.shape)
        return eta

class NoisyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf, noise_std=0.0, seed=None):
        self.base_clf = base_clf
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        
    def fit(self, X, y):
        self.base_clf.fit(X, y)
        self.classes_ = self.base_clf.classes_
        return self
        
    def predict_proba(self, X):
        proba = self.base_clf.predict_proba(X)
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, size=proba.shape[0])
            p_new = np.clip(proba[:, 1] + noise, 0, 1)
            proba = np.column_stack([1 - p_new, p_new])
        return proba


# MAIN EXPERIMENT

def main(args):
    redirect_warnings_to_file(Path("notes/figs/stress_tests/noise_injection/stress_test_warnings.log"))
    dataset= args.dataset
    fairreg_root = "../unaware-fair-reg" 
    out_dir = Path("notes/figs/stress_tests/noise_injection")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #runs
    n_runs = 5  
    
    # Isolated Metrics vs Noise
    noise_levels_smooth = np.linspace(0.0, 0.5, 6)
    fixed_lambda = 0.5
    
    # Accuracy-Fairness Trade-offs
    tradeoff_noise_levels = [0.0, 0.05, 0.1, 0.2]
    lambda_grid = [1e-4, 0.1, 0.5, 1.0, 10.0]
    
    iso_csv_path = out_dir / f"isolation_data_{dataset}.csv"
    trade_csv_path = out_dir / f"tradeoff_data_{dataset}.csv"
    

    if iso_csv_path.exists() and trade_csv_path.exists():
        print(f"\n--- Found existing data for {dataset}! Skipping computation. ---")
        df_iso_agg = pd.read_csv(iso_csv_path)
        df_trade_agg = pd.read_csv(trade_csv_path)
    
    else:
        print(f"\n--- No existing data found for {dataset}. Running computations... ---")
        print(f"Loading {dataset} dataset...")
        X, S, y, scaler, _, _, _, _ = load_dataset(dataset, fairreg_root=fairreg_root, data_scaling=True, full_data=True)
        
        X_train, X_test, S_train, S_test, y_train, y_test = train_test_split(
            X, S, y, test_size=0.2, stratify=S, random_state=42
        )
        
        isolation_results = []
        tradeoff_results = []
        
        print(f"\n--- Running Rigorous Stress Tests ({n_runs} runs per point) ---")
        for run_idx in tqdm(range(n_runs), desc="Random Seeds"):
            
            # 1. Isolation Experiment (Fixed Lambda)
            for noise in tqdm(noise_levels_smooth, desc="Isolation Noise Levels", leave=False):
                for noise_type in ["eta_only", "posterior_only"]:
                    
                    noise_eta = noise if noise_type == "eta_only" else 0.0
                    noise_clf = noise if noise_type == "posterior_only" else 0.0
                    
                    base_lr = LinearRegression(fit_intercept=True)
                    base_clf = LogisticRegression(max_iter=2000)
                    
                    noisy_reg = NoisyRegressor(base_lr, noise_std=noise_eta, seed=run_idx)
                    noisy_clf = NoisyClassifier(base_clf, noise_std=noise_clf, seed=run_idx)
                    
                    ot_model = UnawareFairRegressor(
                        base_regressor=noisy_reg, base_classifier=noisy_clf,
                        relax_lambda=fixed_lambda, continuous_map="rf", verbose=False
                    )
                    
                    ot_model.fit(X_train, S_train, y_train)
                    y_pred_scaled = ot_model.predict(X_test)
                    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[:, 0]
                    y_test_eval = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))[:, 0]
                    
                    isolation_results.append({
                        "Run": run_idx,
                        "Noise_Type": noise_type,
                        "Noise_STD": noise,
                        "MSE": mean_squared_error(y_test_eval, y_pred),
                        "W2": w2_gap_between_groups(y_pred, S_test, squared=False)
                    })

            # 2. Trade-off Experiment (Sweeping Lambda)
            for noise in tqdm(tradeoff_noise_levels, desc="Trade-off Noise Levels", leave=False):
                for noise_type in ["eta_only", "posterior_only"]:
                    for lam in tqdm(lambda_grid, desc="Lambda Grid", leave=False):
                        
                        noise_eta = noise if noise_type == "eta_only" else 0.0
                        noise_clf = noise if noise_type == "posterior_only" else 0.0
                        
                        base_lr = LinearRegression(fit_intercept=True)
                        base_clf = LogisticRegression(max_iter=2000)
                        
                        noisy_reg = NoisyRegressor(base_lr, noise_std=noise_eta, seed=run_idx)
                        noisy_clf = NoisyClassifier(base_clf, noise_std=noise_clf, seed=run_idx)
                        
                        ot_model = UnawareFairRegressor(
                            base_regressor=noisy_reg, base_classifier=noisy_clf,
                            relax_lambda=lam, continuous_map="rf", verbose=False
                        )
                        
                        ot_model.fit(X_train, S_train, y_train)
                        y_pred_scaled = ot_model.predict(X_test)
                        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[:, 0]
                        y_test_eval = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))[:, 0]
                        
                        tradeoff_results.append({
                            "Run": run_idx,
                            "Noise_Type": noise_type,
                            "Noise_STD": noise,
                            "Lambda": lam,
                            "MSE": mean_squared_error(y_test_eval, y_pred),
                            "W2": w2_gap_between_groups(y_pred, S_test, squared=False)
                        })

        df_iso = pd.DataFrame(isolation_results)
        df_iso_agg = df_iso.groupby(["Noise_Type", "Noise_STD"]).agg(
            MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
            W2_mean=("W2", "mean"), W2_std=("W2", "std")
        ).reset_index()
        df_iso_agg.to_csv(iso_csv_path, index=False)

        df_trade = pd.DataFrame(tradeoff_results)
        df_trade_agg = df_trade.groupby(["Noise_Type", "Noise_STD", "Lambda"]).agg(
            MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"), 
            W2_mean=("W2", "mean"), W2_std=("W2", "std")
        ).reset_index()
        df_trade_agg.to_csv(trade_csv_path, index=False)

    # METRICS DEGRADATION

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors_iso = {"eta_only": "tab:blue", "posterior_only": "tab:orange"}
    labels_iso = {"eta_only": r"Noise on $\eta$", "posterior_only": r"Noise on $\mathbb{P}(S|X)$"}
    
    for noise_type in ["eta_only", "posterior_only"]:
        sub = df_iso_agg[df_iso_agg["Noise_Type"] == noise_type]
        x_vals = sub["Noise_STD"]
        
        ax1.plot(x_vals, sub["MSE_mean"], marker='o', color=colors_iso[noise_type], label=labels_iso[noise_type], lw=2)
        ax1.fill_between(x_vals, sub["MSE_mean"] - sub["MSE_std"], sub["MSE_mean"] + sub["MSE_std"], color=colors_iso[noise_type], alpha=0.2)
        
        ax2.plot(x_vals, sub["W2_mean"], marker='o', color=colors_iso[noise_type], label=labels_iso[noise_type], lw=2)
        ax2.fill_between(x_vals, sub["W2_mean"] - sub["W2_std"], sub["W2_mean"] + sub["W2_std"], color=colors_iso[noise_type], alpha=0.2)

    ax1.set_title("MSE vs. Noise")
    ax1.set_xlabel("Noise (STD)")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(r"$W_2$ vs. Noise")
    ax2.set_xlabel("Noise (STD)")
    ax2.set_ylabel(r"$W_2$")
    ax2.legend()
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax2.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    plot1_path = out_dir / f"stress_test_isolation_metrics_{dataset}.pdf"
    fig1.savefig(plot1_path, bbox_inches='tight', transparent=True)
    print(f"\nSaved Image 1: {plot1_path}")

    #ACCURACY-FAIRNESS TRADEOFFS

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    tradeoff_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    # Noise on ETA
    for idx, noise in enumerate(tradeoff_noise_levels):
        sub_eta = df_trade_agg[(df_trade_agg["Noise_Type"] == "eta_only") & (df_trade_agg["Noise_STD"] == noise)].sort_values(by="W2_mean")
        ax3.plot(sub_eta["W2_mean"], sub_eta["MSE_mean"], marker='^', color=tradeoff_colors[idx], label=f"Noise: {noise}", lw=2)
        # Add shaded error bars for MSE
        ax3.fill_between(sub_eta["W2_mean"], 
                         sub_eta["MSE_mean"] - sub_eta["MSE_std"], 
                         sub_eta["MSE_mean"] + sub_eta["MSE_std"], 
                         color=tradeoff_colors[idx], alpha=0.15)
    ax3.set_title(r"Noise on $\eta$")
    ax3.set_xlabel(r"$W_2$")
    ax3.set_ylabel("MSE")
    ax3.legend()
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax3.grid(True, alpha=0.3)

    # Noise on P(S|X)
    for idx, noise in enumerate(tradeoff_noise_levels):
        sub_clf = df_trade_agg[(df_trade_agg["Noise_Type"] == "posterior_only") & (df_trade_agg["Noise_STD"] == noise)].sort_values(by="W2_mean")
        ax4.plot(sub_clf["W2_mean"], sub_clf["MSE_mean"], marker='^', color=tradeoff_colors[idx], label=f"Noise STD: {noise}", lw=2)
        ax4.fill_between(sub_clf["W2_mean"], 
                         sub_clf["MSE_mean"] - sub_clf["MSE_std"], 
                         sub_clf["MSE_mean"] + sub_clf["MSE_std"], 
                         color=tradeoff_colors[idx], alpha=0.15)
    ax4.set_title(r"Noise on $\mathbb{P}(S|X)$")
    ax4.set_xlabel(r"$W_2$")
    ax4.legend()
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax4.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    plot2_path = out_dir / f"stress_test_tradeoffs_{dataset}.pdf"
    fig2.savefig(plot2_path, bbox_inches='tight', transparent=True)
    print(f"Saved Image 2: {plot2_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rigorous stress tests for Fair-Unaware OT.")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="lawschool", 
        choices=["adult", "lawschool", "communities", "synthetic"],
        help="The name of the dataset to run the stress test on."
    )
    
    args = parser.parse_args()
    main(args)