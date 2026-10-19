import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import argparse
import sys

from fair_unaware import UnawareFairRegressor
from fair_unaware.baselines import AwareWassersteinBarycenterDPRegressor
from fair_unaware.utils import compute_group_priors
from fair_unaware.metrics import w2_gap_between_groups

out_dir = Path("notes/figs/toy_comparison/empirical_levels/shift")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory ready at: {out_dir}")

parser = argparse.ArgumentParser(description="Run OT Fairness Toy Experiment")
parser.add_argument("--replot", action="store_true", help="Skip computation and replot from saved .npz files.")
args = parser.parse_args()

out_dir = Path("notes/figs/toy_comparison/empirical_levels/shift")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory ready at: {out_dir}")


class LocalOracle1DClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=0.6): 
        self.gamma = gamma
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        X1 = X[:, 0]
        logits = (2.0 * self.gamma * X1) / (1.0 - self.gamma**2)
        prob_1 = expit(logits)
        return np.column_stack([1.0 - prob_1, prob_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class LocalOracle1DRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X1 = X[:, 0]
        return 3.0 * np.tanh(1.5 * X1) 


def make_synthetic_dataset_1d(n=5000, gamma=0.6, random_state=42): 
    rng = np.random.RandomState(random_state)
    S = rng.binomial(1, 0.5, size=n)
    Z_S = (S - 0.5) * 2.0
    X1 = gamma * Z_S + np.sqrt(1.0 - gamma**2) * rng.normal(loc=0.0, scale=1.0, size=n)
    X = X1.reshape(-1, 1)
    noise = rng.normal(scale=0.5, size=n)
    Y = 3.0 * np.tanh(1.5 * X1) + noise 
    return X, S, Y

X, S, Y = make_synthetic_dataset_1d()
priors = compute_group_priors(S)

clf_oracle = LocalOracle1DClassifier(gamma=0.6) 
reg_oracle = LocalOracle1DRegressor()


print("Pre-computing Baseline and Strict models...")
y_base = reg_oracle.fit(X, Y).predict(X)
w2_base = w2_gap_between_groups(y_base, S)

aware_strict = AwareWassersteinBarycenterDPRegressor(base_regressor=reg_oracle, random_state=42)
aware_strict.fit(X, S, Y, X_unlab=X, S_unlab=S, group_priors=priors)
y_strict = aware_strict.predict(X, S)
w2_strict = w2_gap_between_groups(y_strict, S)



def match_unaware_l2(target_w2):
    """Find the Unaware L2 lambda that yields the target W2 gap."""
    def obj(log_lam):
        lam = np.exp(log_lam)
        model = UnawareFairRegressor(
            base_regressor=reg_oracle, base_classifier=clf_oracle,
            relax_lambda=lam, penalty="l2", 
            continuous_map="rf", continuous_epsilon=0.05, 
            delta_tol=1e-10, sinkhorn_reg=None
        )
        model.fit(X, S, Y)
        current_w2 = w2_gap_between_groups(model.predict(X), S)
        return (current_w2 - target_w2)**2


    res = minimize_scalar(obj, bounds=(np.log(1e-4), np.log(1e5)), method='bounded')
    return np.exp(res.x)

def match_aware_l2(target_w2):
    """Find the interpolation alpha that yields the target W2 gap."""
    def obj(alpha):
        y = (1.0 - alpha) * y_base + alpha * y_strict
        return (w2_gap_between_groups(y, S) - target_w2)**2
    res = minimize_scalar(obj, bounds=(0.0, 1.0), method='bounded')
    return res.x



def match_aware_tv(target_w2):
    """
    Find the exact hard-thresholding tau that yields the target W2 gap.
    Since hard-thresholding is a step function, we test the exact unique 
    displacement distances present in the dataset.
    """
    diff_abs = np.abs(y_strict - y_base)
    

    tau_candidates = np.unique(diff_abs)

    tau_candidates = np.concatenate(([0.0], tau_candidates, [np.max(diff_abs) + 1e-3]))
    
    best_tau = 0.0
    best_error = float('inf')
    
    for t in tau_candidates:
        y = np.where(diff_abs <= t, y_strict, y_base)
        current_w2 = w2_gap_between_groups(y, S)
        error = abs(current_w2 - target_w2)
        
        if error < best_error:
            best_error = error
            best_tau = t
            
    return best_tau

def match_unaware_tv(target_w2):
    """
    Find the Unaware TV lambda using a robust coarse-to-fine grid search 
    to avoid getting trapped in flat regions of the TV penalty landscape.
    """

    lam_grid = np.logspace(-2, 3, 40)  
    best_lam = lam_grid[0]
    best_error = float('inf')
    
    for lam in lam_grid:
        model = UnawareFairRegressor(
            base_regressor=reg_oracle, base_classifier=clf_oracle,
            relax_lambda=lam, penalty="tv", 
            continuous_map="rf", continuous_epsilon=0.05, 
            delta_tol=1e-10, sinkhorn_reg=None
        )
        model.fit(X, S, Y)
        current_w2 = w2_gap_between_groups(model.predict(X), S)
        error = abs(current_w2 - target_w2)
        
        if error < best_error:
            best_error = error
            best_lam = lam
            
    def obj(lam):
        model = UnawareFairRegressor(
            base_regressor=reg_oracle, base_classifier=clf_oracle,
            relax_lambda=lam, penalty="tv", 
            continuous_map="rf", continuous_epsilon=0.05, 
            delta_tol=1e-10, sinkhorn_reg=None
        )
        model.fit(X, S, Y)
        return (w2_gap_between_groups(model.predict(X), S) - target_w2)**2

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(obj, bounds=(best_lam * 0.5, best_lam * 2.0), method='bounded')
    return res.x


def plot_curve_on_ax(ax, X_vals, y_pred_vals, S_vals, title, show_ylabel=False):
    X_1d = X_vals.ravel()
    x_range = np.linspace(-6.0, 6.0, 100)
    
    # Base ERM curve
    ax.plot(x_range, 3.0 * np.tanh(1.5 * x_range), color='gray', linestyle='--', alpha=0.4, linewidth=1, label='ERM')
    
    # Group curves
    colors = ['#ff7f0e', '#1f77b4']
    for group, color, label in [(1, colors[1], 'S=1'), (0, colors[0], 'S=0')]:
        mask = (S_vals == group)
        sorted_idx = np.argsort(X_1d[mask])
        ax.plot(X_1d[mask][sorted_idx], y_pred_vals[mask][sorted_idx], color=color, linewidth=2, label=label)
    
    ax.set_title(title)
    ax.set_xlabel(r"Feature $X$")
    
    # Only show Y label on the first plot
    if show_ylabel:
        ax.set_ylabel(r"$\tilde{Y}$", rotation=0, labelpad=10)
    
    # Limits and removing tick numbers
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0, y=0)
    ax.grid(True, alpha=0.2, linewidth=0.5)

def plot_violins_on_ax(ax, scenarios, S_vals, title):
    dfs = []
    for name, preds in scenarios:
        dfs.append(pd.DataFrame({'Prediction': preds, 'S': S_vals, 'Method': name}))
    
    df_all = pd.concat(dfs)
    df_all['Method'] = pd.Categorical(df_all['Method'], categories=[s[0] for s in scenarios], ordered=True)
    df_all['Group'] = df_all['S'].map({0: 'S=0', 1: 'S=1'})

    sns.violinplot(
        data=df_all, x='Method', y='Prediction', hue='Group', 
        split=True, inner='quartile', linewidth=0.5,
        palette=['#ff7f0e', '#1f77b4'], alpha=0.7, bw_adjust=0.5, ax=ax
    )

    for collection in ax.collections:
        collection.set_zorder(2)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])


def plot_composite_row(X, S, y_erm, y_w2, y_tv, pct, prefix, filename):
    """
    Plot a composite row of 3 curves + 1 violin for a given scenario.
    """
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))
    
    plot_curve_on_ax(axes[0], X, y_erm, S, "ERM", show_ylabel=True)
    plot_curve_on_ax(axes[1], X, y_w2, S, f"{prefix} $W_2$ ({pct:.0f}%)", show_ylabel=False)
    plot_curve_on_ax(axes[2], X, y_tv, S, f"{prefix} TV ({pct:.0f}%)", show_ylabel=False)
    

    scenarios = [('ERM', y_erm), (f'$W_2$', y_w2), (f'TV', y_tv)]
    plot_violins_on_ax(axes[3], scenarios, S, "Marginals")
    
    axes[3].set_xticks(range(len(scenarios)))
    axes[3].set_xticklabels([s[0] for s in scenarios], rotation=30, ha="right")
    
    for i in range(1, 4):
        if axes[i].get_legend() is not None:
            axes[i].legend_.remove()
            
    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()


def plot_evolution_violins(S_vals, w2_scenarios, tv_scenarios, y_base, filename, n_trajectories=12): 

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.5)) 
    
    n_per_group = n_trajectories // 2
    

    idx_0 = np.where(S_vals == 0)[0]
    sorted_idx_0 = idx_0[np.argsort(y_base[idx_0])]
    selected_0 = sorted_idx_0[np.linspace(0, len(sorted_idx_0) - 1, n_per_group).astype(int)]
    
    idx_1 = np.where(S_vals == 1)[0]
    sorted_idx_1 = idx_1[np.argsort(y_base[idx_1])]

    selected_1 = sorted_idx_1[np.linspace(0, len(sorted_idx_1) - 1, n_per_group).astype(int)]
    

    indices = np.concatenate([selected_0, selected_1])

    
    for ax, scenarios, title in zip(axes, [w2_scenarios, tv_scenarios], ["$W_2$ relaxation", "TV relaxation"]):
        

        x_coords = np.arange(len(scenarios))
        for idx in indices:
            y_coords = [scen[1][idx] for scen in scenarios]
            color = '#1f77b4' if S_vals[idx] == 1 else '#ff7f0e'

            ax.plot(x_coords, y_coords, color=color, alpha=0.5, linewidth=0.8, zorder=0)


        plot_violins_on_ax(ax, scenarios, S_vals, title)
        

        x_labels = ["Unfair"] + [""] * (len(scenarios) - 2) + ["Fair"]
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(x_labels)
        

        if ax == axes[0] and ax.get_legend() is not None:
            ax.legend_.remove()
        elif ax == axes[1] and ax.get_legend() is not None:
            ax.legend(title="", loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


# Naive Plug-in Failure
print("Pre-computing Hard Plug-in model (S_hat) for toy illustration...")
S_hat = clf_oracle.predict(X)
priors_hat = compute_group_priors(S_hat)

plugin_strict = AwareWassersteinBarycenterDPRegressor(base_regressor=reg_oracle, random_state=42)
plugin_strict.fit(X, S_hat, Y, X_unlab=X, S_unlab=S_hat, group_priors=priors_hat)
y_plugin_strict = plugin_strict.predict(X, S_hat)

print("Pre-computing strict Unaware Model (Our) for toy illustration...")
our_strict_unaware = UnawareFairRegressor(
    base_regressor=reg_oracle, 
    base_classifier=clf_oracle,
    relax_lambda=None,  
    penalty="l2", 
    continuous_map="rf", 
    continuous_epsilon=0.05, 
    delta_tol=1e-10, 
    sinkhorn_reg=None
)
our_strict_unaware.fit(X, S, Y)
y_our_strict_unaware = our_strict_unaware.predict(X)


def plot_plugin_failure_toy_violins(S_vals, y_base, y_our, y_plugin, filename):
    fig, ax = plt.subplots(figsize=(3, 2.5)) 
    

    scenarios = [
        ("ERM", y_base), 
        ("FAUST-U", y_our), 
        (r"Plug-in", y_plugin)
    ]
    
    plot_violins_on_ax(ax, scenarios, S_vals, "")
    
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([scen[0] for scen in scenarios])
    
    if ax.get_legend() is not None:
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

toy_filename = out_dir / "plugin_failure_toy.pdf"
plot_plugin_failure_toy_violins(S, y_base, y_our_strict_unaware, y_plugin_strict, toy_filename)
print(f"Saved theoretical toy illustration to: {toy_filename}")


#+++=====================
if args.replot:
    print("\n--- REPLOTTING FROM SAVED DATA ---")
    
    evol_data = []
    S_global, y_erm_global = None, None

    for sub_dir in sorted(out_dir.glob("function/level_*")):
        data_file = sub_dir / "plot_data.npz"
        if not data_file.exists():
            continue
            
        print(f"Regenerating plots for: {sub_dir.name}")
        data = np.load(data_file)
        
        X, S, y_base = data['X'], data['S'], data['y_base']
        y_unaware_l2, y_unaware_tv = data['y_unaware_l2'], data['y_unaware_tv']
        y_aware_l2, y_aware_tv = data['y_aware_l2'], data['y_aware_tv']
        pct = float(data['pct_unfairness'])
        
        S_global, y_erm_global = S, y_base
        
        plot_composite_row(X, S, y_base, y_unaware_l2, y_unaware_tv, pct, "Unaware", sub_dir / "unaware_composite.pdf")
        plot_composite_row(X, S, y_base, y_aware_l2, y_aware_tv, pct, "Aware", sub_dir / "aware_composite.pdf")
        
        evol_data.append({
            'pct': pct, 'y_unaware_l2': y_unaware_l2, 'y_unaware_tv': y_unaware_tv,
            'y_aware_l2': y_aware_l2, 'y_aware_tv': y_aware_tv
        })


    if S_global is not None and len(evol_data) > 0:
        print("\nGenerating Global Evolution Plots...")

        evol_data = sorted(evol_data, key=lambda x: x['pct'], reverse=True)
        
        un_w2_scen = [("Unfair", y_erm_global)] + [(f"{d['pct']:.0f}%", d['y_unaware_l2']) for d in evol_data]
        un_tv_scen = [("Unfair", y_erm_global)] + [(f"{d['pct']:.0f}%", d['y_unaware_tv']) for d in evol_data]
        aw_w2_scen = [("Unfair", y_erm_global)] + [(f"{d['pct']:.0f}%", d['y_aware_l2']) for d in evol_data]
        aw_tv_scen = [("Unfair", y_erm_global)] + [(f"{d['pct']:.0f}%", d['y_aware_tv']) for d in evol_data]
        
        plot_evolution_violins(
            S_global, un_w2_scen, un_tv_scen, y_erm_global, 
            out_dir / "unaware_evolution_violins_traj.pdf"
        )
        plot_evolution_violins(
            S_global, aw_w2_scen, aw_tv_scen, y_erm_global, 
            out_dir / "aware_evolution_violins_traj.pdf"
        )

    print("\nDone! All plots regenerated instantly.")
    sys.exit(0)

target_pcts = [80.0, 60.0, 40.0, 20.0, 0.0]


for idx, pct in enumerate(target_pcts):
    print(f"\n[{idx+1}/{len(target_pcts)}] Solving for {pct:.0f}% Unfairness...")
    
    #calculate exact target W2
    target_w2 = w2_base * (pct / 100.0)
    
    if pct == 0.0:
        y_aware_l2 = y_strict
        y_aware_tv = y_strict
    else:
        alpha_l2 = match_aware_l2(target_w2)
        y_aware_l2 = (1.0 - alpha_l2) * y_base + alpha_l2 * y_strict
        
        tau_tv = match_aware_tv(target_w2)
        diff_abs = np.abs(y_strict - y_base)
        y_aware_tv = np.where(diff_abs <= tau_tv, y_strict, y_base)
    

    target_w2_solver = max(target_w2, 1e-4)
    
    lam_l2 = match_unaware_l2(target_w2_solver)
    ot_l2 = UnawareFairRegressor(
        base_regressor=reg_oracle, base_classifier=clf_oracle,
        relax_lambda=lam_l2, penalty="l2", 
        continuous_map="rf", continuous_epsilon=0.05, 
        delta_tol=1e-10, sinkhorn_reg=None
    )
    ot_l2.fit(X, S, Y)
    y_unaware_l2 = ot_l2.predict(X)
    
    lam_tv = match_unaware_tv(target_w2_solver)
    ot_tv = UnawareFairRegressor(
        base_regressor=reg_oracle, base_classifier=clf_oracle,
        relax_lambda=lam_tv, penalty="tv", 
        continuous_map="rf", continuous_epsilon=0.05, 
        delta_tol=1e-10, sinkhorn_reg=None
    )
    ot_tv.fit(X, S, Y)
    y_unaware_tv = ot_tv.predict(X)

    sub_dir = out_dir / f"function/level_{idx+1:02d}_Unfairness_{pct:.0f}pct"
    sub_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sub_dir / "plot_data.npz",
        X=X, S=S, y_base=y_base,
        y_unaware_l2=y_unaware_l2, y_unaware_tv=y_unaware_tv,
        y_aware_l2=y_aware_l2, y_aware_tv=y_aware_tv,
        pct_unfairness=pct
    )
    
    plot_composite_row(X, S, y_base, y_unaware_l2, y_unaware_tv, pct, "Unaware", sub_dir / "unaware_composite.pdf")
    plot_composite_row(X, S, y_base, y_aware_l2, y_aware_tv, pct, "Aware", sub_dir / "aware_composite.pdf")

print("\nAll 5 exact percentage levels successfully generated! Run with --replot to view the final evolution plots.")

