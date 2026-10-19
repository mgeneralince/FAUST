import sys
import types
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fair_unaware.metrics import tv_gap, dp_cdf_max_gap

class DummyFairRegClass:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        pass

class DummyFairRegModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")
        return DummyFairRegClass

sys.modules['FairReg'] = DummyFairRegModule('FairReg')



def generate_bin_configs(y_all, n_bins_list=[20, 50, 100]):
    """Generates fixed and adaptive bin configurations based on the data."""
    configs = {}
    y_min, y_max = np.min(y_all), np.max(y_all)
    
    # 1. Fixed bins (Linear spacing)
    for n in n_bins_list:
        configs[f"Fixed ({n})"] = np.linspace(y_min, y_max, n)
        
    # 2. Adaptive bins (Quantile spacing) 50 quatiles
    configs["Adaptive (50)"] = np.unique(np.quantile(y_all, np.linspace(0, 1, 50)))
    
    return configs
def plot_metric_sensitivity(df_summary, dataset_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # order on the axis 
    order = ["Fixed (20)", "Fixed (50)", "Fixed (100)", "Adaptive (50)"]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    colors = {
        "ERM Unfair": "black", 
        "FairReg": "tab:purple", 
        "Frappe": "tab:pink", 
        "OT Unaware": "tab:blue",
        "OT Aware": "tab:green"
    }
    markers = {
        "ERM Unfair": "o", 
        "FairReg": "^", 
        "Frappe": "D", 
        "OT Unaware": "s",
        "OT Aware": "v"
    }
    
    methods_in_df = df_summary["Method"].unique()
    
    for method in colors.keys():
        if method not in methods_in_df:
            continue
            
        sub = df_summary[df_summary["Method"] == method].copy()
        if sub.empty: 
            continue
            
        sub["Bin Config"] = pd.Categorical(sub["Bin Config"], categories=order, ordered=True)
        sub = sub.sort_values("Bin Config")
        
        x = sub["Bin Config"]
        

        if method == "Frappe":
            label_name = "FRAPPÉ"
        elif method == "OT Unaware":
            label_name = "FAUST-U"
        elif method == "OT Aware":
            label_name = "FAUST-A"
        else:
            label_name = method
            
        # TV 
        axes[0].plot(x, sub["TV_mean"], marker=markers[method], color=colors[method], label=label_name, lw=2)
        axes[0].fill_between(x, sub["TV_mean"] - sub["TV_std"], 
                             sub["TV_mean"] + sub["TV_std"], color=colors[method], alpha=0.15)
        
        # KS 
        axes[1].plot(x, sub["KS_mean"], marker=markers[method], color=colors[method], label=label_name, lw=2)
        axes[1].fill_between(x, sub["KS_mean"] - sub["KS_std"], 
                             sub["KS_mean"] + sub["KS_std"], color=colors[method], alpha=0.15)

    # TV Plot
    axes[0].set_ylabel("TV Distance")
    axes[0].set_title("Sensitivity of TV to Grid Size")
    axes[0].grid(True, alpha=0.3)
    
    # KS Plot
    axes[1].set_ylabel("KS Distance")
    axes[1].set_title("Sensitivity of KS to Grid Size")
    axes[1].grid(True, alpha=0.3)
    
    handles, labels = axes[1].get_legend_handles_labels()


    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()
    
    plot_path = out_dir / f"metric_sensitivity_{dataset_name}.pdf"
    fig.savefig(plot_path, bbox_inches='tight', transparent=True)
    print(f"Saved plot: {plot_path}")

def run_metric_sensitivity(results_dir):
    results_dir = Path(results_dir)
    run_dirs = sorted(results_dir.glob("run_*"))
    
    if not run_dirs:
        print(f"No runs found in {results_dir}")
        return None
        
    all_results = []
    
    for run_path in run_dirs:
        plot_data_file = run_path / "plot_data.joblib"
        if not plot_data_file.exists():
            continue
            
        #load the data
        data = joblib.load(plot_data_file)
        preds = data["preds"] 
        S_test = data["split"]["S_test"] 
        
        #sombining predictions
        y_all = np.concatenate(list(preds.values()))
        bin_configs = generate_bin_configs(y_all)
        
        methods_to_eval = ["ERM Unfair", "FairReg", "Frappe", "OT Unaware", "OT Aware"]
        
        for method in methods_to_eval:
            if method not in preds:
                continue
            
            y_pred = preds[method]
            
            for config_name, bins in bin_configs.items():
                tv = tv_gap(y_pred, S_test, bins=bins)
                ks = dp_cdf_max_gap(y_pred, S_test, bins=bins)
                
                all_results.append({
                    "Run": run_path.name,
                    "Method": method,
                    "Bin Config": config_name,
                    "TV": tv,
                    "KS": ks
                })
                
    if not all_results:
        return None
        
    #aggregate runs
    df = pd.DataFrame(all_results)
    summary = df.groupby(["Method", "Bin Config"]).agg(
        TV_mean=("TV", "mean"),
        TV_std=("TV", "std"),
        KS_mean=("KS", "mean"),
        KS_std=("KS", "std")
    ).reset_index()
    
    return summary


if __name__ == "__main__":
    from pathlib import Path
    
    datasets = {
        "lawschool": "notes/figs/compare_ot_vs_fairreg/lawschool/full/rf",
        "communities": "notes/figs/compare_ot_vs_fairreg/communities/full/rf",
        "adult": "notes/figs/compare_ot_vs_fairreg/adult/full/rf",
        "synthetic": "notes/figs/compare_ot_vs_fairreg/synthetic/synthetic_n10000/rf"
    }
    
    for dataset_name, target_dir in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        if not Path(target_dir).exists():
            print(f"Warning: Directory {target_dir} not found. Skipping.")
            continue
            
        df_summary = run_metric_sensitivity(target_dir)
        
        if df_summary is not None:
            out_dir = Path("notes/figs/stress_tests/metric_sensitivity_plots")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Format and save the LaTeX table to a file
            latex_str = df_summary.to_latex(index=False, float_format="%.3f")
            tex_path = out_dir / f"metric_sensitivity_{dataset_name}.tex"
            with open(tex_path, "w") as f:
                f.write(latex_str)
            print(f"Saved LaTeX table to: {tex_path}")
            
            # 2. Generate the publication plot!
            plot_metric_sensitivity(df_summary, dataset_name, out_dir)