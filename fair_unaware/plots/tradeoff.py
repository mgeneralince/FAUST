import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.lines as mlines


def plot_on_ax(
    ax,
    results_dict: dict,
    base_df,
    metric_key: str,
    xlabel: str,
    title: str,
    color_by_method: dict,
    methods_to_plot=None,
    show_ylabel: bool = True,
    crop_frappe: bool = False,
):
    """
    Plots the MSE vs Fairness tradeoff curves with 95% Confidence Intervals on a specific ax.
    """
    if methods_to_plot is None:
        methods_to_plot = list(results_dict.keys())

    for method in methods_to_plot:
        if method not in results_dict: 
            continue
            
        df = results_dict[method]
        df_mean = df.groupby("eps").mean(numeric_only=True).reset_index()
        
        has_multiple_runs = 'run' in df.columns and df['run'].nunique() > 1
        if has_multiple_runs:
            df_std = df.groupby("eps").std(numeric_only=True).reset_index().fillna(0.0)
            df_count = df.groupby("eps").count().reset_index()
            ci_x = 1.96 * df_std[metric_key] / np.sqrt(df_count[metric_key])
            ci_y = 1.96 * df_std["mse"] / np.sqrt(df_count["mse"])
        else:
            ci_x = ci_y = np.zeros(len(df_mean))

        x_vals = df_mean[metric_key].values
        y_vals = df_mean["mse"].values
        params = df_mean["eps"].values
        
        color = color_by_method.get(method, "black")
        
        ax.plot(x_vals, y_vals, marker='.', markersize=3,label=method, color=color, alpha=0.6, zorder=3)
        
        if has_multiple_runs:
            ax.errorbar(x_vals, y_vals, xerr=ci_x, yerr=ci_y, fmt='none', color=color, alpha=0.2, zorder=2)

        # STRICT MODELS STARS
        strict_mask = None
        method_lower = method.lower()

        if "fairreg" in method_lower:
            strict_mask = np.isclose(params, 0.0)
        elif "faust-a w2" in method_lower or "plug-in" in method_lower or "frappe" in method_lower or 'FRAPPÉ' in method_lower:
            strict_mask = (params == np.max(params))
        elif "faust-u" in method_lower or "faust-a tv" in method_lower:
            strict_mask = np.isinf(params)
            if not strict_mask.any():
                strict_mask = (params == np.max(params))

        if crop_frappe:
                    # True fairest - lowest unfairness (*)
            if len(x_vals) > 0:
                min_x_idx = np.argmin(x_vals)
                fx, fy = x_vals[min_x_idx], y_vals[min_x_idx]
                
                # Hollow star for Plug-in, solid for others
                if "plug-in" in method_lower:
                    star_face = 'none'       
                    star_edge = color        
                    star_lw = 1.5           
                    star_alpha = 0.9
                else:
                    star_face = color
                    star_edge = 'black'
                    star_lw = 0.8
                    star_alpha = 0.9
                
                ax.scatter(fx, fy, marker='*', s=250, facecolor=star_face, edgecolor=star_edge, linewidths=star_lw, alpha=star_alpha, zorder=7)


        else: 
            if strict_mask is not None and strict_mask.any():
                zx, zy = x_vals[strict_mask][0], y_vals[strict_mask][0]
                clean_name = method.replace(' (relaxed)', '')
                
                #Shat
                if "plug-in" in method_lower:
                    star_face = 'none'       
                    star_edge = color        
                    star_lw = 1.5           
                    star_alpha = 0.9
                else:
                    star_face = color
                    star_edge = 'black'
                    star_lw = 1
                    star_alpha = 0.8
                
                # 1. Highest penalty (+)
                ax.scatter(zx, zy, marker='+', s=150, color=color, linewidths=2.5, alpha=1.0, zorder=7)
                
                # 2. True fairest - lowest unfairness (x)
                min_x_idx = np.argmin(x_vals)
                fx, fy = x_vals[min_x_idx], y_vals[min_x_idx]
                ax.scatter(fx, fy, marker='x', s=75, color=color, linewidths=2.5, alpha=1.0, zorder=7)


    # BASE MODEL (ERM UNFAIR)
    if base_df is not None and not base_df.empty:
        label_base = "ERM"
        b_mean = base_df.mean(numeric_only=True)
        
        ax.axvline(x=b_mean[metric_key], color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
            
        ax.scatter(b_mean[metric_key], b_mean["mse"], marker='o', s=100, facecolor='black', edgecolor='white', linewidth=1, label=label_base, zorder=8)
        
        if 'run' in base_df.columns and base_df['run'].nunique() > 1:
            b_std = base_df.std(numeric_only=True).fillna(0.0)
            b_count = len(base_df)
            ax.errorbar(b_mean[metric_key], b_mean["mse"], 
                        xerr=1.96*b_std[metric_key]/np.sqrt(b_count), 
                        yerr=1.96*b_std["mse"]/np.sqrt(b_count), 
                        fmt='none', color='black', alpha=0.3, zorder=5)
            

    if crop_frappe:
        min_y = float('inf')
        max_y = -float('inf')
        
        for m in methods_to_plot:
            if m in results_dict and "frappe" not in m.lower() and "FRAPPÉ" not in m.lower():
                df_m = results_dict[m]
                df_mean = df_m.groupby("eps").mean(numeric_only=True)["mse"]
                if 'run' in df_m.columns and df_m['run'].nunique() > 1:
                    df_std = df_m.groupby("eps").std(numeric_only=True)["mse"].fillna(0.0)
                    df_count = df_m.groupby("eps").count()["mse"]
                    ci_y = 1.96 * df_std / np.sqrt(df_count)
                    m_max = (df_mean + ci_y).max()
                    m_min = (df_mean - ci_y).min()
                else:
                    m_max = df_mean.max()
                    m_min = df_mean.min()
                max_y = max(max_y, m_max)
                min_y = min(min_y, m_min)
                
        if base_df is not None and not base_df.empty:
            b_mean = base_df["mse"].mean()
            if 'run' in base_df.columns and base_df['run'].nunique() > 1:
                b_ci = 1.96 * base_df["mse"].std() / np.sqrt(len(base_df))
                max_y = max(max_y, b_mean + b_ci)
                min_y = min(min_y, b_mean - b_ci)
            else:
                max_y = max(max_y, b_mean)
                min_y = min(min_y, b_mean)

        if max_y > min_y:
            margin = (max_y - min_y) * 0.05  
            ax.set_ylim(bottom=min_y - margin, top=max_y + margin)

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    

    if show_ylabel:
        ax.set_ylabel("MSE")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False) 
        
    ax.grid(True, alpha=0.3)

def plot_custom_tradeoff_curve(
    results_dict: dict,
    base_df,
    metric_key: str,
    save_path,
    xlabel: str,
    title: str,
    color_by_method: dict,
    targets_dict: dict,
    methods_to_plot=None,
    crop_frappe: bool = False,
):
    """
    Plots the MSE vs Fairness tradeoff curves with 95% Confidence Intervals (error bars).
    Highlights the "Strictly Fair" (non-relaxed) endpoints as large colored stars.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if methods_to_plot is None:
        methods_to_plot = results_dict.keys()

    for method in methods_to_plot:
        if method not in results_dict: 
            continue
            
        df = results_dict[method]
        

        df_mean = df.groupby("eps").mean(numeric_only=True).reset_index()
        
        #CI
        has_multiple_runs = 'run' in df.columns and df['run'].nunique() > 1
        if has_multiple_runs:
            df_std = df.groupby("eps").std(numeric_only=True).reset_index().fillna(0.0)
            df_count = df.groupby("eps").count().reset_index()
            ci_x = 1.96 * df_std[metric_key] / np.sqrt(df_count[metric_key])
            ci_y = 1.96 * df_std["mse"] / np.sqrt(df_count["mse"])
        else:
            ci_x = ci_y = np.zeros(len(df_mean))

        x_vals = df_mean[metric_key].values
        y_vals = df_mean["mse"].values
        params = df_mean["eps"].values
        
        color = color_by_method.get(method, "black")
        

        ax.plot(x_vals, y_vals, marker='.', label=method, color=color, alpha=0.6, zorder=3)
        

        if has_multiple_runs:
            ax.errorbar(x_vals, y_vals, xerr=ci_x, yerr=ci_y, fmt='none', color=color, alpha=0.2, zorder=2)


        # strict_mask = None
        method_lower = method.lower()
        
        if "fairreg" in method_lower:
            strict_mask = np.isclose(params, 0.0)
        elif "faust-a w2" in method_lower or "plug-in" in method_lower or "frappe" in method_lower or 'FRAPPÉ' in method_lower:
            strict_mask = (params == np.max(params))
        elif "faust-u" in method_lower or "faust-a tv" in method_lower:
            strict_mask = np.isinf(params)
            if not strict_mask.any():
                strict_mask = (params == np.max(params))


        # if strict_mask is not None and strict_mask.any():
        #     zx, zy = x_vals[strict_mask][0], y_vals[strict_mask][0]
        #     clean_name = method.replace(' (relaxed)', '')

        #     # 1. Highest penalty (+)
        #     ax.scatter(zx, zy, marker='+', s=200, color=color, linewidths=2.5, alpha=1.0, zorder=7)
            
        #     # 2. True fairest - lowest unfairness (x)
        #     min_x_idx = np.argmin(x_vals)
        #     fx, fy = x_vals[min_x_idx], y_vals[min_x_idx]
        #     ax.scatter(fx, fy, marker='x', s=100, color=color, linewidths=2.5, alpha=1.0, zorder=7)

        # True fairest - lowest unfairness (*)
        if len(x_vals) > 0:
            min_x_idx = np.argmin(x_vals)
            fx, fy = x_vals[min_x_idx], y_vals[min_x_idx]
            
            # Hollow star for Plug-in, solid for others
            if "plug-in" in method_lower:
                star_face = 'none'       
                star_edge = color        
                star_lw = 1.5           
                star_alpha = 0.9
            else:
                star_face = color
                star_edge = 'black'
                star_lw = 0.8
                star_alpha = 0.9
                
            ax.scatter(fx, fy, marker='*', s=300, facecolor=star_face, edgecolor=star_edge, linewidths=star_lw, alpha=star_alpha, zorder=7)

            
    # BASE MODEL (ERM UNFAIR)
    if base_df is not None and not base_df.empty:
        label_base = base_df["method"].iloc[0] if "method" in base_df.columns else "ERM Unfair"
        b_mean = base_df.mean(numeric_only=True)
        
        ax.axvline(x=b_mean[metric_key], color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1, label="Base Unfairness")
        
        if "mse_const" in b_mean:
            ax.axhline(y=b_mean["mse_const"], color='black', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1, label="Constant Predictor")
            
        ax.scatter(b_mean[metric_key], b_mean["mse"], marker='*', s=300, facecolor='black', edgecolor='white', linewidth=0.5, label=label_base, zorder=8)
        
        if 'run' in base_df.columns and base_df['run'].nunique() > 1:
            b_std = base_df.std(numeric_only=True).fillna(0.0)
            b_count = len(base_df)
            ax.errorbar(b_mean[metric_key], b_mean["mse"], 
                        xerr=1.96*b_std[metric_key]/np.sqrt(b_count), 
                        yerr=1.96*b_std["mse"]/np.sqrt(b_count), 
                        fmt='none', color='black', alpha=0.3, zorder=5)
            
    if crop_frappe:
        min_y = float('inf')
        max_y = -float('inf')
        
        for m in methods_to_plot:
            if m in results_dict and "frappe" not in m.lower() and "FRAPPÉ" not in m.lower():
                df_m = results_dict[m]
                df_mean = df_m.groupby("eps").mean(numeric_only=True)["mse"]
                if 'run' in df_m.columns and df_m['run'].nunique() > 1:
                    df_std = df_m.groupby("eps").std(numeric_only=True)["mse"].fillna(0.0)
                    df_count = df_m.groupby("eps").count()["mse"]
                    ci_y = 1.96 * df_std / np.sqrt(df_count)
                    m_max = (df_mean + ci_y).max()
                    m_min = (df_mean - ci_y).min()
                else:
                    m_max = df_mean.max()
                    m_min = df_mean.min()
                max_y = max(max_y, m_max)
                min_y = min(min_y, m_min)
                
        if base_df is not None and not base_df.empty:
            b_mean = base_df["mse"].mean()
            if 'run' in base_df.columns and base_df['run'].nunique() > 1:
                b_ci = 1.96 * base_df["mse"].std() / np.sqrt(len(base_df))
                max_y = max(max_y, b_mean + b_ci)
                min_y = min(min_y, b_mean - b_ci)
            else:
                max_y = max(max_y, b_mean)
                min_y = min(min_y, b_mean)

        if max_y > min_y:
            margin = (max_y - min_y) * 0.05  
            ax.set_ylim(bottom=min_y - margin, top=max_y + margin)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("MSE")
    ax.set_title(title)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    if crop_frappe:
        # Inject generic handle at the end of the legend
        handle_star = mlines.Line2D([], [], color='none', marker='*', markerfacecolor='gray', markeredgecolor='black', markersize=12, markeredgewidth=0.5)
        final_handles = list(by_label.values()) + [handle_star]
        final_labels = list(by_label.keys()) + ['Fairest']
    

    else:
        #Inject generic handles at the end of the legend
        handle_plus = mlines.Line2D([], [], color='none', marker='+', markeredgecolor='gray', markersize=11, markeredgewidth=2)
        handle_x = mlines.Line2D([], [], color='none', marker='x', markeredgecolor='gray', markersize=8, markeredgewidth=2)
        final_handles = list(by_label.values()) + [handle_plus, handle_x]
        final_labels = list(by_label.keys()) + ['Max Pen.', 'Fairest']
        

        
    ax.legend(final_handles, final_labels, fontsize=9)
        
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        
    plt.close()
