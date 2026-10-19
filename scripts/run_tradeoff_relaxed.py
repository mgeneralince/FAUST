import argparse
from json import load
from pathlib import Path
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import logging
import warnings


import matplotlib.lines as mlines

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fair_unaware.baselines import AwareWassersteinBarycenterDPRegressor
from fair_unaware.utils import compute_group_priors
from fair_unaware import UnawareFairRegressor
from fair_unaware.metrics import w2_gap_between_groups, dp_cdf_max_gap, tv_gap
from fair_unaware.plots.tradeoff import plot_on_ax, plot_custom_tradeoff_curve
import matplotlib.lines as mlines
from data.synthetic import make_synthetic_dataset
from fair_unaware.baselines.fairreg_adapter import get_fairreg_objects, load_dataset
import matplotlib.pyplot as plt
from fair_unaware.baselines.frappe import FrappeRegressor

FAIRREG_EPS_GRID = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, 10000, 60)]))

# handles fairreg warnings 
def redirect_warnings_to_file(warnings_path: Path):
    """Capture tous les RuntimeWarnings et les envoie dans un fichier."""
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


    
def main(*, 
         dataset,
         full_data,
         no_scaling,
         lam_grid_ot, 
         lam_grid_tv,
         eps_grid_fairreg, 
         out_dir, 
         nb_samples, 
         method, 
         cont_map,
         fairreg_root, 
         T_fairreg, 
         merge_train_unlab_for_non_fairreg: bool = False,
         load_results: str = None, 
         num_runs: int = 5,
         crop_frappe: bool = False): 

    if load_results is not None and Path(load_results).exists():
        run_dir = Path(load_results).parent
    else:
        out_root= Path(out_dir)
        run_dir = out_root / "relaxed"/dataset

    run_dir.mkdir(parents=True, exist_ok=True)

    redirect_warnings_to_file(run_dir / "warnings.log")

    base_name = f"tradeoff_compare_{dataset}_n{nb_samples}"
    csv_path = run_dir / f"{base_name}.csv"
    fig_w2_path = run_dir / f"{base_name}_mse_vs_w2.pdf"
    fig_dp_path = run_dir / f"{base_name}_mse_vs_dp.pdf"
    fig_tv_path = run_dir / f"{base_name}_mse_vs_tv.pdf"


    FR_NAME = "FairReg"
    FRAPPE_NAME = "FRAPPE"
    OT_NAME = "FAUST-U W2"
    OT_TV_NAME = "FAUST-U TV"
    AWARE_NAME  = "FAUST-A W2"
    AWARE_TV_NAME = "FAUST-A TV" 
    AWARE_SHAT_NAME = r"Plug-In W2"
    BASE_NAME = "ERM"

    if load_results is not None and Path(load_results).exists():
        print(f"Loading existing results from {load_results}...")
        df = pd.read_csv(load_results)


        if "dp" in df.columns:
            df.rename(columns={"dp": "ks"}, inplace=True)

        match = re.search(r"_n(\d+)", str(load_results))
        if match:
            nb_samples = int(match.group(1))
            print(f"Auto-detected nb_samples = {nb_samples} from filename.")
            
        base_name = f"tradeoff_compare_fairreg_vs_ot_n{nb_samples}"
        fig_w2_path = run_dir / f"{base_name}_mse_vs_w2.pdf"
        fig_dp_path = run_dir / f"{base_name}_mse_vs_dp.pdf"
        fig_tv_path = run_dir / f"{base_name}_mse_vs_tv.pdf"
        

        def map_method_name(name):
            n = str(name).lower()
            if "fairreg" in n: return FR_NAME
            elif "frappe" in n: return FRAPPE_NAME
            elif ("unaware" in n or "ot-u" in n or "faust-u" in n) and "tv" in n: return OT_TV_NAME
            elif ("unaware" in n or "ot-u" in n or "faust-u" in n): return OT_NAME
            elif "s-hat" in n or "s_hat" in n or "hat{s}" in n or "plug-in" in n: return AWARE_SHAT_NAME 
            elif ("aware" in n or "ot-a" in n or "faust-a" in n) and "tv" in n: return AWARE_TV_NAME
            elif ("aware" in n or "ot-a" in n or "faust-a" in n): return AWARE_NAME
            elif "base" in n or "erm" in n or "unfair" in n: return BASE_NAME
            return name
        df["method"] = df["method"].apply(map_method_name)

        

    else:
        print(f"Running experiments from scratch ({num_runs} runs for CI) with dataset={dataset}...")
        rows = []
        FairReg, _DP_unfairness = get_fairreg_objects(fairreg_root)

        # --- 1. GLOBALLY LOAD AND SCALE DATA ---
        if dataset == "synthetic":
            X_all, S_all, Y_unscaled = make_synthetic_dataset(n=nb_samples, random_state=42)
            scaler_all = MinMaxScaler(feature_range=(-1, 1))
            Y_all = scaler_all.fit_transform(Y_unscaled.reshape(-1, 1)).flatten()
            B_val = 1.0
        else:
            X_all, S_all, Y_all, scaler_all, B_val, K_val, p_val, priors_val = load_dataset(
                dataset, fairreg_root=fairreg_root, data_scaling=not no_scaling, full_data=full_data
            )
            if not hasattr(S_all, "unique"):
                S_all = pd.Series(np.asarray(S_all).ravel())

        def inv_scale(y_arr):
            if scaler_all is not None:
                return scaler_all.inverse_transform(np.asarray(y_arr).reshape(-1, 1)).flatten()
            return np.asarray(y_arr)

        for run_idx in tqdm(range(num_runs), desc="Runs"):
            seed = 42 + run_idx
            
            if dataset == "synthetic":
                X, S, Y_unscaled = make_synthetic_dataset(n=nb_samples, random_state=seed)
                Y = scaler_all.transform(Y_unscaled.reshape(-1, 1)).flatten()
            else:
                X, S, Y = X_all, S_all, Y_all
            X_tr, X_te, S_tr, S_te, Y_tr, Y_te = train_test_split(X, S, Y, test_size=0.3, random_state=seed, stratify=S)
            X_lab, X_unlab, S_lab, S_unlab, Y_lab, Y_unlab = train_test_split(X_tr, S_tr, Y_tr, test_size=0.5, random_state=seed, stratify=S_tr)
            
            if merge_train_unlab_for_non_fairreg:
                X_ot_train = np.concatenate([X_lab, X_unlab], axis=0)
                S_ot_train = np.concatenate([S_lab, S_unlab], axis=0)
                Y_ot_train = np.concatenate([Y_lab, Y_unlab], axis=0)
                X_ot_unlab, S_ot_unlab = X_ot_train, S_ot_train
            else:
                X_ot_train, S_ot_train, Y_ot_train = X_lab, S_lab, Y_lab
                X_ot_unlab, S_ot_unlab = X_unlab, S_unlab

            #to turn the number of epoch into the paameter T from FairReg 
            T_dynamic = T_fairreg if T_fairreg is not None else 30 * len(X_unlab)
            print(f"FairReg T (dynamic): {T_dynamic} (based on {len(X_unlab)} unlab samples)")


            reg_ref = LinearRegression().fit(X_lab, Y_lab)
            y_for_bins_scaled = np.concatenate([reg_ref.predict(X_lab), reg_ref.predict(X_te)])
            
            # Unscale bins and Y_te so W2/MSE are in real-world units
            y_for_bins = inv_scale(y_for_bins_scaled)
            bins = np.linspace(np.min(y_for_bins) - 1e-5, np.max(y_for_bins) + 1e-5, 50)       
            Y_te_eval = inv_scale(Y_te)

            # --- A. Base Model (Unaware ERM) ---
            #"Black Star"
            reg_ot_base = LinearRegression().fit(X_ot_train, Y_ot_train)
            
            y_base_ot_scaled = reg_ot_base.predict(X_te)
            y_base_ot = inv_scale(y_base_ot_scaled)
            
            y_const = np.mean(inv_scale(Y_ot_train))
            mse_const = float(np.mean((y_const - Y_te_eval) ** 2))

            rows.append({"run": run_idx, "method": BASE_NAME, "eps": np.nan,
                "mse": float(np.mean((y_base_ot - Y_te_eval) ** 2)),
                "w2": float(w2_gap_between_groups(y_base_ot, S_te, squared=False)),
                "ks": float(dp_cdf_max_gap(y_base_ot, S_te, bins=bins)),
                "tv": float(tv_gap(y_base_ot, S_te, bins=bins)),
                "mse_const": mse_const 
            })

            # --- B. Unaware OT (W2) ---
            sinkhorn_reg = None if method == "OT" else 0.05
            for lam in tqdm(lam_grid_ot, desc="OT Unaware (L2)", leave=False):
                model = UnawareFairRegressor(base_regressor=LinearRegression(), base_classifier=LogisticRegression(max_iter=2000), 
                                             sinkhorn_reg=sinkhorn_reg, continuous_map=cont_map, relax_lambda=float(lam), verbose=False)
                model.fit(X_ot_train, S_ot_train, Y_ot_train, X_unlab=X_ot_unlab)
                
                yhat_scaled = model.predict(X_te)
                yhat = inv_scale(yhat_scaled)
                
                rows.append({"run": run_idx, "method": OT_NAME, "eps": float(lam),
                    "mse": float(np.mean((yhat - Y_te_eval) ** 2)),
                    "w2": float(w2_gap_between_groups(yhat, S_te, squared=False)),
                    "ks": float(dp_cdf_max_gap(yhat, S_te, bins=bins)),
                    "tv": float(tv_gap(yhat, S_te, bins=bins))
                })

            # --- B. bis Unaware OT (TV) ---
            for lam in tqdm(lam_grid_tv, desc="OT Unaware (TV)", leave=False):
                model = UnawareFairRegressor(
                    base_regressor=LinearRegression(), base_classifier=LogisticRegression(max_iter=2000), 
                    sinkhorn_reg=sinkhorn_reg, continuous_map=cont_map, 
                    relax_lambda=float(lam), penalty="tv", verbose=False 
                )
                model.fit(X_ot_train, S_ot_train, Y_ot_train, X_unlab=X_ot_unlab)
                
                yhat_scaled = model.predict(X_te)
                yhat = inv_scale(yhat_scaled)
                
                rows.append({"run": run_idx, "method": OT_TV_NAME, "eps": float(lam),
                    "mse": float(np.mean((yhat - Y_te_eval) ** 2)),
                    "w2": float(w2_gap_between_groups(yhat, S_te, squared=False)),
                    "ks": float(dp_cdf_max_gap(yhat, S_te, bins=bins)),
                    "tv": float(tv_gap(yhat, S_te, bins=bins))
                })

            # --- B. ter FRAPPÉ ---
            # FRAPPÉ uses a lambda scaling similar to OT, so we can reuse lam_grid_ot
            for lam in tqdm(lam_grid_frappe, desc="FRAPPÉ", leave=False):
                frappe_model = FrappeRegressor(
                    base_regressor=LinearRegression(), 
                    hidden_dims=(64,), 
                    lam=float(lam),
                    epochs=100,
                    lr=0.01
                )
                
                frappe_model.fit(
                    X_ot_train, 
                    S_ot_train, 
                    Y_ot_train, 
                    X_unlab=X_ot_unlab, 
                    S_unlab=S_ot_unlab
                )
                
                yhat_scaled = frappe_model.predict(X_te)
                yhat = inv_scale(yhat_scaled)
                
                rows.append({
                    "run": run_idx, 
                    "method": FRAPPE_NAME, 
                    "eps": float(lam),
                    "mse": float(np.mean((yhat - Y_te_eval) ** 2)),
                    "w2": float(w2_gap_between_groups(yhat, S_te, squared=False)),
                    "ks": float(dp_cdf_max_gap(yhat, S_te, bins=bins)),
                    "tv": float(tv_gap(yhat, S_te, bins=bins))
                })
            # --- C. FairReg ---
            Y_lab_real = inv_scale(Y_lab)
            

            local_scaler = MinMaxScaler(feature_range=(-1, 1))
            Y_lab_scaled = local_scaler.fit_transform(Y_lab_real.reshape(-1, 1)).flatten()
            
            reg_fr = LinearRegression().fit(X_lab, Y_lab_scaled)
            clf_fr = LogisticRegression(max_iter=2000).fit(X_lab, S_lab)
            p_vec = np.array([(S_lab == c).mean() for c in clf_fr.classes_], dtype=float)

            for eps in tqdm(eps_grid_fairreg, desc="FairReg", leave=False):
                eps_vec = [float(eps)] * len(clf_fr.classes_)

                fr = FairReg(reg_fr, clf_fr, B=1.0, K=len(clf_fr.classes_), p=p_vec, eps=eps_vec, T=T_dynamic, keep_history=False)
                fr.fit(X_unlab, beta="auto", L="auto", alg={"base": "SGD3", "method": "ACSA2"})
                
                yhat_scaled = fr.predict(X_te)

                yhat = local_scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).flatten()
                
                rows.append({"run": run_idx, "method": FR_NAME, "eps": float(eps),
                    "mse": float(np.mean((yhat - Y_te_eval) ** 2)),
                    "w2": float(w2_gap_between_groups(yhat, S_te, squared=False)),
                    "ks": float(dp_cdf_max_gap(yhat, S_te, bins=bins)),
                    "tv": float(tv_gap(yhat, S_te, bins=bins))
                })

            # --- D. Aware OT ---
            priors_ot = compute_group_priors(S_ot_train)
            aware_strict = AwareWassersteinBarycenterDPRegressor(base_regressor=reg_ot_base, random_state=seed)
            aware_strict.fit(X_ot_train, S_ot_train, Y_ot_train, X_unlab=X_ot_unlab, S_unlab=S_ot_unlab, group_priors=priors_ot)
            
            y_strict_scaled = aware_strict.predict(X_te, S_te)

            # --- D.1 Aware OT (W2) ---
            for lam in tqdm(np.linspace(0.0, 1.0, 40), desc="OT Aware (L2)", leave=False):
                #interpolation as in Chzhen and Schreuder 
                yhat_relaxed_scaled = (1.0 - lam) * y_base_ot_scaled + lam * y_strict_scaled 
                yhat_relaxed = inv_scale(yhat_relaxed_scaled)
                
                rows.append({"run": run_idx, "method": AWARE_NAME, "eps": float(lam), 
                    "mse": float(np.mean((yhat_relaxed - Y_te_eval) ** 2)),
                    "w2": float(w2_gap_between_groups(yhat_relaxed, S_te, squared=False)),
                    "ks": float(dp_cdf_max_gap(yhat_relaxed, S_te, bins=bins)),
                    "tv": float(tv_gap(yhat_relaxed, S_te, bins=bins))
                })

            # --- D.2 Aware OT (TV) ---
            for lam in tqdm(lam_grid_tv, desc="OT Aware (TV)", leave=False):
                lam_val = float(lam)
                
                # convert the penalty 'lam' to 'tau' matching the TV as in the paper 
                if np.isinf(lam_val):
                    tau = 0.0      
                elif lam_val == 0.0:
                    tau = np.inf   
                else:
                    tau = 1.0 / lam_val
                
                diff = y_strict_scaled - y_base_ot_scaled
                yhat_relaxed_scaled = y_base_ot_scaled + np.sign(diff) * np.maximum(np.abs(diff) - tau, 0.0)
                yhat_relaxed = inv_scale(yhat_relaxed_scaled)
                
                rows.append({"run": run_idx, "method": AWARE_TV_NAME, "eps": lam_val, 
                    "mse": float(np.mean((yhat_relaxed - Y_te_eval) ** 2)),
                    "w2": float(w2_gap_between_groups(yhat_relaxed, S_te, squared=False)),
                    "ks": float(dp_cdf_max_gap(yhat_relaxed, S_te, bins=bins)),
                    "tv": float(tv_gap(yhat_relaxed, S_te, bins=bins))
                })
            # --- D.3 Aware OT W2 with Estimated S (OT S-hat W2) ---

            s_estimator = LogisticRegression(max_iter=2000).fit(X_ot_train, S_ot_train)

            #aware Shat 
            aware_shat = AwareWassersteinBarycenterDPRegressor(
                base_regressor=reg_ot_base, 
                random_state=seed, 
                estimate_S=True,
                s_estimator=s_estimator
            )
            
            aware_shat.fit(X_ot_train, S_ot_train, Y_ot_train, X_unlab=X_ot_unlab, S_unlab=None, group_priors=None)
            
            S_te_hat = s_estimator.predict(X_te)
            y_strict_shat_scaled = aware_shat.predict(X_te, S_te_hat)

            for lam in tqdm(np.linspace(0.0, 1.0, 40), desc="OT S-hat (L2)", leave=False):
                #interpolation to construct relaxation as descirbed in Chzhen and Schreuder
                yhat_relaxed_scaled = (1.0 - lam) * y_base_ot_scaled + lam * y_strict_shat_scaled 
                yhat_relaxed = inv_scale(yhat_relaxed_scaled)
                
                rows.append({"run": run_idx, "method": AWARE_SHAT_NAME, "eps": float(lam), 
                    "mse": float(np.mean((yhat_relaxed - Y_te_eval) ** 2)),
                    "w2": float(w2_gap_between_groups(yhat_relaxed, S_te, squared=False)),
                    "ks": float(dp_cdf_max_gap(yhat_relaxed, S_te, bins=bins)),
                    "tv": float(tv_gap(yhat_relaxed, S_te, bins=bins))
                })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print("Saved:", csv_path)


    results = {}  
    for name in [FR_NAME, OT_NAME, OT_TV_NAME, AWARE_NAME, AWARE_TV_NAME, AWARE_SHAT_NAME, FRAPPE_NAME]: 
        sub = df[df["method"] == name]
        if not sub.empty:
            results[name] = sub

            
    color_by_method = {
    FR_NAME: "tab:purple",
    OT_NAME: "tab:blue",             
    AWARE_NAME: "tab:green",         
    OT_TV_NAME: "tab:red",           
    AWARE_TV_NAME: "tab:orange",
    AWARE_SHAT_NAME: "tab:cyan",
    FRAPPE_NAME: "tab:pink"  
}
    
    targets_dict = {
        FR_NAME: ([0.0, 0.01, 0.1, 0.5, 1.0, 3.0, 6.0], r"\epsilon"),
        OT_NAME: ([1e-1, 1e2, 1e5, 1e8, 1e11, float('inf')], r"\lambda"),
        AWARE_NAME: ([0.25, 0.5, 0.75, 1.0], r"\lambda"),
        OT_TV_NAME: ([1e-2, 1e-1, 1.0, 10.0, float('inf')], r"\lambda_{TV}"),
        AWARE_TV_NAME: ([1e-2, 1e-1, 1.0, 10.0, float('inf')], r"\lambda_{TV}"),
        AWARE_SHAT_NAME: ([0.25, 0.5, 0.75, 1.0], r"\lambda") 
    }

    targets_dict[FRAPPE_NAME] = ([1e-4, 1e-2, 1.0, 10.0, 100.0], r"\lambda")

    base_df = df[df["method"] == BASE_NAME]


    plot_custom_tradeoff_curve(results, base_df, "w2", fig_w2_path, "W2 gap between groups", f"MSE vs W2 Tradeoff ", color_by_method, targets_dict, crop_frappe= crop_frappe)
    plot_custom_tradeoff_curve(results, base_df, "ks", fig_dp_path, "DP (CDF max gap)", f"MSE vs DP Tradeoff ", color_by_method, targets_dict, crop_frappe=crop_frappe)
    plot_custom_tradeoff_curve(results, base_df, "tv", fig_tv_path, "TV gap between groups", f"MSE vs TV Tradeoff ", color_by_method, targets_dict, crop_frappe=crop_frappe)

    fig_w2_fr_path = run_dir / f"{base_name}_fairreg_only_mse_vs_w2.pdf"
    fig_dp_fr_path = run_dir / f"{base_name}_fairreg_only_mse_vs_dp.pdf"
    fig_tv_path_fr_path = run_dir / f"{base_name}_fairreg_only_mse_vs_tv.pdf"

    plot_custom_tradeoff_curve(results, base_df, "w2", fig_w2_fr_path, "W2 gap between groups", f"FairReg only: MSE vs W2 ", color_by_method, targets_dict, methods_to_plot=[FR_NAME], crop_frappe=crop_frappe)
    plot_custom_tradeoff_curve(results, base_df, "ks", fig_dp_fr_path, "DP (CDF max gap)", f"FairReg only: MSE vs DP ", color_by_method, targets_dict, methods_to_plot=[FR_NAME], crop_frappe=crop_frappe)
    plot_custom_tradeoff_curve(results, base_df, "tv", fig_tv_path_fr_path, "TV gap between groups", f"FairReg only: MSE vs TV ", color_by_method, targets_dict, methods_to_plot=[FR_NAME], crop_frappe=crop_frappe)

    print("Saved plots to:", run_dir)


    results = {}
    for name in [FR_NAME, OT_NAME, OT_TV_NAME, AWARE_NAME, AWARE_TV_NAME, AWARE_SHAT_NAME, FRAPPE_NAME]: 
        sub = df[df["method"] == name]
        if not sub.empty:
            results[name] = sub
    

    color_by_method = {
        FR_NAME: "tab:purple",
        OT_NAME: "tab:blue",             
        AWARE_NAME: "tab:green",         
        OT_TV_NAME: "tab:red",           
        AWARE_TV_NAME: "tab:orange",
        AWARE_SHAT_NAME: "tab:cyan",
        FRAPPE_NAME: "tab:pink"
    }

    base_df = df[df["method"] == BASE_NAME]

#combinesdd
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    plot_on_ax(axes[0], results, base_df, "ks", "KS", "MSE vs KS", color_by_method, show_ylabel=True, crop_frappe=crop_frappe)
    plot_on_ax(axes[1], results, base_df, "w2", "W2", "MSE vs W2", color_by_method, show_ylabel=False, crop_frappe=crop_frappe)
    plot_on_ax(axes[2], results, base_df, "tv", "TV", "MSE vs TV", color_by_method, show_ylabel=False, crop_frappe=crop_frappe)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    if crop_frappe: 
        handle_star = mlines.Line2D([], [], color='none', marker='*', markerfacecolor='gray', markeredgecolor='black', markersize=12, markeredgewidth=0.5)
        final_handles = list(by_label.values()) + [handle_star]
        final_labels = list(by_label.keys()) + ['Fairest']
    else:
        handle_plus = mlines.Line2D([], [], color='none', marker='+', markeredgecolor='gray', markersize=11, markeredgewidth=2)
        handle_x = mlines.Line2D([], [], color='none', marker='x', markeredgecolor='gray', markersize=8, markeredgewidth=2)
        final_handles = list(by_label.values()) + [handle_plus, handle_x]
        final_labels = list(by_label.keys()) + ['Max Pen.', 'Fairest']


    fig.legend(final_handles, final_labels, loc='center left', bbox_to_anchor=(0.98, 0.5))


    plt.tight_layout()
    fig.subplots_adjust(right=0.96)
    
    fig_combined_path = run_dir / f"{base_name}_combined.pdf"
    plt.savefig(fig_combined_path, bbox_inches="tight")
    plt.close()

    # FAIRREG ONLY
    fig_fr, axes_fr = plt.subplots(1, 3, figsize=(8.5, 2.5))

    plot_on_ax(axes_fr[0], results, base_df, "ks", "KS gap", "FairReg: MSE vs KS", color_by_method, methods_to_plot=[FR_NAME], show_ylabel=True, crop_frappe=crop_frappe)
    plot_on_ax(axes_fr[1], results, base_df, "w2", "W2 gap", "FairReg: MSE vs W2", color_by_method, methods_to_plot=[FR_NAME], show_ylabel=False, crop_frappe=crop_frappe)
    plot_on_ax(axes_fr[2], results, base_df, "tv", "TV gap", "FairReg: MSE vs TV", color_by_method, methods_to_plot=[FR_NAME], show_ylabel=False, crop_frappe=crop_frappe)

    handles_fr, labels_fr = axes_fr[0].get_legend_handles_labels()
    by_label_fr = dict(zip(labels_fr, handles_fr))

    
    if crop_frappe:
        final_handles_fr = list(by_label_fr.values()) + [handle_star]
        final_labels_fr = list(by_label_fr.keys()) + ['Fairest']

    else:
        final_handles_fr = list(by_label_fr.values()) + [handle_plus, handle_x]
        final_labels_fr = list(by_label_fr.keys()) + ['Max Pen.', 'Fairest']


    fig_fr.legend(final_handles_fr, final_labels_fr, loc='center left', bbox_to_anchor=(0.98, 0.5))
    plt.tight_layout()
    fig_fr.subplots_adjust(right=0.96)

    fig_fr_combined_path = run_dir / f"{base_name}_fairreg_only_combined.pdf"
    plt.savefig(fig_fr_combined_path, bbox_inches="tight")
    plt.close()

    print("Saved combined plots to:", run_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["adult", "lawschool", "communities", "synthetic"], default="synthetic")
    p.add_argument("--full_data", action="store_true", default=True, help="Use the full dataset.")
    p.add_argument("--no_scaling", action="store_true", help="Disable target scaling.")
    
    p.add_argument("--nb_samples", type=int, default=1500)
    p.add_argument("--num_runs", type=int, default=10, help="Number of random seeds for Confidence Intervals.")
    p.add_argument("--out_dir", type=str, default="notes/figs")
    p.add_argument("--method", choices=["OT", "Sinkhorn"], default="OT")
    p.add_argument("--cont_map", choices=["rf", "entropic"], default="rf")
    p.add_argument("--eps_ot_min", type=float, default=1e-3)
    p.add_argument("--eps_ot_max", type=float, default=1.0)
    p.add_argument("--eps_ot_n", type=int, default=40)
    p.add_argument("--eps_user_ot", type=float, default=None)
    p.add_argument("--eps_user_fairreg", type=float, default=None)
    p.add_argument("--fairreg_root", type=str, required=True)
    p.add_argument("--T_fairreg", type=int, default=None)
    p.add_argument("--merge_train_unlab_for_non_fairreg", action="store_true")
    p.add_argument("--load_results", type=str, default=None, help="Path to a results CSV to skip training.")
    p.add_argument("--crop_frappe", action="store_true", help="Crop FRAPPÉ results to match the range of other methods.")


    args = p.parse_args()
    lam_grid_ot = np.geomspace(1e-6, 1e11, args.eps_ot_n)
    lam_grid_ot = np.append(lam_grid_ot, [float('inf')]) 

    lam_grid_tv = np.geomspace(1e-4, 1e2, 40)
    lam_grid_tv = np.append(lam_grid_tv, [float('inf')])

    lam_grid_frappe = np.concatenate([[0.0], np.geomspace(1e-4, 1e2, 40)])

    

    main(
            dataset=args.dataset,
            full_data=args.full_data,
            no_scaling=args.no_scaling,
            lam_grid_ot=lam_grid_ot, 
            lam_grid_tv=lam_grid_tv,
            eps_grid_fairreg=FAIRREG_EPS_GRID,
            out_dir=args.out_dir,
            nb_samples=args.nb_samples,
            method=args.method,
            cont_map=args.cont_map,
            fairreg_root=args.fairreg_root,
            T_fairreg=args.T_fairreg,
            merge_train_unlab_for_non_fairreg=args.merge_train_unlab_for_non_fairreg,
            load_results=args.load_results, 
            num_runs=args.num_runs, 
            crop_frappe=args.crop_frappe,
        )

