import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ..plots.visualization import (
    plot_eta_delta_scatter,
    plot_prediction_distributions,
    plot_transport_plan,
    plot_rank_reordering,
    plot_base_vs_fair_scatter,
    plot_aggregated_plan_by_eta,
    plot_cost_preference_for_x1,   
    plot_min_cost_for_plus_log,
    plot_min_cost_hard_vs_easy,
    plot_min_cost_easy_only,
    plot_binned_cost_by_eta,
    plot_eta_delta_scatter_colored_by_y,
    plot_eta_delta_with_transport_lines,
    plot_predictions_barycenter_links,
    scatter_XY_colored_by_S,
)

from fair_unaware.metrics import rank_metrics_by_group, auc_report, metrics_to_latex, dp_mean_gap
from fair_unaware.models import compute_delta_from_posteriors

def plot_basic_diagnostics(
    model,
    X,
    S,
    Y,
    show=True,
    save_prefix=None,
    with_ot_matrices=False,
    save_transport=False,         
    save_suffix="pdf",
    figsize=(12, 8),
):

    """
    High-level wrapper that generates a few standard plots
    for a fitted UnawareFairRegressor.

    Parameters
    ----------
    model : UnawareFairRegressor
    X, S, Y : arrays
        Data used for diagnostics (usually the training set).
    show : bool
        Whether to call plt.show() at the end.
    save_prefix : str or None
        If not None, save figures as pdf with this prefix.
    with_ot_matrices : bool
        If True, also plot full C / pi heatmaps (big matrices).
    """
    # 1. Base vs fair predictions on the same X
    y_base = model.predict_base(X)
    y_fair = model.predict(X)

    # Recompute (eta, delta) on the provided X (meaningful if X is a held-out set)
    eta_plot = y_base
    proba_plot = model.base_classifier_.predict_proba(X)
    delta_plot = compute_delta_from_posteriors(
        proba_plot,
        model.base_classifier_.classes_,
        model.group_priors_,
    )


    def _save(fig, stem: str, *, subdir: str | None = None):
        """
        Save figure using save_prefix as a path prefix:
        - core:   <parent>/<prefix>_<stem>.<suffix>
        - debug:  <parent>/debug/<prefix>_<stem>.<suffix>
        - tables: <parent>/tables/<prefix>_<stem>.<suffix>
        """
        if not save_prefix:
            if not show:
                plt.close(fig)
            return None

        sp = Path(save_prefix)
        out_dir = sp.parent
        prefix = sp.name

        if subdir is not None:
            out_dir = out_dir / subdir

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{stem}.{save_suffix}"

        fig.savefig(path, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return path


    # (a) (eta, delta) scatter
    fig1, ax1 = plt.subplots()
    plot_eta_delta_scatter(eta_plot, delta_plot, S, ax=ax1)

    _save(fig1, "eta_delta_scatter")
    
    # (a) bis (eta, delta) scatter colored by y
    fig1b, ax1b = plt.subplots(figsize= figsize)
    plot_eta_delta_scatter_colored_by_y(eta_plot, delta_plot, Y, ax=ax1b)

    _save(fig1b, "eta_delta_colored_by_Y")


    # (a) ter (eta, delta) scatter with transport lines
    # total version to check everything is ok
    if save_transport:
        fig_t_all, ax_t_all = plt.subplots(figsize=(12, 8))
        plot_eta_delta_with_transport_lines(
            model.h_plus_, model.d_plus_,
            model.h_minus_, model.d_minus_,
            model.pi_,
            ax=ax_t_all,
            title="(eta, delta) with OT pairings (all)",
        )
        _save(fig_t_all, "eta_delta_transport_lines_all", subdir="transport")


    # niceversion where we can actually see something
    fig_t, ax_t = plt.subplots(figsize=(12, 8))
    plot_eta_delta_with_transport_lines(
        model.h_plus_, model.d_plus_,
        model.h_minus_, model.d_minus_,
        model.pi_,
        ax=ax_t,
        title="(eta, delta) with main OT pairings",
        max_lines=150,
        min_mass_quantile=0.7,
    )
    _save(fig_t, "eta_delta_transport_lines")



    # (a) quater predictions barycenter links
    # total
    if save_transport:
        fig_b_all, ax_b_all = plt.subplots(figsize=(12, 8))
        plot_predictions_barycenter_links(
            y_base, y_fair, S,
            n_bins=40,
            ax=ax_b_all,
            title="Base vs barycenter (per group) — all moves",
        )
        _save(fig_b_all, "barycenter_links_all", subdir="transport")


    #filtered
    fig_b, ax_b = plt.subplots(figsize=(12, 8))
    plot_predictions_barycenter_links(
        y_base, y_fair, S,
        n_bins=40,
        ax=ax_b,
        title="Base vs barycenter (per group)",
        max_lines=150,
        min_shift_quantile=0.7,
    )
    _save(fig_b, "barycenter_links")


    # (b) prediction distributions per group
    fig2, ax2 = plt.subplots()
    plot_prediction_distributions(y_base, None, S, ax=ax2)
    _save(fig2, "base_pred_dists")

    # (b) bis prediction distributions per group (before vs after)
    fig2b, ax2b = plt.subplots()
    plot_prediction_distributions(y_base, y_fair, S, ax=ax2b)
    _save(fig2b, "pred_dists")

    # (b) ter scatter X-Y 
    fig2c, axes2c = scatter_XY_colored_by_S(X, Y, S) 
    _save(fig2c, "scatter_XY_colored_by_S")

    # (c) base vs fair scatter
    fig3, ax3 = plt.subplots()
    plot_base_vs_fair_scatter(y_base, y_fair, S, ax=ax3)
    _save(fig3, "base_vs_fair_scatter")

    # (d) Within-group rank reordering (for each group)
    groups = model.group_labels_
    fig4, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4))
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        plot_rank_reordering(y_base, y_fair, S, group=g, ax=ax)

    _save(fig4, "rank_reordering")

    # (c) bis ordering metrics bars
    #  metrics: ordering + AUC 
    order = rank_metrics_by_group(y_base, y_fair, S, topk_frac=0.05)
    auc = auc_report(Y, S, y_eta=y_base, y_corr=y_fair)

    # Print quick summary
    print("[diagnostics] ordering overall:", order["overall"])
    print("[diagnostics] auc:", auc)
    latex = metrics_to_latex(order, auc, topk_frac=0.05, nd=3)

    if save_prefix:
        sp = Path(save_prefix)
        tables_dir = sp.parent / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tables_path = tables_dir / f"{sp.name}_metrics_tables.tex"
        with open(tables_path, "w") as f:
            f.write(latex)

    # ave MSE + DP summary (base vs fair)
    try:
        y_fair = model.predict(X)
    except Exception:
        y_fair = None

    y_base = None
    if hasattr(model, "base_regressor_") and model.base_regressor_ is not None:
        try:
            y_base = model.base_regressor_.predict(X)
        except Exception:
            y_base = None

    def _safe(x):
        x = float(x)
        return x if np.isfinite(x) else np.nan

    mse_base = np.nan
    dp_base = np.nan
    if y_base is not None:
        mse_base = _safe(np.mean((y_base - Y) ** 2))
        dp_base = _safe(dp_mean_gap(y_base, S))

    mse_fair = np.nan
    dp_fair = np.nan
    if y_fair is not None:
        mse_fair = _safe(np.mean((y_fair - Y) ** 2))
        dp_fair = _safe(dp_mean_gap(y_fair, S))

    cont_map = getattr(model, "continuous_map", None)
    cont_eps = getattr(model, "continuous_epsilon", None)
    cont_knn = getattr(model, "continuous_knn", None)

    metrics_path = tables_dir / f"{sp.name}_metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("cont_map,cont_epsilon,cont_knn,sinkhorn_reg,n,mse_base,dp_base,mse_fair,dp_fair\n")
        f.write(
            f"{cont_map},{cont_eps},{cont_knn},{model.sinkhorn_reg},{len(X)},"
            f"{mse_base:.2f},{dp_base:.2f},{mse_fair:.2f},{dp_fair:.2f}\n"
        )




    # (e) aggregated OT plan by eta bins 
    if getattr(model, "store_diagnostics", False) and hasattr(model, "pi_") and hasattr(model, "h_plus_"):
        fig5, ax5 = plt.subplots()
        plot_aggregated_plan_by_eta(
            model.pi_,
            model.h_plus_,
            model.h_minus_,
            n_bins=100,
            ax=ax5,
            title="Aggregated transport (eta bins)",
        )
        _save(fig5, "agg_plan_eta")

        #also keep the full matrices (huge but detailed pb is somethimes we don't see anything)
        if with_ot_matrices and hasattr(model, "C_") and save_transport:
            fig6, (ax6, ax7) = plt.subplots(1, 2, figsize=(10, 4))
            plot_transport_plan(
                model.C_,
                model.pi_,
                h_plus=model.h_plus_,
                h_minus=model.h_minus_,
                ax_cost=ax6,
                ax_plan=ax7,
            )
            _save(fig6, "ot_matrices_full", subdir="transport")
    
    
            # (f) Cost-preference region for a specific x1 in μ⁺
        #choose a representative plus index, e.g. median eta
        if save_transport and len(model.h_plus_) > 0 and len(model.h_minus_) > 0:
            # pick plus point with eta closest to median to have a typique eta score 
            h_plus = np.asarray(model.h_plus_)
            median_eta = np.median(h_plus)
            i_plus = int(np.argmin(np.abs(h_plus - median_eta)))

            fig7, ax7 = plt.subplots()
            plot_cost_preference_for_x1(
                model.h_plus_,
                model.d_plus_,
                model.h_minus_,
                model.d_minus_,
                i_plus=i_plus,
                j_ref=None,  #automatic choice of the cheaoest point rom minus
                ax=ax7,
                title="Region where c(x1, x) > c(x1, x_ref)",
            )
            _save(fig7, "cost_preference_x1", subdir="transport")
        # (g) Various min cost diga but not very informative maybe i will delete them 
            if getattr(model, "store_diagnostics", False) and hasattr(model, "C_"):
                #log-scale minimal cost for all plus points
                fig1, ax1 = plt.subplots()
                plot_min_cost_for_plus_log(model.C_, model.h_plus_, model.d_plus_, ax=ax1)
                _save(fig1, "min_cost_plus_log", subdir="transport")

                #hard vs easy (top 5% hard)
                fig2, ax2 = plt.subplots()
                plot_min_cost_hard_vs_easy(model.C_, model.h_plus_, model.d_plus_,
                                        quantile=0.95, ax=ax2)
                _save(fig2, "min_cost_hard_vs_easy", subdir="transport")

                #easy-only, log colors
                fig3, ax3 = plt.subplots()
                plot_min_cost_easy_only(model.C_, model.h_plus_, model.d_plus_,
                                        quantile=0.95, ax=ax3)
                _save(fig3, "min_cost_easy_only", subdir="transport")

                #binned average cost heatmap
                fig4, ax4 = plt.subplots()
                plot_binned_cost_by_eta(model.C_, model.h_plus_, model.h_minus_, ax=ax4)
                _save(fig4, "binned_cost_by_eta", subdir="transport")
                
                #Save timings
    if hasattr(model, "timings_") and save_prefix:
        sp = Path(save_prefix)
        tables_dir = sp.parent / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        timings_path = tables_dir / f"{sp.name}_timings.csv"
        with open(timings_path, "w") as f:
            f.write("Step,Time (s)\n")
            for k, v in model.timings_.items():
                f.write(f"{k},{v:.6f}\n")


    if show:
        plt.show()
    else:
        plt.close("all")

