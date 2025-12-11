import matplotlib.pyplot as plt
import numpy as np
from .visualization import (
    plot_eta_delta_scatter,
    plot_prediction_distributions,
    plot_transport_plan,
    plot_rank_reordering,
    plot_base_vs_fair_scatter,
    plot_aggregated_plan_by_eta,
    plot_cost_preference_for_x1,   
    plot_global_best_partners,
    plot_min_cost_for_plus,
    plot_min_cost_for_plus_log,
    plot_min_cost_hard_vs_easy,
    plot_min_cost_easy_only,
    plot_binned_cost_by_eta,
    plot_eta_delta_scatter_colored_by_y,
    plot_eta_delta_with_transport_lines,
    plot_predictions_barycenter_links
)



# def plot_basic_diagnostics(model, X, S, Y, show=True, save_prefix=None):
#     """
#     High-level wrapper that generates a few standard plots
#     for a fitted UnawareFairRegressor.
#     """
#     # 1. Base vs fair predictions on the *same* X
#     y_base = model.predict_base(X)
#     y_fair = model.predict(X)

#     # (a) (eta, delta) scatter
#     fig1, ax1 = plt.subplots()
#     plot_eta_delta_scatter(model.eta_hat_, model.delta_hat_, S, ax=ax1)
#     if save_prefix:
#         fig1.savefig(f"{save_prefix}_eta_delta.png", bbox_inches="tight")

#     # (b) prediction distributions per group
#     fig2, ax2 = plt.subplots()
#     plot_prediction_distributions(y_base, y_fair, S, ax=ax2)
#     if save_prefix:
#         fig2.savefig(f"{save_prefix}_pred_dists.png", bbox_inches="tight")

#     # (c) OT internals (if available)
#     if getattr(model, "store_diagnostics", False) and hasattr(model, "C_") and hasattr(model, "pi_"):
#         fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
#         plot_transport_plan(
#             model.C_,
#             model.pi_,
#             h_plus=model.h_plus_,
#             h_minus=model.h_minus_,
#             ax_cost=ax3,
#             ax_plan=ax4,
#         )
#         if save_prefix:
#             fig3.savefig(f"{save_prefix}_ot_matrices.png", bbox_inches="tight")


#         ###### DEBGGGGG #######
#         pi = model.pi_
#         C = model.C_

#         print("pi shape:", pi.shape)
#         print("sum(pi):", pi.sum())
#         print("non-zero entries:", (pi > 0).sum())

#         print("row sums (first 5):", pi.sum(axis=1)[:5])
#         print("col sums (first 5):", pi.sum(axis=0)[:5])
#         print("a (first 5):", model.a_[:5])
#         print("b (first 5):", model.b_[:5])

#         ###### DEBGGGGG #######

#         plot_transport_plan(model.C_, model.pi_, ax_cost=ax3, ax_plan=ax4)
#         if save_prefix:
#             fig3.savefig(f"{save_prefix}_ot_matrices.png", bbox_inches="tight")
        
#         fig, ax = plt.subplots()
#         plot_base_vs_fair_scatter(y_base, y_fair, S, ax=ax)


#     # (d) Within-group rank reordering (for each group)
#     groups = model.group_labels_
#     fig4, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4))
#     if len(groups) == 1:
#         axes = [axes]

#     for ax, g in zip(axes, groups):
#         plot_rank_reordering(model.eta_hat_, y_fair, S, group=g, ax=ax)

#     if save_prefix:
#         fig4.savefig(f"{save_prefix}_rank_reordering.png", bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close("all")

def plot_basic_diagnostics(
    model,
    X,
    S,
    Y,
    show=True,
    save_prefix=None,
    with_ot_matrices=False,
    save_suffix="png",
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
        If not None, save figures as PNG with this prefix.
    with_ot_matrices : bool
        If True, also plot full C / pi heatmaps (big matrices).
    """
    # 1. Base vs fair predictions on the same X
    y_base = model.predict_base(X)
    y_fair = model.predict(X)

    # (a) (eta, delta) scatter
    fig1, ax1 = plt.subplots()
    plot_eta_delta_scatter(model.eta_hat_, model.delta_hat_, S, ax=ax1)
    if save_prefix:
        fig1.savefig(f"{save_prefix}_eta_delta.{save_suffix}", bbox_inches="tight")
    
    # (a) bis (eta, delta) scatter colored by y
    fig1b, ax1b = plt.subplots(figsize= figsize)
    plot_eta_delta_scatter_colored_by_y(model.eta_hat_, 
                                        model.delta_hat_, Y, ax=ax1b)
    if save_prefix:
        fig1b.savefig(f"{save_prefix}_eta_delta_colored_by_Y.{save_suffix}", 
                      bbox_inches="tight")
    
    # (a) ter (eta, delta) scatter with transport lines
    fig1c, ax1c = plt.subplots()
    plot_eta_delta_with_transport_lines(
        model.h_plus_,    
        model.d_plus_,    
        model.h_minus_,   
        model.d_minus_,   
        model.pi_,        
        ax=ax1c,
        title="(eta, delta) with main OT pairings",
    )
    if save_prefix:
        fig1c.savefig(
            f"{save_prefix}_eta_delta_transport_lines.{save_suffix}",
            bbox_inches="tight",
        )

    fig1c, ax1c = plt.subplots(figsize= (12, 8))
    plot_eta_delta_with_transport_lines(
        model.h_plus_,
        model.d_plus_,
        model.h_minus_,
        model.d_minus_,
        model.pi_,
        ax=ax1c,
        title="(eta, delta) with main OT pairings",
        max_lines=100,           # show at most ...lines
        min_mass_quantile=0.7,   # keep only 30% strongest couplings
    )
    if save_prefix:
        fig1c.savefig(
            f"{save_prefix}_eta_delta_transport_lines.{save_suffix}",
            bbox_inches="tight",
        )


    # (a) quater predictions barycenter links
    fig1d, ax1d = plt.subplots(figsize= (12, 8))
    plot_predictions_barycenter_links(
        model.eta_hat_,          
        model.fair_predictions_, 
        S,                       
        n_bins=40,               
        ax=ax1d,
        title="Base vs barycenter (per group)",
    )
    if save_prefix:
        fig1d.savefig(
            f"{save_prefix}_barycenter_links.{save_suffix}",
            bbox_inches="tight",
        )

    fig1d, ax1d = plt.subplots(figsize=(12, 8))
    plot_predictions_barycenter_links(
        model.eta_hat_,
        model.fair_predictions_,
        S,
        n_bins=40,
        ax=ax1d,
        title="Base vs barycenter (per group)",
        max_lines=150,           # at most 500 lines total
        min_shift_quantile=0.7,  # only the 30% largest moves
    )
    if save_prefix:
        fig1d.savefig(
            f"{save_prefix}_barycenter_links.{save_suffix}",
            bbox_inches="tight",
        )


    # (b) prediction distributions per group
    fig2, ax2 = plt.subplots()
    plot_prediction_distributions(y_base, None, S, ax=ax2)
    if save_prefix:
        fig2.savefig(f"{save_prefix}_base_pred_dists.{save_suffix}", bbox_inches="tight")

    # (b) bis prediction distributions per group (before vs after)
    fig2b, ax2b = plt.subplots()
    plot_prediction_distributions(y_base, y_fair, S, ax=ax2b)
    if save_prefix:
        fig2b.savefig(f"{save_prefix}_pred_dists.{save_suffix}", bbox_inches="tight")

    # (c) base vs fair scatter
    fig3, ax3 = plt.subplots()
    plot_base_vs_fair_scatter(y_base, y_fair, S, ax=ax3)
    if save_prefix:
        fig3.savefig(f"{save_prefix}_base_vs_fair_scatter.{save_suffix}", bbox_inches="tight")

    # (d) Within-group rank reordering (for each group)
    groups = model.group_labels_
    fig4, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4))
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        plot_rank_reordering(model.eta_hat_, y_fair, S, group=g, ax=ax)

    if save_prefix:
        fig4.savefig(f"{save_prefix}_rank_reordering.{save_suffix}", bbox_inches="tight")

    # (e) aggregated OT plan by eta bins (small KxK heatmap)
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
        if save_prefix:
            fig5.savefig(f"{save_prefix}_agg_plan_eta.{save_suffix}", bbox_inches="tight")

        # Optional: also keep the full matrices, behind a flag
        if with_ot_matrices and hasattr(model, "C_"):
            fig6, (ax6, ax7) = plt.subplots(1, 2, figsize=(10, 4))
            plot_transport_plan(
                model.C_,
                model.pi_,
                h_plus=model.h_plus_,
                h_minus=model.h_minus_,
                ax_cost=ax6,
                ax_plan=ax7,
            )
            if save_prefix:
                fig6.savefig(f"{save_prefix}_ot_matrices_full.png", bbox_inches="tight")
    
            # (f) Cost-preference region for a specific x1 in μ⁺
        # Choose a representative plus index, e.g. median eta
        if len(model.h_plus_) > 0 and len(model.h_minus_) > 0:
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
                j_ref=None,  # let the function chose the cheapest minus partner
                ax=ax7,
                title="Region where c(x1, x) > c(x1, x_ref)",
            )
            if save_prefix:
                fig7.savefig(
                    f"{save_prefix}_cost_preference_x1.png",
                    bbox_inches="tight",
                )

            # if getattr(model, "store_diagnostics", False) and hasattr(model, "C_"):
            #     fig_bp, ax_bp = plt.subplots()
            #     plot_global_best_partners(
            #         model.C_,
            #         model.h_minus_,
            #         model.d_minus_,
            #         ax=ax_bp,
            #         title="Global attractiveness of minus points",
            #     )
            #     if save_prefix:
            #         fig_bp.savefig(f"{save_prefix}_global_best_partners.png", bbox_inches="tight")

            # if getattr(model, "store_diagnostics", False) and hasattr(model, "C_"):
            #     fig_mc, ax_mc = plt.subplots()
            #     plot_min_cost_for_plus(
            #         model.C_,
            #         model.h_plus_,
            #         model.d_plus_,
            #         ax=ax_mc,
            #         title="Minimal cost per plus point",
            #     )
            #     if save_prefix:
            #         fig_mc.savefig(
            #             f"{save_prefix}_min_cost_plus.png",
            #             bbox_inches="tight",
            #         )

            if getattr(model, "store_diagnostics", False) and hasattr(model, "C_"):
                # 1) log-scale minimal cost for all plus points
                fig1, ax1 = plt.subplots()
                plot_min_cost_for_plus_log(model.C_, model.h_plus_, model.d_plus_, ax=ax1)
                if save_prefix:
                    fig1.savefig(f"{save_prefix}_min_cost_plus_log.png", bbox_inches="tight")

                # 2) hard vs easy (top 5% hard)
                fig2, ax2 = plt.subplots()
                plot_min_cost_hard_vs_easy(model.C_, model.h_plus_, model.d_plus_,
                                        quantile=0.95, ax=ax2)
                if save_prefix:
                    fig2.savefig(f"{save_prefix}_min_cost_hard_vs_easy.png", bbox_inches="tight")

                # 3) easy-only, log colors
                fig3, ax3 = plt.subplots()
                plot_min_cost_easy_only(model.C_, model.h_plus_, model.d_plus_,
                                        quantile=0.95, ax=ax3)
                if save_prefix:
                    fig3.savefig(f"{save_prefix}_min_cost_easy_only.png", bbox_inches="tight")

                # 4) binned average cost heatmap
                fig4, ax4 = plt.subplots()
                plot_binned_cost_by_eta(model.C_, model.h_plus_, model.h_minus_, ax=ax4)
                if save_prefix:
                    fig4.savefig(f"{save_prefix}_binned_cost_eta.png", bbox_inches="tight")
                
                #5. Save timings in to a table if available
    if hasattr(model, "timings_") and save_prefix:
        with open(f"{save_prefix}_timings.txt", "w") as f:
            f.write("Step,Time (s)\n")
            for k, v in model.timings_.items():
                f.write(f"{k},{v:.6f}\n")





    if show:
        plt.show()
    else:
        plt.close("all")

