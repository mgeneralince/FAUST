from os import name
from cv2 import log
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import math
import pandas as pd



def plot_eta_delta_scatter(eta, delta, S, ax=None, title="(eta, delta) scatter"):
    """
    Scatter plot of (eta, delta), colored by group S.
    """
    if ax is None:
        fig, ax = plt.subplots()

    eta = np.asarray(eta)
    delta = np.asarray(delta)
    S = np.asarray(S)

    groups = np.unique(S)
    for g in groups:
        mask = (S == g)
        ax.scatter(eta[mask], delta[mask], label=f"S={g}", alpha=0.6)

    ax.set_xlabel("eta_hat")
    ax.set_ylabel("delta_hat")
    ax.set_title(title)
    ax.legend()
    return ax


def scatter_XY_colored_by_S(
    X, Y, S,
    axes=None,
    title_prefix="X vs Y",
    sharey=True,
    max_cols=4,
    col_width=4.0,
    row_height=3.5,
    alpha=0.5,
    s=8,                
    legend="single",     
    rasterized=True,
):
    """
    Grid scatter plots: for each column j of X, plot X[:, j] vs Y, colored by S.

    If axes is None, creates a grid with up to max_cols columns and enough rows.
    """

    X = np.asarray(X)
    Y = np.asarray(Y)
    S = np.asarray(S)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, d = X.shape
    if not (len(Y) == n and len(S) == n):
        raise ValueError(f"Length mismatch: X has n={n}, len(Y)={len(Y)}, len(S)={len(S)}")

    if axes is None:
        ncols = min(max_cols, d)
        nrows = int(math.ceil(d / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(col_width * ncols, row_height * nrows),
            sharey=sharey
        )
    else:
        axes = np.array(axes, dtype=object)
        fig = axes.ravel()[0].figure

    axes = np.array(axes, dtype=object).ravel()

    groups = np.unique(S)
    legend_handles = None

    for j in range(d):
        ax = axes[j]
        for g in groups:
            mask = (S == g)
            sc = ax.scatter(
                X[mask, j], Y[mask],
                label=f"S={g}",
                alpha=alpha, s=s,
                rasterized=rasterized
            )
        ax.set_xlabel(f"X[:, {j}]")
        if sharey and j % (min(max_cols, d)) == 0:
            ax.set_ylabel("Y")
        ax.set_title(f"{title_prefix} (col {j})", fontsize=10)

        if legend == "each":
            ax.legend()

        if legend == "single" and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()


    for k in range(d, len(axes)):
        axes[k].axis("off")


    if legend == "single" and legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", ncol=min(len(groups), 5),
            frameon=False
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95) if legend == "single" else None)
    return fig, axes


#not every link, only those with significant mass
def plot_eta_delta_with_transport_lines(
    h_plus,
    d_plus,
    h_minus,
    d_minus,
    pi,
    ax=None,
    title="(eta, delta) with main OT pairings",
    max_lines=None,
    min_mass_quantile=None,
):
    """
    ...
    max_lines : int or None
        If not None, randomly subsample at most this many plus points
        for drawing lines.
    min_mass_quantile : float in (0,1) or None
        If not None, only draw lines for plus points whose main coupling
        mass is >= this quantile of main_mass.
    """
    if ax is None:
        fig, ax = plt.subplots()

    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)
    h_minus = np.asarray(h_minus)
    d_minus = np.asarray(d_minus)
    pi = np.asarray(pi)

    n_plus = h_plus.shape[0]


    ax.scatter(h_plus, d_plus, label="plus (Δ>0)", alpha=0.7)
    ax.scatter(h_minus, d_minus, label="minus (Δ<0)", alpha=0.7)

    if pi.size > 0:

        j_star = np.argmax(pi, axis=1)  
        main_mass = pi[np.arange(n_plus), j_star]
        max_mass = main_mass.max() if main_mass.max() > 0 else 1.0

        idx_plus = np.arange(n_plus)

        #keep only plus points with sufficiently large main_mass
        if min_mass_quantile is not None:
            thresh = np.quantile(main_mass, min_mass_quantile)
            idx_plus = idx_plus[main_mass >= thresh]

        #random subsampling if still too many
        if max_lines is not None and len(idx_plus) > max_lines:
            rng = np.random.RandomState(0)  # deterministic for reproducibility
            idx_plus = rng.choice(idx_plus, size=max_lines, replace=False)

        # draw lines only for selected plus indices
        for i in idx_plus:
            j = j_star[i]
            x1, y1 = h_plus[i], d_plus[i]
            x2, y2 = h_minus[j], d_minus[j]
            alpha = 0.1 + 0.6 * (main_mass[i] / max_mass)
            ax.plot([x1, x2], [y1, y2], color="k", linewidth=0.5, alpha=alpha)

    ax.set_xlabel("eta_hat")
    ax.set_ylabel("delta_hat")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_eta_delta_scatter_colored_by_y(
    eta,
    delta,
    y,
    ax=None,
    title="(eta, delta) scatter colored by Y",
):
    """
    Scatter plot of (eta, delta), colored by the true target Y.

    Parameters
    ----------
    eta : array-like of shape (n_samples,)
        Base predictions (eta_hat).
    delta : array-like of shape (n_samples,)
        Delta_hat values.
    y : array-like of shape (n_samples,)
        True target values used for coloring.
    """
    if ax is None:
        fig, ax = plt.subplots()

    eta = np.asarray(eta)
    delta = np.asarray(delta)
    y = np.asarray(y)

    sc = ax.scatter(eta, delta, c=y, alpha=0.7)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Y")

    ax.set_xlabel("eta_hat")
    ax.set_ylabel("delta_hat")
    ax.set_title(title)

    return ax


def plot_predictions_barycenter_links(
    eta,
    f_fair,
    S,
    n_bins=30,
    ax=None,
    title="Base predictions, barycenter histogram, and links",
    max_lines=None,
    min_shift_quantile=None,
):
    """
    ...
    max_lines : int or None
        If not None, randomly subsample at most this many lines.
    min_shift_quantile : float in (0,1) or None
        If not None, only draw lines for points whose shift
        |eta - f_fair| is >= this quantile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    eta = np.asarray(eta)
    f_fair = np.asarray(f_fair)
    S = np.asarray(S)

    groups = np.unique(S)
    if len(groups) != 2:
        raise ValueError("plot_predictions_barycenter_links is designed for exactly 2 groups.")


    y_pos = {groups[0]: 0.0, groups[1]: 1.0}
    y_bary = 0.5


    y_scatter = np.empty_like(eta, dtype=float)
    for idx, g in enumerate(groups):
        mask = (S == g)
        jitter = 0.03 * np.random.randn(mask.sum())
        y_scatter[mask] = y_pos[g] + jitter
        ax.scatter(
            eta[mask],
            y_scatter[mask],
            alpha=0.7,
            label=f"S={g}",
        )


    hist, bin_edges = np.histogram(f_fair, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if len(bin_centers) > 1:
        bin_width = bin_edges[1] - bin_edges[0]
    else:
        bin_width = 1.0

    if hist.max() > 0:
        hist_scaled = hist / hist.max() * 0.4
    else:
        hist_scaled = hist

    for h, x_c, h_scaled in zip(hist, bin_centers, hist_scaled):
        if h == 0:
            continue
        bottom = y_bary - h_scaled / 2.0
        rect = plt.Rectangle(
            (x_c - bin_width / 2.0, bottom),
            bin_width,
            h_scaled,
            alpha=0.3,
            color="grey",
        )
        ax.add_patch(rect)


    n = len(eta)
    idx_all = np.arange(n)


    if min_shift_quantile is not None:
        shifts = np.abs(eta - f_fair)
        thresh = np.quantile(shifts, min_shift_quantile)
        idx_all = idx_all[shifts >= thresh]


    if max_lines is not None and len(idx_all) > max_lines:
        rng = np.random.RandomState(0)
        idx_all = rng.choice(idx_all, size=max_lines, replace=False)

    bin_idx = np.digitize(f_fair, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for i in idx_all:
        x0 = eta[i]
        y0 = y_scatter[i]
        x1 = bin_centers[bin_idx[i]]
        y1 = y_bary
        ax.plot([x0, x1], [y0, y1], color="k", linewidth=0.5, alpha=0.3)

    ax.set_yticks([y_pos[groups[0]], y_bary, y_pos[groups[1]]])
    ax.set_yticklabels([f"S={groups[0]}", "barycenter", f"S={groups[1]}"])
    ax.set_xlabel("prediction")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return ax


def plot_prediction_distributions(y_base, y_fair, S, ax=None, title="Predictions by group (KDE)"):
    """
    Smooth distributions (KDE) of predictions by group, before vs after fairness.
    """

    y_base = np.asarray(y_base)
    y_fair = None if y_fair is None else np.asarray(y_fair)
    S = np.asarray(S)
    groups = np.unique(S)

    data_min = y_base.min() if y_fair is None else min(y_base.min(), y_fair.min())
    data_max = y_base.max() if y_fair is None else max(y_base.max(), y_fair.max())
    x = np.linspace(data_min, data_max, 512)

    for g in groups:
        mask = (S == g)

        # KDE for base
        if np.sum(mask) > 1:
            kde_base = gaussian_kde(y_base[mask])
            ax.plot(x, kde_base(x), alpha=0.8, label=f"ERM Unfair, S={g}")
        else:
            ax.axvline(y_base[mask][0], color="C0", alpha=0.5, label=f"ERM Unfair, S={g}")

        # KDE for fair
        if y_fair is not None:
            if np.sum(mask) > 1:
                kde_fair = gaussian_kde(y_fair[mask])
                ax.plot(x, kde_fair(x), alpha=0.8, linestyle="--", label=f"OT Unaware, S={g}")
            else:
                ax.axvline(y_fair[mask][0], color="C1", alpha=0.5, linestyle="--", label=f"OT Unaware, S={g}")

    ax.set_xlabel("prediction")
    ax.set_ylabel("density")

    ax.set_yticks([])

    ax.set_xticks([])
    ax.set_title(title)
    ax.legend()
    return ax


def plot_prediction_ecdf(y_base, y_fair, S, ax=None, title="Predictions by group (ECDF)"):
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    y_base = np.asarray(y_base).ravel()
    y_fair = None if y_fair is None else np.asarray(y_fair).ravel()
    S = np.asarray(S).ravel()
    groups = np.unique(S)

    def _ecdf(vals):
        vals = np.sort(vals)
        n = len(vals)
        y = np.arange(1, n + 1) / n
        return vals, y

    for g in groups:
        mask = (S == g)
        xb, yb = _ecdf(y_base[mask])
        ax.plot(xb, yb, label=f"ERM Unfair, S={g}", alpha=0.9)

        if y_fair is not None:
            xf, yf = _ecdf(y_fair[mask])
            ax.plot(xf, yf, linestyle="--", label=f"OT Unaware, S={g}", alpha=0.9)

    ax.set_xlabel("prediction")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_transport_plan(
    C,
    pi,
    h_plus=None,
    h_minus=None,
    ax_cost=None,
    ax_plan=None,
    use_log=True,
):
    """
    Heatmaps for cost matrix C and transport plan pi.

    - Optionally sorts rows/cols by eta (h_plus, h_minus) to reveal structure.
    - Optionally uses log scale for the plan to make tiny non-zero entries visible.
    """
    C = np.asarray(C)
    pi = np.asarray(pi)

    if ax_cost is None or ax_plan is None:
        fig, (ax_cost, ax_plan) = plt.subplots(1, 2, figsize=(10, 4))

    if h_plus is not None:
        h_plus = np.asarray(h_plus)
        idx_plus = np.argsort(h_plus)
    else:
        idx_plus = np.arange(C.shape[0])

    if h_minus is not None:
        h_minus = np.asarray(h_minus)
        idx_minus = np.argsort(h_minus)
    else:
        idx_minus = np.arange(C.shape[1])

    C_sorted = C[np.ix_(idx_plus, idx_minus)]
    pi_sorted = pi[np.ix_(idx_plus, idx_minus)]

    im0 = ax_cost.imshow(C_sorted, aspect="auto")
    ax_cost.set_title("Cost matrix C (sorted)")
    plt.colorbar(im0, ax=ax_cost, fraction=0.046, pad=0.04)

    if use_log:
        positive = pi_sorted[pi_sorted > 0]
        if positive.size > 0:
            vmin = max(positive.min(), 1e-6)
            vmax = positive.max()
        else:
            vmin, vmax = 1e-6, 1.0

        im1 = ax_plan.imshow(
            pi_sorted + 1e-12,
            aspect="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
    else:
        im1 = ax_plan.imshow(pi_sorted, aspect="auto")

    ax_plan.set_title("Transport plan pi (sorted, log scale)" if use_log else "Transport plan pi (sorted)")
    plt.colorbar(im1, ax=ax_plan, fraction=0.046, pad=0.04)

    return ax_cost, ax_plan

def plot_rank_reordering(eta, y_fair, S, group, ax=None, title=None):
    """
    Within-group rank scatter: rank(eta) vs rank(fair_pred) for a given group.
    """
    if ax is None:
        fig, ax = plt.subplots()

    eta = np.asarray(eta)
    y_fair = np.asarray(y_fair)
    S = np.asarray(S)

    mask = (S == group)
    eta_g = eta[mask]
    fair_g = y_fair[mask]

    ##ranks in [0, 1]
    ranks_eta = np.argsort(np.argsort(eta_g)) / (len(eta_g) - 1)
    ranks_fair = np.argsort(np.argsort(fair_g)) / (len(fair_g) - 1)

    ax.scatter(ranks_eta, ranks_fair, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)

    ax.set_xlabel("rank of eta within group")
    ax.set_ylabel("rank of fair prediction within group")
    if title is None:
        title = f"Within-group rank reordering (S={group})"
    ax.set_title(title)

    return ax

def plot_base_vs_fair_scatter(y_base, y_fair, S, ax=None, title="Base vs fair predictions"):
    """
    Scatter plot: x = base prediction (eta_hat), y = fair prediction (f_hat),
    colored by sensitive group S.
    """
    if ax is None:
        fig, ax = plt.subplots()

    y_base = np.asarray(y_base)
    y_fair = np.asarray(y_fair)
    S = np.asarray(S)

    groups = np.unique(S)
    for g in groups:
        mask = (S == g)
        ax.scatter(y_base[mask], y_fair[mask], alpha=0.5, label=f"S={g}")

    lo = min(y_base.min(), y_fair.min())
    hi = max(y_base.max(), y_fair.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("base prediction eta_hat")
    ax.set_ylabel("fair prediction f_hat")
    ax.set_title(title)
    ax.legend()
    return ax

def aggregate_plan_by_eta_bins(pi, h_plus, h_minus, n_bins=10):
    """
    Aggregate a transport plan pi into a small n_bins x n_bins matrix
    by binning plus/minus points according to their eta values.

    Parameters
    ----------
    pi : (n_plus, n_minus) array
        Transport plan.
    h_plus : (n_plus,) array
        eta for plus points.
    h_minus : (n_minus,) array
        eta for minus points.
    n_bins : int
        Number of eta bins on each side.

    (Returns
    -------
    agg : (n_bins, n_bins) array
        Aggregated mass in each (bin_plus, bin_minus) cell.
    bin_edges : (n_bins + 1,) array
        Common eta bin edges used for both sides.)

        Returns
    -------
    agg : (n_bins, n_bins) array
        Aggregated mass in each (bin_plus, bin_minus) cell.
    bin_edges_plus : (n_bins + 1,) array
        Eta bin edges for plus side.
    bin_edges_minus : (n_bins + 1,) array
        Eta bin edges for minus side.

    """
    pi = np.asarray(pi)
    h_plus = np.asarray(h_plus)
    h_minus = np.asarray(h_minus)
    bin_edges_plus = np.quantile(h_plus,  np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges_minus = np.quantile(h_minus, np.linspace(0.0, 1.0, n_bins + 1))

    plus_bins = np.digitize(h_plus, bin_edges_plus) - 1
    minus_bins = np.digitize(h_minus, bin_edges_minus) - 1

    # DEBUG clip just in case something falls exactly on the last edge
    plus_bins = np.clip(plus_bins, 0, n_bins - 1)
    minus_bins = np.clip(minus_bins, 0, n_bins - 1)

    agg = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(pi.shape[0]):
        bi = plus_bins[i]
        row = pi[i]
        for j in np.nonzero(row)[0]:
            bj = minus_bins[j]
            if 0 <= bi < n_bins and 0 <= bj < n_bins:
                agg[bi, bj] += row[j]

    return agg, bin_edges_plus, bin_edges_minus 


def plot_aggregated_plan_by_eta(pi, h_plus, h_minus, n_bins=10, ax=None,
                                title="Aggregated transport by eta bins"):
    """
    (Plot a small n_bins x n_bins heatmap of aggregated transport mass,
    where rows/cols correspond to eta-bins for plus/minus points.)

        Plot a small n_bins x n_bins heatmap of aggregated transport mass,
    where rows correspond to eta-bins (plus side, sorted ascending)
    and columns to eta-bins (minus side, sorted ascending),
    in the same "direction" as the full pi plot.
    """
    if ax is None:
        fig, ax = plt.subplots()


    agg, bin_edges_plus, bin_edges_minus = aggregate_plan_by_eta_bins(
        pi, h_plus, h_minus, n_bins=n_bins
    )

    im = ax.imshow(agg, origin="upper", aspect="auto")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    centers_plus = 0.5 * (bin_edges_plus[:-1] + bin_edges_plus[1:])
    centers_minus = 0.5 * (bin_edges_minus[:-1] + bin_edges_minus[1:])
    tick_idx = np.linspace(0, n_bins - 1, min(n_bins, 5), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{centers_plus[i]:.2f}" for i in tick_idx])
    ax.set_xticklabels([f"{centers_minus[i]:.2f}" for i in tick_idx], rotation=45)


    ax.set_xlabel("eta-bin (minus side)")
    ax.set_ylabel("eta-bin (plus side)")
    return ax


def single_pair_cost(h1, d1, h2, d2, eps=1e-12):
    """
    Cost c(x1, x2) = (h2 - h1)^2 / (|d1| + |d2|).

    Parameters
    ----------
    h1, d1 : float
        (eta, delta) for x1.
    h2, d2 : float
        (eta, delta) for x2.

    Returns
    -------
    float
        OT cost between the two points.
    """
    return (h2 - h1) ** 2 / (abs(d1) + abs(d2) + eps)

def plot_cost_preference_for_x1(
    h_plus,
    d_plus,
    h_minus,
    d_minus,
    i_plus,
    j_ref=None,
    ax=None,
    title=None,
):
    """
    Visualize regions where c(x1, x) > c(x1, x_ref) for a fixed x1 in μ⁺
    and candidate x in μ⁻.

    Parameters
    ----------
    h_plus, d_plus : arrays of shape (n_plus,)
        eta and delta for μ⁺ points.
    h_minus, d_minus : arrays of shape (n_minus,)
        eta and delta for μ⁻ points.
    i_plus : int
        Index of the chosen x1 in μ⁺.
    j_ref : int or None
        Index of reference x_ref in μ⁻ (x3 in the paper).
        If None, we choose the minus point with minimal cost to x1.
    ax : matplotlib axis or None
    title : str or None

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)
    h_minus = np.asarray(h_minus)
    d_minus = np.asarray(d_minus)


    h1 = h_plus[i_plus]
    d1 = d_plus[i_plus]

    denom = np.abs(d1) + np.abs(d_minus) + 1e-12
    costs = (h_minus - h1) ** 2 / denom

    if j_ref is None:
        j_ref = np.argmin(costs)  
    cost_ref = costs[j_ref]

    # Region indicator: where c(x1, x) > c(x1, x_ref)
    mask_high = costs > cost_ref  #
    mask_low = ~mask_high         

    # Plot minus points in (eta, delta)-space
    ax.scatter(
        h_minus[mask_low],
        d_minus[mask_low],
        c="C0",
        alpha=0.6,
        label="c(x1, x) ≤ c(x1, x_ref)",
    )
    ax.scatter(
        h_minus[mask_high],
        d_minus[mask_high],
        c="C1",
        alpha=0.6,
        label="c(x1, x) > c(x1, x_ref)",
    )

    # Highlight x_ref
    ax.scatter(
        h_minus[j_ref],
        d_minus[j_ref],
        c="k",
        marker="x",
        s=80,
        label="x_ref (minus)",
    )

    # Highlight x1 (source) as well, even though it's from μ⁺
    ax.scatter(
        h1,
        d1,
        c="k",
        marker="*",
        s=120,
        label="x1 (plus)",
    )

    ax.set_xlabel("eta (minus side)")
    ax.set_ylabel("delta (minus side)")
    if title is None:
        title = f"Region where c(x1, x) > c(x1, x_ref) (i_plus={i_plus}, j_ref={j_ref})"
    ax.set_title(title)
    ax.legend(loc="best")

    return ax


def plot_global_best_partners(
    C,
    h_minus,
    d_minus,
    ax=None,
    title="Global attractiveness of minus points (best partners count)",
):
    """
    For each plus point i, find the cheapest minus partner j* = argmin_j C[i,j].
    Count how often each minus j is selected, and plot minus points in (eta, delta)
    colored by this count.

    Parameters
    ----------
    C : (n_plus, n_minus) array
        Cost matrix between plus and minus points.
    h_minus, d_minus : (n_minus,) arrays
        eta and delta for minus points.
    ax : matplotlib axis or None
    title : str

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    C = np.asarray(C)
    h_minus = np.asarray(h_minus)
    d_minus = np.asarray(d_minus)

    # For each plus index i, cheapest minus index j*
    j_best = np.argmin(C, axis=1)  # shape (n_plus,)

    # Count how often each minus j is chosen
    n_minus = C.shape[1]
    counts = np.bincount(j_best, minlength=n_minus)  # shape (n_minus,)

    # To avoid everything being dark when counts=0, add a tiny epsilon if you want
    # counts = counts + 1e-12

    sc = ax.scatter(
        h_minus,
        d_minus,
        c=counts,
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="nb. times j is best partner")

    ax.set_xlabel("eta (minus side)")
    ax.set_ylabel("delta (minus side)")
    ax.set_title(title)
    return ax


def plot_min_cost_for_plus(
    C,
    h_plus,
    d_plus,
    ax=None,
    title="Minimal cost to match each plus point",
):
    """
    For each plus point i, compute min_j C[i,j] and plot (eta_plus, delta_plus)
    colored by this minimal cost.

    Parameters
    ----------
    C : (n_plus, n_minus) array
        Cost matrix between plus and minus points.
    h_plus, d_plus : (n_plus,) arrays
        eta and delta for plus points.
    ax : matplotlib axis or None
    title : str

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    C = np.asarray(C)
    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)

    # minimal cost for each plus point
    c_min = C.min(axis=1)   # shape (n_plus,)

    sc = ax.scatter(
        h_plus,
        d_plus,
        c=c_min,
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="min_j c(x_i^+, x_j^-)")

    ax.set_xlabel("eta (plus side)")
    ax.set_ylabel("delta (plus side)")
    ax.set_title(title)
    return ax


def plot_min_cost_for_plus_log(
    C,
    h_plus,
    d_plus,
    ax=None,
    title="Minimal cost per plus point (log scale)",
):
    """
    For each plus point i, compute min_j C[i,j] and plot (eta_plus, delta_plus)
    colored by this minimal cost, on a log color scale.
    """
    if ax is None:
        fig, ax = plt.subplots()

    C = np.asarray(C)
    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)

    # minimal cost for each plus point
    c_min = C.min(axis=1)  # shape (n_plus,)

    # avoid zeros for LogNorm
    positive = c_min[c_min > 0]
    if positive.size > 0:
        min_pos = positive.min()
        c_min_plot = np.where(c_min > 0, c_min, min_pos)
        norm = LogNorm(vmin=min_pos, vmax=positive.max())
    else:
        # degenerate case: everything is 0
        c_min_plot = np.ones_like(c_min)
        norm = None

    sc = ax.scatter(
        h_plus,
        d_plus,
        c=c_min_plot,
        cmap="viridis",
        norm=norm,
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="min_j c(x_i^+, x_j^-)")

    ax.set_xlabel("eta (plus side)")
    ax.set_ylabel("delta (plus side)")
    ax.set_title(title)
    return ax


def plot_min_cost_hard_vs_easy(
    C,
    h_plus,
    d_plus,
    quantile=0.95,
    ax=None,
    title=None,
):
    """
    Plot plus points, highlighting the 'hard' ones (with largest minimal cost)
    and fading the 'easy' ones in grey.

    Parameters
    ----------
    quantile : float in (0,1)
        Points with c_min >= quantile(c_min) are considered 'hard'.
    """
    if ax is None:
        fig, ax = plt.subplots()

    C = np.asarray(C)
    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)

    c_min = C.min(axis=1)

    # threshold between easy and hard
    q = np.quantile(c_min, quantile)
    hard = c_min >= q


    ax.scatter(
        h_plus[~hard],
        d_plus[~hard],
        color="lightgrey",
        alpha=0.4,
        label="easy (low min cost)",
    )

    # hard points in color
    sc = ax.scatter(
        h_plus[hard],
        d_plus[hard],
        c=c_min[hard],
        cmap="viridis",
        alpha=0.9,
        label=f"hard (c_min ≥ q_{quantile:.2f})",
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="min_j c(x_i^+, x_j^-)")

    ax.set_xlabel("eta (plus side)")
    ax.set_ylabel("delta (plus side)")
    if title is None:
        title = f"Hard vs easy plus points (top {int((1-quantile)*100)}% in min cost)"
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_min_cost_easy_only(
    C,
    h_plus,
    d_plus,
    quantile=0.95,
    ax=None,
    title=None,
):
    """
    Plot only the 'easy' plus points (c_min below a given quantile),
    with a log color scale so you can see the variation among them.
    """
    if ax is None:
        fig, ax = plt.subplots()

    C = np.asarray(C)
    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)

    c_min = C.min(axis=1)
    q = np.quantile(c_min, quantile)
    easy = c_min < q

    c_easy = c_min[easy]

    positive = c_easy[c_easy > 0]
    if positive.size > 0:
        min_pos = positive.min()
        c_plot = np.where(c_easy > 0, c_easy, min_pos)
        norm = LogNorm(vmin=min_pos, vmax=positive.max())
    else:
        c_plot = np.ones_like(c_easy)
        norm = None

    sc = ax.scatter(
        h_plus[easy],
        d_plus[easy],
        c=c_plot,
        cmap="viridis",
        norm=norm,
        alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="min_j c(x_i^+, x_j^-) (easy pts)")

    ax.set_xlabel("eta (plus side)")
    ax.set_ylabel("delta (plus side)")
    if title is None:
        title = f"Easy plus points only (c_min < q_{quantile:.2f}, log scale)"
    ax.set_title(title)
    return ax

def plot_binned_cost_by_eta(
    C,
    h_plus,
    h_minus,
    n_bins=10,
    ax=None,
    title="Average cost by eta bins (plus vs minus)",
):
    """
    Aggregate the cost matrix C into an n_bins x n_bins heatmap:
    rows = eta-bins on plus side, columns = eta-bins on minus side,
    values = average cost C_ij within each bin pair.
    """
    if ax is None:
        fig, ax = plt.subplots()

    C = np.asarray(C)
    h_plus = np.asarray(h_plus)
    h_minus = np.asarray(h_minus)

    # separate quantile bins for plus and minus (like in aggregated plan)
    bin_edges_plus = np.quantile(h_plus,  np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges_minus = np.quantile(h_minus, np.linspace(0.0, 1.0, n_bins + 1))

    plus_bins = np.digitize(h_plus,  bin_edges_plus)  - 1
    minus_bins = np.digitize(h_minus, bin_edges_minus) - 1

    plus_bins = np.clip(plus_bins, 0, n_bins - 1)
    minus_bins = np.clip(minus_bins, 0, n_bins - 1)

    agg_sum = np.zeros((n_bins, n_bins), dtype=float)
    agg_count = np.zeros((n_bins, n_bins), dtype=float)

    n_plus, n_minus = C.shape
    for i in range(n_plus):
        bi = plus_bins[i]
        row = C[i]
        for j in range(n_minus):
            bj = minus_bins[j]
            if 0 <= bi < n_bins and 0 <= bj < n_bins:
                agg_sum[bi, bj] += row[j]
                agg_count[bi, bj] += 1.0

    # avoid division by 0
    with np.errstate(divide="ignore", invalid="ignore"):
        agg_mean = np.where(agg_count > 0, agg_sum / agg_count, np.nan)

    im = ax.imshow(agg_mean, origin="upper", aspect="auto")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean cost C_ij")

    centers_plus = 0.5 * (bin_edges_plus[:-1] + bin_edges_plus[1:])
    centers_minus = 0.5 * (bin_edges_minus[:-1] + bin_edges_minus[1:])
    tick_idx = np.linspace(0, n_bins - 1, min(n_bins, 5), dtype=int)

    ax.set_yticks(tick_idx)
    ax.set_xticks(tick_idx)
    ax.set_yticklabels([f"{centers_plus[i]:.2f}" for i in tick_idx])
    ax.set_xticklabels([f"{centers_minus[i]:.2f}" for i in tick_idx], rotation=45)

    ax.set_xlabel("eta-bin (minus side)")
    ax.set_ylabel("eta-bin (plus side)")
    return ax


def plot_ordering_metrics_bars(order_by_group, ax=None, metric="kendall_tau", title=None):
    """
    Barplot for one ordering metric across groups (+ overall).
    order_by_group: dict from rank_metrics_by_group(...)
    metric in {"kendall_tau","spearman","mean_abs_rank_disp","topk_jaccard"}
    """
    if ax is None:
        fig, ax = plt.subplots()

    labels = list(order_by_group.keys())
    values = [order_by_group[k].get(metric, np.nan) for k in labels]

    ax.bar(labels, values)
    ax.set_xlabel("Group")
    ax.set_ylabel(metric)

    if metric in ("kendall_tau", "spearman"):
        ax.set_ylim(-1, 1)
        ax.axhline(0, linewidth=1)
    else:
        ax.set_ylim(0, 1)

    ax.set_xticklabels(labels, rotation=45, ha="right")

    if title is None:
        title = f"Ordering metric: {metric}"
    ax.set_title(title)
    return ax


def plot_auc_report_bars(auc_rep, ax=None, title="AUC summary"):
    """
    Barplot of AUC numbers in auc_rep (dict from auc_report).
    """
    if ax is None:
        fig, ax = plt.subplots()

    labels = list(auc_rep.keys())
    values = [auc_rep[k] for k in labels]

    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    return ax


def plot_gamma_sweep(rows, save_path=None):
    gam = [r["gamma"] for r in rows]
    mse_b = [r["mse_base"] for r in rows]
    mse_f = [r["mse_fair"] for r in rows]
    dp_b  = [r["dp_base"] for r in rows]
    dp_f  = [r["dp_fair"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    axes[0].plot(gam, mse_b, marker="o", label="base")
    axes[0].plot(gam, mse_f, marker="o", label="fair")
    axes[0].set_xlabel("gamma (X separability by S)")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].set_title("MSE vs gamma")

    axes[1].plot(gam, dp_b, marker="o", label="base")
    axes[1].plot(gam, dp_f, marker="o", label="fair")
    axes[1].set_xlabel("gamma (X separability by S)")
    axes[1].set_ylabel(r"$|\mathbb{E}[\hat y|S=0]-\mathbb{E}[\hat y|S=1]|$")
    axes[1].legend()
    axes[1].set_title("Unfairness (DP mean gap) vs gamma")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig



def plot_gamma_sweep_multi(
    rows, save_path, *,
    x_key="gamma",
    dp_keys=("w2_base", "w2_ot", "w2_aware", "w2_fairreg", "w2_frappe", "w2_shat"),
    mse_keys=("mse_base", "mse_ot", "mse_aware", "mse_fairreg", "mse_frappe", "mse_shat"),
    dp_base_key="w2_base",
    mse_base_key="mse_base",
    relative=True,         
    relative_mode="pct",    # "pct" or "ratio"
    eps=1e-12,
    marker=None,           
    title_prefix="Confounded sweep",
    dp_ylabel=r"$W_2$",
    log_x=True,
    log_y=True,
    labels_dict=None,     
    colors_dict=None,
    linestyles_dict=None,
):
    
    df = pd.DataFrame(rows)

    def _rel(y, base):
        base_safe = np.maximum(np.abs(base), eps)
        if relative_mode == "ratio":
            return y / base_safe
        if relative_mode == "pct":
            return 100.0 * (y - base) / base_safe
        raise ValueError("relative_mode must be 'pct' or 'ratio'")

    if relative:
        for k in dp_keys:
            df[k] = _rel(df[k], df[dp_base_key])
        for k in mse_keys:
            df[k] = _rel(df[k], df[mse_base_key])

    df_mean = df.groupby(x_key).mean().reset_index()
    df_std = df.groupby(x_key).std().reset_index()
    df_count = df.groupby(x_key).count().reset_index()
    df_ci = 1.96 * df_std / np.sqrt(df_count)

    gammas = df_mean[x_key].values

    name_map = labels_dict if labels_dict is not None else {}
    color_map = colors_dict if colors_dict is not None else {}

    # COMBINED FIGURE for the paper 
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2)) 

    # MSE Plot (Left) 
    ax_mse = axes[0]
    for k in mse_keys:
        y_mean = df_mean[k].values
        y_ci = df_ci[k].values
        raw_name = k.replace("mse_", "")
        
        label = name_map.get(raw_name, raw_name)
        color = color_map.get(raw_name, None)
        ls = linestyles_dict.get(raw_name, None) if linestyles_dict else '-'
        
        line, = ax_mse.plot(gammas, y_mean, marker=marker, label=label, color=color, linestyle=ls)
        ax_mse.fill_between(gammas, y_mean - y_ci, y_mean + y_ci, color=line.get_color(), alpha=0.2)
    
    if log_x: ax_mse.set_xscale("log")
    if log_y: ax_mse.set_yscale("log")
    ax_mse.set_xlabel(r"$\gamma$")
    ax_mse.set_ylabel("MSE" if not relative else ("MSE / base" if relative_mode == "ratio" else "% change of MSE vs base"))
    ax_mse.set_title(r"MSE vs $\gamma$")
    ax_mse.grid(True, alpha=0.3)

    # W2 Plot (Right) 
    ax_w2 = axes[1]
    for k in dp_keys:
        y_mean = df_mean[k].values
        y_ci = df_ci[k].values
        raw_name = k.replace("w2_", "").replace("dp_", "").replace("var_", "")
        
        label = name_map.get(raw_name, raw_name)
        color = color_map.get(raw_name, None)
        ls = linestyles_dict.get(raw_name, None) if linestyles_dict else '-'
        
        line, = ax_w2.plot(gammas, y_mean, marker=marker, label=label, color=color, linestyle=ls)
        ax_w2.fill_between(gammas, y_mean - y_ci, y_mean + y_ci, color=line.get_color(), alpha=0.2)
        
    if log_y: ax_w2.set_yscale("log")
    ax_w2.set_xscale("log" if log_x else "linear")
    ax_w2.set_xlabel(r"$\gamma$")
    ax_w2.set_ylabel(dp_ylabel if not relative else (f"{dp_ylabel} / base" if relative_mode == "ratio" else f"% change of {dp_ylabel} vs base"))
    ax_w2.set_title(f"{dp_ylabel} vs "+r"$\gamma$")
    ax_w2.grid(True, alpha=0.3)


    handles, labels = ax_mse.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.98, 0.5))



    fig.tight_layout()
    fig.subplots_adjust(right=0.96) 


    suffix = "_rel" if relative else "_abs"
    combined_path = str(save_path).replace(".pdf", f"{suffix}_combined.pdf")
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)

    return combined_path


def plot_gamma_sweep_relative_curves(
    rows,
    save_path=None,
    *,
    x_key="gamma",
    mse_base_key="mse_base",
    w2_base_key="w2_base",
    relative="pct",   # "pct" or "ratio"
    eps=1e-3,
    title_prefix="Confounded sweep",
    log_x=True,
    log_y=True,
):
    """
    Plots (relative MSE vs gamma) and (relative W2 vs gamma) as curves with 95% CI.
    Relative is computed w.r.t. the base at the SAME gamma for each run.
    """

    df = pd.DataFrame(rows)


    keys = list(df.columns)
    mse_keys = [k for k in keys if k.startswith("mse_") and k != mse_base_key]
    w2_keys  = [k for k in keys if k.startswith("w2_") and k != w2_base_key]

    def _rel(y, base):
        base_safe = np.maximum(np.abs(base), eps)
        if relative == "ratio":
            return y / base_safe
        if relative == "pct":
            return 100.0 * (y - base) / base_safe
        raise ValueError("relative must be 'pct' or 'ratio'")


    for k in mse_keys:
        df[k] = _rel(df[k], df[mse_base_key])
    for k in w2_keys:
        df[k] = _rel(df[k], df[w2_base_key])


    df_mean = df.groupby(x_key).mean().reset_index()
    df_std = df.groupby(x_key).std().reset_index()
    df_count = df.groupby(x_key).count().reset_index()

    # Calculate 95% CI
    df_ci = (1.96 * df_std / np.sqrt(df_count)).fillna(0.0)

    gammas = df_mean[x_key].values
    name_map = {
        "base": "ERM", 
        "ot": "FAUST-U", 
        "aware": "FAUST-A", 
        "fairreg": "FairReg", 
        "frappe": "FRAPPÉ", 
        "shat": "Plug-In W2"
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    if log_x:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
    if log_y:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")

    # MSE relative 
    for k in mse_keys:
        y_mean = df_mean[k].values
        y_ci = df_ci[k].values
        raw_name = k.replace("mse_", "")
        
        line, = axes[0].plot(gammas, y_mean, label=name_map.get(raw_name, raw_name))  
        axes[0].fill_between(gammas, y_mean - y_ci, y_mean + y_ci, alpha=0.2, color=line.get_color())

    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_title("Relative MSE vs $\\gamma$")
    axes[0].set_ylabel("MSE / base" if relative == "ratio" else "% change vs base")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    # W2 relative 
    for k in w2_keys:
        y_mean = df_mean[k].values
        y_ci = df_ci[k].values
        raw_name = k.replace("w2_", "")
        
        line, = axes[1].plot(gammas, y_mean, label=name_map.get(raw_name, raw_name))  
        axes[1].fill_between(gammas, y_mean - y_ci, y_mean + y_ci, alpha=0.2, color=line.get_color())

    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_title("Relative $W_2$ vs $\\gamma$")
    axes[1].set_ylabel(r"$W_2$ / base" if relative == "ratio" else "% change vs base")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)


    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig

def plot_mse_vs_w2_tradeoff(
    rows,
    save_path=None,
    *,
    x_key="gamma",
    w2_keys=("w2_base", "w2_ot", "w2_aware", "w2_fairreg", "w2_frappe", "w2_shat"),
    mse_keys=("mse_base", "mse_ot", "mse_aware", "mse_fairreg", "mse_frappe", "mse_shat"),
    title="Accuracy vs Fairness Trade-off",
    w2_max=1.5,
    mse_max=None,
    labels_dict=None    
):
    df = pd.DataFrame(rows)
    
    # Calculate Mean, Std, and  CI
    df_mean = df.groupby(x_key).mean().reset_index()
    df_std = df.groupby(x_key).std().reset_index().fillna(0.0)
    df_count = df.groupby(x_key).count().reset_index()
    df_ci = (1.96 * df_std / np.sqrt(df_count)).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    
    if labels_dict is not None:
        name_map = labels_dict
    else:
        name_map = {
            "base": "ERM", 
            "ot": "FAUST-U", 
            "aware": "FAUST-A", 
            "fairreg": "FairReg", 
            "frappe": "FRAPPÉ", 
            "shat": "Plug-In W2"
        }
    gammas = df_mean[x_key].values
    
    alphas = np.linspace(0.2, 1.0, len(gammas))

    for w2_k, mse_k in zip(w2_keys, mse_keys):
        w2_vals = df_mean[w2_k].values
        mse_vals = df_mean[mse_k].values
        w2_ci = df_ci[w2_k].values
        mse_ci = df_ci[mse_k].values
        
        raw_name = w2_k.replace("w2_", "")
        label = name_map.get(raw_name, raw_name)
        
        #dummy invisible line just to claim a color and populate the legend
        dummy_line, = ax.plot([], [], marker='o', label=label, linestyle='-')
        color = dummy_line.get_color()
        
        #Plot the points and segments with an increasing alpha
        for i in range(len(gammas)):
            # Draw a line segment connecting to the previous point
            if i > 0:
                ax.plot(
                    [w2_vals[i-1], w2_vals[i]], 
                    [mse_vals[i-1], mse_vals[i]], 
                    color=color, alpha=alphas[i], linestyle='-'
                )

    ax.set_xlabel(r"Fairness Violation ($W_2 \downarrow$)")
    ax.set_ylabel(r"Error ($\mathrm{MSE} \downarrow$)")
    ax.set_title(title)
    

    ax.set_xscale("log")
    ax.set_yscale("log")
    

    if w2_max is not None:
        ax.set_xlim(right=w2_max)
    if mse_max is not None:
        ax.set_ylim(top=mse_max)
        
    ax.legend()
    ax.grid(True, alpha=0.3, which="both", ls="--")
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        
    return save_path

def plot_fairreg_convergence(df_conv, dataset_name, save_path):
    """
    Plots FairReg convergence metrics as a function of Epochs.
    Averages over multiple runs and plots a 95% Confidence Interval.
    """

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.5))
    
    x_col = 'Epoch' if 'Epoch' in df_conv.columns else 'T'
    x_label = "Number of Epochs" if x_col == 'Epoch' else "Number of Iterations (T)"
    

    df_mean = df_conv.groupby(x_col).mean().reset_index()
    

    if 'run' in df_conv.columns and df_conv['run'].nunique() > 1:
        df_std = df_conv.groupby(x_col).std().reset_index().fillna(0.0)
        df_count = df_conv.groupby(x_col).count().reset_index()
        df_ci = 1.96 * df_std / np.sqrt(df_count) 
    else:
        df_ci = df_mean.copy()
        for col in df_ci.columns:
            if col != x_col: df_ci[col] = 0.0

    metrics_config = [
        ('MSE', "MSE", r"$\mathrm{MSE}$", 'tab:blue', 'o'),
        ('DP_max', "KS", "KS", 'tab:red', 's'),
        ('W2', r"$W_2$", r"$W_2$", 'tab:green', '^'),
        ('Time_sec', "Total Training Time", "Time (seconds)", 'tab:purple', 'd')
    ]

    for i, (col, title, ylabel, color, marker) in enumerate(metrics_config):
        ax = axes[i]
        if col not in df_mean.columns:
            ax.set_visible(False)
            continue
            
        x_vals = df_mean[x_col]
        y_mean = df_mean[col]
        y_ci = df_ci[col]
        

        ax.plot(x_vals, y_mean, marker=marker, markersize=2, linewidth=1.5, color=color)
        
        ax.fill_between(x_vals, y_mean - y_ci, y_mean + y_ci, color=color, alpha=0.2)
        
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

            
        if col == 'Time_sec':
            ax.set_yscale('log')

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()