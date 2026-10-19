# fair_unaware/metrics.py
import numpy as np
from scipy.stats import rankdata, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import combinations


# def dp_cdf_gaps(y_pred, S, bins):
#     """
#     DP metric for regression: sup_t | P(Yhat<=t|S=0) - P(Yhat<=t|S=1) |
#     returned as a dict over the provided bins (like the external FairReg code did).
#     """
#     y_pred = np.asarray(y_pred).ravel()
#     S = np.asarray(S).ravel()
#     bins = np.asarray(bins).ravel()

#     s_vals = np.unique(S)
#     if s_vals.size != 2:
#         raise ValueError("dp_cdf_gaps currently expects exactly 2 groups.")

#     s0, s1 = s_vals[0], s_vals[1]
#     y0 = y_pred[S == s0]
#     y1 = y_pred[S == s1]
#     if y0.size == 0 or y1.size == 0:
#         raise ValueError("Both groups must have at least one sample to compute DP gaps.")

#     gaps = {}
#     for t in bins:
#         c0 = float(np.mean(y0 <= t))
#         c1 = float(np.mean(y1 <= t))
#         gaps[float(t)] = abs(c0 - c1)
#     return gaps

def dp_cdf_gaps(y_pred, S, bins):
    """
    DP metric for regression: sup_t max_{i,j} | P(Yhat<=t|S=i) - P(Yhat<=t|S=j) |
    returned as a dict over the provided bins.
    """
    y_pred = np.asarray(y_pred).ravel()
    S = np.asarray(S).ravel()
    bins = np.asarray(bins).ravel()

    groups = np.unique(S)
    if groups.size < 2:
        raise ValueError("dp_cdf_gaps expects at least 2 groups.")

    group_cdfs = {}
    for g in groups:
        y_g = y_pred[S == g]
        if y_g.size == 0:
            raise ValueError(f"Group {g} has no samples.")
        # Precompute CDF for this group across all bins
        group_cdfs[g] = np.array([float(np.mean(y_g <= t)) for t in bins])

    gaps = {}
    for i, t in enumerate(bins):
        # For multiple groups, the max pairwise gap is simply max(CDF) - min(CDF)
        vals_at_t = [group_cdfs[g][i] for g in groups]
        gaps[float(t)] = max(vals_at_t) - min(vals_at_t)
        
    return gaps


def dp_cdf_max_gap(y_pred, S, bins):
    gaps = dp_cdf_gaps(y_pred, S, bins=bins)
    return float(max(gaps.values()))


# def tv_gap(y, S, bins):
#     """
#     Computes the Total Variation (TV) distance between the empirical 
#     distributions of predictions for the two groups, using fixed bins.
#     """
#     y = np.asarray(y)
#     S = np.asarray(S)
#     groups = np.unique(S)
    
#     if len(groups) != 2:
#         raise ValueError("TV gap is only implemented for binary groups here.")
        
#     y0 = y[S == groups[0]]
#     y1 = y[S == groups[1]]
    

#     y0_clipped = np.clip(y0, bins[0], bins[-1])
#     y1_clipped = np.clip(y1, bins[0], bins[-1])
    
#     p0, _ = np.histogram(y0_clipped, bins=bins)
#     p1, _ = np.histogram(y1_clipped, bins=bins)
    
#     p0 = p0 / max(len(y0), 1)
#     p1 = p1 / max(len(y1), 1)
    
#     return 0.5 * float(np.sum(np.abs(p0 - p1)))


def tv_gap(y, S, bins):
    """
    Computes the maximum Total Variation (TV) distance between the empirical 
    distributions of predictions across all pairs of sensitive groups.
    """
    y = np.asarray(y)
    S = np.asarray(S)
    groups = np.unique(S)
    
    if len(groups) < 2:
        raise ValueError("TV gap requires at least 2 groups.")

    group_hists = {}
    for g in groups:
        y_g = y[S == g]
        y_g_clipped = np.clip(y_g, bins[0], bins[-1])
        p_g, _ = np.histogram(y_g_clipped, bins=bins)
        group_hists[g] = p_g / max(len(y_g), 1)
        
    max_tv = 0.0
    for g1, g2 in combinations(groups, 2):
        tv = 0.5 * float(np.sum(np.abs(group_hists[g1] - group_hists[g2])))
        if tv > max_tv:
            max_tv = tv

    return max_tv

def w2_1d(a, b, n_grid=200):
    """
    1D Wasserstein-2 distance between samples a and b using quantile approximation.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return np.nan
    qs = np.linspace(0.0, 1.0, n_grid)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))

def w2_1d_weighted(
    x: np.ndarray,
    wx: np.ndarray,
    y: np.ndarray,
    wy: np.ndarray,
    eps: float = 1e-15,
    *,
    squared: bool = True,
) -> float:
    """
    Weighted 1D Wasserstein-2 between two empirical distributions.

    squared=True  -> returns W2^2
    squared=False -> returns W2
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    wx = np.asarray(wx, float)
    wy = np.asarray(wy, float)

    # Sort each support
    ix = np.argsort(x)
    iy = np.argsort(y)
    x = x[ix]
    wx = wx[ix]
    y = y[iy]
    wy = wy[iy]

    # Normalize weights
    wx = np.clip(wx, 0.0, np.inf)
    wy = np.clip(wy, 0.0, np.inf)
    wx = wx / max(wx.sum(), eps)
    wy = wy / max(wy.sum(), eps)

    # cumulative weights (quantile levels)
    cx = np.cumsum(wx)
    cy = np.cumsum(wy)

    # Force endpoints to be exactly 1.0 (prevents grid containing 1.0 while cumsum ends < 1)
    cx[-1] = 1.0
    cy[-1] = 1.0

    # merge the cumulative grids
    grid = np.unique(np.concatenate([cx, cy]))
    grid = np.clip(grid, 0.0, 1.0)

    # for each t in grid, pick smallest index with cumsum >= t
    ix = np.searchsorted(cx, grid, side="left")
    iy = np.searchsorted(cy, grid, side="left")

    # IMPORTANT: clip indices to valid range (handles any tiny numeric overshoot)
    ix = np.clip(ix, 0, len(x) - 1)
    iy = np.clip(iy, 0, len(y) - 1)

    qx = x[ix]
    qy = y[iy]

    # integral over [0,1] of squared quantile gaps
    dt = np.diff(np.concatenate([[0.0], grid]))
    w2_sq = float(np.sum(((qx - qy) ** 2) * dt))

    return w2_sq if squared else float(np.sqrt(w2_sq))

# def w2_gap_between_groups(
#     y_hat: np.ndarray,
#     s: np.ndarray,
#     *,
#     squared: bool = False,
#     eps: float = 1e-15,
# ) -> float:
#     """
#     Wasserstein gap between the two sensitive groups.

#     Returns W2^2 by default (set squared=False to get W2).
#     """
#     y_hat = np.asarray(y_hat, float).ravel()
#     s = np.asarray(s).ravel()

#     vals = np.unique(s)
#     if len(vals) != 2:
#         raise ValueError(f"Expected binary groups; got {vals}")

#     g0, g1 = vals[0], vals[1]
#     x0 = y_hat[s == g0]
#     x1 = y_hat[s == g1]

#     wx0 = np.full_like(x0, 1.0 / max(len(x0), 1), dtype=float)
#     wx1 = np.full_like(x1, 1.0 / max(len(x1), 1), dtype=float)

#     return float(w2_1d_weighted(x0, wx0, x1, wx1, eps=eps, squared=squared))

def w2_gap_between_groups(
    y_hat: np.ndarray,
    s: np.ndarray,
    *,
    squared: bool = False,
    eps: float = 1e-15,
) -> float:
    """
    Maximum Wasserstein gap across all pairs of sensitive groups.

    Returns W2^2 by default (set squared=False to get W2).
    """
    y_hat = np.asarray(y_hat, float).ravel()
    s = np.asarray(s).ravel()

    groups = np.unique(s)
    if len(groups) < 2:
        raise ValueError(f"Expected at least 2 groups; got {groups}")

    max_w2 = 0.0
    for g1, g2 in combinations(groups, 2):
        x0 = y_hat[s == g1]
        x1 = y_hat[s == g2]

        wx0 = np.full_like(x0, 1.0 / max(len(x0), 1), dtype=float)
        wx1 = np.full_like(x1, 1.0 / max(len(x1), 1), dtype=float)

        w2_val = float(w2_1d_weighted(x0, wx0, x1, wx1, eps=eps, squared=squared))
        if w2_val > max_w2:
            max_w2 = w2_val

    return max_w2


# sanity checks 
def _as_1d(x):
    return np.asarray(x).ravel()


def _is_binary(y):
    y = _as_1d(y)
    return np.unique(y).size == 2

def _safe_auc(y_true, y_score):
    """ROC AUC safe: returns np.nan if undefined (non-binary or single-class)."""
    y_true = _as_1d(y_true)
    y_score = _as_1d(y_score)
    if not _is_binary(y_true):
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan

# def dp_mean_gap(y_pred, S):
#     return float(abs(y_pred[S == 0].mean() - y_pred[S == 1].mean()))

def dp_mean_gap(y_pred, S):
    """Maximum absolute difference between the mean predictions of any two groups."""
    y_pred = np.asarray(y_pred).ravel()
    S = np.asarray(S).ravel()
    
    groups = np.unique(S)
    if len(groups) < 2:
        raise ValueError("DP mean gap requires at least 2 groups.")
        
    means = [y_pred[S == g].mean() for g in groups]
    return float(max(means) - min(means))


# Ordering / ranking metrics
def rank_metrics(y_ref, y_new, topk_frac=0.05):
    """
    Compare ordering between y_ref and y_new on the same individuals.

    Returns:
      - spearman in [-1, 1]
      - kendall_tau (tau-b) in [-1, 1]
      - mean_abs_rank_disp in [0, 1]
      - topk_jaccard in [0, 1]
    """
    y_ref = _as_1d(y_ref)
    y_new = _as_1d(y_new)
    n = len(y_ref)

    if n <= 1:
        return {
            "spearman": np.nan,
            "kendall_tau": np.nan,
            "mean_abs_rank_disp": np.nan,
            "topk_jaccard": np.nan,
        }

    sp = spearmanr(y_ref, y_new).correlation
    kt = kendalltau(y_ref, y_new, variant="b").correlation

    r_ref = rankdata(y_ref, method="average")
    r_new = rankdata(y_new, method="average")
    disp = float(np.mean(np.abs(r_ref - r_new)) / (n - 1))

    k = max(1, int(topk_frac * n))
    A = set(np.argsort(y_ref)[-k:])
    B = set(np.argsort(y_new)[-k:])
    jacc = float(len(A & B) / max(1, len(A | B)))

    return {
        "spearman": float(sp) if sp is not None else np.nan,
        "kendall_tau": float(kt) if kt is not None else np.nan,
        "mean_abs_rank_disp": disp,
        "topk_jaccard": jacc,
    }


def rank_metrics_by_group(y_ref, y_new, S, topk_frac=0.05):
    """
    Returns dict:
      {
        "overall": {...},
        "S=<g1>": {...},
        ...
      }
    """
    y_ref = _as_1d(y_ref)
    y_new = _as_1d(y_new)
    S = _as_1d(S)

    out = {"overall": rank_metrics(y_ref, y_new, topk_frac=topk_frac)}
    for g in np.unique(S):
        m = (S == g)
        out[f"S={g}"] = rank_metrics(y_ref[m], y_new[m], topk_frac=topk_frac)
    return out



# Leakage AUC 
def auc_report(Y, S, y_eta, y_corr):
    """
    Leakage AUCs (only):
      - AUC(S | Y)
      - AUC(S | eta)
      - AUC(S | corr)

    Note: AUC(Y | score) is intentionally NOT computed here, because Y can be continuous.
    """
    Y = _as_1d(Y)
    S = _as_1d(S)
    y_eta = _as_1d(y_eta)
    y_corr = _as_1d(y_corr)

    return {
        "auc_S_from_Y": _safe_auc(S, Y),
        "auc_S_from_eta": _safe_auc(S, y_eta),
        "auc_S_from_corr": _safe_auc(S, y_corr),
    }


# Latex export 
def _fmt(x, nd=3):
    if x is None:
        return "--"
    try:
        if np.isnan(x):
            return "--"
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


def ordering_to_latex_table(
    ordering_by_group,
    topk_frac=0.05,
    nd=3,
    caption_long="Ordering stability between $\\hat\\eta$ and corrected scores.",
    caption_short="Ordering stability between eta-hat and corrected scores.",
    label="tab:ordering",
):
    """
    Hyperref-safe caption:
      - caption_short (no math) used for PDF bookmarks / list of tables
      - caption_long can contain math
    """
    topk_pct = int(round(100 * topk_frac))
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(
        f"Subset & Spearman $\\rho$ & Kendall $\\tau_b$ & Mean $|\\Delta\\mathrm{{rank}}|$ & Top-{topk_pct}\\% Jaccard \\\\"
    )
    lines.append("\\midrule")
    for key in ordering_by_group.keys():
        m = ordering_by_group[key]
        lines.append(
            f"{key} & {_fmt(m.get('spearman'), nd)} & {_fmt(m.get('kendall_tau'), nd)} & "
            f"{_fmt(m.get('mean_abs_rank_disp'), nd)} & {_fmt(m.get('topk_jaccard'), nd)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption[{caption_short}]{{{caption_long}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def auc_to_latex_table(
    auc_rep,
    nd=3,
    caption_long="Leakage AUC: ability to infer $S$ from a scalar score (values near 0.5 indicate low leakage).",
    caption_short="Leakage AUC summary.",
    label="tab:auc",
):
    """
    Produces a paper-friendly table (no python keys).
    Uses hyperref-safe caption_short for bookmarks.
    """
    rows = [
        ("auc_S_from_Y", r"$\mathrm{AUC}(S\mid Y)$"),
        ("auc_S_from_eta", r"$\mathrm{AUC}(S\mid \hat{\eta})$"),
        ("auc_S_from_corr", r"$\mathrm{AUC}(S\mid \tilde{y})$"),
    ]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Quantity & AUC \\\\")
    lines.append("\\midrule")
    for key, latex_name in rows:
        v = auc_rep.get(key, np.nan)
        lines.append(f"{latex_name} & {_fmt(v, nd)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption[{caption_short}]{{{caption_long}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def metrics_to_latex(ordering_by_group, auc_rep, topk_frac=0.05, nd=3):
    return (
        ordering_to_latex_table(ordering_by_group, topk_frac=topk_frac, nd=nd)
        + "\n\n"
        + auc_to_latex_table(auc_rep, nd=nd)
        + "\n"
    )
