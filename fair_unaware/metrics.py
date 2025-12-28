# fair_unaware/metrics.py
import numpy as np
from scipy.stats import rankdata, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
import numpy as np


def dp_cdf_gaps(y_pred, S, bins):
    """
    DP metric for regression: sup_t | P(Yhat<=t|S=0) - P(Yhat<=t|S=1) |
    returned as a dict over the provided bins (like the external FairReg code did).
    """
    y_pred = np.asarray(y_pred).ravel()
    S = np.asarray(S).ravel()
    bins = np.asarray(bins).ravel()

    s_vals = np.unique(S)
    if s_vals.size != 2:
        raise ValueError("dp_cdf_gaps currently expects exactly 2 groups.")

    s0, s1 = s_vals[0], s_vals[1]
    y0 = y_pred[S == s0]
    y1 = y_pred[S == s1]
    if y0.size == 0 or y1.size == 0:
        raise ValueError("Both groups must have at least one sample to compute DP gaps.")

    gaps = {}
    for t in bins:
        c0 = float(np.mean(y0 <= t))
        c1 = float(np.mean(y1 <= t))
        gaps[float(t)] = abs(c0 - c1)
    return gaps


def dp_cdf_max_gap(y_pred, S, bins):
    gaps = dp_cdf_gaps(y_pred, S, bins=bins)
    return float(max(gaps.values()))


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




# sanity checks 
def _as_1d(x):
    return np.asarray(x).ravel()


def _is_binary(y):
    y = _as_1d(y)
    return np.unique(y).size == 2


def dp_mean_gap(y_pred, S):
    return float(abs(y_pred[S == 0].mean() - y_pred[S == 1].mean()))


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
    # order 
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
