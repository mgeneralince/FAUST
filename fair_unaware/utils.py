from curses import raw
import numpy as np
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
import warnings

def load_dataset(name, data_dir='data'):
    """
    Returns (X, S, y) for the requested dataset.
    Assumes data_dir is relative to the project root or absolute.
    """

    if not os.path.exists(data_dir):

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_path, 'data')

    if name == 'adult':
        df = pd.read_csv(os.path.join(data_dir, 'adult.csv'))
        S = (df['gender'] == 'Male').astype(float)
        y = (df['income'] == '>50K').astype(float)
        X = pd.get_dummies(df.drop(columns=['income', 'gender']), drop_first=True)

    elif name == 'lawschool':
        df = pd.read_csv(os.path.join(data_dir, 'lawschool.csv'))
        S = (df['race'] == 'White').astype(float)
        y = df['gpa'] # Regression target
        X = pd.get_dummies(df.drop(columns=['gpa', 'race']), drop_first=True)

    elif name == 'communities':
        df = pd.read_csv(os.path.join(data_dir, 'communities.csv'))
        #processing for Communities 
        S_cont = df['racePctWhite']
        S = (S_cont > S_cont.median()).astype(float)
        y = df['ViolentCrimesPerPop']
        drop_cols = ['ViolentCrimesPerPop', 'racePctWhite', 'communityname', 'state']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=[np.number]).fillna(0)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return X.astype(float).values, S.astype(float).values, y.astype(float).values



def check_binary_groups(s):
    """
    Ensure that the sensitive attribute s has exactly two distinct values.

    Returns
    -------
    unique_groups : np.ndarray
        Sorted unique group labels (length 2).
    """
    s = np.asarray(s)
    unique = np.unique(s)
    if unique.size != 2:
        raise ValueError(
            f"Expected exactly 2 groups in S, got {unique.size}: {unique}."
        )
    return unique


def compute_group_priors(s):
    """
    Compute empirical group priors P(S = s) from the sample.

    Parameters
    ----------
    s : array-like of shape (n_samples,)

    Returns
    -------
    priors : dict
        Mapping from group label -> prior probability.
    """
    s = np.asarray(s)
    unique, counts = np.unique(s, return_counts=True)
    n = len(s)
    priors = {label: count / n for label, count in zip(unique, counts)}
    return priors


## for FairReg tuning

def _parse_grid(spec: str, auto_value: float, cast=float):
    """
    spec: 'auto,0.5x,1x,2x,10'
    Returns list of casted values, with 'x' interpreted as multiplier of auto_value.
    """
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok == "auto":
            out.append(cast(auto_value))
        elif tok.endswith("x"):
            out.append(cast(float(tok[:-1]) * auto_value))
        else:
            out.append(cast(float(tok)))
    # deduplicate keep order
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def _common_bins(y_eval, nbins: int):
    """
    Build a fixed bin grid for DP computation (shared across all L).
    We use evenly spaced bins over the observed y range.
    """
    y_min = float(np.min(y_eval))
    y_max = float(np.max(y_eval))
    if y_min == y_max:
        y_max = y_min + 1e-6
    return np.linspace(y_min, y_max, nbins)

def plot_risk_unfairness_curve_annotated(points, title, savepath, annotate_frontier=True):

    xs = np.array([p["unfair"] for p in points], dtype=float)
    ys = np.array([p["risk"] for p in points], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.scatter(xs, ys, alpha=0.25, s=30)
    idx = np.argsort(xs)
    frontier = []
    best_risk = np.inf
    for j in idx:
        if ys[j] < best_risk - 1e-12:
            frontier.append(j)
            best_risk = ys[j]

    #to highlight pareto frontier 
    fx, fy = xs[frontier], ys[frontier]
    plt.plot(fx, fy, marker="o", linewidth=1.5, color= "tab:orange")

    #annotationon the frontier points
    if annotate_frontier:
        for j in frontier:
            p = points[j]
            label = f"L={int(p['L'])}, β={float(p['beta']):.2g}"
            plt.annotate(
                label,
                (p["unfair"], p["risk"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", alpha=0.6),
            )

    plt.xlabel("Unfairness (max DP gap)")
    plt.ylabel("Risk (MSE)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def tune_fairreg(
    reg, clf, B, K, p, eps_vec, T,
    X_unlab_fit,
    X_val, S_val, y_val_eval,
    dp_bins,
    L_list, beta_list,
    FairRegClass,
    dp_unfairness_fn,      
    tune_objective="feasible_min_risk",
    penalty_lambda=100.0
):
    rows = []
    best = None
    best_score = None

    for L in L_list:
        L = int(max(1, L))
        for beta in beta_list:
            beta = float(beta)

            fr = FairRegClass(reg, clf, B=B, K=K, p=p, eps=eps_vec, T=T, keep_history=False)
            fr.fit(X_unlab_fit, beta=beta, L=L, alg={"base": "SGD3", "method": "ACSA2"})

            y_pred_val = fr.predict(X_val)

            risk = float(np.mean((y_val_eval - y_pred_val) ** 2))
            dp = dp_unfairness_fn(y_pred_val, S_val, bins=dp_bins)
            unfair = float(max(dp.values()))

            row = {"L": L, "beta": beta, "risk": risk, "unfair": unfair}
            rows.append(row)

            if tune_objective == "feasible_min_risk":
                feasible = (unfair <= eps_vec[0])
                score = risk if feasible else np.inf
            else:
                viol = max(0.0, unfair - eps_vec[0])
                score = risk + penalty_lambda * (viol ** 2)

            if best_score is None or score < best_score:
                best_score = score
                best = (L, beta)

    return best, rows

def relaxed_suffix(relax_epsilon):
    if relax_epsilon is None:
        return Path()  # suffix vide => path inchangé
    eps = float(relax_epsilon)
    if eps <= 0:
        raise ValueError("--relax_epsilon must be > 0")
    return Path("relaxed") / f"eps{eps:g}"


# generate latex tables in compare_ot_vs_fairreg

def _latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}")
             .replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&")
             .replace("#", r"\#")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("$", r"\$")
            )

def _first_two_floats(s: str) -> Optional[Tuple[float, float]]:
    # robust: grabs the first two numbers, regardless of "±" or spacing
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])

def _render_metrics_table(stats: Dict[str, Tuple[float, float]],
                          caption: str,
                          label: Optional[str] = None,
                          precision: int = 4) -> str:
    nice = {
        "mse_base": "MSE (base)",
        "mse_fairreg": "MSE (FairReg)",
        "mse_ot": "MSE (OT)",
        "mse_aware": "MSE (aware)",
        "dp_base_max": r"DP$_{\max}$ (base)",
        "dp_fairreg_max": r"DP$_{\max}$ (FairReg)",
        "dp_ot_max": r"DP$_{\max}$ (OT)",
        "dp_aware_max": r"DP$_{\max}$ (aware)",
        "ks_base": "KS (base)",
        "ks_fairreg": "KS (FairReg)",
        "ks_ot": "KS (OT)",
        "ks_aware": "KS (aware)",
        "dp_base_unlab_max": r"DP$_{\max}$ (base, unlab)",
        "dp_fairreg_unlab_max": r"DP$_{\max}$ (FairReg, unlab)",
        "dp_ot_unlab_max": r"DP$_{\max}$ (OT, unlab)",
        "dp_aware_unlab_max": r"DP$_{\max}$ (aware, unlab)",
        "ks_base_unlab": "KS (base, unlab)",
        "ks_fairreg_unlab": "KS (FairReg, unlab)",
        "ks_ot_unlab": "KS (OT, unlab)",
        "ks_aware_unlab": "KS (aware, unlab)",
    }

    preferred_order = [
        "mse_base","mse_fairreg","mse_ot","mse_aware",
        "dp_base_max","dp_fairreg_max","dp_ot_max","dp_aware_max",
        "ks_base","ks_fairreg","ks_ot","ks_aware",
        "dp_base_unlab_max","dp_fairreg_unlab_max","dp_ot_unlab_max","dp_aware_unlab_max",
        "ks_base_unlab","ks_fairreg_unlab","ks_ot_unlab","ks_aware_unlab",
    ]
    keys = [k for k in preferred_order if k in stats] + [k for k in stats if k not in preferred_order]

    rows = []
    for k in keys:
        mean, std = stats[k]
        label_k = nice.get(k, _latex_escape(k))
        rows.append(f"{label_k} & {mean:.{precision}f} & {std:.{precision}f} \\\\")
    rows_str = "\n".join(rows)

    lab = f"\\label{{{label}}}\n" if label else ""
    return rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
{lab}\begin{{tabular}}{{l S[table-format=1.4] @{{\,\(\pm\)\,}} S[table-format=1.4]}}
\toprule
Metric & \multicolumn{{2}}{{c}}{{Value}} \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
def _render_metrics_table_pivot(stats: Dict[str, Tuple[float, float]],
                                caption: str,
                                label: Optional[str] = None,
                                precision: int = 4, 
                                raw_df: Optional[pd.DataFrame] = None
                                ) -> str:
    """
    Pivot summary stats into a table with:
      rows    = methods (Base, FairReg, FairReg(auto), OT, Aware)
      columns = metrics (MSE, DP_max, KS, W2), each displayed as mean ± std.
    """

    def metric_name_from_key(k: str) -> Optional[str]:
        if k.startswith("mse_"):
            return r"$\mathrm{MSE}\,\downarrow$"
        if k.startswith("dp_") and k.endswith("_max"):
            return r"$\mathrm{DP}_{\max}\,\downarrow$"
        if k.startswith("ks_"):
            return r"$\mathrm{KS}\,\downarrow$"
        if k.startswith("w2_"):
            return r"$W_2\,\downarrow$"
        if k.startswith("acc_"):
            return r"$\mathrm{Acc}\,\uparrow$"
        return None


    def method_name_from_key(k: str) -> Optional[str]:
        kk = k
        for pref in ("mse_", "dp_", "ks_", "w2_"):
            if kk.startswith(pref):
                kk = kk[len(pref):]
                break
        kk = kk.replace("_unlab", "")
        kk = kk.replace("_max", "")

        # order matters
                # order matters
        if kk.startswith("fairreg_auto"):
            return "FairReg (auto)"
        if kk.startswith("fairreg"):
            return "FairReg"
        if kk.startswith("frappe"):
            return "FRAPPE"
        if kk.startswith("base_full"):
            return "Base (train+unlab)"
        if kk.startswith("base"):
            return "Base (train)"
        if kk.startswith("ot"):
            return "OT"
        if kk.startswith("aware_hat_hard"):
            return r"Aware ($\hat S$ hard)"
        if kk.startswith("aware_hat_soft"):
            return r"Aware ($\hat S$ soft)"
        if kk.startswith("aware"):
            return "Aware"

        return None

    metric_order = [
    r"$\mathrm{MSE}\,\downarrow$",
    r"$\mathrm{DP}_{\max}\,\downarrow$",
    r"$\mathrm{KS}\,\downarrow$",
    r"$W_2\,\downarrow$",
    r"$\mathrm{Acc}\,\uparrow$",
]

    method_order = [
    "Base (train)", "Base (train+unlab)", "FairReg", "FairReg (auto)", "FRAPPE", "OT", "Aware",
    r"Aware ($\hat S$ hard)",
    r"Aware ($\hat S$ soft)",
]



    # method -> metric -> (mean, std)
    pivot: Dict[str, Dict[str, Tuple[float, float]]] = {}
    present_metrics = set()
    present_methods = set()
    cell_col = {}  

    for k, (mu, sd) in stats.items():
        met = metric_name_from_key(k)
        meth = method_name_from_key(k)
        if met is None or meth is None:
            continue
        pivot.setdefault(meth, {})[met] = (float(mu), float(sd))
        present_metrics.add(met)
        present_methods.add(meth)
        cell_col[(meth, met)] = k  # to retrieve original key if needed for rw_df


    metrics = [m for m in metric_order if m in present_metrics]

    methods = [m for m in method_order if m in present_methods]

    if not metrics or not methods:
        return ""
    
    # decide best method for each metric using the MEAN (mu)
    # default: if header contains \downarrow => smaller is better, else if \uparrow => larger is better
    best_by_metric: Dict[str, str] = {}

    for met in metrics:

        vals = []
        for meth in methods:
            if met in pivot.get(meth, {}):
                mu, _sd = pivot[meth][met]
                vals.append((mu, meth))

        if not vals:
            continue

        if r"\uparrow" in met:
            best_by_metric[met] = max(vals, key=lambda t: t[0])[1]
        else:
            best_by_metric[met] = min(vals, key=lambda t: t[0])[1]

    colspec = "l " + " ".join(
        [f"S[table-format=1.{precision}] @{{\\,\\(\\pm\\)\\,}} S[table-format=1.{precision}]"
         for _ in metrics]
    )

    header1 = "Method " + "".join([f"& \\multicolumn{{2}}{{c}}{{{met}}} " for met in metrics]) + r"\\"
    header2 = " " + "".join([r"& \multicolumn{1}{c}{mean} & \multicolumn{1}{c}{std} " for _ in metrics]) + r"\\"

    rows = []
    for meth in methods:
        row = [meth]
        for met in metrics:
            if met in pivot.get(meth, {}):
                mu, sd = pivot[meth][met]
                mu_s = f"{mu:.{precision}f}"
                sd_s = f"{sd:.{precision}f}"


                is_best = (best_by_metric.get(met) == meth)
                is_tied = False

                # Wilcoxon signed-rank test
                if raw_df is not None:
                    best_meth = best_by_metric.get(met)
                    best_k = cell_col.get((best_meth, met)) if best_meth else None
                    current_k = cell_col.get((meth, met))
                    
                    if best_k and current_k and best_k in raw_df.columns and current_k in raw_df.columns:
                        mask = ~(raw_df[best_k].isna() | raw_df[current_k].isna())
                        arr_best = raw_df.loc[mask, best_k].to_numpy(dtype=float)
                        arr_curr = raw_df.loc[mask, current_k].to_numpy(dtype=float)
                        if len(arr_best) >= 3: # Wilcoxon needs a few samples
                            diff = arr_best - arr_curr
                            if np.allclose(diff, 0):
                                is_tied = True
                            else:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    try:
                                        res = wilcoxon(arr_best, arr_curr)
                                        is_tied = (res.pvalue >= 0.05)
                                    except Exception:
                                        is_tied = False

                if is_best:
                    row += [rf"{{\color{{blue}}\bfseries {mu_s}}}", rf"{{\color{{blue}}\bfseries {sd_s}}}"]
                elif is_tied:
                    row += [rf"{{\color{{blue}} {mu_s}}}", rf"{{\color{{blue}} {sd_s}}}"]
                else:
                    row += [mu_s, sd_s]
            else:
                row += ["", ""]
        rows.append(" & ".join(row) + r" \\")

    lab = f"\\label{{{label}}}\n" if label else ""
    return rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
{lab}\begin{{tabular}}{{{colspec}}}
\toprule
{header1}
{header2}
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

#to output table with relative metrics in latex 

_METHODS_FOR_RELATIVE = (
    "aware_hat_hard",
    "aware_hat_soft",
    "fairreg_auto",
    "fairreg",
    "frappe",
    "ot",
    "aware",
    "base",
)
_METHODS_RE = "|".join(map(re.escape, sorted(_METHODS_FOR_RELATIVE, key=len, reverse=True)))
_REL_COL_RE = re.compile(rf"^(?P<metric>[^_]+)_(?P<method>{_METHODS_RE})(?P<rest>.*)$")


def delta_to_base_df(
    df: 'pd.DataFrame',
    *,
    methods: list[str] | None = None,
    percent: bool = True,
) -> 'pd.DataFrame':
    """
    Compute *differences* to the base method, column-wise.

    For any column that looks like: <metric>_<method><suffix>,
    replace it by: (<metric>_<method><suffix> - <metric>_base<suffix>).

    - Base columns stay **absolute** (unchanged).
    - If percent=True, multiply deltas by 100 for dp/ks/w2 (so gaps in [0,1] become "percentage points").

    This is intentionally NOT a ratio.
    """
    df = df.copy()

    if methods is None:
        # keep in sync with _render_metrics_table_pivot
        methods = ["base", "fairreg", "fairreg_auto", "frappe", "ot", "aware", "aware_hat_hard","aware_hat_soft"]

    methods_sorted = sorted(methods, key=len, reverse=True)

    for col in list(df.columns):
        if col == "run":
            continue
        if "_" not in col:
            continue

        metric, rest = col.split("_", 1)

        # identify the method as a prefix of `rest`
        method = None
        suffix = ""
        for m in methods_sorted:
            if rest == m or rest.startswith(m + "_"):
                method = m
                suffix = rest[len(m):]  # '' or '_max', '_unlab_max', etc.
                break
        if method is None or method == "base":
            continue

        base_col = f"{metric}_base{suffix}"
        if base_col not in df.columns:
            continue

        delta = df[col].astype(float) - df[base_col].astype(float)
        if percent and metric in {"dp", "ks", "w2"}:
            delta = 100.0 * delta
        df[col] = delta

    return df


def write_summary_relative_tex(
    df: 'pd.DataFrame',
    *,
    out_path: str | Path,
    dataset_name: str = "",
    precision: int = 2,
    percent: bool = True,
) -> Path:
    """
    Write a *summary* LaTeX file with **relative-to-base** values (base stays absolute).

    - Base row: absolute mean ± std (same as usual)
    - Other methods: mean ± std of (method - base) computed *per run*
      (multiplied by 100 if percent=True)

    Produces two tables: TEST metrics + UNLAB fairness metrics.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_rel = relative_change_to_base_df(df, percent=percent)

    def _stats_from_cols(cols: list[str]) -> dict[str, tuple[float, float]]:
        out = {}
        for c in cols:
            if c == "run" or c not in df_rel.columns:
                continue
            out[c] = (float(df_rel[c].mean()), float(df_rel[c].std()))
        return out

    # TEST: mse + dp_max + ks + w2
    cols_test = [
        "run",
        "mse_fairreg", "mse_fairreg_auto", "mse_frappe", "mse_ot", "mse_aware",
        "mse_aware_hat_hard", "mse_aware_hat_soft",

        "dp_fairreg_max", "dp_fairreg_auto_max", "dp_frappe_max", "dp_ot_max", "dp_aware_max",
        "dp_aware_hat_hard_max", "dp_aware_hat_soft_max",

        "ks_fairreg", "ks_fairreg_auto", "ks_frappe", "ks_ot", "ks_aware",
        "ks_aware_hat_hard", "ks_aware_hat_soft",

        "w2_fairreg", "w2_fairreg_auto", "w2_frappe", "w2_ot", "w2_aware",
        "w2_aware_hat_hard", "w2_aware_hat_soft",
    ]

    # UNLAB: dp_max + ks + w2
    cols_unlab = [
        "run",
        "dp_fairreg_unlab_max", "dp_fairreg_auto_unlab_max", "dp_frappe_unlab_max",
        "dp_ot_unlab_max", "dp_aware_unlab_max",
        "dp_aware_hat_hard_unlab_max", "dp_aware_hat_soft_unlab_max",

        "ks_fairreg_unlab", "ks_fairreg_auto_unlab", "ks_frappe_unlab",
        "ks_ot_unlab", "ks_aware_unlab",
        "ks_aware_hat_hard_unlab", "ks_aware_hat_soft_unlab",

        "w2_fairreg_unlab", "w2_fairreg_auto_unlab", "w2_frappe_unlab",
        "w2_ot_unlab", "w2_aware_unlab",
        "w2_aware_hat_hard_unlab", "w2_aware_hat_soft_unlab",
    ]



    # Drop auto columns if not present
    cols_test = [c for c in cols_test if c in df_rel.columns or c == "run"]
    cols_unlab = [c for c in cols_unlab if c in df_rel.columns or c == "run"]

    test_stats = _stats_from_cols(cols_test)
    unlab_stats = _stats_from_cols(cols_unlab)

    unit = "percentage points for DP/KS/W2 (MSE is absolute delta)" if percent else "absolute units"
    parts = []
    parts.append("% Auto-generated relative summary (method - base) from runs.csv")
    parts.append("% Requires in LaTeX preamble: \\usepackage{booktabs} \\usepackage{siunitx} \\usepackage{xcolor} \n")
    parts.append(f"% NOTE: Non-base rows are deltas in {unit}. Base row is absolute.\n")

    if test_stats:
        parts.append(_render_metrics_table_pivot(
            test_stats,
            caption=rf"Relative-to-base test metrics for {dataset_name} (Base absolute; others: method$-$Base, mean $\pm$ std over runs).",
            label="tab:test_metrics_relative",
            precision=precision,
            raw_df=df_rel,
            # include_metrics=("mse", "dp_max", "ks", "w2"),
        ))
    if unlab_stats:
        parts.append(_render_metrics_table_pivot(
            unlab_stats,
            caption=rf"Relative-to-base unlabeled fairness metrics for {dataset_name} (Base absolute; others: method$-$Base, mean $\pm$ std over runs).",
            label="tab:unlab_metrics_relative",
            precision=precision,
            raw_df=df_rel,
            # include_metrics=("dp_max", "ks", "w2"),
        ))

    out_path.write_text("\n\n% ===== next table =====\n\n".join(parts), encoding="utf-8")
    return out_path



def relative_change_to_base_df(df, *, eps=1e-12, percent=True):
    out = df.copy()

    baseline_for = {
        "fairreg": "base",
        "fairreg_auto": "base",
        "frappe": "base_full" if "mse_base_full" in df.columns else "base",
        "ot": "base_full" if "mse_base_full" in df.columns else "base",
        "aware": "base_full" if "mse_base_full" in df.columns else "base",
        "aware_hat_hard": "base_full" if "mse_base_full" in df.columns else "base",
        "aware_hat_soft": "base_full" if "mse_base_full" in df.columns else "base",
        "base_full": "base_full",
        "base": "base",
    }

    for col in out.columns:
        if col == "run":
            continue
        m = _REL_COL_RE.match(col)
        if m is None:
            continue

        metric, method, rest = m.group("metric"), m.group("method"), m.group("rest")
        base_method = baseline_for.get(method, "base")
        base_col = f"{metric}_{base_method}{rest}"
        if base_col not in df.columns:
            base_col = f"{metric}_base{rest}"
            if base_col not in df.columns:
                continue

        base = pd.to_numeric(df[base_col], errors="coerce").astype(float)

        if method in {"base", "base_full"}:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            continue

        val = pd.to_numeric(df[col], errors="coerce").astype(float)
        denom = np.maximum(np.abs(base), eps)
        rel = val  / denom   
        if percent:
            rel = 100.0 * rel
        out[col] = rel

    return out



def _render_fairreg_per_run_table(runs: List[dict],
                                 caption: str,
                                 label: Optional[str] = None,
                                 precision: int = 5) -> str:
    lab = f"\\label{{{label}}}\n" if label else ""
    rows = []
    for r in runs:
        rows.append(
            f"{r['run']:02d} & {r['L']} & {r['beta']:.2f} & {r['val_risk']:.{precision}f} & {r['val_unfair']:.{precision}f} \\\\"
        )
    rows_str = "\n".join(rows)

    return rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
{lab}\begin{{tabular}}{{r r S[table-format=3.2] S[table-format=1.{precision}] S[table-format=1.{precision}]}}
\toprule
Run & $L$ & {{$\beta$}} & {{val\_risk}} & {{val\_unfair}} \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

def _render_fairreg_aggregate_table(agg: dict,
                                   caption: str,
                                   label: Optional[str] = None) -> str:
    lab = f"\\label{{{label}}}\n" if label else ""
    return rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
{lab}\begin{{tabular}}{{l r}}
\toprule
Quantity & Value \\
\midrule
Best $L$ (mean $\pm$ std)      & ${agg['best_L_mean']:.3f} \pm {agg['best_L_std']:.3f}$ \\
Best $\beta$ (mean $\pm$ std)  & ${agg['best_beta_mean']:.3f} \pm {agg['best_beta_std']:.3f}$ \\
Mode($L$)                      & ${agg['mode_L']}$ \\
Mode($\beta$)                  & ${agg['mode_beta']}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""

def txt_summary_to_tex(txt_path: str | Path,
                       tex_path: str | Path | None = None,
                       precision_metrics: int = 4,
                       dataset_name: str = "",
                       include_fairreg_per_run: bool = True, 
                       raw_df: Optional[pd.DataFrame] = None
                       ) -> Path:
    """
    Convert your .txt summary into ONE .tex file containing LaTeX tables.
    Returns the output .tex Path.
    """

    print("dataset_name =", dataset_name)
    txt_path = Path(txt_path)
    if tex_path is None:
        tex_path = txt_path.with_suffix(".tex")
    tex_path = Path(tex_path)

    lines = txt_path.read_text(encoding="utf-8").splitlines()

    # Sections
    test_stats: Dict[str, Tuple[float, float]] = {}
    unlab_stats: Dict[str, Tuple[float, float]] = {}
    fairreg_runs: List[dict] = []
    fairreg_agg: dict = {}

    section = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("===") and "TEST metrics" in line:
            section = "test"
            continue
        if line.startswith("===") and "UNLAB fairness metrics" in line:
            section = "unlab"
            continue
        if line.startswith("===") and "FairReg selected hyperparameters (per run)" in line:
            section = "fairreg_per_run"
            continue
        if line.startswith("===") and "FairReg selected hyperparameters (aggregate)" in line:
            section = "fairreg_agg"
            continue
        if line.startswith("==="):
            section = None
            continue

        if section in {"test", "unlab"}:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            parsed = _first_two_floats(rest)
            if parsed is None:
                continue
            if section == "test":
                test_stats[key] = parsed
            else:
                unlab_stats[key] = parsed

        elif section == "fairreg_per_run":
            # example: run 01: L=54, beta=317.45 | val_risk=0.11911 val_unfair=0.137539
            m = re.search(
                r"run\s+(\d+)\s*:\s*L\s*=\s*(\d+)\s*,\s*beta\s*=\s*([0-9.]+)\s*\|\s*val_risk\s*=\s*([0-9.]+)\s*val_unfair\s*=\s*([0-9.]+)",
                line,
                flags=re.IGNORECASE,
            )
            if m:
                fairreg_runs.append({
                    "run": int(m.group(1)),
                    "L": int(m.group(2)),
                    "beta": float(m.group(3)),
                    "val_risk": float(m.group(4)),
                    "val_unfair": float(m.group(5)),
                })

        elif section == "fairreg_agg":
            if ":" not in line:
                continue
            k, rest = line.split(":", 1)
            k = k.strip()

            if k.lower().startswith("best_l"):
                mean_std = _first_two_floats(rest)
                if mean_std:
                    fairreg_agg["best_L_mean"], fairreg_agg["best_L_std"] = mean_std
            elif k.lower().startswith("best_beta"):
                mean_std = _first_two_floats(rest)
                if mean_std:
                    fairreg_agg["best_beta_mean"], fairreg_agg["best_beta_std"] = mean_std
            elif k.lower().startswith("mode(l)"):
                nums = re.findall(r"\d+", rest)
                if nums:
                    fairreg_agg["mode_L"] = int(nums[0])
            elif k.lower().startswith("mode(beta)"):
                nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", rest)
                if nums:
                    # keep as string to preserve exact formatting if needed
                    fairreg_agg["mode_beta"] = nums[0]

    # Build LaTeX output (single file, multiple tables)
    parts = []
    parts.append("% Auto-generated from: " + txt_path.name)
    parts.append("% Requires in LaTeX preamble: \\usepackage{booktabs} \\usepackage{siunitx} \\usepackage{xcolor} \n")

    if test_stats:
        parts.append(_render_metrics_table_pivot(
            test_stats,
            caption=rf"Test metrics for {dataset_name} (mean $\pm$ std over runs).",
            label="tab:test_metrics",
            precision=precision_metrics
        ))
    if unlab_stats:
        parts.append(_render_metrics_table_pivot(
            unlab_stats,
            caption=rf"Unlabeled fairness metrics for {dataset_name} (mean $\pm$ std over runs).",
            label="tab:unlab_metrics",
            precision=precision_metrics
        ))


    if include_fairreg_per_run and fairreg_runs:
        parts.append(_render_fairreg_per_run_table(
            fairreg_runs,
            caption=f"FairReg selected hyperparameters for {dataset_name}(per run).",
            label="tab:fairreg_hparams_runs",
            precision=5
        ))
    if fairreg_agg:
        # fill defaults if missing
        fairreg_agg.setdefault("best_L_mean", 0.0)
        fairreg_agg.setdefault("best_L_std", 0.0)
        fairreg_agg.setdefault("best_beta_mean", 0.0)
        fairreg_agg.setdefault("best_beta_std", 0.0)
        fairreg_agg.setdefault("mode_L", 0)
        fairreg_agg.setdefault("mode_beta", "0")
        parts.append(_render_fairreg_aggregate_table(
            fairreg_agg,
            caption=f"FairReg selected hyperparameters (aggregate) {dataset_name}.",
            label="tab:fairreg_hparams_agg",
        ))

    tex_path.write_text("\n\n% ===== next table =====\n\n".join(parts), encoding="utf-8")
    return tex_path


## csv to latex table

def df_to_latex_table(df: pd.DataFrame, *,
                      caption: str,
                      label: str,
                      precision: int = 3,
                      env: str = "table",
                      placement: str = "t",
                      max_width: str = r"\textwidth") -> str:
    """
    Convert a DataFrame to a LaTeX table using pandas + booktabs, wrapped in adjustbox.
    Requires LaTeX packages: booktabs, adjustbox.
    """
    tab = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.{precision}f}",
    )
    return rf"""\begin{{{env}}}[{placement}]
\centering
\caption{{{caption}}}\label{{{label}}}
\begin{{adjustbox}}{{max width={max_width}}}
{tab}
\end{{adjustbox}}
\end{{{env}}}
"""

def write_runs_tables_tex(df: pd.DataFrame,
                          *,
                          out_path: str | Path,
                          cols_mse: list[str] = None,
                          cols_unlab: list[str] = None,
                          cols_rest: list[str] = None,
                          precision: int = 3,
                          rotate_rest: bool = False,
                          dataset_name: str = "",
                          rename: dict | None = None, 
                          relative_to_base: bool = False, 
                          percent: bool = False,
                          ) -> Path:
    """
    Writes ONE .tex file with 3 tables:
      1) MSE per run
      2) UNLAB fairness per run
      3) REST per run (e.g., DP_test + KS_test)

    If rotate_rest=True, the REST table is sidewaystable (requires rotating).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cols_mse is None:
        cols_mse = ["run", "mse_base", "mse_fairreg", "mse_ot", "mse_aware",
                "mse_aware_hat_hard", "mse_aware_hat_soft"]

    if cols_unlab is None:
        cols_unlab = ["run",
                      "dp_base_unlab_max", "dp_fairreg_unlab_max", "dp_ot_unlab_max", "dp_aware_unlab_max",
                      "ks_base_unlab", "ks_fairreg_unlab", "ks_ot_unlab", "ks_aware_unlab"]
    if cols_rest is None:
        cols_rest = ["run",
                     "dp_base_max", "dp_fairreg_max", "dp_ot_max", "dp_aware_max",
                     "ks_base", "ks_fairreg", "ks_ot", "ks_aware"]
        
    df_use = df.copy()
        
    if relative_to_base:
        df_use = relative_change_to_base_df(df_use, percent=percent)
        df = df_use


    df_mse = df[cols_mse].copy()
    df_rest = df[cols_rest].copy()

    df_unlab = None
    if cols_unlab and len(cols_unlab) > 0:
        df_unlab = df[cols_unlab].copy()

    if rename:
        df_mse.rename(columns=rename, inplace=True)
        df_rest.rename(columns=rename, inplace=True)
        if df_unlab is not None:
            df_unlab.rename(columns=rename, inplace=True)


    parts = []
    parts.append("% Auto-generated LaTeX tables from runs.csv\n"
                 "% Requires: \\usepackage{booktabs} \\usepackage{adjustbox}\n"
                 "% Optional (if sidewaystable used): \\usepackage{rotating}\n")

    parts.append(df_to_latex_table(
        df_mse,
        caption=f"Per-run MSE on the test set {dataset_name}.",
        label="tab:runs_mse",
        precision=precision,
        env="table",
        placement="t",
    ))

    if df_unlab is not None:
        parts.append(df_to_latex_table(
            df_unlab,
            caption=f"Per-run fairness metrics on the unlabeled set {dataset_name}.",
            label="tab:runs_unlab",
            precision=precision,
            env="table",
            placement="t",
        ))

    rest_env = "sidewaystable" if rotate_rest else "table"
    parts.append(df_to_latex_table(
        df_rest,
        caption=f"Per-run DP(max) and KS on the test set {dataset_name}.",
        label="tab:runs_rest",
        precision=precision,
        env=rest_env,
        placement="p" if rotate_rest else "t",
    ))

    out_path.write_text("\n\n% ===== next table =====\n\n".join(parts), encoding="utf-8")
    return out_path



def generate_custom_summary_tex(df, out_abs, out_rel, dataset_name):
    """
    Generates the absolute and relative LaTeX summary tables.
    """
    # 1. models 
    all_models = [
        ("base", "ERM Unfair", "base"),
        ("fairreg", "FairReg", "unaware"),
    ]
    if "mse_fairreg_auto" in df.columns:
        all_models.append(("fairreg_auto", "FairReg (auto)", "unaware"))

    if "mse_frappe" in df.columns:
        all_models.append(("frappe", "FRAPPÉ", "unaware"))

    all_models.append(("ot", "FAUST-U", "unaware"))

    if "mse_aware_hat_hard" in df.columns:
        all_models.append(("aware_hat_hard", r"Plug-In hard", "unaware"))
    if "mse_aware_hat_soft" in df.columns:
        all_models.append(("aware_hat_soft", r"Plug-In soft", "unaware"))

    all_models.append(("aware", "FAUST-A", "aware"))

    # 2. metrics 
    metrics_config = [
        ("MSE $\\downarrow$", "mse_{}"),
        ("$W_2$ $\\downarrow$", "w2_{}"),
        ("TV $\\downarrow$", "tv_{}"),
        ("KS $\\downarrow$", "ks_{}"),
        ("KS$^{\\mathcal{T}}$ $\\downarrow$", "dp_{}_max")
    ]

    def build_table(is_relative):
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{l" + "c" * len(metrics_config) + "}",
            r"\toprule",
            "Method & " + " & ".join([m[0] for m in metrics_config]) + r" \\"
        ]

        # Calculate absolute stats first
        stats = {}
        for m_id, _, _ in all_models:
            stats[m_id] = {}
            for met_name, met_pat in metrics_config:
                col = met_pat.replace("_{}", f"_{m_id}")
                if col not in df.columns: continue
                
                vals = df[col].values
                stats[m_id][met_name] = {
                    "mean": np.mean(vals), 
                    "std": np.std(vals), 
                    "raw": vals
                }

        # If relative, divide means and stds by the safely stored BASE mean
        if is_relative:
            base_means_safe = {}
            for met_name, _ in metrics_config:
                if met_name in stats["base"]:
                    base_means_safe[met_name] = stats["base"][met_name]["mean"] + 1e-9

            for m_id, _, _ in all_models:
                for met_name, _ in metrics_config:
                    if met_name not in stats[m_id]: continue
                    
                    # Use the safely stored base mean
                    b_mean = base_means_safe[met_name]
                    stats[m_id][met_name]["mean"] /= b_mean
                    stats[m_id][met_name]["std"] /= b_mean

        # Find best unaware models based on the processed means
        best_unaware = {}
        for met_name, _ in metrics_config:
            unaware_means = {
                m_id: stats[m_id][met_name]["mean"] 
                for m_id, _, cat in all_models if cat == "unaware" and met_name in stats[m_id]
            }
            if unaware_means:
                best_unaware[met_name] = min(unaware_means, key=unaware_means.get)

        current_cat = None
        for m_id, name_latex, cat in all_models:
            if current_cat is not None and current_cat != cat:
                lines.append(r"\midrule")
            current_cat = cat

            row_strs = [name_latex]
            for met_name, _ in metrics_config:
                if met_name not in stats[m_id]:
                    row_strs.append("-")
                    continue
                    
                mean, std = stats[m_id][met_name]["mean"], stats[m_id][met_name]["std"]
                val_str = f"{mean:.2f} $\\pm$ {std:.2f}"

                # Bolding and Blue for unaware
                if cat == "unaware" and met_name in best_unaware:
                    best_id = best_unaware[met_name]
                    if m_id == best_id:
                        val_str = f"\\textbf{{{val_str}}}"
                    else:
                        try:
                            # Wilcoxon test must always be done on RAW absolute data
                            _, pval = wilcoxon(stats[m_id][met_name]["raw"], stats[best_id][met_name]["raw"])
                            if pval > 0.05:
                                val_str = f"\\textcolor{{blue}}{{{val_str}}}"
                        except:
                            pass
                            
                row_strs.append(val_str)
            
            lines.append(" & ".join(row_strs) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{('Relative' if is_relative else 'Absolute')} metrics for {dataset_name}}}",
            r"\end{table}"
        ])
        return "\n".join(lines)

    with open(out_abs, "w") as f: f.write(build_table(is_relative=False))
    with open(out_rel, "w") as f: f.write(build_table(is_relative=True))
