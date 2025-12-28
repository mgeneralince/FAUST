import numpy as np


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
    import matplotlib.pyplot as plt

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


