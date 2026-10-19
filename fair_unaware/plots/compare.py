import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def moving_average(x, window: int):
    x = np.asarray(x, dtype=float)
    if window is None or window <= 1:
        return x
    window = int(window)
    if len(x) < window:
        return x
    w = np.ones(window) / window
    return np.convolve(x, w, mode="valid")


def plot_fairreg_history(fair_reg, X, S, y, *, data_scaling=False, scaler=None,
                         moving_risk=100, moving_unf=50, title_prefix="", savepath=None):
    """
    Reproduces the 'risk history' + 'unfairness history' plots from their notebooks,
    using FairReg.history(...).
    """
    risk_hist, unf_hist = fair_reg.history(X, S, y, data_scaling=data_scaling, scaler=scaler)

    risk_ma = moving_average(risk_hist, moving_risk)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(risk_ma)
    axes[0].set_title(f"{title_prefix}FairReg: risk history (moving avg={moving_risk})")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("risk")

    groups = sorted(list(unf_hist[0].keys()))
    for g in groups:
        series = np.array([d[g] for d in unf_hist], dtype=float)
        series_ma = moving_average(series, moving_unf)
        axes[1].plot(series_ma, label=f"S={g}")
    axes[1].set_title(f"{title_prefix}FairReg: unfairness history (moving avg={moving_unf})")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("unfairness (max CDF gap)")
    axes[1].legend()

    fig.tight_layout()

    if savepath is not None:
        _ensure_dir(os.path.dirname(savepath))
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_pred_histograms_by_group(preds, S, *, bins, title="", savepath=None):
    """
    preds: dict label -> y_pred (1D)
    S: pandas Series or array-like (0/1)
    bins: same bins/grid you use for DP_unfairness
    """
    S = pd.Series(S).reset_index(drop=True)
    groups = sorted(S.unique().tolist())

    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, yhat) in zip(axes, preds.items()):
        yhat = np.asarray(yhat)
        for g in groups:
            ax.hist(yhat[S.values == g], bins=bins, density=True, alpha=0.5, label=f"S={g}")
        ax.set_title(name)
        ax.set_xlabel("prediction")
        ax.set_ylabel("density")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()

    if savepath is not None:
        _ensure_dir(os.path.dirname(savepath))
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_pred_kde_by_group(preds, S, *, title="", savepath=None):
    """
    KDE version (requires scipy). Falls back to hist if scipy isn't available.
    """
    if not _HAS_SCIPY:
        return None
    S = pd.Series(S).reset_index(drop=True)
    groups = sorted(S.unique().tolist())

    n = len(preds)
    cols = min(3, n)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharey=True)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).reshape(-1)

    for ax, (name, yhat) in zip(axes[:n], preds.items()):
        yhat = np.asarray(yhat)
        xs = np.linspace(np.min(yhat), np.max(yhat), 300)

        for g in groups:
            yg = yhat[S.values == g]
            if len(yg) < 5:
                continue
            kde = gaussian_kde(yg)
            ax.plot(xs, kde(xs), label=f"S={g}")

        ax.set_title(name)
        ax.set_xlabel("prediction")
        ax.set_ylabel("density")
        ax.legend()

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()

    if savepath is not None:
        _ensure_dir(os.path.dirname(savepath))
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig



def plot_true_vs_pred(y_true, preds, *, title="", savepath=None):
    """
    y_true: 1D
    preds: dict label -> y_pred (1D)
    """
    y_true = np.asarray(y_true)
    n = len(preds)
    cols = min(3, n)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).reshape(-1)

    # Plot each model
    for ax, (name, yhat) in zip(axes[:n], preds.items()):
        yhat = np.asarray(yhat)
        ax.scatter(y_true, yhat, s=8, alpha=0.6)
        lo = min(y_true.min(), yhat.min())
        hi = max(y_true.max(), yhat.max())
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")

    # Hide unused axes 
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()

    if savepath is not None:
        _ensure_dir(os.path.dirname(savepath))
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig




def plot_tradeoff_curve(points, *, title="", savepath=None):
    """
    points: dict label -> list of (unfairness_scalar, risk_scalar)
    plots risk vs unfairness
    """
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    for name, xy in points.items():
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            continue
        ax.plot(xy[:, 0], xy[:, 1], marker="o", label=name)
    ax.set_xlabel("unfairness (max DP CDF gap)")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if savepath is not None:
        _ensure_dir(os.path.dirname(savepath))
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig






