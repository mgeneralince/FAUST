import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from fair_unaware.utils import compute_group_priors


class AwareWassersteinBarycenterDPRegressor:
    """
    Aware DP-fair regression baseline (Chzhen et al. 2020 barycenter post-processing).

    Fit per-group regressors eta_s(x) ~ E[Y|X=x,S=s].
    Define u = F_s(eta_s(x)) (empirical CDF within group s),
    then output g(x,s) = sum_{s'} p_{s'} Q_{s'}(u) (mixture of quantiles).

    """

    def __init__(self, base_regressor=None, random_state=0):
        self.base_regressor = base_regressor
        self.random_state = int(random_state)

    @staticmethod
    def _ensure_2d(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _cdf_from_sorted(sorted_vals, v):
        """Empirical CDF F(v)=P(V<=v), using right-continuous convention."""
        v = np.asarray(v)
        n = sorted_vals.size
        idx = np.searchsorted(sorted_vals, v, side="right")
        return idx / max(n, 1)

    @staticmethod
    def _quantile_from_sorted(sorted_vals, u):
        """Empirical quantile via linear interpolation on a [0,1] grid."""
        u = np.asarray(u)
        n = sorted_vals.size
        if n <= 1:
            return np.full_like(u, float(sorted_vals[0]) if n == 1 else 0.0, dtype=float)
        grid = np.linspace(0.0, 1.0, n)
        return np.interp(u, grid, sorted_vals)

    def _default_regressor(self):
        return LinearRegression(fit_intercept=True)

    def fit(self, X_train, S_train, y_train, *, X_unlab=None, S_unlab=None, group_priors=None):
        X_train = self._ensure_2d(X_train)
        S_train = np.asarray(S_train).ravel()
        y_train = np.asarray(y_train).ravel()

        if X_unlab is None:
            X_unlab = X_train
            S_unlab = S_train
        X_unlab = self._ensure_2d(X_unlab)
        S_unlab = np.asarray(S_unlab).ravel()

        self.groups_ = np.unique(S_train)
        self.group_priors_ = compute_group_priors(S_train) if group_priors is None else dict(group_priors)

        # fit eta_s per group
        self.models_ = {}
        for s in self.groups_:
            m = (S_train == s)
            if not np.any(m):
                raise ValueError(f"No labeled samples for group {s}.")
            model = self._default_regressor() if self.base_regressor is None else clone(self.base_regressor)
            model.fit(X_train[m], y_train[m])
            self.models_[s] = model

        #build sorted predictions per group on unlabeled pool (for CDF/quantiles)
        self.sorted_preds_ = {}
        for s in self.groups_:
            mu = (S_unlab == s)
            if np.any(mu):
                preds = self.models_[s].predict(X_unlab[mu])
            else:
                preds = self.models_[s].predict(X_train[S_train == s])
            self.sorted_preds_[s] = np.sort(np.asarray(preds, dtype=float).ravel())

        return self

    def predict_base(self, X, S):
        """Unconstrained aware predictor eta_s(x)."""
        X = self._ensure_2d(X)
        S = np.asarray(S).ravel()
        out = np.empty(X.shape[0], dtype=float)
        for s in self.groups_:
            m = (S == s)
            if np.any(m):
                out[m] = self.models_[s].predict(X[m])
        return out

    def predict(self, X, S):
        """DP-fair aware predictor g(x,s) via barycenter quantiles."""
        X = self._ensure_2d(X)
        S = np.asarray(S).ravel()
        base = self.predict_base(X, S)

        groups = list(self.groups_)
        w = np.array([self.group_priors_.get(s, 0.0) for s in groups], dtype=float)
        w = w / w.sum()

        out = np.empty_like(base, dtype=float)
        for s in self.groups_:
            m = (S == s)
            if not np.any(m):
                continue
            u = self._cdf_from_sorted(self.sorted_preds_[s], base[m])

            bary_q = np.zeros_like(u, dtype=float)
            for wi, s2 in zip(w, groups):
                bary_q += wi * self._quantile_from_sorted(self.sorted_preds_[s2], u)

            out[m] = bary_q

        return out
