import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from fair_unaware.utils import compute_group_priors

from sklearn.base import clone  



class AwareWassersteinBarycenterDPRegressor:
    """
    Aware DP-fair regression baseline (Chzhen et al. 2020 barycenter post-processing).

    Fit per-group regressors eta_s(x) ~ E[Y|X=x,S=s].
    Define u = F_s(eta_s(x)) (empirical CDF within group s),
    then output g(x,s) = sum_{s'} p_{s'} Q_{s'}(u) (mixture of quantiles).

    """

    def __init__(
        self,
        base_regressor=None,
        random_state: int = 0,
        *,
        estimate_S: bool = False,
        s_estimator=None,
    ):

        self.base_regressor = base_regressor
        self.random_state = int(random_state)

        #for the estimated aware models
        self.estimate_S = estimate_S
        self.s_estimator = s_estimator

    def _fit_s_estimator(self, X, S):
        if self.s_estimator is None:
            from sklearn.linear_model import LogisticRegression
            est = LogisticRegression(max_iter=2000)
        else:
            est = clone(self.s_estimator)

        est.fit(X, S)
        self.s_estimator_ = est

    def _get_S_used(self, X, S=None):
        """
        Return the sensitive attribute to *use* internally.
        - If estimate_S=False: requires S.
        - If estimate_S=True: uses S if provided, else uses Ŝ(X).
        """
        if not self.estimate_S:
            if S is None:
                raise ValueError("S must be provided when estimate_S=False.")
            return np.asarray(S).ravel()

        # estimate_S=True
        if S is not None:
            return np.asarray(S).ravel()
        return self.s_estimator_.predict(X).ravel()


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
        u = np.asarray(u, dtype=float)
        n = sorted_vals.size
        if n <= 1:
            return np.full_like(u, float(sorted_vals[0]) if n == 1 else 0.0, dtype=float)

        grid = (np.arange(1, n + 1, dtype=float) / n)
        return np.interp(u, grid, sorted_vals, left=sorted_vals[0], right=sorted_vals[-1])

    def _default_regressor(self):
        return LinearRegression(fit_intercept=True)


    def fit(self, X_train, S_train, y_train, *, X_unlab=None, S_unlab=None, group_priors=None):
        X_train = self._ensure_2d(X_train)
        S_train = np.asarray(S_train).ravel()
        y_train = np.asarray(y_train).ravel()

        # Decide which S we use (true S or Ŝ)
        if self.estimate_S:
            self._fit_s_estimator(X_train, S_train)
            S_used_train = self._get_S_used(X_train, S=None)   # Ŝ(X_train)
        else:
            S_used_train = S_train

        if X_unlab is None:
            X_unlab = X_train
        X_unlab = self._ensure_2d(X_unlab)

        if self.estimate_S:
            S_used_unlab = self._get_S_used(X_unlab, S=None)   # Ŝ(X_unlab)
        else:
            if S_unlab is None:
                S_unlab = S_train
            S_used_unlab = np.asarray(S_unlab).ravel()

        self.groups_ = np.unique(S_used_train)

        if group_priors is None:
            self.group_priors_ = compute_group_priors(S_used_train)
        else:
            self.group_priors_ = dict(group_priors)

        if np.unique(S_used_train).size < 2:
            raise ValueError("S_used_train has <2 groups (Ŝ collapsed). Cannot fit barycenter.")


        self.models_ = {}
        for s in self.groups_:
            m = (S_used_train == s)
            if not np.any(m):
                raise ValueError(f"No labeled samples for group {s}.")
            model = self._default_regressor() if self.base_regressor is None else clone(self.base_regressor)
            model.fit(X_train[m], y_train[m])
            self.models_[s] = model


        self.sorted_preds_ = {}
        for s in self.groups_:
            mu = (S_used_unlab == s)
            if np.any(mu):
                preds = self.models_[s].predict(X_unlab[mu])
            else:
                preds = self.models_[s].predict(X_train[S_used_train == s])
            self.sorted_preds_[s] = np.sort(np.asarray(preds, dtype=float).ravel())

        return self


    def predict_base(self, X, S=None):
        """Unconstrained aware predictor eta_s(x)."""
        X = self._ensure_2d(X)
        # S = np.asarray(S).ravel()
        S = self._get_S_used(X, S)
        out = np.empty(X.shape[0], dtype=float)
        for s in self.groups_:
            m = (S == s)
            if np.any(m):
                out[m] = self.models_[s].predict(X[m])
        return out

    def predict(self, X, S=None):
        """DP-fair aware predictor g(x,s) via barycenter quantiles."""
        X = self._ensure_2d(X)
        # S = np.asarray(S).ravel()
        S = self._get_S_used(X, S)
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
    
    def predict_relaxed(self, X, S=None, lam=0.0, penalty="l2"):
        """
        Produce relaxed Aware DP predictions.
        - penalty="l2": Wasserstein linear interpolation.
        - penalty="tv": Hard-thresholding (L0 penalty from Lemma 1).
        """
        y_base = self.predict_base(X, S)
        y_strict = self.predict(X, S)
        
        if penalty == "l2":
            return (1.0 - lam) * y_base + lam * y_strict
            
        elif penalty == "tv":
            if np.isinf(lam):
                tau = np.inf   
            elif lam == 0.0:
                tau = 0.0      
            else:
                tau = lam      
                
            diff = y_strict - y_base
            
            return np.where(np.abs(diff) <= tau, y_strict, y_base)
            
        else:
            raise ValueError("penalty must be 'l2' or 'tv'")
    
    def predict_marginalized(self, X, s_estimator=None):
        """
        Calculate tilde_eta(x) = sum_s P(S=s|X=x) * eta_s(x)
        to have the same base astimator than unaware
        """
        X = self._ensure_2d(X)
        
        # 1. get pi(x)
        if s_estimator is None:
            if not hasattr(self, 's_estimator_') or self.s_estimator_ is None:
                raise ValueError("Veuillez fournir un s_estimator (ex: LogisticRegression déjà fittée).")
            s_estimator = self.s_estimator_
            
        probas = s_estimator.predict_proba(X)
        classes = s_estimator.classes_
        
        # 2. ponderated sum 
        tilde_eta = np.zeros(X.shape[0], dtype=float)
        for idx, s in enumerate(classes):
            if s in self.models_:  
                eta_s = self.models_[s].predict(X)
                tilde_eta += probas[:, idx] * eta_s
                
        return tilde_eta
