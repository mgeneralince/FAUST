import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from scipy.special import expit

class OracleSyntheticClassifier(BaseEstimator, ClassifierMixin):
    """True P(S=1 | X) Bayes predictor for the 2D synthetic dataset."""
    def __init__(self, gamma=0.5):
        self.gamma = gamma
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        # S depends ONLY on X1, so we ignore X2 for the classifier
        X1 = X[:, 0]
        
        if self.gamma >= 0.999:
            prob_1 = (X1 > 0).astype(float)
        elif self.gamma <= 0.001:
            prob_1 = np.full_like(X1, 0.5)
        else:
            logits = (2.0 * self.gamma * X1) / (1.0 - self.gamma**2)
            prob_1 = expit(logits)
            
        return np.column_stack([1.0 - prob_1, prob_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    

class OracleSyntheticUnawareRegressor(BaseEstimator, RegressorMixin):
    """True E[Y | X] Bayes predictor for the 2D synthetic dataset."""
    def __init__(self, gamma=0.5, beta=None):
        self.gamma = gamma
        self.beta = beta

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X1 = X[:, 0]
        X2 = X[:, 1]
        return 2.0 * X1 - 1.0 * X2

class OracleSyntheticAwareRegressor(BaseEstimator, RegressorMixin):
    """True E[Y | X, S] Bayes predictor for the 2D synthetic dataset."""
    def __init__(self, beta=None):
        self.beta = beta

    def fit(self, X, y):
        return self

    def predict(self, X):
        X1 = X[:, 0]
        X2 = X[:, 1]
        return 2.0 * X1 - 1.0 * X2

def fit_eta_model(X, y, model=None, random_state=0):
    """
    Fit a regression model eta(x) ≈ E[Y | X=x].

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    model : sklearn-like regressor or None
        If None, a RandomForestRegressor is created.
    random_state : int

    Returns
    -------
    model : fitted regressor
    eta_hat : np.ndarray of shape (n_samples,)
    """
    if model is None:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
        )
    model.fit(X, y)
    eta_hat = model.predict(X)
    return model, eta_hat


def fit_s_classifier(X, s, clf=None, random_state=0):
    """
    Fit a classifier to estimate posteriors P(S | X).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    s : array-like of shape (n_samples,)
    clf : sklearn-like classifier or None
        If None, a RandomForestClassifier is created.
    random_state : int

    Returns
    -------
    clf : fitted classifier
    proba : np.ndarray of shape (n_samples, n_classes)
        Estimated class probabilities for each sample.
    """
    if clf is None:
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        )
    clf.fit(X, s)
    proba = clf.predict_proba(X)
    return clf, proba


def compute_delta_from_posteriors(proba, classes, group_priors, eps=1e-12):
    """
    Compute a proxy for Δ(x) from posteriors P(S | X) and priors.

    The paper uses
      Δ(x) ∝ P(S=1|X=x)/p1 - P(S=2|X=x)/p2
    for two groups. Here we generalize to the two class labels found
    in `classes` (assumed length 2).

    Parameters
    ----------
    proba : np.ndarray of shape (n_samples, 2)
        Posterior probabilities P(S=classes[k] | X).
    classes : np.ndarray of shape (2,)
        Class labels in the order of columns of `proba`.
    group_priors : dict
        Mapping from label -> prior probability P(S=label).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    delta : np.ndarray of shape (n_samples,)
    """
    if len(classes) != 2:
        raise ValueError(
            f"Expected exactly 2 classes, got {len(classes)}: {classes}"
        )

    c0, c1 = classes[0], classes[1]
    p0 = group_priors[c0]
    p1 = group_priors[c1]

    p_s0_x = proba[:, 0]
    p_s1_x = proba[:, 1]

    # THEO 4: Δ(x) ∝ P(S=c0|X=x)/p0 - P(S=c1|X=x)/p1
    delta = p_s0_x / (p0 + eps) - p_s1_x / (p1 + eps)
    return delta

def compute_multiclass_delta(proba, classes, group_priors, eps=1e-12):
    """
    Compute the generalized proxy for Δ_k(x) for K >= 2 groups.
    
    Formula: Δ_k(x) = P(S=k | X=x) / p_k - 1
    This quantifies how over/under-represented group k is at point x.

    Parameters
    ----------
    proba : np.ndarray of shape (n_samples, K)
        Posterior probabilities P(S=classes[k] | X).
    classes : np.ndarray of shape (K,)
        Class labels in the order of columns of `proba`.
    group_priors : dict
        Mapping from label -> prior probability P(S=label).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    delta_multi : np.ndarray of shape (n_samples, K)
    """
    n_samples, K = proba.shape
    delta_multi = np.zeros_like(proba, dtype=float)
    
    for k_idx in range(K):
        c_k = classes[k_idx]
        p_k = group_priors[c_k]
        
        # Generalized Delta for group k
        delta_multi[:, k_idx] = (proba[:, k_idx] / (p_k + eps)) - 1.0
        
    return delta_multi


class EntropicKernelRegressor:
    """Non-parametric 'entropic' smoother on Z=(eta, delta).

    Predicts y(Z) = Σ_i w_i(Z) y_i with weights
        w_i(Z) ∝ exp( -||Z - Z_i||^2 / epsilon ).

    Parameters
    ----------
    epsilon : float
        Temperature (>0). Smaller => sharper; larger => smoother.
    knn : int or None
        If set, restrict to knn closest points for speed.
    eps : float
        Numerical jitter.
    """

    def __init__(self, epsilon: float = 0.05, knn: int | None = None, eps: float = 1e-12):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 for EntropicKernelRegressor.")
        self.epsilon = float(epsilon)
        self.knn = None if knn is None else int(knn)
        self.eps = float(eps)
        self.Z_ = None
        self.y_ = None

    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2D, got shape={Z.shape}")
        if Z.shape[0] != y.shape[0]:
            raise ValueError("Z and y must have the same number of rows.")
        self.Z_ = Z
        self.y_ = y
        return self

    def predict(self, Z_new):
        if self.Z_ is None:
            raise RuntimeError("EntropicKernelRegressor must be fit before predict().")
        Z_new = np.asarray(Z_new, dtype=float)
        if Z_new.ndim == 1:
            Z_new = Z_new.reshape(1, -1)

        Z = self.Z_
        # dist2 = ||z||^2 + ||Zi||^2 - 2 z·Zi
        z2 = (Z_new ** 2).sum(axis=1, keepdims=True)
        Z2 = (Z ** 2).sum(axis=1)[None, :]
        dist2 = z2 + Z2 - 2.0 * (Z_new @ Z.T)

        if self.knn is not None and 0 < self.knn < Z.shape[0]:
            idx = np.argpartition(dist2, kth=self.knn - 1, axis=1)[:, : self.knn]
            out = np.empty(Z_new.shape[0], dtype=float)
            for r in range(Z_new.shape[0]):
                d = dist2[r, idx[r]]
                logits = -d / self.epsilon
                logits = logits - np.max(logits)
                w = np.exp(logits)
                w = w / (w.sum() + self.eps)
                out[r] = np.dot(w, self.y_[idx[r]])
            return out

        logits = -dist2 / self.epsilon
        logits = logits - logits.max(axis=1, keepdims=True)
        W = np.exp(logits)
        W = W / (W.sum(axis=1, keepdims=True) + self.eps)
        return W @ self.y_

