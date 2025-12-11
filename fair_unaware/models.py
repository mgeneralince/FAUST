import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


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
