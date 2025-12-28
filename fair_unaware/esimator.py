import warnings
import numpy as np

import time 

from sklearn.ensemble import RandomForestRegressor

from .utils import check_binary_groups, compute_group_priors
from .models import (
    fit_eta_model,
    fit_s_classifier,
    compute_delta_from_posteriors,
    EntropicKernelRegressor,
)
from .ot.barycenter import (
    build_plus_minus_measures,
    compute_cost_matrix,
    solve_ot_plan,
    compute_barycentric_predictions_from_plan,
)




class UnawareFairRegressor:
    """
    Naive implementation of the unawareness + DP fair regression
    construction .

    Pipeline in fit(X, S, y):
    -------------------------
    1. Fit base regressor η(X) on (X, y).
    2. Fit classifier P(S | X) on (X, S).
    3. Compute Δ(X) from posteriors and group priors.
    4. Build empirical μ⁺, μ⁻ from (η_i, Δ_i).
    5. Solve OT(μ⁺, μ⁻) with custom cost.
    6. Construct fair predictions at training points via barycentric
       map and conditional expectations.
    7. Fit a final 2D regressor on features (η, Δ) -> fair predictions.

    predict(X) applies the sequence:
    - η_new = base_regressor(X)
    - P(S | X) via classifier
    - Δ_new from posteriors
    - fair_regressor(η_new, Δ_new)

    This is intentionally *naive* and aimed at illustrating the
    algorithm, not at scale or numerical robustness.
    """

    def __init__(
        self,
        base_regressor=None,
        base_classifier=None,
        fair_regressor=None,
        delta_tol=1e-6,
        sinkhorn_reg=None,
        continuous_map="rf",
        continuous_epsilon=0.05,
        continuous_knn=None,
        random_state=0,
        verbose=False,
        store_diagnostics=False,
    ):
        """
        Parameters
        ----------
        base_regressor : sklearn-like regressor or None
            Used to approximate η(X). If None, RandomForestRegressor is used.
        base_classifier : sklearn-like classifier or None
            Used to approximate P(S | X). If None, RandomForestClassifier is used.
        fair_regressor : sklearn-like regressor or None
            Final regressor on 2D features (η, Δ). If None,
            a RandomForestRegressor is used.
        delta_tol : float
            Threshold below which |Δ| is considered ~0 for building μ⁺, μ⁻.
        sinkhorn_reg : float or None
            If None, use exact OT (emd). Otherwise, use Sinkhorn with this
            entropic regularization parameter.
        random_state : int
        verbose : bool
        """
        self.base_regressor = base_regressor
        self.base_classifier = base_classifier
        self.fair_regressor = fair_regressor
        self.delta_tol = delta_tol
        self.sinkhorn_reg = sinkhorn_reg
        self.continuous_map = continuous_map
        self.continuous_epsilon = continuous_epsilon
        self.continuous_knn = continuous_knn
        self.random_state = random_state
        self.verbose = verbose
        self.store_diagnostics = store_diagnostics
        self.timings_ = {} 

        # Attributes set in fit
        self.group_labels_ = None
        self.group_priors_ = None
        self.base_regressor_ = None
        self.base_classifier_ = None
        self.fair_regressor_ = None
        self.classes_ = None

        self.eta_hat_ = None
        self.delta_hat_ = None
        self.fair_predictions_ = None
        self.fairness_active_ = False

        #disgnostics attirbutes 
        self.X_ = None
        self.S_ = None

        self.plus_idx_ = None
        self.minus_idx_ = None
        self.h_plus_ = None
        self.d_plus_ = None
        self.a_ = None
        self.h_minus_ = None
        self.d_minus_ = None
        self.b_ = None
        self.C_ = None
        self.pi_ = None
        self.f_plus_ = None
        self.f_minus_ = None

    def fit(self, X_train, S_train, y_train, X_unlab=None, group_priors=None):
        """
        Fit the unawareness fair regressor using the FairReg-style protocol:

        - Fit η and P(S|X) on the labeled training set (X_train, S_train, y_train)
        - Build the OT/barycenter correction on X_unlab (unlabeled features only)
        - Fit the final (η, Δ) -> f regressor on that unlabeled/correction sample

        Parameters
        ----------
        X_train : (n_train, n_features)
        S_train : (n_train,)
        y_train : (n_train,)
        X_unlab : (n_unlab, n_features) or None
            If None, falls back to using X_train (backward compatible).
        group_priors : dict or None
            Optional mapping {label -> P(S=label)}. If None, estimated from S_train.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train)
        S_train = np.asarray(S_train)
        y_train = np.asarray(y_train)

        if X_unlab is None:
            X_fair = X_train
        else:
            X_fair = np.asarray(X_unlab)

        n_train = X_train.shape[0]

        # store labeled training set (onlyfor diagnostics/plots)
        self.X_ = X_train
        self.S_ = S_train

        if S_train.shape[0] != n_train or y_train.shape[0] != n_train:
            raise ValueError("X_train, S_train, y_train must have the same number of samples.")

        # 1) Check binary groups and priors
        self.group_labels_ = check_binary_groups(S_train)
        self.group_priors_ = group_priors if group_priors is not None else compute_group_priors(S_train)

        # 2) Fit eta(X) on TRAIN only
        t0 = time.perf_counter()
        self.base_regressor_, _ = fit_eta_model(
            X_train, y_train, model=self.base_regressor, random_state=self.random_state
        )
        self.timings_["base_model_fit"] = time.perf_counter() - t0
        if self.verbose:
            print("[fit] Fitted base regressor for eta on train.")

        # 3) Fit classifier P(S|X) on TRAIN only
        t0 = time.perf_counter()
        self.base_classifier_, _ = fit_s_classifier(
            X_train, S_train, clf=self.base_classifier, random_state=self.random_state
        )
        self.classes_ = self.base_classifier_.classes_
        self.timings_["s_classifier_fit"] = time.perf_counter() - t0
        if self.verbose:
            print("[fit] Fitted classifier for P(S|X) on train. Classes:", self.classes_)

        # 4) Compute (eta_hat, delta_hat) on FAIRNESS/CALIBRATION sample X_fair (X_unlab)
        t0 = time.perf_counter()
        eta_hat_fair = self.base_regressor_.predict(X_fair)
        proba_fair = self.base_classifier_.predict_proba(X_fair)
        delta_hat_fair = compute_delta_from_posteriors(
            proba_fair, self.classes_, self.group_priors_
        )
        self.timings_["delta_computation"] = time.perf_counter() - t0
        if self.verbose:
            print("[fit] Computed eta and delta on X_unlab (or X_train if X_unlab=None).")

        self.eta_hat_ = eta_hat_fair
        self.delta_hat_ = delta_hat_fair

        # 5) Build μ⁺, μ⁻ on X_fair
        t0 = time.perf_counter()
        measures = build_plus_minus_measures(
            eta_hat_fair, delta_hat_fair, delta_tol=self.delta_tol
        )
        plus_idx = measures["plus_idx"]
        minus_idx = measures["minus_idx"]
        a = measures["a"]
        b = measures["b"]
        self.timings_["measures_construction"] = time.perf_counter() - t0

        if self.store_diagnostics:
            self.plus_idx_ = plus_idx
            self.minus_idx_ = minus_idx
            self.h_plus_ = measures["h_plus"]
            self.d_plus_ = measures["d_plus"]
            self.a_ = a
            self.h_minus_ = measures["h_minus"]
            self.d_minus_ = measures["d_minus"]
            self.b_ = b

        # small check in case OT cannot be built 
        if a is None or b is None or len(plus_idx) == 0 or len(minus_idx) == 0:
            warnings.warn(
                "Could not build both μ⁺ and μ⁻ (too few points with positive/negative Δ). "
                "Falling back to unconstrained regressor."
            )
            self.fair_regressor_ = None
            self.fair_predictions_ = eta_hat_fair.copy()
            self.fairness_active_ = False
            return self

        h_plus, d_plus = measures["h_plus"], measures["d_plus"]
        h_minus, d_minus = measures["h_minus"], measures["d_minus"]

        # 6) OT plan on X_fair
        t0 = time.perf_counter()
        C = compute_cost_matrix(h_plus, d_plus, h_minus, d_minus)
        self.timings_["cost_matrix_computation"] = time.perf_counter() - t0
        if self.verbose:
            print("[fit] Computed OT cost matrix:", C.shape)

        t0 = time.perf_counter()
        pi = solve_ot_plan(a, b, C, sinkhorn_reg=self.sinkhorn_reg)
        self.timings_["ot_plan_solution"] = time.perf_counter() - t0
        if self.verbose:
            print("[fit] Solved OT plan.")

        if self.store_diagnostics:
            self.C_ = C
            self.pi_ = pi

        # 7) Barycentric predictions at supports
        t0 = time.perf_counter()
        f_plus, f_minus = compute_barycentric_predictions_from_plan(
            pi, h_plus, d_plus, h_minus, d_minus, a, b
        )
        self.timings_["barycentric_predictions"] = time.perf_counter() - t0
        if self.verbose:
            print("[fit] Computed barycentric predictions.")

        if self.store_diagnostics:
            self.f_plus_ = f_plus
            self.f_minus_ = f_minus

        # 8) Build f_target on X_fair
        f_target = eta_hat_fair.copy()
        f_target[plus_idx] = f_plus
        f_target[minus_idx] = f_minus

        # 9) Fit final 2D regressor (eta, delta) -> f_target on X_fair
        t0 = time.perf_counter()
        Z = np.column_stack([eta_hat_fair, delta_hat_fair])

        # NEW: choose continuous mapping
        if self.fair_regressor is not None:
            # user-provided regressor overrides everything
            self.fair_regressor_ = self.fair_regressor

        elif self.continuous_map == "rf":
            self.fair_regressor_ = RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_state,
            )

        elif self.continuous_map == "entropic":
            self.fair_regressor_ = EntropicKernelRegressor(
                epsilon=self.continuous_epsilon,
                knn=self.continuous_knn,
            )

        else:
            raise ValueError(
                f"Unknown continuous_map={self.continuous_map!r}. Expected 'rf' or 'entropic'."
            )

        self.fair_regressor_.fit(Z, f_target)
        self.fair_predictions_ = f_target
        self.fairness_active_ = True
        self.timings_["fair_regressor_fit"] = time.perf_counter() - t0

        if self.verbose:
            print("[fit] Fitted final 2D fair regressor on (eta, delta).")

        return self
    
    
    def predict(self, X):
        """
        Predict with the fair regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        X = np.asarray(X)
        if self.base_regressor_ is None:
            raise RuntimeError("The model must be fit before calling predict().")

        #if fairness had a problem, just use base regressor
        if not self.fairness_active_ or self.fair_regressor_ is None:
            return self.base_regressor_.predict(X)

        t0 = time.perf_counter()

        # 1. eta(X)
        eta_new = self.base_regressor_.predict(X)
        # 2.P(S | X)
        proba_new = self.base_classifier_.predict_proba(X)
        # 3. Δ(X)
        delta_new = compute_delta_from_posteriors(
            proba_new, self.classes_, self.group_priors_
        )
        # 4.dair regression on (eta, Δ)
        Z_new = np.column_stack([eta_new, delta_new])
        y_pred = self.fair_regressor_.predict(Z_new)

        self.timings_["predict_total"] = time.perf_counter() - t0
        return y_pred

    def predict_base(self, X):
        """
        Predict with the base (unconstrained) regressor η(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        if self.base_regressor_ is None:
            raise RuntimeError("The model must be fit before calling predict_base().")
        return self.base_regressor_.predict(X)
