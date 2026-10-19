import numpy as np
from fair_unaware import UnawareFairRegressor
from fair_unaware.baselines import AwareWassersteinBarycenterDPRegressor
from fair_unaware.models import compute_delta_from_posteriors
from sklearn.model_selection import train_test_split
from fair_unaware.metrics import w2_gap_between_groups, dp_cdf_max_gap
from fair_unaware.utils import compute_group_priors
from sklearn.linear_model import LinearRegression, LogisticRegression
from fair_unaware.baselines.fairreg_adapter import get_fairreg_objects
from sklearn.metrics import mean_squared_error as mse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from fair_unaware.baselines.frappe import FrappeRegressor
        


from fair_unaware.models import (
    OracleSyntheticClassifier, 
    OracleSyntheticUnawareRegressor, 
    OracleSyntheticAwareRegressor
)




def run_one(method, X, S, Y, seed=42, return_artifacts=False,
            cont_map="rf", cont_epsilon=0.05, cont_knn=None):

    X = np.asarray(X)
    S = np.asarray(S).ravel()
    Y = np.asarray(Y).ravel()

    X_tr, X_te, S_tr, S_te, Y_tr, Y_te = train_test_split(
        X, S, Y,
        test_size=0.3,
        random_state=42,
        stratify=S
    )

    sinkhorn_reg = None if method == "OT" else 0.05

    fair_model = UnawareFairRegressor(
        base_regressor=None,
        base_classifier=None,
        fair_regressor=None,  

        continuous_map=cont_map,
        continuous_epsilon=cont_epsilon,
        continuous_knn=cont_knn,

        sinkhorn_reg=sinkhorn_reg,
        delta_tol=1e-3, 
        random_state=seed,
        verbose=False,
        store_diagnostics=False,
    )

    fair_model.fit(X_tr, S_tr, Y_tr)

    # Predictions on TEST 
    y_base = fair_model.predict_base(X_te)
    y_fair = fair_model.predict(X_te)

    metrics = {
        "mse_base": float(np.mean((y_base - Y_te) ** 2)),
        "mse_fair": float(np.mean((y_fair - Y_te) ** 2)),
        "dp_base": dp_cdf_max_gap(y_base, S_te),
        "dp_fair": dp_cdf_max_gap(y_fair, S_te),
        "w2_base": w2_gap_between_groups(y_base, S_te),
        "w2_fair": w2_gap_between_groups(y_fair, S_te),
    }

    if return_artifacts:
        # (eta, delta) on TEST
        eta_te = y_base
        proba_te = fair_model.base_classifier_.predict_proba(X_te)
        delta_te = compute_delta_from_posteriors(
            proba_te,
            fair_model.base_classifier_.classes_,
            fair_model.group_priors_,
        )
        return metrics, eta_te, delta_te, S_te, Y_te, y_base, y_fair

    return metrics

def _auto_L_beta(n_unlab: int, L_arg: str, beta_arg: str):
    L_auto = int(np.sqrt(max(n_unlab, 1)))
    beta_auto = 0.5 * np.sqrt(max(n_unlab, 2)) * np.log2(max(n_unlab, 2))
    L = L_auto if str(L_arg).lower() == "auto" else int(float(L_arg))
    beta = beta_auto if str(beta_arg).lower() == "auto" else float(beta_arg)
    return L, beta


def _parse_eps_vec(eps_list: str, K: int) -> np.ndarray:
    vals = [float(x) for x in str(eps_list).split(",") if x.strip() != ""]
    if len(vals) == 1:
        return np.array([vals[0]] * K, dtype=float)
    if len(vals) == K:
        return np.array(vals, dtype=float)
    raise ValueError(f"--fairreg_eps_list must have length 1 or K={K}, got {len(vals)}")



def within_group_variance(y_pred, S):
    """Calculates the average variance of predictions within each sensitive group."""
    y_pred = np.asarray(y_pred).ravel()
    S = np.asarray(S).ravel()
    groups = np.unique(S)
    
    #calculate variance for each group, then take the mean
    variances = [np.var(y_pred[S == g]) for g in groups]
    return float(np.mean(variances))


def run_one_all_methods(
    method: str,
    X,
    S,
    Y,
    *,
    gamma: float = 1.0,
    beta: float = 1.0,
    use_oracle: bool = False,
    seed: int = 42,
    return_artifacts: bool = False,
    cont_map: str = "rf",
    cont_epsilon: float = 0.05,
    cont_knn: int | None = None,
    sinkhorn_reg: float | None = None,
    delta_tol: float = 1e-6,
    # FairReg options (same spirit as compare_ot_vs_fairreg)
    fairreg_root: str | None = None,
    fairreg_L: str = "auto",
    fairreg_beta: str = "auto",
    fairreg_T: int | None = None, 
    fairreg_B: int = 1,
    fairreg_eps_list: str = "0.01",
):
    """
    Mirrors compare_ot_vs_fairreg.py style:
      - base models: reg=LinearRegression, clf=LogisticRegression
      - Unaware OT: UnawareFairRegressor(reg, clf, ..., sinkhorn_reg, delta_tol) + fit(..., X_unlab, group_priors)
      - Aware OT: AwareWassersteinBarycenterDPRegressor + fit(..., X_unlab, S_unlab, group_priors)
      - FairReg: FairReg(reg, clf, B,K,p,eps,T) + fit(X_unlab, beta, L, alg={"base":"SGD3","method":"ACSA2"})

    Returns:
      if return_artifacts:
        (metrics, eta_te, delta_te, S_te, Y_te, y_base, y_ot)
      else:
        metrics
    """

    X = np.asarray(X)
    S = np.asarray(S).ravel()
    Y = np.asarray(Y).ravel()

    # same split pattern as in confounded.run_one (70% train, 30% test, stratify by S) for the non fairreg methods 
    X_tr, X_te, S_tr, S_te, Y_tr, Y_te = train_test_split(
        X, S, Y, test_size=0.2, stratify=S, random_state=seed
    )

    #splits for fairreg 
    X_tr_fr, X_unlab_fr, S_tr_fr, S_unlab_fr, Y_tr_fr, Y_unlab_fr = train_test_split(
        X_tr, S_tr, Y_tr, test_size=0.5, stratify=S_tr, random_state=seed
    )

    #base models (like compare_ot_vs_fairreg.py fit_base_models)
    if use_oracle:
        reg = OracleSyntheticUnawareRegressor(gamma=gamma, beta=beta)
        clf = OracleSyntheticClassifier(gamma=gamma)
        aware_base_reg = OracleSyntheticAwareRegressor(beta=beta)
        reg_fr = OracleSyntheticUnawareRegressor(gamma=gamma, beta=beta)
        clf_fr = OracleSyntheticClassifier(gamma=gamma)
    else:
        reg = LinearRegression(fit_intercept=True)
        clf = LogisticRegression(max_iter=2000)
        aware_base_reg = LinearRegression(fit_intercept=True)
        reg_fr = LinearRegression(fit_intercept=True)
        clf_fr = LogisticRegression(max_iter=2000)

    reg.fit(X_tr, Y_tr)
    clf.fit(X_tr, S_tr)


    # priors (used by OT + aware)
    priors = compute_group_priors(S_tr)

    #Unaware OT (same constructor style as compare_ot_vs_fairreg.py fit_ot_model)
    if method == "OT":
        _sink = None
    else:
        _sink = float(sinkhorn_reg) if sinkhorn_reg is not None else 0.01

    ot_model = UnawareFairRegressor(
        base_regressor=reg,
        base_classifier=clf,
        fair_regressor=None,
        continuous_map=cont_map,
        continuous_epsilon=cont_epsilon,
        continuous_knn=cont_knn,
        sinkhorn_reg=_sink,
        delta_tol=delta_tol,
        random_state=seed,
        verbose=False,
        store_diagnostics=False,
    )

    #like compare_ot_vs_fairreg.py: X_unlab + group_priors 
    try:
        ot_model.fit(X_tr, S_tr, Y_tr, X_unlab=X_tr, group_priors=priors)
    except TypeError:
        ot_model.fit(X_tr, S_tr, Y_tr)

    y_ot = ot_model.predict(X_te)

    #eta/delta plot (TEST), like the preious confounded.run_one
    y_base = ot_model.predict_base(X_te)
    eta_te = y_base

    proba_te = ot_model.base_classifier_.predict_proba(X_te)

    delta_te = compute_delta_from_posteriors(
    proba_te,
    ot_model.base_classifier_.classes_,
    ot_model.group_priors_,
)


    #aware OT (same spirit as compare_ot_vs_fairreg.py fit_aware_model) 
    y_aware = np.full_like(Y_te, np.nan, dtype=float)
    try:
        aware = AwareWassersteinBarycenterDPRegressor(
            base_regressor=aware_base_reg,
            random_state=seed,
        )
        aware.fit(
            X_tr, S_tr, Y_tr,
            X_unlab=X_tr, S_unlab=S_tr,
            group_priors=priors,
        )
        y_aware = aware.predict(X_te, S_te)
    except Exception:
        pass
    
    #OT ablation: use the aware model's predictions but plug them into the OT logic instead of the base regressor's predictions
    #o see how much of the gap is due to better eta estimation vs the OT logic itself. 
    # This is a bit hacky but it avoids code duplication and ensures we use the exact same OT logic as in the main OT method. 
    # We just need to create a wrapper that makes the aware model look like a base regressor for the OT code.

    class TildeEtaWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, aware_model, s_estimator):
            self.aware_model = aware_model
            self.s_estimator = s_estimator
        def fit(self, X, y=None, sample_weight=None):
            return self 
        def predict(self, X):
            return self.aware_model.predict_marginalized(X, self.s_estimator)

    y_ot_ablation = np.full_like(Y_te, np.nan, dtype=float)
    if np.isfinite(y_aware).all(): 
        ot_ablation_model = UnawareFairRegressor(
            base_regressor=TildeEtaWrapper(aware_model=aware, s_estimator=clf), 
            base_classifier=clf,
            fair_regressor=None,
            continuous_map=cont_map,
            continuous_epsilon=cont_epsilon,
            continuous_knn=cont_knn,
            sinkhorn_reg=_sink,
            delta_tol=delta_tol,
            random_state=seed,
            verbose=False,
        )
        try:
            ot_ablation_model.fit(X_tr, S_tr, Y_tr, X_unlab=X_tr, group_priors=priors)
        except TypeError:
            ot_ablation_model.fit(X_tr, S_tr, Y_tr)
            
        y_ot_ablation = ot_ablation_model.predict(X_te)

    #Aware OT with Estimated S (OT S-hat W2)
    y_shat = np.full_like(Y_te, np.nan, dtype=float)
    try:
        
        s_estimator = LogisticRegression(max_iter=2000).fit(X_tr, S_tr)

        aware_shat = AwareWassersteinBarycenterDPRegressor(
            base_regressor=aware_base_reg, 
            random_state=seed, 
            estimate_S=True,
            s_estimator=s_estimator
        )
        
        aware_shat.fit(
            X_tr, S_tr, Y_tr, 
            X_unlab=X_tr, S_unlab=None, group_priors=None
        )
        
        S_te_hat = s_estimator.predict(X_te)
        y_shat = aware_shat.predict(X_te, S_te_hat)
    except Exception as e:
        print(f"S-hat failed: {e}")
        pass

    #FairReg baseline (same construction logic as compare_ot_vs_fairreg.py)
    y_fairreg = np.full_like(Y_te, np.nan, dtype=float)
    try:
        FairReg, _ = get_fairreg_objects(fairreg_root)

        # 1. T dynamiq (60 epoichs)
        if fairreg_T is not None:
            T_final = int(fairreg_T)
        else:
            T_final = 60 * len(X_unlab_fr)

        # 2. Scaling 
        scaler = MinMaxScaler(feature_range=(-1, 1))
        Y_tr_fr_scaled = scaler.fit_transform(Y_tr_fr.reshape(-1, 1)).flatten()

        # 3. save the oracle from the scaling
        if hasattr(reg_fr, "gamma"): 
            class OracleScaledWrapper(BaseEstimator, RegressorMixin):
                def __init__(self, oracle, scaler):
                    self.oracle = oracle
                    self.scaler = scaler
                def fit(self, X, y=None):
                    return self
                def predict(self, X):
                    raw = self.oracle.predict(X)
                    return self.scaler.transform(raw.reshape(-1, 1)).flatten()
            
            reg_fr_safe = OracleScaledWrapper(reg_fr, scaler)
        else:
            reg_fr.fit(X_tr_fr, Y_tr_fr_scaled)
            reg_fr_safe = reg_fr

        clf_fr.fit(X_tr_fr, S_tr_fr)

        classes = clf.classes_ 
        K = len(classes)
        p_vec = np.array([(np.asarray(S_tr_fr) == c).mean() for c in classes], dtype=float)
        p_vec = np.clip(p_vec, 1e-6, 1 - 1e-6)

        eps_vec = _parse_eps_vec(fairreg_eps_list, K)

        fr = FairReg(
            reg_fr_safe, clf_fr,
            B=1.0,  
            K=K,
            p=p_vec,
            eps=eps_vec,
            T=T_final, 
            keep_history=False,
        )
        L, beta_fr = _auto_L_beta(len(X_unlab_fr), fairreg_L, fairreg_beta)
        fr.fit(X_unlab_fr, beta=beta_fr, L=L, alg={"base": "SGD3", "method": "ACSA2"})
    
        y_fairreg_scaled = fr.predict(X_te)
        y_fairreg = scaler.inverse_transform(y_fairreg_scaled.reshape(-1, 1)).flatten()
        
    except Exception as e:
        print(f"FairReg failed: {e}")
        pass

    # --- FRAPPÉ baseline ---
    y_frappe = np.full_like(Y_te, np.nan, dtype=float)
    try:
        if use_oracle:
            reg_frappe = OracleSyntheticUnawareRegressor(gamma=gamma, beta=beta)
        else:
            reg_frappe = LinearRegression(fit_intercept=True)
            
        frappe_model = FrappeRegressor(
            base_regressor=reg_frappe, 
            hidden_dims=(64,), 
            lam=100.0, # High penalty for strict fairness
            epochs=100,
            lr=0.01
        )
        frappe_model.fit(X_tr, S_tr, Y_tr, X_unlab=X_tr, S_unlab=S_tr)
        y_frappe = frappe_model.predict(X_te)
    except Exception as e:
        print(f"FRAPPÉ failed: {e}")

    # common bins for DPmax (use all finite predictions so bins cover everyone)
    preds_for_bins = [y_base, y_ot, y_aware, y_fairreg, y_ot_ablation, y_shat, y_frappe]
    all_preds = np.concatenate([p.ravel() for p in preds_for_bins if p is not None and np.all(np.isfinite(p))])
    lo, hi = np.min(all_preds), np.max(all_preds)
    if lo == hi:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, 200)

    metrics = {
        "mse_base": float(mse(Y_te, y_base)),
        "dpmax_base": float(dp_cdf_max_gap(y_base, S_te, bins=bins)),
        "w2_base": float(w2_gap_between_groups(y_base, S_te)),
        "var_base": within_group_variance(y_base, S_te), 

        "mse_ot": float(mse(Y_te, y_ot)),
        "dpmax_ot": float(dp_cdf_max_gap(y_ot, S_te, bins=bins)),
        "w2_ot": float(w2_gap_between_groups(y_ot, S_te)),
        "var_ot": within_group_variance(y_ot, S_te),    

        "mse_aware": float(mse(Y_te, y_aware)) if np.isfinite(y_aware).all() else float("nan"),
        "dpmax_aware": float(dp_cdf_max_gap(y_aware, S_te, bins=bins)) if np.isfinite(y_aware).all() else float("nan"),
        "w2_aware": float(w2_gap_between_groups(y_aware, S_te)) if np.isfinite(y_aware).all() else float("nan"),
        "var_aware": within_group_variance(y_aware, S_te) if np.isfinite(y_aware).all() else float("nan"),

        "mse_ablation": float(mse(Y_te, y_ot_ablation)) if np.isfinite(y_ot_ablation).all() else float("nan"),
        "dpmax_ablation": float(dp_cdf_max_gap(y_ot_ablation, S_te, bins=bins)) if np.isfinite(y_ot_ablation).all() else float("nan"),
        "w2_ablation": float(w2_gap_between_groups(y_ot_ablation, S_te)) if np.isfinite(y_ot_ablation).all() else float("nan"),

        "mse_shat": float(mse(Y_te, y_shat)) if np.isfinite(y_shat).all() else float("nan"),
        "dpmax_shat": float(dp_cdf_max_gap(y_shat, S_te, bins=bins)) if np.isfinite(y_shat).all() else float("nan"),
        "w2_shat": float(w2_gap_between_groups(y_shat, S_te)) if np.isfinite(y_shat).all() else float("nan"),
        "var_shat": within_group_variance(y_shat, S_te) if np.isfinite(y_shat).all() else float("nan"),

        "mse_fairreg": float(mse(Y_te, y_fairreg)) if np.isfinite(y_fairreg).all() else float("nan"),
        "dpmax_fairreg": float(dp_cdf_max_gap(y_fairreg, S_te, bins=bins)) if np.isfinite(y_fairreg).all() else float("nan"),
        "w2_fairreg": float(w2_gap_between_groups(y_fairreg, S_te)) if np.isfinite(y_fairreg).all() else float("nan"),
        "var_fairreg": within_group_variance(y_fairreg, S_te) if np.isfinite(y_fairreg).all() else float("nan"), 

        "mse_frappe": float(mse(Y_te, y_frappe)) if np.isfinite(y_frappe).all() else float("nan"),
        "dpmax_frappe": float(dp_cdf_max_gap(y_frappe, S_te, bins=bins)) if np.isfinite(y_frappe).all() else float("nan"),
        "w2_frappe": float(w2_gap_between_groups(y_frappe, S_te)) if np.isfinite(y_frappe).all() else float("nan"),
        "var_frappe": within_group_variance(y_frappe, S_te) if np.isfinite(y_frappe).all() else float("nan"),
    }


    if return_artifacts:
        return metrics, eta_te, delta_te, S_te, Y_te, y_base, y_ot

    return metrics
