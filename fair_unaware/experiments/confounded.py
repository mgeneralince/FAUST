import numpy as np
from fair_unaware import UnawareFairRegressor
from fair_unaware.models import compute_delta_from_posteriors
from sklearn.model_selection import train_test_split
from fair_unaware.metrics import dp_mean_gap

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
        fair_regressor=None,  #none--> continuous_map decides

        continuous_map=cont_map,
        continuous_epsilon=cont_epsilon,
        continuous_knn=cont_knn,

        sinkhorn_reg=sinkhorn_reg,
        delta_tol=1e-3, #tighter tol for confounded setting bc otherwise pb with delta estimation
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
        "dp_base": dp_mean_gap(y_base, S_te),
        "dp_fair": dp_mean_gap(y_fair, S_te),
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
