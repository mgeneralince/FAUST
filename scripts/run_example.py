import os
import argparse
from pathlib import Path


import numpy as np
from sklearn.ensemble import RandomForestRegressor

from fair_unaware import UnawareFairRegressor
from sklearn.model_selection import train_test_split

from fair_unaware.diagnostics import plot_basic_diagnostics
from fair_unaware.metrics import dp_mean_gap   
from data.synthetic import make_synthetic_dataset


def main(method="OT", nb_samples=1000, *, out_dir="notes/figs", prefix="example",
         save_suffix="pdf", show=False, with_ot_matrices=False,
         cont_map="rf", cont_epsilon=0.05, cont_knn=None):

    X, S, Y = make_synthetic_dataset(n=nb_samples, random_state=42)

    X_tr, X_te, S_tr, S_te, Y_tr, Y_te = train_test_split(
        X, S, Y,
        test_size=0.3,
        random_state=42,
        stratify=S
    )

    #unconstrained regressor 
    base_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    base_reg.fit(X_tr, Y_tr)
    y_pred_base = base_reg.predict(X_te)

    mse_base = np.mean((y_pred_base - Y_te) ** 2)
    dp_base = dp_mean_gap(y_pred_base, S_te)


    print("=== Unconstrained regressor ===")
    print(f"MSE      : {mse_base:.3f}")
    print(f"DP gap   : {dp_base:.3f}")

    # Unawareness fair regressor
    if method not in {"OT", "Sinkhorn"}:
        raise ValueError(f"Unknown method: {method}")
    sinkhorn_reg = None if method == "OT" else 0.05
    fair_model = UnawareFairRegressor(
        base_regressor=None,
        base_classifier=None,
        fair_regressor=None,  #None== continuous_map decides

        continuous_map=cont_map,
        continuous_epsilon=cont_epsilon,
        continuous_knn=cont_knn,

        delta_tol=1e-6,
        sinkhorn_reg=sinkhorn_reg,
        random_state=42,
        verbose=True,
        store_diagnostics=True,
    )


    fair_model.fit(X_tr, S_tr, Y_tr)
    y_pred_fair = fair_model.predict(X_te)

    mse_fair = np.mean((y_pred_fair - Y_te) ** 2)
    dp_fair = dp_mean_gap(y_pred_fair, S_te)

    print("\n=== Unawareness fair regressor ===")
    print(f"MSE      : {mse_fair:.3f}")
    print(f"DP gap   : {dp_fair:.3f}")

        #generate basicplots (with_ot_matrices= matrices non agrégées mais on voit rien) 
    print("\nGenerating diagnostic plots (including OT internals)...")
    fig_root = Path(out_dir)
    run_dir = fig_root / method / f"{cont_map}" / str(nb_samples)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = run_dir / prefix


    plot_basic_diagnostics(
        fair_model,
        X_te,
        S_te,
        Y_te,
        show=show,
        save_prefix=str(save_prefix),
        with_ot_matrices=with_ot_matrices,
        save_suffix=save_suffix,
        save_transport=True,
    )
    
    #store timings 
    print("Timing breakdown (in seconds):")
    for k, v in fair_model.timings_.items():
        print(f"  {k:25s}: {v:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Unawareness Fair Regression Example")
    parser.add_argument("--method", type=str, choices=["OT", "Sinkhorn"], default="OT",
                        help="Choose the method for fair regression: 'OT' for Optimal Transport, 'Sinkhorn' for Sinkhorn regularization.")
    parser.add_argument("--nb_samples", type=int, default=1000,
                        help="Number of samples in the synthetic dataset.")
    parser.add_argument("--out_dir", default="notes/figs")   
    parser.add_argument("--prefix", default="example")       
    parser.add_argument("--save_suffix", default="pdf")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--with_ot_matrices", action="store_true", default=True)
    parser.add_argument("--cont_map", choices=["rf", "entropic"], default="rf")
    parser.add_argument("--cont_epsilon", type=float, default=0.05)
    parser.add_argument("--cont_knn", type=int, default=None)

    args = parser.parse_args()

    main(
    method=args.method,
    nb_samples=args.nb_samples,
    out_dir=args.out_dir,
    prefix=args.prefix,
    save_suffix=args.save_suffix,
    show=args.show,
    with_ot_matrices=args.with_ot_matrices,
    cont_map=args.cont_map,
    cont_epsilon=args.cont_epsilon,
    cont_knn=args.cont_knn,
)

