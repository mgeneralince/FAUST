import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from fair_unaware import UnawareFairRegressor
from fair_unaware.diagnostics import plot_basic_diagnostics


os.makedirs("notes/figs", exist_ok=True)

def make_synthetic_dataset(n=1000, random_state=0):
    """
    Simple synthetic dataset:
    - S ~ Bernoulli(0.5)
    - X1 ~ N(S, 1), X2 ~ N(0, 1)
    - Y = 2*X1 + S + noise

    Groups have different conditional distributions of Y and X, so
    unconstrained predictions will typically not satisfy DP.
    """
    rng = np.random.RandomState(random_state)
    S = rng.binomial(1, 0.5, size=n)

    X1 = rng.normal(loc=S, scale=1.0, size=n)
    X2 = rng.normal(loc=0.0, scale=1.0, size=n)
    X = np.column_stack([X1, X2])

    noise = rng.normal(scale=0.5, size=n)
    Y = 2.0 * X1 + S + noise

    return X, S, Y


def main(method="OT", nb_samples=1000):
    X, S, Y = make_synthetic_dataset(n=nb_samples, random_state=42)

    #unconstrained regressor 
    base_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    base_reg.fit(X, Y)
    y_pred_base = base_reg.predict(X)

    #simple DP stats: mean prediction per group
    base_mean_0 = y_pred_base[S == 0].mean()
    base_mean_1 = y_pred_base[S == 1].mean()

    print("=== Unconstrained regressor ===")
    print(f"Mean prediction for S=0: {base_mean_0:.3f}")
    print(f"Mean prediction for S=1: {base_mean_1:.3f}")
    print(f"Difference in means: {abs(base_mean_0 - base_mean_1):.3f}")
    print()

    # Unawareness fair regressor
    if method == "OT":
        sinkhorn_reg = None
        fair_model = UnawareFairRegressor(
            base_regressor=None,     # RF for eta
            base_classifier=None,    # RF for P(S|X)
            fair_regressor=None,     # RF on (eta, delta)
            delta_tol=1e-6,
            sinkhorn_reg=sinkhorn_reg,    
            random_state=42,
            verbose=True,
            store_diagnostics=True,  #ofr the PLOTS
        )
    elif method == "Sinkhorn":
        sinkhorn_reg = 0.05
        fair_model = UnawareFairRegressor(
            base_regressor=None,     
            base_classifier=None,    
            fair_regressor=None,     
            delta_tol=1e-6,
            sinkhorn_reg=sinkhorn_reg,      
            random_state=42,
            verbose=True,
            store_diagnostics=True,  
        )


    fair_model.fit(X, S, Y)
    y_pred_fair = fair_model.predict(X)

    fair_mean_0 = y_pred_fair[S == 0].mean()
    fair_mean_1 = y_pred_fair[S == 1].mean()

    print("\n=== Unawareness fair regressor ===")
    print(f"Mean prediction for S=0: {fair_mean_0:.3f}")
    print(f"Mean prediction for S=1: {fair_mean_1:.3f}")
    print(f"Difference in means: {abs(fair_mean_0 - fair_mean_1):.3f}")

    #MSE
    mse_base = np.mean((y_pred_base - Y) ** 2)
    mse_fair = np.mean((y_pred_fair - Y) ** 2)
    print(f"\nMSE (base) : {mse_base:.3f}")
    print(f"MSE (fair) : {mse_fair:.3f}")

        #generate basicplots (with_ot_matrices= matrices non agrégées mais on voit rien) 
    if method == "OT":
        print("\nGenerating diagnostic plots (including OT internals)...")
        plot_basic_diagnostics(
            fair_model,
            X,
            S,
            Y,
            show=False,
            save_prefix=f"notes/figs/{method}/{args.nb_samples}/example",
            with_ot_matrices=True,   
            save_suffix="pdf"
        )
    elif method == "Sinkhorn":
        print("\nGenerating diagnostic plots (including Sinkhorn internals)...")
        plot_basic_diagnostics(
            fair_model,
            X,
            S,
            Y,
            show=True,
            save_prefix=f"notes/figs/{method}/{args.nb_samples}/example",
            with_ot_matrices=True,  
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
    args = parser.parse_args()
    os.makedirs(f"notes/figs/{args.method}/{args.nb_samples}", exist_ok=True)
    main(method=args.method, nb_samples=args.nb_samples)
