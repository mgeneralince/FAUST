import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.synthetic import make_synthetic_dataset
from fair_unaware.experiments import run_one
from fair_unaware.plots.visualization import (
    plot_gamma_sweep,          
    scatter_XY_colored_by_S,   
    plot_eta_delta_scatter,  
)

#remove points from the name otherwise we have a problem roger 
def fmt_g(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["OT", "Sinkhorn"], default="OT")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma_min", type=float, default=0.0)
    p.add_argument("--gamma_max", type=float, default=2.0)
    p.add_argument("--gamma_steps", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="notes/figs/")
    p.add_argument("--cont_map", choices=["rf", "entropic"], default="rf")
    p.add_argument("--cont_epsilon", type=float, default=0.05)
    p.add_argument("--cont_knn", type=int, default=None)

    args = p.parse_args()

    fig_root = Path(args.out_dir)
    exp_root = fig_root / "confounded" /  f"{args.cont_map}"
    xy_dir = exp_root / "scatter_XY_by_gamma"
    ed_dir = exp_root / "eta_delta_by_gamma"
    tables_dir = exp_root / "tables"


    for d in [xy_dir, ed_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)


    gammas = np.linspace(args.gamma_min, args.gamma_max, args.gamma_steps)

    rows = []
    for g in gammas:
        #same data generation but X_1= N(gamma * S, 1) and Y= 2*X_1 + beta*S + noise
        X, S, Y = make_synthetic_dataset(
            n=args.n,
            gamma=float(g),
            beta=float(args.beta),
            random_state=args.seed,
        )
        #usual run with entropic or rf mapper
        metrics, eta_hat, delta_hat, S_plot, Y_plot, y_base, y_fair = run_one(
            args.method, X, S, Y,
            seed=args.seed,
            return_artifacts=True,
            cont_map=args.cont_map,
            cont_epsilon=args.cont_epsilon,
            cont_knn=args.cont_knn,
        )


        metrics["gamma"] = float(g)
        rows.append(metrics)

        # --- X vs Y scatter ---
        fig_xy, _ = scatter_XY_colored_by_S(
            X, Y, S,
            title_prefix=f"X vs Y (gamma={g:.2f}, beta={args.beta})",
            max_cols=4,
            alpha=0.5,
            s=8,
            legend="single",
            rasterized=True
        )
        xy_path = xy_dir / f"scatter_XY_gamma{fmt_g(g)}_beta{args.beta}.pdf"
        fig_xy.savefig(xy_path),
        plt.close(fig_xy)

        # --- (eta, delta) scatter ---
        fig_ed, ax = plt.subplots(figsize=(5, 4))
        plot_eta_delta_scatter(
            eta_hat, delta_hat, S_plot,
            ax=ax,
            title=f"(eta, delta) TEST (gamma={g:.2f}, beta={args.beta})"
        )
        ed_path = ed_dir / f"eta_delta_gamma{fmt_g(g)}_beta{args.beta}.pdf"
        fig_ed.savefig(ed_path)
        plt.close(fig_ed)

    csv_path = tables_dir / f"sweep_{args.method}_n{args.n}_beta{args.beta}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


    fig_path = exp_root / f"sweep_{args.method}_n{args.n}_beta{args.beta}_{args.gamma_steps}g.pdf"

    plot_gamma_sweep(rows, save_path=fig_path)

    print("Saved:", csv_path)
    print("Saved:", fig_path)
    print("Saved XY scatters in:", xy_dir)
    print("Saved (eta,delta) scatters in:", ed_dir)

if __name__ == "__main__":
    main()
