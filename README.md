# FAUST

Code and artifacts for **Geometry of Relaxed Fair Regression: A Unified
Framework for Aware and Unaware Settings** using
[Python Optimal Transport (POT)](https://pythonot.github.io) and
basic `scikit-learn` models.

## Citation 
```
@article{lince2026geometryrelaxedfairregression,
      title={Geometry of Relaxed Fair Regression: A Unified Framework for Aware and Unaware Settings}, 
      author={M. Generali Lince and V. Divol and R. Flamary and S. Gaucher and P. Loiseau},
      year={2026},
      eprint={2605.28233},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.28233}, 
}
```

## High-level idea

Given data $(X, S, Y)$ with:
- $X$ : non-sensitive features
- $S \in \{0,1\}$: sensitive group
- $Y \in \mathbb{R}$: regression target

we want a predictor $f(X)$ that

1. Minimizes squared loss $\mathbb{E}[(Y - f(X))^2]$,
2. Satisfies **Demographic Parity**: the distribution of $f(X)$ is
   the same across groups (unawareness: $f$ cannot see $S$ directly).

The paper shows that (in the 2-group case) the optimal fair regressor
can be obtained by solving an OT barycenter problem in the space of
$(\eta(X), \Delta(X))$, where:

- $\eta(X) = \mathbb{E}[Y \mid X]$ : Bayes regressor,
- $\Delta(X)$ is a signed quantity built from
  posteriors $P(S=s \mid X)$ and group priors.

In this repo, we:

1. **Estimate**:
   - $\hat\eta(X)$ via a regression model (RandomForestRegressor),
   - $P(S \mid X)$ via a classifier (RandomForestClassifier),
   - $\hat\Delta(X)$ from those posteriors.

2. **Build empirical measures** $\mu_+$ and $\mu_-$ on
   $\Omega = \{(h,d) : d \neq 0\}$ from samples of
   $(h,d) = (\hat\eta_i, \hat\Delta_i)$ with $d > 0$ and $d < 0$
   (Jordan decomposition idea from the paper).

3. **Solve an OT problem** between $\mu_+$ and $\mu_-$ with POT:
   - Cost:
     $$
       C\big( (h_1,d_1),(h_2,d_2) \big)
       = \frac{(h_2 - h_1)^2}{|d_1| + |d_2|}.
     $$
   - Using either exact `emd` or entropic `sinkhorn`.

4. **Construct a discrete barycenter**: from an optimal coupling $\pi$,
   we map each pair of support points to a barycenter value
   $$
     m((h_1,d_1),(h_2,d_2))
     = \frac{h_1/|d_1| + h_2/|d_2|}{1/|d_1| + 1/|d_2|},
   $$
   and average these with weights from $\pi$ to define **fair
   predictions at the training points**.

5. **Fit a final 2D regressor**:
   - Input features: $(\hat\eta_i, \hat\Delta_i)$,
   - Targets: fair predictions from step 4.
   - Result: a smooth predictor $\hat f(x)$ that depends only on
     $(\hat\eta(x), \hat\Delta(x))$.

Predictions for new points:
1. Compute $\hat\eta(x)$ via the base regressor.
2. Compute $P(S \mid x)$ via the classifier; build $\hat\Delta(x)$.
3. Apply the final 2D regressor to $(\hat\eta(x), \hat\Delta(x))$.

## Code overview

- `fair_unaware/utils.py`  
  Small helper utilities: checking the groups and computing group priors.

- `fair_unaware/models.py`  
  Wraps the estimation of:
  - the regression model for $\eta(X)$,
  - the classifier for $P(S \mid X)$,
  - and the computation of $\Delta(X)$.

- `fair_unaware/ot_barycenter.py`  
  Core OT logic:
  - Building empirical measures $\mu_+$ and $\mu_-$,
  - Constructing the cost matrix,
  - Solving OT with POT,
  - Computing barycentric fair predictions from the optimal plan.

- `fair_unaware/predictor.py`  
  High-level `UnawareFairRegressor` class:
  - `fit(X, S, y)` runs the full pipeline and trains a final 2D regressor.
  - `predict(X)` uses the trained pipeline to produce fair predictions.

- `scripts/run_example.py`  
  Generates a toy dataset, trains the unconstrained regressor and the
  `UnawareFairRegressor`, and prints basic DP statistics
  (mean prediction per group before/after).

- `scripts/run_relaxed_tradeoff.py`  
  Sweeps the regularization parameter $\lambda$ for OT and plots the
  tradeoff between MSE and fairness (W2 gap).


# Fair Unaware Regression — Experiment Scripts

This repository contains experiment scripts for **Geometry of Relaxed Fair Regression: A Unified Framework for Aware and Unaware Settings**, using Optimal Transport (OT) to enforce distributional fairness constraints without access to sensitive attributes at prediction time.

The core library is `fair_unaware`, which implements `UnawareFairRegressor` and an aware baseline (`AwareWassersteinBarycenterDPRegressor`). The scripts in this repo run benchmarks, toy comparisons, and tradeoff analyses comparing OT-based methods against the FairReg baseline.

---

## Repository Structure

```
.
├── scripts/
│   ├── compare_ot_vs_fairreg.py        # Main benchmark on real & synthetic datasets
│   ├── run_full_benchmark.py           # Multi-base-model benchmark with LaTeX output
│   ├── run_tradeoff_relaxed.py         # Fairness/accuracy tradeoff curves (λ sweep)
│   ├── run_toy_comparison_metrics.py   # 1D toy experiment with exact W2 targets
│   └── run_example_confounded.py       # Gamma sweep on the confounded DGP
├── data/
│   └── synthetic.py                    # Synthetic dataset generators
└── fair_unaware/                       # Core library (installed separately)
```

---

## Methods

All methods are **post-processors**: they take the predictions of a shared unconstrained base regressor η̂(X) and transform them to enforce fairness. No method modifies the base model itself.

| Key | Full name | Description |
|---|---|---|
| `base` / `ERM` | Unconstrained ERM | Raw predictions of the base regressor η̂(X). Unfair reference point. |
| `ot` / `OT Unaware` | **OT Unaware** (this work) | Lifts predictions into a 2D pseudo-measure space via the signed density-ratio Δ(x) = P(S=1\|x)/π₁ − P(S=0\|x)/π₀, then solves an exact OT plan between μ⁺ and μ⁻ and reads off fair predictions via barycentric projection. S is **never observed at test time**. |
| `aware` / `OT Aware` | OT Aware ([Chzhen et al., 2020](https://arxiv.org/abs/2007.14265)) | Computes the Wasserstein-2 barycenter of the per-group prediction distributions, with S **observed at inference**. This is the theoretical optimum and serves as an upper bound for all unaware methods. |
| `fairreg` / `FairReg` | FairReg ([Tarturyan et al., 2024](https://arxiv.org/abs/2407.15453)) | Grid-based adversarial post-processor. Learns a randomized prediction rule over a discrete grid Ŷ_L ⊂ [−B, B] by minimizing an entropically regularized dual objective via stochastic gradient (AC-SA2). Does not observe S at test time. |
| `frappe` / `FRAPPÉ` | FRAPPÉ ([Tifrea et al., 2024](https://arxiv.org/abs/2312.02592)) | Learns an additive neural network module T(X) over fixed base predictions to minimize MSE plus a Gaussian MMD Demographic Parity penalty. Does not observe S at test time. |
| `ablation` / `OT-U (η̃)` | OT Unaware ablation | Same OT Unaware pipeline, but the base regressor η̂(X) is replaced by a **marginalized version of the aware predictor**: η̃(x) = Σ_s P̂(S=s\|x) T_s(η̂(x)). Tests whether performance gains come from the OT geometry or from a better base signal. |
| `shat` / `OT Ŝ W₂` | Plug-in hard baseline | Applies the **aware** OT maps T_s but substitutes a hard predicted label Ŝ(x) = argmax P̂(S\|x) for the unobserved S at test time. A naive plug-in approach; fails when classification errors corrupt the transport plan. |
**Fairness metrics** used throughout: W₂ gap between group marginals, TV gap, KS statistic, and DP_max (CDF-based demographic parity).

---

## Scripts

### `compare_ot_vs_fairreg.py`

The main comparison script. Runs multiple independent trials on a dataset, fits all methods, evaluates on a held-out test set and unlabeled pool, and produces LaTeX summary tables.

**Datasets:** `adult`, `lawschool`, `communities`, `synthetic`

```bash
python scripts/compare_ot_vs_fairreg.py \
    --dataset lawschool \
    --num 10 \
    --cont_map rf \
    --train_size 0.4 \
    --test_size 0.2
```


**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `lawschool` | Dataset to use |
| `--num` | `10` | Number of independent runs |
| `--cont_map` | `rf` | Continuous map estimator: `rf` or `entropic` |
| `--cont_epsilon` | `0.05` | Regularisation for entropic OT map |
| `--sinkhorn_reg` | `None` | Sinkhorn regularisation (uses exact OT if None) |
| `--delta_tol` | `1e-6` | Convergence tolerance |
| `--train_size` | `0.4` | Fraction of data used as labelled training set |
| `--test_size` | `0.2` | Fraction of data held out for evaluation |
| `--tune_fairreg` | `False` | Enable hyperparameter tuning for FairReg on a validation split |
| `--L_grid` | `auto,0.5x,1x,2x` | Candidate L values for FairReg tuning |
| `--beta_grid` | `auto,0.25x,0.5x,1x,2x,4x` | Candidate beta values for FairReg tuning |
| `--make_plots` | `False` | Generate per-run diagnostic plots |
| `--make_tradeoff` | `False` | Generate fairness/accuracy tradeoff plots |
| `--load_results` | `None` | Path to a saved `runs.csv` to skip training and regenerate outputs |
| `--out_dir` | `notes/figs/compare_ot_vs_fairreg` | Output directory |

**Outputs** (under `--out_dir/<dataset>/`):
- `runs.csv` — raw metrics for all runs
- `summary/<dataset>/runs.tex` — per-run LaTeX table
- `summary/<dataset>/summary.tex` / `summary_relative.tex` — aggregate LaTeX tables
- `summary/<dataset>/fairreg_selected_params.csv` — chosen FairReg hyperparameters per run

---

### `run_full_benchmark.py`

Extends `compare_ot_vs_fairreg.py` by sweeping over multiple **base model** families (e.g. linear, random forest, gradient boosting) and producing a consolidated multi-model LaTeX table.

```bash
python scripts/run_full_benchmark.py \
    --dataset adult \
    --num 5 \
    --cont_map rf
```

Outputs a single `benchmark_table.tex` with `\multirow` formatting grouped by base model.

---

### `run_tradeoff_relaxed.py`


Traces the **MSE vs. fairness tradeoff curve** by sweeping the regularisation parameter λ for OT (L2 and TV penalties), FairReg, and FRAPPÉ.

```bash
python scripts/run_tradeoff_relaxed.py \
    --dataset lawschool \
    --num_runs 5 \
    --cont_map rf \
    --out_dir notes/figs/tradeoff
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | — | Dataset name |
| `--lam_grid_ot` | — | Comma-separated λ values for OT (L2 penalty) |
| `--lam_grid_tv` | — | Comma-separated λ values for OT (TV penalty) |
| `--eps_grid_fairreg` | log-spaced | ε values for the FairReg tradeoff |
| `--num_runs` | `5` | Number of independent runs per λ |
| `--method` | `OT` | Transport solver: `OT` or `Sinkhorn` |
| `--load_results` | `None` | Path to saved CSV to skip recomputation |
| `--out_dir` | — | Output directory for plots |
| `--crop_frappe` | `False` | Crop the plot axes to the OT bounds. |

**Outputs:** Tradeoff curve PDF plots (MSE vs. W₂ / DP_max) saved under `--out_dir`.

---

### `run_toy_comparison_metrics.py`

Runs a controlled **1D synthetic experiment** to visualise the exact geometric effect of the OT relaxations. Two penalty types are compared in both the aware and unaware settings:

- **W₂ (L2) relaxation:** The cost matrix is modified to `C_λ(x₁,x₂) = [λ/(1+λ(a₁+a₂))] · (h₁−h₂)²`, which induces a smooth global interpolation — all predictions are gently shifted toward the fair barycenter proportionally to their density-ratio weights.
- **TV relaxation:** The cost is `C_λ(x₁,x₂) = min(λ, (h₁−h₂)²/(a₁+a₂))`, which acts as a hard-thresholding operator — pairs whose disparity cost falls below λ are perfectly merged to their weighted mean, while all others are left completely unchanged.

Both penalties are calibrated at five exact W₂ unfairness levels (0%, 20%, 40%, 60%, 80% of the baseline gap) via a numerical line search over λ, so comparisons are equitable.

```bash
python scripts/run_toy_comparison_metrics.py
```

**Outputs** (under `notes/figs/toy_comparison/empirical_levels/shift/`):
- Per-level `unaware_composite.pdf` / `aware_composite.pdf` — prediction curves + marginal violin plots
- `unaware_evolution_violins_traj.pdf` / `aware_evolution_violins_traj.pdf` — full fairness evolution with individual trajectories overlaid
- `plot_data.npz` — cached arrays for fast replotting with `--replot` 
---

### `run_example_confounded.py`

Sweeps **γ (confounding strength)** on a synthetic confounded dataset, evaluating all methods as γ varies from near-zero to near-one. Runs are parallelised with `joblib`.

```bash
python scripts/run_example_confounded.py \
    --gamma_min 0.01 \
    --gamma_max 0.99 \
    --gamma_steps 9 \
    --num_runs 10 \
    --n 1000 \
    --cont_map rf \
    --use_oracle        # optional: use Bayes-optimal base models computed with the exact formula 
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--gamma_min/max` | `0.01` / `0.99` | Range of confounding strength γ |
| `--gamma_steps` | `9` | Number of γ values (100 in the paper)|
| `--num_runs` | `10` | Runs per γ value |
| `--n` | `1000` | Samples per dataset |
| `--use_oracle` | `False` | Use oracle (Bayes-optimal) base models |
| `--cont_map` | `rf` | Continuous map estimator |
| `--load_results` | `None` | Path to saved CSV to skip recomputation |
| `--out_dir` | `notes/figs/` | Root output directory |

**Outputs** (under `notes/figs/confounded/<oracle_folder>/<cont_map>/`):
- Relative and absolute gamma-sweep PDFs
- MSE vs. W₂ tradeoff scatter plot
- Within-group variance sweep plot
- LaTeX table (`tables/sweep_*.tex`)
- Raw results CSV

---

## Installation

The scripts depend on the `fair_unaware` package. Install it (and optional FairReg baseline) before running:

```bash
pip install -e .
pip install -r requirements.txt
```


To use the **FairReg** baseline, pass `--fairreg_root /path/to/unaware-fair-reg`.

---

## Fairness Metrics Reference

| Symbol | Name | Description |
|---|---|---|
| W₂ gap | Wasserstein-2 gap | W₂ distance between per-group prediction marginals.|
| TV | Total Variation gap | L1 distance between per-group CDFs. |
| KS | Kolmogorov–Smirnov | Max absolute CDF difference between groups. |
| DP_max | Demographic Parity max | Maximum KS gap swept over a threshold grid. |
| MSE | Mean Squared Error | Prediction accuracy relative to the true label Y. |
---

