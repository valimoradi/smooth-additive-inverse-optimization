# Consumer-Choice Experiment (§4.1, Figures 1–2)

A revealed-preference study. A consumer chooses a bundle `x >= 0` by solving

```
max_x  U(x) - p^T x
```

for a price vector `p` and a concave utility `U`. From bundles observed under many price
vectors we impute `U` under four structural classes, then predict held-out bundles by
re-solving the forward problem with the imputed utility. Error is the test-set relative
L2 error of the predicted bundle, averaged over 200 held-out price vectors.

## The four classes

|               | no smoothness | β-smooth          |
| ------------- | ------------- | ----------------- |
| no additivity | Convex only   | Smooth            |
| additive      | Additive      | Additive + Smooth |

Additive classes are recovered and **predicted** good by good (per-coordinate components);
smooth classes carry a β-Lipschitz gradient with `β*` chosen per training size by bisection
(Model 2 of the paper) and reported for every cell in `results/`.

## Data

Two ground-truth utilities, each drawn once:

* `smooth` (Figure 1): `U(x) = 40 Σ_i log(a_i x_i + b_i)`, `a_i ~ U[1,1000]`, `b_i ~ U[2,60]`.
* `kicks3` (Figure 2): `U(x) = 40 min{U_1, U_2, U_3}`, three coupled-log branches over the
  same goods (nonadditive and nonsmooth). Branches 2–3 permute the slope pool of branch 1 and
  rescale goods 2 and 4 by 1.5 in opposite directions; the interaction matrix `W` couples
  eight pairs, seven symmetrically.

600 price vectors `p ~ U[8,20]^5`; 200 held out for testing, 400 form the training pool.
Two synthetic anchors fix level and scale: `U(0) = 0` and `U(1000·1) = 1000`. Training sets
are nested prefixes of a random ordering of the pool (rep 0 is the generated order; reps
1–19 use seed `1000 + r`), sizes 20, 40, …, 200. Two regimes: `not-perturbed` (exact
optimal bundles) and `perturbed` (training bundles scaled coordinatewise by
`1 + U[-0.05, 0.05]`; the test set stays exact).

The caches in `data/` hold the exact generated data used for the paper.

## Certificate

All four classes use the exact inverse-optimality certificate (C8 of the paper) with a
free witness per observation; the additive classes use its componentwise form. Stage 1
minimizes ε at `β̄ = 1000`, `ε* = ε₀ + 1e-6`; Stage 3 is the paper's conservative selection.

## Run

```
# Figures 1 and 2, 20 replications, all sizes (the run behind the paper; ~3 days on 10 cores)
CONSUMER_WORKERS=10 python run_parallel.py --utilities smooth kicks3 --reps 20 \
    --out-name paper --cert c8 --cache-dir data

# Rebuild the paper panels from the shipped results
python make_paper_panels.py results figures --exclude results_excluded_cells.txt
```

`run_parallel.py` checkpoints every cell to `results_<out-name>_c8/_cells_checkpoint.csv`
and can be re-entered; `_cells_log.jsonl` records every solver call of every cell.

## What is shipped in `results/`

The complete run: 3,193 of 3,200 cells (2 utilities × 2 regimes × 4 classes × 10 sizes ×
20 replications). Seven Additive+Smooth cells (all `kicks3`, N ∈ {140, 180, 200}) raised
`Stage-3 solve failed after beta escalation` deterministically and are absent. Four further
Additive+Smooth cells (`smooth`, N = 200, listed in `results_excluded_cells.txt`) solved but
with the Model-2 bisection terminating inside a band of MOSEK solver failures (identical
`β* = 4.1015625` in all four, against a median of 409 for their siblings); the paper's
figures exclude them, and `make_paper_panels.py --exclude` reproduces that choice. The
checkpoint ships every cell, so either variant can be rebuilt.

`results_<utility>_<regime>_reps.csv` are the per-cell values in wide form;
`results_<utility>_<regime>.csv` are the per-size means.

## Expected results (mean test Rel-L2 over 20 replications)

Figure 1, additive smooth truth:

| N   | Convex only | Additive | Smooth | Additive + Smooth |
| --- | ----------- | -------- | ------ | ----------------- |
| 20  | 0.275 / 0.275 | 0.114 / 0.117 | 0.120 / 0.129 | 0.093 / 0.096 |
| 100 | 0.179 / 0.178 | 0.028 / 0.036 | 0.081 / 0.108 | 0.024 / 0.034 |
| 200 | 0.153 / 0.151 | 0.013 / 0.023 | 0.072 / 0.109 | 0.008 / 0.023 |

Figure 2, nonadditive nonsmooth truth:

| N   | Convex only | Additive | Smooth | Additive + Smooth |
| --- | ----------- | -------- | ------ | ----------------- |
| 20  | 0.326 / 0.324 | 0.159 / 0.160 | 0.190 / 0.195 | 0.148 / 0.147 |
| 100 | 0.197 / 0.197 | 0.122 / 0.120 | 0.123 / 0.133 | 0.107 / 0.108 |
| 200 | 0.160 / 0.159 | 0.120 / 0.121 | 0.102 / 0.113 | 0.109 / 0.108 |

(not-perturbed / perturbed; Additive + Smooth at N = 200 excludes the four listed cells.)

## Layout

```
consumer_inverse_optimization.py   models, certificates, predictors, data generation, plotting
run_parallel.py                    multiprocess driver with per-cell checkpointing (used for the paper)
make_paper_panels.py               paper panels from a checkpoint; ±2 SE bands
data/                              generated data caches (smooth, kicks3)
results/                           the complete paper run: checkpoint, solver log, CSVs
results_excluded_cells.txt         the four cells excluded from the figures, with the reason
figures/                           the paper's four panels and legend
```
