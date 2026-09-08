# Consumer-Choice Experiment (§4.1)

A revealed-preference study. A consumer chooses a bundle `x >= 0` by solving the
forward problem

```
min_x  p^T x - U(x)
```

for a price vector `p` and a concave utility `U`. We observe optimal bundles for
many price vectors and impute `U` (equivalently the convex cost `f = -U`) under
different structural assumptions, then predict held-out bundles by re-solving the
forward problem with the imputed objective. Error is reported as test MAPE versus
the training-set size.

## The four models (a 2×2 ablation)

|               | no smoothness          | β-smooth            |
| ------------- | ---------------------- | ------------------- |
| no additivity | Convex only (Li 2019)  | Smooth              |
| additive      | Additive               | Additive + Smooth   |

Additivity is imposed in the inverse **solve** (per-coordinate concavity);
smoothness adds a β-Lipschitz gradient. Combining both gives the best
out-of-sample prediction.

## Formulation: no approximation

Smoothness is the only place a `β` appears, and it is **solved for**, never
approximated by a large constant:

* **Convex only** and **Additive** carry **no `β` term**. Prediction uses a
  convex-combination forward resolve from the recovered values `δ` alone.
* **Smooth** and **Additive + Smooth** enforce a β-Lipschitz gradient; the
  smoothness `β*` is found by bisection (the smallest feasible value — the most
  smoothness the data admits) and reported per training size in the results CSV.
  Prediction uses the perspective/conjugate forward resolve with `δ`, `λ`, `β*`.

## Optimal vs. perturbed (the point of the experiment)

Both panels come from **one generated dataset**. The unperturbed panel feeds the
**optimal** training bundles to the solvers; the perturbed panel feeds the same
bundles after a **±5% multiplicative perturbation of the training inputs only**.
The held-out test set is the clean optimum in both cases, so the pair isolates
how prediction error grows when observed decisions are noisy. The convex-hull
baseline is nearly noise-insensitive, while the smooth models degrade most.

## Configuration

Ground-truth utility `U(x) = sum_i log(a_i x_i + b_i)`. The data-generation draw
order (prices, then `a`, then `b`) reproduces the paper figures.

| parameter      | value                                       |
| -------------- | ------------------------------------------- |
| products       | 5                                           |
| gamma          | 40                                          |
| prices         | Uniform[8, 20]                              |
| a, b           | Uniform[1, 1000], Uniform[2, 60]            |
| scenarios      | 600 (200 test, the rest train pool)         |
| anchors        | two synthetic: p=1000 → x=0, p=0 → x=1000   |
| perturbation   | training bundles × (1 + Uniform[−.05, .05]) |
| training sizes | 20, 40, …, 260                              |
| seed           | 42                                          |

## Run

```
# Both panels (unperturbed + perturbed) from one dataset:
python consumer_inverse_optimization.py --utilities smooth

# Misspecification panels (non-additive, non-smooth ground truth):
python consumer_inverse_optimization.py --utilities nonsmooth

# Quick smoke test:
python consumer_inverse_optimization.py --utilities smooth --sizes 20 60 100 --test-subset 40

# Optional parallel driver (the Smooth model is slow at large N):
python run_parallel.py
```

The sequential script writes figures and CSVs to `consumer_figures/` by default
(transient, git-ignored). The parallel driver (`run_parallel.py`) writes figures to
`figures/` and CSVs to `results/`, matching the committed canonical outputs.

Outputs: `smooth-additive-not-perturbed.pdf`, `smooth-additive-perturbed.pdf`,
`legend.pdf`, and `results_smooth_<regime>.csv` (per-size MAPE and `β*`).

## Expected results (test MAPE %)

**Unperturbed (optimal training inputs):**

| N   | Convex only | Additive | Smooth | Additive + Smooth |
| --- | ----------- | -------- | ------ | ----------------- |
| 20  | 17.69       | 14.46    | 6.94   | 8.94              |
| 100 | 12.01       | 10.32    | 6.84   | 4.97              |
| 260 | 9.98        | 8.38     | 5.77   | 3.90              |

**Perturbed (±5% training inputs):**

| N   | Convex only | Additive | Smooth | Additive + Smooth |
| --- | ----------- | -------- | ------ | ----------------- |
| 20  | 17.60       | 14.72    | 10.87  | 8.46              |
| 100 | 11.99       | 10.73    | 10.37  | 9.52              |
| 260 | 9.96        | 9.20     | 8.36   | 6.48              |

The Convex-only and Additive curves reproduce the published figures essentially
exactly. The Smooth / Additive+Smooth curves are somewhat lower than the original
fixed-β notebook because `β` is solved rather than approximated; the ordering
(Additive+Smooth best, Convex-only worst) and the perturbation effect are
unchanged. The Smooth (optimal) curve is mildly non-monotone at N=40/80 because
`β*` is selected per training size — this is a genuine property of the method,
not smoothed away.

## Layout

```
consumer_inverse_optimization.py   # single, self-contained implementation
run_parallel.py                    # optional multiprocess driver (same results)
figures/                           # generated panels + optimal-vs-perturbed overlay
results/                           # per-size MAPE + β* CSVs
reference/
  Log-2-additive-...csv            # ground-truth numbers behind the published figure
  notebooks/                       # original notebooks (provenance)
```
