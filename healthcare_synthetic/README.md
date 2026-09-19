# Synthetic Radiotherapy Experiment (§4.2, Figures 3, 4, 7)

A synthetic test bed. A ground-truth forward planner with quadratic, linear and exponential
organ-at-risk penalties produces optimal plans for randomly generated 2-D anatomies
(55 × 55 voxel grid, 100 beamlets, Gaussian beamlet kernels). From observed plans the
method recovers the three penalty functions under the 2 × 2 ablation (additivity ×
smoothness) and is scored by held-out beamlet prediction error (Rel-L2) over the test
patients that admit a feasible plan.

## `paper_run/` — what produced the paper's results

| Item | Contents |
| --- | --- |
| `Cancer simulation code seed 142.ipynb` | the notebook that generated the data (seed 142), ran all four recoveries at N = 20, …, 200 and evaluated them. Cells 8–10 generate the cohort; 22/27/29/31 recover Additive+Smooth / Additive / Smooth / Convex-only; 26/28/30/32 evaluate them. |
| `results_high_std/` (`_v3`) | recovered `δ`, `λ`, `β*`, `ε*` for Additive + Smooth |
| `results_high_std_add_nonsmooth_lp/` (`_lp`) | Additive (LP, GUROBI) |
| `results_high_std_non_add_smooth/` (`_vS`) | Smooth |
| `results_high_std_non_add_nonsmooth/` (`_vNS`) | Convex only (β = 10⁵ conic program) |
| `fig4_data/` | the four Figure 4 curves as CSV |
| `figure_scripts/` | `make_combined_mape.py` (Figure 4), `make_true_vs_retrieved.py` (Figure 3), `make_fig3_zoom.py` (Figure 7); default paths point at the folders above |

The generated cohort itself (`ml_data_high_std/`, 2.6 GB) is not shipped: running the
notebook's generation cells reproduces it from seed 142.

Recovery uses the first-order certificate (Appendix B.2, exact at ε = 0); `ε*` is the
Stage-1 solver residual plus 10⁻⁶ and is stored per N in `epsilon_star_N*.npy`. Stage 3 is
the paper's conservative selection. Per class:

| class | Stage 1 `β̄` | Model 2 | returned `β*` |
| --- | --- | --- | --- |
| Additive + Smooth | 2000 | bisection on [0.01, 10], gap 0.11 | 0.0880 at every N (the lower end of the final bracket) |
| Smooth | 2000 | bisection on [0.01, 20], gap 0.2 | 0.1662 for N ≥ 40; at N = 20 the original run returned 42.036 after solver exceptions, and Figure 4 uses the refit in `results_high_std_non_add_smooth_N20_bisection_refit/` (`β* = 0.1662`) |
| Additive | — (LP) | — | — |
| Convex only | fixed β = 10⁵ | — | 10⁵ |

Seven of the 400 generated anatomies admit no feasible plan: three are held-out cases and
are not scored (the test mean is over 197 patients); four are in the training pool and
enter the recovery through their outcomes only.

## Scripted re-implementation

`config.py`, `forward_model.py`, `anchor_search.py`, `generate_cohort.py`, `data_pipeline.py`,
`recovery.py`, `predict.py`, `plots.py`, `run_generation.py`, `run_experiment.py` are a
module-by-module rewrite of the notebook:

```bash
python run_generation.py    # Stage A: regenerate the cohort from seed 142 (slow)
python run_experiment.py    # Stage B: filter -> recovery -> prediction -> plots
```

Two differences from `paper_run/` to be aware of:

* its `nonadd_nonsmooth_lp` model (labelled "Convex only" in `run_panel.py` / `plots.py`)
  is a max-of-tangents LP, not the β = 10⁵ conic program the paper's Figure 4 uses;
  `nonadd_nonsmooth` is the paper's model;
* it applies one unified validity filter to cohort and anchors (the notebook gates the
  anchors separately; both pass, so the data are identical).

The `*_outputs/` folders hold this pipeline's prediction CSVs. The paper's numbers are the
ones in `paper_run/`.

## Requirements

Python 3, `numpy`, `scipy`, `cvxpy`, `pandas`, `matplotlib`, `scikit-learn`; **MOSEK** for
the smooth classes and the Convex-only conic program, **GUROBI** for the additive LP.
