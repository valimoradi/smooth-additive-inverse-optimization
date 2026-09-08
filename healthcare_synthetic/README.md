# Synthetic Healthcare Inverse-Optimization Experiment (Section 4.2)

Clean, runnable implementation of the synthetic radiotherapy test bed used in the
"Healthcare application" subsection. A ground-truth forward planner (quadratic + linear +
exponential organ-at-risk penalties) generates plans for randomly sampled anatomies; the
inverse-optimization method then recovers the structural penalty functions from observed
plans and is evaluated on held-out patients across four ablation models.

## Models (2x2 ablation)

| key                | additive? | smooth? | recovery                                   | solver |
|--------------------|-----------|---------|--------------------------------------------|--------|
| `add_smooth`       | yes       | yes     | 3-stage (min eps -> bisect beta* -> max delta) | MOSEK  |
| `add_lp`           | yes       | no      | 2-stage pure LP                            | GUROBI |
| `nonadd_smooth`    | no        | yes     | 3-stage                                    | MOSEK  |
| `nonadd_nonsmooth` | no        | no      | 2-stage, forced large beta                 | MOSEK  |

## Layout

```
healthcare_synthetic/
  config.py          all constants, paths, seed (GLOBAL_SEED = 142)
  forward_model.py   anatomy + dose-matrix generation, ground-truth forward solve, Z_hat
  anchor_search.py   best/worst-case anchor search (cells 8/9)
  generate_cohort.py cohort generation + aggregation (cells 10/13)
  data_pipeline.py   UNIFIED 5-check filter + train/test split + anchor augmentation
  recovery.py        inverse-optimization recovery for the 4 models (cells 22/27/29/31)
  predict.py         imputed-forward prediction + beamlet-error CSVs (cells 26/28/30/32)
  plots.py           combined-MAPE and true-vs-retrieved figures (cells 34/58)
  run_generation.py  Stage A orchestrator (heavy; regenerates local data)
  run_experiment.py  Stage B orchestrator (filter -> recovery -> predict -> plots)
```

## Reproducing the figures

```bash
cd healthcare_synthetic
# Stage A (optional): regenerate the local data from scratch (slow; seed 142).
python run_generation.py

# Stage B: produce the figures. --skip-data reuses an existing ml_data split.
python run_experiment.py
```

Outputs:
- `combined_plots/combined_mape_vs_training_size_OR_style.png`
- `final_comparison_plots/true_vs_retrieved_1x3_large.png`

## Data policy

The generated data (per-patient plans `patient_data_high_std/`, aggregated cohort
`patient_data_clean_high_std/`, train/test split `ml_data_high_std/`) is large and kept
**local only** (git-ignored). Set `SYNTH_DATA_ROOT` to point the code at a data location.

The following are committed to the repository for reproducibility:
- `results_high_std/`, `results_high_std_add_nonsmooth_lp/`, etc. — recovered parameters
  (`.npy` files, small) for all four models at N=20–400.
- `add_smooth_outputs/`, `add_nonsmooth_lp_outputs/`, etc. — prediction-error CSVs for
  all four models at N=20–400. The paper figure uses N=20–400.

Figure output directories (`combined_plots/`, `final_comparison_plots/`) are git-ignored
and regenerated locally by `run_experiment.py`. The canonical paper figures are in
`submission/figures/` at the repo root.

## Notes on faithfulness / fixes

This code was verified to reproduce the paper's cached results, with two corrections:

1. **Unified filter.** A single 5-check validity filter (`check_patient_validity`:
   duplicate-anatomy, micro-organ, 3-sigma size outlier, PTV-OAR overlap, bladder >= 0.5*PTV)
   is the sole data gate, applied to BOTH the cohort and the two anchors. In the original
   notebook the cohort used the 5-check while the anchors only passed a 2-check anatomy gate;
   the anchors pass the 5-check, so the produced `ml_data` is identical.

2. **`add_lp` normalization gauge.** `U_MAX = 1000` (matching the paper's parameters).
   `delta` and `lambda` scale linearly with `U_MAX`, so it is a free gauge and predictions
   are invariant to it.

3. **`add_smooth` small-N robustness.** MOSEK 11 throws on the small (e.g. N=20) stage-3
   problem at 1e-6 tolerances; the recovery retries the same beta at 1e-5, which solves it
   at the natural smallest-feasible beta* (no escalation needed). Larger N still solve at
   1e-6 first, so their parameters reproduce exactly.

## Requirements

Python 3, `numpy`, `scipy`, `cvxpy`, `pandas`, `matplotlib`, `scikit-learn`, and licensed
solvers **MOSEK** and **GUROBI**.
