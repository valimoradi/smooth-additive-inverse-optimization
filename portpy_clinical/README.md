# Clinical-Scale Radiotherapy Study (§4.3, Figures 5–6, Table 3)

Prostate cases from the public [PortPy](https://github.com/PortPy-Project/PortPy) dataset.
A forward planner with per-voxel quadratic overdose penalties on six structures
(equation 4.12 of the paper) produces the reference plans. The recovery imputes the six
shared penalty functions from nested cohorts of N = 3, 4, 5, 6 patients, and the recovered
objective is used to predict plans for 20 held-out patients, against KKT-residual
benchmarks restricted to linear, quadratic and cubic penalties.

## What is shipped

| Path | Contents |
| --- | --- |
| `inverse/`, `forward/`, `recovery/`, `data/`, `config.py` | the model: per-voxel outcomes, C5 adjacent-pair smoothness, C6 monotonicity with the sorted slope ordering, C7 two-anchor normalization, the per-patient dual with complementary slackness at the observed optimum and the linking `ν = λ` (Appendix B.3), the C8′ certificate, and the perspective-form prediction SOCP (Appendix B.4) |
| `results/nested_N{3,4,5,6}_fullleak_nopin/` | the recovered `δ_k`, `λ_k`, outcomes `Z_k`, `β*`, `β_used`, `ε*` behind Figure 5 |
| `results/prediction_nested_fullleak_nopin_dedup/` | predicted doses for the 20 test patients behind Figure 6 |
| `results/parametric_comparison/` | the benchmark fits and the Figure 6 metrics |
| `test_patients_canonical.txt` | the 20 held-out patient ids |
| `_fl_recover_nopin.sh`, `_fl_pred_nopin.sh`, `_fl_pred_dedup.sh` | the scripts that produced the folders above, as run |

Not shipped: the PortPy patient data (download from PortPy) and the derived forward caches
`results/all_patients/forward_cache_*.pkl` (31–130 MB each), which the pipeline below
rebuilds.

## Pipeline, as run for the paper

```bash
# 1. Forward plans for the cohort pool and the test patients (needs PortPy data)
python solve_all_forward.py

# 2. Synthetic best/worst anchors and the nested training caches
python build_synthetic_anchors.py
python build_nested_caches.py

# 3. Recovery, N = 3..6 (forward_cache_nested_N{N}.pkl holds the 2 anchors + N patients):
#    Stage 1 minimizes eps at beta_bar = 40 (= 2 max_k alpha_k, the generating curvature);
#    Stage 2 bisects beta* downward from 100 with floor 1.0 and gap 0.11;
#    Stage 3 is the conservative selection. Anchors are pinned to their exact forward objectives.
bash _fl_recover_nopin.sh

# 4. Prediction for the 20 test patients, with the anchor reduction of Appendix B.4
bash _fl_pred_dedup.sh          # (_fl_pred_nopin.sh: without the reduction)

# 5. Parametric benchmarks, the Figure 6 metrics, and Figure 5 (the env vars select the
#    published directories; the scripts' defaults point at the earlier *_fullleak_anchor runs)
python run_parametric_comparison.py
FL_DOSES_DIR=results/prediction_nested_fullleak_nopin_dedup python make_prediction_metrics_exact.py
FL_RUN_DIR='nested_N{N}_fullleak_nopin' python plot_summary_grid_fullleak.py
```

The shell scripts read the two anchor objectives from
`results/all_patients/forward_cache_nested_N5.pkl` and export them as `U_MIN_ANCHOR` /
`U_MAX_ANCHOR`; they call the interpreter as `python3.11.exe` (edit `PY=` for another name).
Voxels are downsampled to about 15 × 15 × 5 mm. The four real cohort patients are 44, 35,
20, 37 (in that nesting order); the anchors are constructed from patients 61 (best) and
6 (worst).

## Requirements

PortPy, `numpy`, `scipy`, `cvxpy`, `pandas`, `matplotlib`, and **MOSEK** (required; there is
no fallback for the conic programs).
