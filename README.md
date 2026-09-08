# Smooth and Additive Nonparametric Inverse Optimization

Reference code for the paper *Learning Convex Objectives from Decisions: Smooth
and Additive Nonparametric Inverse Optimization* (Valimoradi and Li). Given
observed optimal decisions, the methods impute a convex objective (equivalently
a concave utility) **without a parametric form**, optionally enforcing
**additivity** and/or **smoothness** (a β-Lipschitz gradient), and predict
out-of-sample decisions by re-solving the forward problem with the imputed
objective.

The repository is organized by the paper's experiments.

## Components

| Directory | Paper section | Status |
| --- | --- | --- |
| [`consumer_choice/`](consumer_choice/) | §4.1 Learning a consumer's utility | available |
| [`healthcare_synthetic/`](healthcare_synthetic/) | §4.2 Healthcare application (synthetic) | available |
| [`portpy_clinical/`](portpy_clinical/) | §4.3 Radiotherapy planning (PortPy) | available |

### `consumer_choice/`

A revealed-preference study: a consumer chooses bundles by minimizing
`p^T x - U(x)`; from observed bundles we impute `U` under four structural
settings (a 2×2 ablation of additivity × smoothness) and report test MAPE versus
training-set size. The experiment is run twice from a **single dataset** — once
on the optimal training decisions and once on the same decisions after a ±5%
perturbation — to show how prediction error grows when observed decisions are
noisy. See [`consumer_choice/README.md`](consumer_choice/README.md).

### `healthcare_synthetic/`

A synthetic radiotherapy test bed: a ground-truth forward planner with quadratic,
linear, and exponential organ-at-risk penalties produces optimal plans for randomly
sampled 2-D anatomies. From observed plans the method imputes the structural penalty
functions under the same 2×2 ablation (additivity × smoothness) and reports held-out
beamlet-prediction error versus training-set size, alongside the recovered-vs-true
penalty functions. See [`healthcare_synthetic/README.md`](healthcare_synthetic/README.md).

### `portpy_clinical/`

The clinical-scale study on prostate cases from the public
[PortPy](https://github.com/PortPy-Project/PortPy) dataset. A ground-truth
forward planner with per-voxel quadratic organ-at-risk penalties produces
optimal plans; the recovery imputes the six penalty components from observed
plans on a nested cohort and predicts beamlet weights for held-out patients.
The three-stage driver is `run_stage1_nested.py` (feasibility), 
`run_stage2_nested.py` (smoothness bisection) and `run_stage3_nested_auto.py`
(recovery), with `run_prediction.py` for out-of-sample evaluation. The PortPy
dataset is not redistributed here; see their repository for access.

## Install

```
pip install -r requirements.txt
```

MOSEK is the primary solver (a free academic license is available); the code
falls back to ECOS (bundled with CVXPY) when MOSEK is absent.

## License

MIT — see [`LICENSE`](LICENSE).
