# Beyond Parametric Inverse Optimization

Code and results for the paper *Beyond Parametric Inverse Optimization* (Valimoradi and Li).
Given observed optimal decisions, the method imputes a convex objective (equivalently a
concave utility) **without a parametric form**, optionally enforcing **additivity** and/or
**smoothness** (a β-Lipschitz gradient), and predicts out-of-sample decisions by re-solving
the forward problem with the imputed objective.

The repository is organized by the paper's three experiments (Section 4).

| Directory | Paper | What is here |
| --- | --- | --- |
| [`consumer_choice/`](consumer_choice/) | §4.1, Figures 1–2 | code, data caches, the full replicated run (3,198 cells) and the figures built from it |
| [`healthcare_synthetic/`](healthcare_synthetic/) | §4.2, Figures 3, 4, 7 | the notebook that produced the paper's results, its recovered parameters, and the figure scripts; plus a scripted re-implementation |
| [`portpy_clinical/`](portpy_clinical/) | §4.3, Figures 5–6, Table 3 | code, the scripts that produced the published runs, the recovered parameters (N = 3–6) and the prediction outputs |

Each directory has its own README with the exact commands.

## Install

```
pip install -r requirements.txt
```

**MOSEK is required** for every smooth (conic) program; a free academic license is available
at mosek.com. **GUROBI** is used for the additive LP class in the consumer and synthetic
studies. There is no solver fallback: without MOSEK the smooth classes do not run.

## Data

* `consumer_choice/` ships its generated data caches (small).
* `healthcare_synthetic/` does not ship the generated cohort (2.6 GB); the notebook regenerates
  it from seed 142. The recovered parameters behind the figures are shipped.
* `portpy_clinical/` does not redistribute the PortPy dataset or the derived forward caches
  (31–130 MB each); see that README for how to rebuild them.

## License

MIT — see [`LICENSE`](LICENSE).
