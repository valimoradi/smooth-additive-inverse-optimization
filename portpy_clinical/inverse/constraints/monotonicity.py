"""
inverse/constraints/monotonicity.py
=====================================
Constraint C6 — monotone non-decreasing cost functions.

Paper reference: per_voxel_formulation.tex, eq:c6 (and main paper
Proposition 5, C6):

    lambda_k_(j) >= 0,   j = 1, ..., M_k,  k = 1, ..., K.

For a univariate convex function on observations sorted by z, the
subgradient lambda must be monotone non-decreasing along z. This is
algebraically implied by the smoothness constraint C5 (summing C5 in
both directions for an adjacent (i, j) pair with z_j > z_i forces
lambda_i <= lambda_j), but the implication relies on C5 holding with
zero slack. MOSEK is an interior-point solver with finite primal/dual
feasibility tolerances (~1e-8); with M_k > 10^4 observations per
function, the per-pair slack accumulates and lets the solver return a
lambda trajectory that dips slightly out of order at some adjacent
indices. The recovered envelope is then a different convex hull than
the one obtained when monotonicity is enforced explicitly.

We therefore impose the explicit ordering

    lambda_k[sorted j+1] >= lambda_k[sorted j]

in addition to lambda_k >= 0. The two together are the numerically
robust realisation of the paper's eq:c6 plus the convexity ordering
that C5 implies in exact arithmetic.
"""

import cvxpy as cp
import numpy as np

from config import N_FUNCTIONS


def build_c6_monotonicity(lambda_funcs: list, Z_binned=None) -> list:
    """
    Return C6 monotonicity constraints on lambda.

    lambda_k >= 0  -- paper's eq:c6 (g_k non-decreasing).
    lambda_k sorted ascending in z -- the convexity ordering. Implied
    by C5 in exact arithmetic; enforced explicitly for numerical
    conditioning under MOSEK's finite tolerances on the per-voxel
    formulation (M_k can exceed 10^4).
    """
    constraints = []
    for k in range(N_FUNCTIONS):
        constraints.append(lambda_funcs[k] >= 0)

        if Z_binned is not None and len(Z_binned[k]) >= 2:
            idx_sorted = np.argsort(Z_binned[k])
            constraints.append(
                lambda_funcs[k][idx_sorted[1:]] >= lambda_funcs[k][idx_sorted[:-1]]
            )

    return constraints
