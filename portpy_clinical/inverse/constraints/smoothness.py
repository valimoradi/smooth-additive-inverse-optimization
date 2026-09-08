"""
inverse/constraints/smoothness.py
==================================
Constraint C5 — smooth convexity (per-voxel version).

For each function k and each pair of adjacent observations (sorted by overdose),
enforces:
  g_k(o_j) >= g_k(o_i) + lambda_i * (o_j - o_i) + (1/2*beta) * (lambda_i - lambda_j)^2

Now operates on M_k observations per function (pooled across patients and voxels)
instead of N observations.
"""

import cvxpy as cp
import numpy as np

from config import N_FUNCTIONS


def build_c5_smoothness(
    delta_funcs: list,
    lambda_funcs: list,
    Z_per_func: list,
    beta_val: float,
) -> list:
    """
    Build smooth-convexity constraints for all K functions.

    Parameters
    ----------
    delta_funcs  : list of K cp.Variables, each shape (M_k,)
    lambda_funcs : list of K cp.Variables, each shape (M_k,)
    Z_per_func   : list of K arrays, each shape (M_k,) — overdose observations
    beta_val     : smoothness parameter

    Returns
    -------
    list of CVXPY constraints
    """
    if beta_val <= 0:
        raise ValueError(f"beta_val must be positive; got {beta_val}")

    beta_inv = 1.0 / (2.0 * beta_val)
    constraints = []

    for k in range(N_FUNCTIONS):
        z_k = Z_per_func[k]   # binned grid points
        M_k = len(z_k)
        if M_k < 2:
            continue

        idx_sorted = np.argsort(z_k)  # sort bins by z
        i_idx = idx_sorted[:-1]
        j_idx = idx_sorted[1:]

        dz = z_k[j_idx] - z_k[i_idx]

        delta_k = delta_funcs[k]
        lam_k = lambda_funcs[k]

        delta_i = delta_k[i_idx]
        delta_j = delta_k[j_idx]
        lam_i = lam_k[i_idx]
        lam_j = lam_k[j_idx]

        q_term = beta_inv * cp.square(lam_i - lam_j)

        # C5 forward: g_k(o_j) >= g_k(o_i) + lambda_i * (o_j - o_i) + quad
        constraints.append(delta_j - delta_i - cp.multiply(lam_i, dz) >= q_term)

        # C5 backward: g_k(o_i) >= g_k(o_j) + lambda_j * (o_i - o_j) + quad
        constraints.append(delta_i - delta_j - cp.multiply(lam_j, -dz) >= q_term)

    return constraints
