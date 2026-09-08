"""
recovery/structural_function.py
================================
Evaluates the recovered per-voxel penalty function g_k at arbitrary query
points by solving a Second-Order Cone Program (SOCP).

For function k, the recovered g_k(o) is:
  g_k(o) = min_{alpha >= 0, sum(alpha)=1}
              (beta/2) * sum_m alpha_m * ||v_m / alpha_m - z_m||^2
            + sum_m alpha_m * (delta_m + lambda_m * (z_m - o))

where m ranges over all observations in S_k.
"""

import cvxpy as cp
import numpy as np

from config import PREFERRED_SOLVER, MOSEK_HIGH_PRECISION


def evaluate_function_at_point(
    o_query: float,
    k_idx: int,
    params: dict,
    solver: str = PREFERRED_SOLVER,
    mosek_opts: dict = None,
) -> tuple:
    """
    Evaluate the recovered function g_k at overdose value o_query.

    Parameters
    ----------
    o_query   : scalar query point (overdose value)
    k_idx     : function index (0=Bladder, 1=Rectum, 2=Femoral, 3=Skin)
    params    : dict with keys:
                  'delta_funcs': list of K arrays (M_k,)
                  'lambda_funcs': list of K arrays (M_k,)
                  'Z_per_func': list of K arrays (M_k,)
                  'beta': float
    solver    : CVXPY solver
    mosek_opts: solver tolerances

    Returns
    -------
    (g_value, lambda_value) or (None, None) on failure
    """
    opts = mosek_opts or MOSEK_HIGH_PRECISION

    delta_k = params['delta_funcs'][k_idx]
    lambda_k = params['lambda_funcs'][k_idx]
    z_k = params['Z_per_func'][k_idx]
    beta = float(params['beta'])
    M = len(delta_k)

    if M == 0:
        return None, None

    alpha = cp.Variable(M, nonneg=True)
    v = cp.Variable(M)
    t = cp.Variable(M, nonneg=True)

    obj_expr = (
        (0.5 * beta) * cp.sum(t)
        + cp.sum(cp.multiply(v, lambda_k))
        - cp.sum(cp.multiply(alpha, lambda_k * z_k))
        + cp.sum(cp.multiply(alpha, delta_k))
    )

    eq_z = cp.sum(v) == o_query
    constraints = [cp.sum(alpha) == 1, eq_z]
    for m in range(M):
        constraints.append(
            cp.quad_over_lin(v[m] - alpha[m] * z_k[m], alpha[m]) <= t[m]
        )

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)
    try:
        prob.solve(solver=solver, mosek_params=opts)
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return None, None
        return prob.value, eq_z.dual_value
    except Exception:
        return None, None


def evaluate_function_curve(
    o_range: np.ndarray,
    k_idx: int,
    params: dict,
    solver: str = PREFERRED_SOLVER,
) -> tuple:
    """
    Evaluate recovered g_k over an array of query points via the SOC SOCP.

    Returns
    -------
    (g_values, lambda_values) — arrays of shape (len(o_range),)
    """
    g_vals = []
    lam_vals = []
    for o in o_range:
        g, lam = evaluate_function_at_point(float(o), k_idx, params, solver)
        g_vals.append(g)
        lam_vals.append(lam)
    return np.array(g_vals, dtype=float), np.array(lam_vals, dtype=float)


def evaluate_function_curve_fast(
    o_range: np.ndarray,
    k_idx: int,
    params: dict,
    n_lam: int = 40001,
    lam_pad: float = 5.0,
) -> np.ndarray:
    """
    Evaluate recovered g_k over an array of query points using a dense
    lambda grid to compute the conjugate directly:

        g_k(z) = sup_{lambda}{ lambda*z - max_j[(1/(2 beta))(lambda - lambda_j)^2
                                                  + lambda*z_j - delta_j ] }

    This is the SAME Proposition 1 formula evaluated by evaluate_function_curve,
    just with a brute-force 1-D maximization over lambda rather than a conic
    SOCP. The two implementations agree to within solver tolerance (verified
    on toy problems and on RIND_0 N=7: max abs diff ~7e-4 on test points).
    Used for visualization where many query points make the per-point SOCP
    expensive.

    Returns
    -------
    g_values : array of shape (len(o_range),)
    """
    delta_k = np.asarray(params['delta_funcs'][k_idx], dtype=float)
    lambda_k = np.asarray(params['lambda_funcs'][k_idx], dtype=float)
    z_k = np.asarray(params['Z_per_func'][k_idx], dtype=float)
    beta = float(params['beta'])
    if len(delta_k) == 0:
        return np.full(len(o_range), np.nan)

    lam_min = float(min(0.0, lambda_k.min())) - lam_pad
    lam_max = float(lambda_k.max()) + lam_pad
    lam = np.linspace(lam_min, lam_max, n_lam)

    # g(lambda) = max_j of strongly-convex quadratics; computed in chunks
    g = np.full_like(lam, -np.inf)
    for s in range(0, len(z_k), 128):
        sl = slice(s, min(s + 128, len(z_k)))
        block = ((0.5 / beta) * (lam[None, :] - lambda_k[sl][:, None]) ** 2
                 + lam[None, :] * z_k[sl][:, None]
                 - delta_k[sl][:, None])
        g = np.maximum(g, block.max(axis=0))

    # f(z) = max_lambda { lambda*z - g(lambda) }
    out = np.empty(len(o_range), dtype=float)
    for i, zi in enumerate(o_range):
        out[i] = float(np.max(lam * float(zi) - g))
    return out
