"""
_lp_solver.py
=============
Light-memory prediction evaluator: the MAX-OF-AFFINES (beta -> infinity) limit of
the recovered penalty. The recovered f_k is the beta-smooth conjugate whose
tangent triples (z_j, delta_j, lambda_j) define the convex envelope
    f_k^inf(z) = max_j { delta_j + lambda_j (z - z_j) }.
The paper's SOC evaluator adds the beta=41 quadratic-perspective smoothing on top;
this LP form drops that smoothing (upper-envelope only). It needs only t_v per
voxel + n_v*M LINEAR constraints (no per-anchor alpha/v_persp/aux matrices), so it
canonicalizes/solves in ~1GB instead of ~50GB -- the only way the largest patients
fit 64GB. Validated against the exact SOC solves in _validate_lp.py before use.
"""
import threading
import numpy as np
import cvxpy as cp

from config import (OAR_KEYS, OAR_GROUPS, DOSE_THRESHOLDS,
                    PTV_PRESCRIBED, PTV_MAX, MOSEK_TOLERANCES)


def solve_prediction_lp(D, organ_indices, recovered_funcs, beta_val=None,
                        verbose=False, mosek_tol=1e-5, timeout=3600.0):
    n_voxels, n_beamlets = D.shape
    w = cp.Variable(n_beamlets, nonneg=True)
    dose = D @ w

    constraints = []
    obj_terms = []

    idx_ptv = organ_indices['ptv']
    if len(idx_ptv) > 0:
        constraints.append(dose[idx_ptv] >= PTV_PRESCRIBED)
        constraints.append(dose[idx_ptv] <= PTV_MAX)

    for oar_key in OAR_KEYS:
        idx = organ_indices.get(oar_key, np.array([], dtype=int))
        if len(idx) == 0:
            continue

        func_idx = OAR_GROUPS[oar_key]['func_idx']
        rf = recovered_funcs[func_idx]
        theta = DOSE_THRESHOLDS[oar_key]

        n_v = len(idx)
        z_anchors = np.asarray(rf['z_hat'], dtype=float)
        delta_anchors = np.asarray(rf['delta'], dtype=float)
        lambda_anchors = np.asarray(rf['lam'], dtype=float)

        z_v = cp.Variable(n_v, nonneg=True)
        constraints.append(z_v >= dose[idx] - theta)

        t_v = cp.Variable(n_v)
        # max-of-affines: t_v[v] >= delta_j + lambda_j*(z_v[v] - z_j)  for all j
        # intercepts b_j = delta_j - lambda_j*z_j ; vectorized (n_v, M) inequality
        b = delta_anchors - lambda_anchors * z_anchors            # (M,)
        constraints.append(
            cp.reshape(t_v, (n_v, 1)) >=
            cp.reshape(z_v, (n_v, 1)) @ lambda_anchors[None, :] + b[None, :]
        )
        obj_terms.append(cp.sum(t_v))

    if not obj_terms:
        return None

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)

    tol = {**MOSEK_TOLERANCES,
           'MSK_DPAR_OPTIMIZER_MAX_TIME': max(timeout * 0.8, 480.0),
           'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   mosek_tol,
           'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   mosek_tol,
           'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': mosek_tol}
    _exc = [None]

    def _solve():
        try:
            prob.solve(solver='MOSEK', verbose=verbose, mosek_params=tol)
        except Exception as e:
            _exc[0] = e

    t = threading.Thread(target=_solve, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None
    if _exc[0] is not None:
        if isinstance(_exc[0], (cp.SolverError, MemoryError)):
            return None
        raise _exc[0]
    if prob.status in ('optimal', 'optimal_inaccurate'):
        return w.value
    return None
