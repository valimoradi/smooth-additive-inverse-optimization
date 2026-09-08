"""
_fast_solver.py
===============
solve_prediction_fast: byte-for-byte the SAME optimization problem as
run_prediction.solve_prediction (paper Proposition 1 / appendix SOC perspective
form), but the per-anchor rotated-SOC constraints are built in ONE vectorized
cp.SOC call over all (voxel, anchor) pairs instead of a Python `for j in
range(M)` loop. For the full-leakage recovery M reaches ~1472 per OAR, and the
loop makes CVXPY canonicalization (single-threaded Python) the dominant cost --
tens of minutes per patient before MOSEK even starts. The cone SET is identical,
so the optimum is identical; only the construction is faster. Verified against
the loop version in _verify_fast.py.
"""
import threading
import numpy as np
import cvxpy as cp

from config import (OAR_KEYS, OAR_GROUPS, DOSE_THRESHOLDS,
                    PTV_PRESCRIBED, PTV_MAX, MOSEK_TOLERANCES)


def thin_envelope(z, delta, lam, M_max):
    """Coarsen a sorted convex tangent set to <= M_max lines by Douglas-Peucker
    simplification of the (z, delta) polyline (endpoints always kept). Removes
    near-collinear tangents whose deletion changes f_k by < the DP tolerance,
    so the beta-smooth conjugate is preserved to that tolerance while the cone
    count (and hence solve memory) drops. Returns thinned (z, delta, lam)."""
    n = len(z)
    if n <= M_max:
        return z, delta, lam
    # binary-search a vertical tolerance that yields <= M_max kept points
    rng = float(delta.max() - delta.min()) or 1.0
    lo, hi = 0.0, rng
    keep = np.ones(n, dtype=bool)
    for _ in range(40):
        tol = 0.5 * (lo + hi)
        kept = _dp_mask(z, delta, tol)
        if kept.sum() > M_max:
            lo = tol
        else:
            hi = tol
            keep = kept
    idx = np.where(keep)[0]
    return z[idx], delta[idx], lam[idx]


def _dp_mask(z, y, tol):
    """Douglas-Peucker mask on polyline (z,y) with vertical tolerance tol."""
    n = len(z)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        za, zb, ya, yb = z[a], z[b], y[a], y[b]
        dz = zb - za
        if abs(dz) < 1e-12:
            continue
        interp = ya + (z[a:b + 1] - za) / dz * (yb - ya)
        err = np.abs(y[a:b + 1] - interp)
        k = int(np.argmax(err))
        if err[k] > tol:
            keep[a + k] = True
            stack.append((a, a + k))
            stack.append((a + k, b))
    return keep


def solve_prediction_fast(D, organ_indices, recovered_funcs, beta_val,
                          verbose=False, mosek_tol=1e-5, timeout=3600.0,
                          thin_m=0):
    n_voxels, n_beamlets = D.shape
    w = cp.Variable(n_beamlets, nonneg=True)
    dose = D @ w

    constraints = []
    obj_terms = []

    idx_ptv = organ_indices['ptv']
    if len(idx_ptv) > 0:
        constraints.append(dose[idx_ptv] >= PTV_PRESCRIBED)
        constraints.append(dose[idx_ptv] <= PTV_MAX)

    BETA = float(beta_val)

    for oar_key in OAR_KEYS:
        idx = organ_indices.get(oar_key, np.array([], dtype=int))
        if len(idx) == 0:
            continue

        func_idx = OAR_GROUPS[oar_key]['func_idx']
        rf = recovered_funcs[func_idx]
        theta = DOSE_THRESHOLDS[oar_key]

        n_v = len(idx)
        M = len(rf['z_hat'])
        z_anchors = np.asarray(rf['z_hat'], dtype=float)
        delta_anchors = np.asarray(rf['delta'], dtype=float)
        lambda_anchors = np.asarray(rf['lam'], dtype=float)

        if thin_m and len(z_anchors) > thin_m:
            z_anchors, delta_anchors, lambda_anchors = thin_envelope(
                z_anchors, delta_anchors, lambda_anchors, thin_m)
            M = len(z_anchors)

        z_v = cp.Variable(n_v, nonneg=True)
        constraints.append(z_v >= dose[idx] - theta)

        alpha = cp.Variable((n_v, M), nonneg=True)
        v_persp = cp.Variable((n_v, M))
        aux = cp.Variable((n_v, M), nonneg=True)

        constraints.append(cp.sum(alpha, axis=1) == 1)
        constraints.append(cp.sum(v_persp, axis=1) >= z_v)

        # Vectorized rotated SOC over ALL (v, j): (v_p - alpha*z_j)^2 <= alpha*aux
        #   <=>  || 2(v_p - alpha z_j) , (alpha - aux) ||_2 <= alpha + aux
        # Identical cones to the per-j loop; flattened C-order so the (v,j)
        # pairing of U, S, T is preserved.
        U = v_persp - cp.multiply(alpha, z_anchors[None, :])   # (n_v, M)
        Uf = cp.reshape(U, (n_v * M,), order='C')
        Sf = cp.reshape(alpha, (n_v * M,), order='C')
        Tf = cp.reshape(aux, (n_v * M,), order='C')
        constraints.append(cp.SOC(Sf + Tf, cp.vstack([2 * Uf, Sf - Tf]), axis=0))

        obj_terms.append(
            (BETA / 2.0) * cp.sum(aux)
            + cp.sum(cp.multiply(v_persp, lambda_anchors[None, :]))
            - cp.sum(cp.multiply(alpha, (lambda_anchors * z_anchors)[None, :]))
            + cp.sum(cp.multiply(alpha, delta_anchors[None, :]))
        )

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
