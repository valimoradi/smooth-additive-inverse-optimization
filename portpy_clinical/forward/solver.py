"""
forward/solver.py
=================
Per-voxel forward optimization solver for the IMRT dose planning problem.

Objective (per-voxel penalties on the K=6 OAR components only):
  min  sum_k alpha_k * sum_{v in OAR_k} max(0, d_v - theta_k)^2

Subject to:
  - dose[v] >= PTV_PRESCRIBED  for v in PTV   (hard lower bound)
  - dose[v] <= PTV_MAX         for v in PTV   (hard upper bound)
  - w >= 0

PTV dose is controlled by hard inequality constraints (no slack variables),
matching the formulation in equation (8) of the paper. Earlier drafts of
this solver carried additional soft PTV penalty terms with weights 1e5 / 1e4;
those were dominated by the hard bounds (any feasible plan zeroes them out)
and have been removed for clarity. Numerical results are unchanged.
"""

import cvxpy as cp
import numpy as np

from config import (
    FORWARD_PARAMS, DOSE_THRESHOLDS, PTV_PRESCRIBED, PTV_MAX,
    OAR_KEYS, PREFERRED_SOLVER, MOSEK_TOLERANCES,
)


def solve_forward_problem(
    D,
    organ_indices: dict,
    forward_params: dict = None,
    dose_thresholds: dict = None,
    solver: str = PREFERRED_SOLVER,
    mosek_params: dict = None,
) -> dict:
    """
    Solve the per-voxel forward dose-optimization problem.

    Parameters
    ----------
    D              : dose-influence matrix (n_voxels, n_beamlets), sparse or dense
    organ_indices  : dict with keys 'ptv','bladder','rectum','femur_l','femur_r','skin'
    forward_params : override for FORWARD_PARAMS
    dose_thresholds: override for DOSE_THRESHOLDS
    solver         : preferred CVXPY solver

    Returns
    -------
    dict with 'w', 'obj_value', 'status', 'dose', 'overdose'
    """
    fp = forward_params or FORWARD_PARAMS
    dt = dose_thresholds or DOSE_THRESHOLDS

    n_voxels, n_beamlets = D.shape
    w = cp.Variable(n_beamlets, nonneg=True, name="w")

    dose = D @ w

    constraints = []
    obj_terms = []

    # PTV: hard inequality constraints only (matches eq. (8) in the paper).
    idx_ptv = organ_indices['ptv']
    overdose_vars = {}
    if len(idx_ptv) > 0:
        constraints.append(dose[idx_ptv] >= PTV_PRESCRIBED)
        constraints.append(dose[idx_ptv] <= PTV_MAX)

    # Per-voxel OAR overdose variables and objective terms

    for oar_key in OAR_KEYS:
        idx = organ_indices.get(oar_key, np.array([], dtype=int))
        if len(idx) == 0:
            continue

        alpha = fp[f'alpha_{oar_key}']
        theta = dt[oar_key]

        o_v = cp.Variable(len(idx), nonneg=True, name=f"o_{oar_key}")
        constraints.append(o_v >= dose[idx] - theta)

        # Per-voxel quadratic: alpha * sum_v o_v^2
        obj_terms.append(alpha * cp.sum_squares(o_v))
        overdose_vars[oar_key] = o_v

    objective = cp.Minimize(sum(obj_terms)) if obj_terms else cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    mosek_opts = mosek_params or MOSEK_TOLERANCES
    try:
        problem.solve(solver='MOSEK', verbose=False, mosek_params=mosek_opts)
    except Exception as exc:
        return {'w': None, 'obj_value': None, 'status': f'Failed: {exc}',
                'dose': None, 'overdose': None}

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return {'w': None, 'obj_value': None, 'status': problem.status,
                'dose': None, 'overdose': None}

    w_opt = w.value
    dose_opt = np.asarray(D @ w_opt).ravel()

    # Extract per-voxel overdose values
    overdose = {}
    for oar_key, o_var in overdose_vars.items():
        overdose[oar_key] = o_var.value

    return {
        'w': w_opt,
        'obj_value': problem.value,
        'status': problem.status,
        'dose': dose_opt,
        'overdose': overdose,
    }


def compute_true_penalty(dose_values: np.ndarray, threshold: float, alpha: float) -> np.ndarray:
    """
    Compute the known per-voxel penalty function: g(o_v) = alpha * o_v^2
    where o_v = max(0, d_v - threshold).

    Returns array of penalty values, one per voxel.
    """
    overdose = np.maximum(0, dose_values - threshold)
    return alpha * overdose ** 2


def compute_true_penalty_curve(o_range: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the ground-truth penalty as a function of overdose: g(o) = alpha * o^2.
    Used for plotting comparison against recovered function.
    """
    return alpha * o_range ** 2
