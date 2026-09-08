"""
inverse/constraints/normalization.py
=====================================
Constraint C7 — anchor-based normalization.

The recovered functions g_k are pinned at two patients:
  delta_global[worst_patient] = U_MAX
  delta_global[best_patient]  = U_MIN   (must be supplied by the caller)

delta_global[i] = sum_k sum_{v in V_k^(i)} delta_k[phi(i,v)]
  = total imputed penalty for patient i across all functions.

The inverse problem is solved purely through the non-parametric
(delta, lambda) representation. The constraint code here NEVER reads
the parametric form of g_k (no alpha, no z^2). Best/worst patient
INDICES are needed to remove the affine ambiguity that C5/C6 leave
in g_k, but the indices themselves are scalars chosen by the caller.

U_MIN must be a non-negative scalar chosen a priori (the paper uses 0).
Forward-objective-derived anchors (e.g., U_MAX * obj_best / obj_worst)
are a form of information leakage from the optimal forward objective
into the inverse problem and are NOT supported by this code; passing
U_MIN=None will raise.
"""

import cvxpy as cp
import numpy as np

from config import N_FUNCTIONS


def build_c7_normalization(
    variables: dict,
    outcome_data: dict,
    norm_settings: dict,
    anchor_indices: dict,
) -> list:
    """
    Build anchor normalization + delta_global linking constraints.

    Parameters
    ----------
    variables      : dict from get_model_variables()
    outcome_data   : dict from extract_per_voxel_outcomes()
    norm_settings  : dict with 'U_MAX'
    anchor_indices : dict with 'best_idx' and 'worst_idx' (indices into patients list)
    """
    u_max = norm_settings['U_MAX']
    delta_funcs = variables['delta_funcs']
    delta_global = variables['delta_global']
    patient_voxel_map = outcome_data['patient_voxel_map']
    N = delta_global.shape[0]

    constraints = []

    # Build delta_global[i] = sum of delta values for patient i's voxels
    # across all functions
    per_patient_maps = _build_per_patient_delta_sum(patient_voxel_map, N)

    for i in range(N):
        expr = 0
        has_terms = False
        for k in range(N_FUNCTIONS):
            obs_indices = per_patient_maps[i].get(k, [])
            if len(obs_indices) > 0:
                obs_indices = np.array(obs_indices, dtype=int)
                expr = expr + cp.sum(delta_funcs[k][obs_indices])
                has_terms = True
        if has_terms:
            constraints.append(delta_global[i] == expr)
        else:
            constraints.append(delta_global[i] == 0)

    # Anchor normalization.
    # delta_global[worst] = U_MAX sets the scale.
    # delta_global[best]  = U_MIN (must be supplied by the caller; the paper
    #                     uses 0). No forward-objective information may enter
    #                     here; passing U_MIN=None is an error.
    best_idx = anchor_indices['best_idx']
    worst_idx = anchor_indices['worst_idx']

    u_min = norm_settings.get('U_MIN', None)
    if u_min is None:
        raise ValueError(
            "norm_settings['U_MIN'] is required. Set U_MIN to a non-negative "
            "scalar chosen a priori (the paper uses 0). The earlier fallback "
            "to U_MAX * (best_obj / worst_obj) has been removed because that "
            "ratio leaks the forward objective value into the inverse problem."
        )
    best_anchor = u_min
    constraints.append(delta_global[best_idx] == best_anchor)
    constraints.append(delta_global[worst_idx] == u_max)

    # Normalization is imposed on the summed values only. A componentwise
    # equality on each f_k is not applied: it follows from the summed form
    # only when the lower anchor is zero, and under exact-objective anchors
    # (U_MIN > 0) it would further restrict the objective class.

    return constraints


def _build_per_patient_delta_sum(patient_voxel_map, N):
    """
    For each patient i and function k, collect the observation indices
    that belong to patient i in function k's observation array.

    Returns list of N dicts: maps[i][k] = list of global observation indices
    """
    maps = [{} for _ in range(N)]
    for k in range(N_FUNCTIONS):
        for global_idx, (pat_idx, oar_key, v_local) in enumerate(patient_voxel_map[k]):
            if k not in maps[pat_idx]:
                maps[pat_idx][k] = []
            maps[pat_idx][k].append(global_idx)
    return maps
