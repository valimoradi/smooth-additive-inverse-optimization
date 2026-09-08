"""
inverse/outcomes.py
===================
Per-voxel outcome extraction for the inverse optimization model.

For each patient i and each OAR voxel v, the outcome is:
  z_{k,v} = max(0, d_v - theta_k)

These are pooled across patients into per-function outcome sets S_k.
No binning — one delta/lambda variable per observation as per the paper.
"""

import numpy as np

from config import DOSE_THRESHOLDS, OAR_KEYS, OAR_GROUPS, N_FUNCTIONS


def subsample_patient_voxels(patients: list[dict], max_voxels_per_oar: int,
                             seed: int = 42) -> list[dict]:
    """
    Create modified patient dicts with subsampled organ_indices.

    For each patient and each OAR, if the number of voxels exceeds
    max_voxels_per_oar, randomly select a subset. The D matrix,
    dose, w, etc. stay the same — only organ_indices is narrowed.

    This ensures that all downstream code (variables, stationarity, CS,
    linking, C8) sees a consistent smaller problem: fewer nu variables,
    fewer rows in D_oar^T @ nu, fewer observations, all matched.
    """
    rng = np.random.RandomState(seed)
    out = []
    for p in patients:
        new_organ_indices = {}
        for key, idx in p['organ_indices'].items():
            idx = np.asarray(idx)
            if key != 'ptv' and len(idx) > max_voxels_per_oar:
                sel = rng.choice(len(idx), size=max_voxels_per_oar, replace=False)
                sel.sort()
                new_organ_indices[key] = idx[sel]
            else:
                new_organ_indices[key] = idx
        new_p = dict(p)
        new_p['organ_indices'] = new_organ_indices
        out.append(new_p)
    return out


def extract_per_voxel_outcomes(patients: list[dict], w_solutions: list[np.ndarray],
                               max_voxels_per_oar: int = None, seed: int = 42) -> dict:
    """
    Extract per-voxel overdose outcomes from solved patients.

    Parameters
    ----------
    max_voxels_per_oar : if set, subsample organ_indices in each patient
                         to at most this many voxels per OAR. This modifies
                         the patients list in-place so that variables.py and
                         dual.py see the same reduced voxel set.
    seed               : random seed for reproducible subsampling.

    Returns
    -------
    dict with:
      'Z_per_func': list of K arrays, each shape (M_k,)
      'obs_per_func': list of K ints
      'patient_voxel_map': list of K lists of (patient_idx, oar_key, local_voxel_idx)
      'total_obs': int
    """
    outcomes_by_func = [[] for _ in range(N_FUNCTIONS)]
    map_by_func = [[] for _ in range(N_FUNCTIONS)]

    for i, (pdata, w) in enumerate(zip(patients, w_solutions)):
        dose = pdata.get('dose')
        if dose is None:
            dose = np.asarray(pdata['D'] @ w).ravel()
        organ_indices = pdata['organ_indices']

        for oar_key in OAR_KEYS:
            idx = organ_indices.get(oar_key, np.array([], dtype=int))
            if len(idx) == 0:
                continue

            func_idx = OAR_GROUPS[oar_key]['func_idx']
            theta = DOSE_THRESHOLDS[oar_key]

            overdose = np.maximum(0, dose[idx] - theta)

            for v_local, o_val in enumerate(overdose):
                outcomes_by_func[func_idx].append(float(o_val))
                map_by_func[func_idx].append((i, oar_key, v_local))

    obs_per_func = [len(outcomes_by_func[k]) for k in range(N_FUNCTIONS)]

    Z_per_func = []
    for k in range(N_FUNCTIONS):
        Z_per_func.append(np.array(outcomes_by_func[k]))

    return {
        'Z_per_func': Z_per_func,
        'obs_per_func': obs_per_func,
        'patient_voxel_map': map_by_func,
        'total_obs': sum(obs_per_func),
    }


def find_anchor_patients(
    all_patients: list[dict],
    all_w_solutions: list[np.ndarray],
    all_obj_values: list[float],
) -> dict:
    """
    Find best-case (lowest obj) and worst-case (highest obj) patients.
    These anchor the normalization: delta_global[best] = 0, delta_global[worst] = U_MAX.
    """
    obj_arr = np.array(all_obj_values)
    best_idx = int(np.argmin(obj_arr))
    worst_idx = int(np.argmax(obj_arr))

    return {
        'best_idx': best_idx,
        'worst_idx': worst_idx,
        'best_obj': float(obj_arr[best_idx]),
        'worst_obj': float(obj_arr[worst_idx]),
    }
