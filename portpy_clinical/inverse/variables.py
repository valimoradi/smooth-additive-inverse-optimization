"""
inverse/variables.py
====================
CVXPY decision variables for the per-voxel inverse optimization problem.

Per-function variables (one per observation, no binning):
  delta_k   : (M_k,)  function values g_k(z_{k,m})
  lambda_k  : (M_k,)  sub-gradients g_k'(z_{k,m})

Global variable:
  delta_global : (N,)  total imputed penalty per patient
                       delta_global[i] = sum_k sum_{v in V_k^(i)} delta_k[phi(i,v)]

Per-patient dual variables:
  mu_1, mu_2 : PTV dose constraint duals (hard constraints ensure coverage)
  nu_{oar}   : OAR overdose duals
  sigma      : beamlet non-negativity duals
"""

import cvxpy as cp
import numpy as np

from config import N_FUNCTIONS, OAR_KEYS


def get_model_variables(
    outcome_data: dict,
    patients: list[dict],
) -> dict:
    obs_per_func = outcome_data['obs_per_func']
    N = len(patients)
    K = N_FUNCTIONS

    # Per-function primal variables — one per observation (no binning)
    delta_funcs = []
    lambda_funcs = []
    for k in range(K):
        M_k = obs_per_func[k]
        delta_funcs.append(cp.Variable(M_k, nonneg=True, name=f"delta_{k}"))
        lambda_funcs.append(cp.Variable(M_k, nonneg=True, name=f"lambda_{k}"))

    # Global imputed objective per patient
    delta_global = cp.Variable(N, nonneg=True, name="delta_global")

    epsilon = cp.Variable(nonneg=True, name="epsilon")

    # Per-patient dual variables
    dual_lists = {'mu_1': [], 'mu_2': [], 'sigma': []}
    for oar_key in OAR_KEYS:
        dual_lists[f'nu_{oar_key}'] = []

    for i in range(N):
        pdata = patients[i]
        organ_indices = pdata['organ_indices']
        n_b = pdata.get('n_beamlets', pdata['D'].shape[1] if 'D' in pdata else 0)

        n_ptv = len(organ_indices.get('ptv', []))
        if n_ptv > 0:
            dual_lists['mu_1'].append(cp.Variable(n_ptv, nonneg=True, name=f"mu1_{i}"))
            dual_lists['mu_2'].append(cp.Variable(n_ptv, nonneg=True, name=f"mu2_{i}"))
        else:
            dual_lists['mu_1'].append(None)
            dual_lists['mu_2'].append(None)

        for oar_key in OAR_KEYS:
            n_oar = len(organ_indices.get(oar_key, []))
            if n_oar > 0:
                dual_lists[f'nu_{oar_key}'].append(
                    cp.Variable(n_oar, nonneg=True, name=f"nu_{oar_key}_{i}")
                )
            else:
                dual_lists[f'nu_{oar_key}'].append(None)

        dual_lists['sigma'].append(cp.Variable(n_b, nonneg=True, name=f"sigma_{i}"))

    return {
        'delta_funcs': delta_funcs,
        'lambda_funcs': lambda_funcs,
        'delta_global': delta_global,
        'epsilon': epsilon,
        **dual_lists,
    }
