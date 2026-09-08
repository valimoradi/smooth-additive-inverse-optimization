"""Run Stage 1 for nested experiment. Uses nested cache files."""
import os, sys, pickle, numpy as np, cvxpy as cp, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MOSEK_TOLERANCES, N_FUNCTIONS
from inverse.outcomes import extract_per_voxel_outcomes
from inverse.variables import get_model_variables
from inverse.constraints.assembler import build_constraints

N = int(sys.argv[1])
cache_file = f'results/all_patients/forward_cache_nested_N{N}.pkl'
res_dir = sys.argv[2] if len(sys.argv) > 2 else f'results/nested_N{N}'
os.makedirs(res_dir, exist_ok=True)

with open(cache_file, 'rb') as f:
    c = pickle.load(f)
patients, w_sols, obj_vals = c['patients'], c['w_solutions'], c['obj_values']
del c; gc.collect()

outcome_data = extract_per_voxel_outcomes(patients, w_sols)
print(f"Total obs: {outcome_data['total_obs']}")

variables = get_model_variables(outcome_data, patients)
anchors = {'best_idx': 0, 'worst_idx': 1,
           'best_obj': float(obj_vals[0]), 'worst_obj': float(obj_vals[1])}
ns = {'U_MAX': float(os.environ.get('U_MAX_ANCHOR', '100000.0')), 'U_MIN': float(os.environ.get('U_MIN_ANCHOR', '0.0'))}
print(f"Normalization: U_MIN={ns['U_MIN']}, U_MAX={ns['U_MAX']} (U_MIN_ANCHOR env override; default 0.0)")

cons = build_constraints(variables, outcome_data, patients, w_sols,
                         beta_val=40.0, epsilon_val=variables['epsilon'],
                         norm_settings=ns, anchor_indices=anchors)
prob = cp.Problem(cp.Minimize(variables['epsilon']), cons)

tol = {**MOSEK_TOLERANCES, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 7200.0}
prob.solve(solver='MOSEK', verbose=True, mosek_params=tol)

eps = float(variables['epsilon'].value)
print(f"Stage 1 status: {prob.status}  eps={eps:.2e}")
np.save(f'{res_dir}/epsilon_star.npy', np.array([eps]))

with open(f'{res_dir}/outcome_data.pkl', 'wb') as f:
    pickle.dump(outcome_data, f)
print("Stage 1 saved.")
