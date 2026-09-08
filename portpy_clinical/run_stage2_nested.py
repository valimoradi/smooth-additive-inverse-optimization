"""Run Stage 2 (bisection) for nested experiment. Uses nested cache files."""
import os, sys, pickle, numpy as np, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MOSEK_TOLERANCES, BISECT_GAP_TOL, BISECT_BETA_LOW
from inverse.outcomes import extract_per_voxel_outcomes
from inverse.feasibility import find_beta_bisect

N = int(sys.argv[1])
beta_init = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
cache_file = f'results/all_patients/forward_cache_nested_N{N}.pkl'
res_dir = sys.argv[3] if len(sys.argv) > 3 else f'results/nested_N{N}'

with open(cache_file, 'rb') as f:
    c = pickle.load(f)
patients, w_sols, obj_vals = c['patients'], c['w_solutions'], c['obj_values']
del c; gc.collect()

with open(f'{res_dir}/outcome_data.pkl', 'rb') as f:
    outcome_data = pickle.load(f)

eps_star = float(np.load(f'{res_dir}/epsilon_star.npy')[0])
eps_s2 = eps_star + 1e-6
print(f"N={N}, eps*={eps_star:.2e}, eps_s2={eps_s2:.2e}")

anchors = {'best_idx': 0, 'worst_idx': 1,
           'best_obj': float(obj_vals[0]), 'worst_obj': float(obj_vals[1])}
ns = {'U_MAX': float(os.environ.get('U_MAX_ANCHOR', '100000.0')), 'U_MIN': float(os.environ.get('U_MIN_ANCHOR', '0.0'))}
print(f"Normalization: U_MIN={ns['U_MIN']}, U_MAX={ns['U_MAX']} (U_MIN_ANCHOR env override; default 0.0)")

print(f"Starting bisection from beta_init={beta_init}")
beta_star = find_beta_bisect(
    outcome_data, patients, w_sols,
    epsilon_val=eps_s2, beta_init=beta_init,
    beta_low=BISECT_BETA_LOW, gap_tol=BISECT_GAP_TOL,
    norm_settings=ns, anchor_indices=anchors,
)

print(f"    (True beta = 2*max(alpha) = 40.0)")
np.save(f'{res_dir}/beta_star.npy', np.array([beta_star]))
print(f"Saved beta_star to {res_dir}/beta_star.npy")
