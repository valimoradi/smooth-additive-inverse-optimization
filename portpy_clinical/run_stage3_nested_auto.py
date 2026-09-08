"""
Run Stage 3 for nested experiment with auto-retry.

Two layers of fallback:
  1. Per-beta tolerance walk: at each beta, try MOSEK at PFEAS=1e-5,
     then 1e-4, then 1e-3 before giving up on that beta. Mirrors
     inverse.feasibility.is_feasible (Stage 2).
  2. Beta escalation: if all tolerance levels fail at a given beta,
     multiply beta by 1.2 and try again, up to max_attempts times.

Uses nested cache files.
"""
import os, sys, pickle, numpy as np, cvxpy as cp, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MOSEK_TOLERANCES, N_FUNCTIONS, FUNC_LABELS, FORWARD_PARAMS
from inverse.variables import get_model_variables
from inverse.constraints.assembler import build_constraints

N = int(sys.argv[1])
cache_file = f'results/all_patients/forward_cache_nested_N{N}.pkl'
res_dir = sys.argv[2] if len(sys.argv) > 2 else f'results/nested_N{N}'
beta_start = float(sys.argv[3]) if len(sys.argv) > 3 else None
max_attempts = int(sys.argv[4]) if len(sys.argv) > 4 else 10

with open(cache_file, 'rb') as f:
    c = pickle.load(f)
patients, w_sols, obj_vals = c['patients'], c['w_solutions'], c['obj_values']
del c; gc.collect()

with open(f'{res_dir}/outcome_data.pkl', 'rb') as f:
    outcome_data = pickle.load(f)

eps_star = float(np.load(f'{res_dir}/epsilon_star.npy')[0])
eps_s3 = eps_star + 1e-6
print(f"eps_star from Stage 1: {eps_star:.2e}")

anchors = {'best_idx': 0, 'worst_idx': 1,
           'best_obj': float(obj_vals[0]), 'worst_obj': float(obj_vals[1])}
ns = {'U_MAX': float(os.environ.get('U_MAX_ANCHOR', '100000.0')), 'U_MIN': float(os.environ.get('U_MIN_ANCHOR', '0.0'))}
print(f"Normalization: U_MIN={ns['U_MIN']}, U_MAX={ns['U_MAX']} (U_MIN_ANCHOR env override; default 0.0)")

# Use beta from Stage 2 if available
if beta_start is None:
    beta_file = f'{res_dir}/beta_star.npy'
    if os.path.exists(beta_file):
        beta_start = float(np.load(beta_file)[0])
        print(f"Using beta* from Stage 2: {beta_start:.4f}")
    else:
        beta_start = 40.0
        print(f"Using default beta: {beta_start}")

TOL_LEVELS = [
    {**MOSEK_TOLERANCES, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 7200.0},
    {**MOSEK_TOLERANCES, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 7200.0,
     'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   1e-4,
     'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   1e-4,
     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-4},
    {**MOSEK_TOLERANCES, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 7200.0,
     'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   1e-3,
     'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   1e-3,
     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-3},
]

beta = beta_start
for attempt in range(1, max_attempts + 1):
    print(f"\n=== Attempt {attempt}: beta={beta:.4f} ===")

    variables_ok = None
    last_status = None
    for tol_idx, tol in enumerate(TOL_LEVELS):
        if tol_idx > 0:
            print(f"  -- tolerance fallback level {tol_idx + 1} "
                  f"(PFEAS={tol['MSK_DPAR_INTPNT_CO_TOL_PFEAS']:.0e})")
        try:
            variables = get_model_variables(outcome_data, patients)
            cons = build_constraints(variables, outcome_data, patients, w_sols,
                                     beta_val=beta, epsilon_val=eps_s3,
                                     norm_settings=ns, anchor_indices=anchors)
            prob = cp.Problem(cp.Maximize(cp.sum(variables['delta_global'])), cons)
            prob.solve(solver='MOSEK', verbose=True, mosek_params=tol)
            last_status = prob.status

            if prob.status in ('optimal', 'optimal_inaccurate'):
                variables_ok = variables
                if tol_idx > 0:
                    print(f"  Stage 3 succeeded at tol level {tol_idx + 1}, "
                          f"status={prob.status}")
                break
            else:
                print(f"  status={prob.status} at tol level {tol_idx + 1}")
                del variables, cons, prob
                gc.collect()
        except cp.SolverError as e:
            print(f"  SolverError at tol level {tol_idx + 1}: {e}")
            try:
                del variables, cons, prob
            except NameError:
                pass
            gc.collect()

    if variables_ok is not None:
        print(f"\nStage 3 SUCCEEDED with beta={beta:.4f}, status={last_status}")
        np.save(f'{res_dir}/beta_used.npy', np.array([beta]))

        for k in range(N_FUNCTIONS):
            np.save(f'{res_dir}/delta_func{k}_N{N}.npy', variables_ok['delta_funcs'][k].value)
            np.save(f'{res_dir}/lambda_func{k}_N{N}.npy', variables_ok['lambda_funcs'][k].value)
            np.save(f'{res_dir}/Z_func{k}_N{N}.npy', outcome_data['Z_per_func'][k])

        print(f"\neps={eps_star:.2e}, beta_used={beta:.4f}")
        for k in range(N_FUNCTIONS):
            z = outcome_data['Z_per_func'][k]
            d = variables_ok['delta_funcs'][k].value
            if d is None:
                continue
            m = z > 0
            afit = np.sum(d[m]*z[m]**2)/np.sum(z[m]**4) if m.sum() > 5 else np.nan
            pred = afit * z**2
            ss_res = np.sum((d - pred)**2)
            ss_tot = np.sum((d - d.mean())**2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            print(f"  {FUNC_LABELS[k]:15s}: R2={r2:.4f}  alpha_fit={afit:.4f}")
        print("Stage 3 saved.")
        sys.exit(0)

    print(f"All tolerance levels failed at beta={beta:.4f} -- increasing beta by 20%")
    beta *= 1.2

print(f"FAILED after {max_attempts} attempts (last beta={beta/1.2:.4f})")
sys.exit(1)
