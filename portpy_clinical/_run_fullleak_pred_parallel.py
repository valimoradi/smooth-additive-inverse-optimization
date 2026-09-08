"""
Parallel driver: score the full-leakage (exact U_min/U_max anchor) recovered
model's PREDICTIONS on the canonical 20 test patients, N=5. Reuses the exact
solve_prediction / recovery-loading from run_prediction.py (unchanged), but caps
MOSEK threads and runs a user-specified subset of patient indices so several
processes can share the machine. Each process writes its own <pid>.npz.

Usage:  python _run_fullleak_pred_parallel.py 0,4,8,12,16
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_prediction as rp
from _fast_solver import solve_prediction_fast

# Cap MOSEK threads so N parallel processes don't oversubscribe 32 cores.
NTHREADS = int(os.environ.get('MOSEK_THREADS', '7'))
rp.MOSEK_TOLERANCES = {**rp.MOSEK_TOLERANCES, 'MSK_IPAR_NUM_THREADS': NTHREADS}
import config
config.MOSEK_TOLERANCES['MSK_IPAR_NUM_THREADS'] = NTHREADS  # picked up by _fast_solver

N = int(os.environ.get('FL_N', '5'))
RES_DIR = os.environ.get('FL_RES_DIR', f'results/nested_N{N}_fullleak_anchor')
DOSE_DIR = os.environ.get('FL_DOSE_DIR', f'results/prediction_nested_fullleak_anchor/doses_N{N}')
CACHE = 'results/all_patients/forward_cache_all.pkl'

indices = [int(x) for x in sys.argv[1].split(',')]

recovered = rp.load_recovered_functions(RES_DIR, N)

# Optional exact anchor dedup: drop envelope lines the solver cannot resolve.
# FL_DEDUP is a RELATIVE tolerance on delta (e.g. 1e-5 = MOSEK's own tol).
# Unset/0 -> no dedup, behaviour identical to before.
_dedup = float(os.environ.get('FL_DEDUP', '0') or 0)
if _dedup > 0:
    from _dedup_anchors import dedup_recovered
    recovered = dedup_recovered(recovered, _dedup, verbose=True)
beta = rp.load_recovered_beta(RES_DIR)

training_ids = rp.get_all_training_ids('nested_fullleak_anchor')
with open('test_patients_canonical.txt') as f:
    fixed = [l.strip() for l in f if l.strip()]
patients, w_sols, _ = rp.select_test_patients(CACHE, training_ids, 20, 123,
                                              fixed_test_ids=fixed)
os.makedirs(DOSE_DIR, exist_ok=True)

# small -> big so the memory-heavy solves land last (few concurrent)
indices = sorted(indices, key=lambda i: np.linalg.norm(w_sols[i]))

for i in indices:
    p = patients[i]; wt = w_sols[i]
    out = f'{DOSE_DIR}/{p["patient_id"]}.npz'
    if os.path.exists(out):
        print(f'[{i}] {p["patient_id"]} exists -> skip', flush=True); continue
    t0 = time.time()
    wp = solve_prediction_fast(p['D'], p['organ_indices'], recovered,
                               beta_val=beta, mosek_tol=1e-5, timeout=21600.0,
                               thin_m=int(os.environ.get('THIN_M', '0')))
    dt = time.time() - t0
    if wp is None:
        print(f'[{i}] {p["patient_id"]} FAILED ({dt:.0f}s)', flush=True); continue
    d_true = p['D'] @ wt; d_pred = p['D'] @ wp
    rel = np.linalg.norm(wp - wt) / max(np.linalg.norm(wt), 1e-10)
    np.savez(out, d_true=d_true, d_pred=d_pred, w_true=wt, w_pred=wp,
             ptv_idx=p['organ_indices'].get('ptv', np.array([], dtype=int)))
    print(f'[{i}] {p["patient_id"]} OK ({dt:.0f}s, rel_err={rel:.4f})', flush=True)

print(f'DONE indices={indices}', flush=True)
