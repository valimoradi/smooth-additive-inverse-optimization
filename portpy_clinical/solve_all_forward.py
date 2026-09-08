"""
solve_all_forward.py
====================
Forward-solve ALL available patients and save a comprehensive cache.
Identifies the patients with lowest and highest forward objectives
for use as dedicated anchor patients in the inverse optimization.
"""

import os
import sys
import time
import gc
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PORTPY_DATA_DIR, FORWARD_PARAMS, DOSE_THRESHOLDS,
    OAR_KEYS, PREFERRED_SOLVER, MOSEK_TOLERANCES,
)
from data.portpy_loader import list_available_patients, load_patients
from forward.solver import solve_forward_problem


def main():
    results_dir = 'results/all_patients'
    os.makedirs(results_dir, exist_ok=True)
    cache_file = os.path.join(results_dir, 'forward_cache_all.pkl')

    # Load existing progress if any
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        solved_ids = set(p['patient_id'] for p in cache['patients'])
        patients_valid = cache['patients']
        w_valid = cache['w_solutions']
        obj_valid = cache['obj_values']
        print(f"Resuming: {len(patients_valid)} patients already solved.")
    else:
        solved_ids = set()
        patients_valid = []
        w_valid = []
        obj_valid = []

    all_patient_ids = list_available_patients(PORTPY_DATA_DIR)
    print(f"Total available: {len(all_patient_ids)}")
    print(f"Already solved:  {len(solved_ids)}")
    print(f"Remaining:       {len(all_patient_ids) - len(solved_ids)}")

    failed = []
    for idx, pid in enumerate(all_patient_ids):
        if pid in solved_ids:
            continue

        print(f"\n[{idx+1}/{len(all_patient_ids)}] {pid} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            loaded = load_patients([pid], PORTPY_DATA_DIR)
            if not loaded:
                print("SKIP (load failed)")
                failed.append((pid, "load failed"))
                continue
            pdata = loaded[0]
        except Exception as exc:
            print(f"SKIP (load exception: {exc})")
            failed.append((pid, f"load: {exc}"))
            gc.collect()
            continue

        try:
            result = solve_forward_problem(pdata['D'], pdata['organ_indices'])
        except Exception as exc:
            print(f"SKIP (solve exception: {exc})")
            failed.append((pid, f"solve: {exc}"))
            gc.collect()
            continue

        if result.get('w') is None:
            print(f"FAILED ({result.get('status', 'unknown')})")
            failed.append((pid, result.get('status', 'unknown')))
            gc.collect()
            continue

        elapsed = time.time() - t0
        w_opt = result['w']
        dose = np.asarray(pdata['D'] @ w_opt).ravel()
        pdata['dose'] = dose
        pdata['n_beamlets'] = pdata['D'].shape[1]

        # Compute forward objective
        obj_i = 0
        for oar_key in OAR_KEYS:
            oar_idx = pdata['organ_indices'].get(oar_key, np.array([], dtype=int))
            if len(oar_idx) == 0:
                continue
            alpha = FORWARD_PARAMS[f'alpha_{oar_key}']
            theta = DOSE_THRESHOLDS[oar_key]
            overdose = np.maximum(0, dose[oar_idx] - theta)
            obj_i += alpha * float(np.sum(overdose ** 2))

        print(f"OK  obj={obj_i:.2f}  ({elapsed:.1f}s)")

        patients_valid.append(pdata)
        w_valid.append(w_opt)
        obj_valid.append(obj_i)
        solved_ids.add(pid)

        # Save progress every 5 patients
        if len(patients_valid) % 5 == 0:
            _save_cache(cache_file, patients_valid, w_valid, obj_valid)
            print(f"  [checkpoint: {len(patients_valid)} patients saved]")

        gc.collect()

    # Final save
    _save_cache(cache_file, patients_valid, w_valid, obj_valid)

    # Report
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(patients_valid)} solved, {len(failed)} failed")
    print(f"{'='*70}")

    if obj_valid:
        obj_arr = np.array(obj_valid)
        idx_min = int(np.argmin(obj_arr))
        idx_max = int(np.argmax(obj_arr))

        print(f"\nObjective range: [{obj_arr.min():.2f}, {obj_arr.max():.2f}]")
        print(f"  LOWEST:  patient {idx_min} = {patients_valid[idx_min]['patient_id']}  "
              f"obj={obj_arr[idx_min]:.2f}")
        print(f"  HIGHEST: patient {idx_max} = {patients_valid[idx_max]['patient_id']}  "
              f"obj={obj_arr[idx_max]:.2f}")

        # Save summary
        summary = {
            'patient_ids': [p['patient_id'] for p in patients_valid],
            'obj_values': obj_valid,
            'best_idx': idx_min,
            'worst_idx': idx_max,
            'best_id': patients_valid[idx_min]['patient_id'],
            'worst_id': patients_valid[idx_max]['patient_id'],
        }
        with open(os.path.join(results_dir, 'summary.pkl'), 'wb') as f:
            pickle.dump(summary, f)

        # Print sorted list
        order = np.argsort(obj_arr)
        print(f"\nAll patients sorted by objective:")
        for rank, i in enumerate(order):
            tag = ''
            if i == idx_min:
                tag = ' <-- LOWEST'
            elif i == idx_max:
                tag = ' <-- HIGHEST'
            print(f"  {rank+1:3d}. {patients_valid[i]['patient_id']:30s}  "
                  f"obj={obj_arr[i]:15.2f}{tag}")

    if failed:
        print(f"\nFailed patients:")
        for pid, reason in failed:
            print(f"  {pid}: {reason}")


def _save_cache(cache_file, patients, w_sols, obj_vals):
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'patients': patients,
            'w_solutions': w_sols,
            'obj_values': obj_vals,
        }, f)


if __name__ == '__main__':
    main()
