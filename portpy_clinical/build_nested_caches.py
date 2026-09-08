"""
build_nested_caches.py
======================
Build nested forward cache files for N=5,6,7,8 where each larger set
contains all patients from the smaller set plus one additional patient.

N=5: 2 anchors + 3 random patients
N=6: same 5 + 1 more
N=7: same 6 + 1 more
N=8: same 7 + 1 more

Usage:
  python build_nested_caches.py [--seed 42]
"""

import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OAR_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-list', type=str, default='5,6,7,8',
                        help='comma-separated cohort sizes to build (default 5,6,7,8). '
                             'N=2 is anchors-only; N-2 real patients are the prefix of '
                             'the handpicked progression.')
    args = parser.parse_args()
    n_list = [int(x) for x in args.n_list.split(',') if x.strip()]

    # Load full forward cache
    all_cache_file = 'results/all_patients/forward_cache_all.pkl'
    print(f"Loading {all_cache_file} ...")
    with open(all_cache_file, 'rb') as f:
        cache = pickle.load(f)

    all_patients = cache['patients']
    all_w = cache['w_solutions']
    all_obj = np.array(cache['obj_values'])
    print(f"  {len(all_patients)} patients total")

    # Load synthetic anchors from the synthetic cache (built by
    # build_synthetic_anchors.py). The synthetic anchors are contour-modified
    # versions of the best/worst real patients; they serve as the lower and
    # upper anchor rows under the U_MIN=0 / U_MAX=1e5 normalization.
    synth_cache_file = 'results/all_patients/forward_cache_synthetic_N5.pkl'
    print(f"Loading synthetic anchors from {synth_cache_file} ...")
    with open(synth_cache_file, 'rb') as f:
        synth_cache = pickle.load(f)
    synth_best = synth_cache['patients'][0]
    synth_worst = synth_cache['patients'][1]
    synth_best_w = synth_cache['w_solutions'][0]
    synth_worst_w = synth_cache['w_solutions'][1]
    synth_best_obj = float(synth_cache['obj_values'][0])
    synth_worst_obj = float(synth_cache['obj_values'][1])
    real_best_real_id = synth_best['patient_id'].replace('_synthetic_best', '')
    real_worst_real_id = synth_worst['patient_id'].replace('_synthetic_worst', '')
    print(f"  BEST  anchor: {synth_best['patient_id']}  obj={synth_best_obj:.2f}")
    print(f"  WORST anchor: {synth_worst['patient_id']}  obj={synth_worst_obj:.2f}")

    # Filter to complete-data REAL patients (excluding the real patients that
    # the synthetic anchors were derived from, to avoid duplicating their D).
    excluded_ids = {real_best_real_id, real_worst_real_id}
    complete_indices = []
    for i, p in enumerate(all_patients):
        if p['patient_id'] in excluded_ids:
            continue
        has_all = all(
            len(p['organ_indices'].get(oar_key, np.array([], dtype=int))) > 0
            for oar_key in OAR_KEYS
        )
        if has_all:
            complete_indices.append(i)

    print(f"  {len(complete_indices)} complete-data real patients available "
          f"(excluding the source patients of the synthetic anchors)")

    # Handpicked nested progression. These six real patients give a forward-
    # objective spread roughly between 10^7 and 2*10^7 and were chosen
    # because the resulting penalty observations spread the support of each
    # recovered f_k across a wider z-range than a random draw at seed=42 did.
    # Order matters: adding patient j at step N=j+5 is what makes the
    # progression nested (cohort(N) is a prefix of cohort(N+1)).
    HANDPICKED_IDS = [
        'Prostate_Patient_44',   # N=5
        'Prostate_Patient_35',   # N=5
        'Prostate_Patient_20',   # N=5
        'Prostate_Patient_37',   # +1 at N=6
        'Prostate_Patient_5',    # +1 at N=7
        'Prostate_Patient_34',   # +1 at N=8
    ]
    id_to_idx = {all_patients[i]['patient_id']: i for i in complete_indices}
    missing = [pid for pid in HANDPICKED_IDS if pid not in id_to_idx]
    if missing:
        raise RuntimeError(
            f"Handpicked patients not found in complete-data pool: {missing}"
        )
    extra_patients = [id_to_idx[pid] for pid in HANDPICKED_IDS]

    print(f"\nNested patient selection (handpicked, seed arg ignored):")
    for j, idx in enumerate(extra_patients):
        print(f"  extra[{j}]: {all_patients[idx]['patient_id']}  obj={all_obj[idx]:.2f}")

    # Build nested sets: [synth_best, synth_worst, real_3, ..., real_N]
    out_dir = 'results/all_patients'
    os.makedirs(out_dir, exist_ok=True)

    for N in n_list:
        n_extra = N - 2
        extra_idx = extra_patients[:n_extra]

        train_patients = [synth_best, synth_worst] + \
                         [all_patients[i] for i in extra_idx]
        train_w = [synth_best_w, synth_worst_w] + \
                  [all_w[i] for i in extra_idx]
        train_obj = [synth_best_obj, synth_worst_obj] + \
                    [float(all_obj[i]) for i in extra_idx]

        cache_file = f'{out_dir}/forward_cache_nested_N{N}.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'patients': train_patients,
                'w_solutions': train_w,
                'obj_values': train_obj,
            }, f)

        print(f"\nN={N}: saved {cache_file}")
        for pos, (p, o) in enumerate(zip(train_patients, train_obj)):
            tag = (' (SYNTH BEST)' if pos == 0
                   else ' (SYNTH WORST)' if pos == 1 else '')
            print(f"  [{pos}] {p['patient_id']:40s}  obj={o:15.2f}{tag}")

    print("\nAll nested caches built with synthetic anchors.")


if __name__ == '__main__':
    main()
