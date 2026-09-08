"""
build_synthetic_anchors.py
==========================
Create synthetic anchor patients by modifying real patient contours.

Worst anchor (highest penalty):
  - Take Patient_6 (already highest real penalty)
  - Expand PTV by adding RIND_0 voxels (physically closest to PTV)
  - Expand OAR contours by adding PTV voxels to each OAR

Best anchor (lowest penalty):
  - Take Patient_61 (already lowest real penalty)
  - Shrink PTV by removing boundary voxels (those overlapping RIND_0)
  - Contract OAR contours by removing voxels that overlap with PTV

D matrix stays real (untouched). Only organ_indices are modified.
Normalization: U_MIN = epsilon (no ratio needed), U_MAX = constant.

Usage:
  python build_synthetic_anchors.py [--ptv-expand-frac 0.5] [--oar-expand-frac 0.5]
"""

import os, sys, argparse, pickle, copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OAR_KEYS


def expand_patient_for_worst(patient, ptv_expand_frac=0.5, oar_expand_frac=0.5):
    """
    Modify a patient to push penalty toward maximum.

    1. Expand PTV by adding a fraction of RIND_0 voxels
    2. Expand each OAR by adding a fraction of PTV voxels
    """
    p = copy.deepcopy(patient)
    oi = p['organ_indices']

    ptv_set = set(oi['ptv'])
    rind0_set = set(oi.get('rind_0', []))

    # 1. Expand PTV: add RIND_0 voxels not already in PTV
    rind0_not_ptv = sorted(rind0_set - ptv_set)
    n_add_ptv = int(len(rind0_not_ptv) * ptv_expand_frac)
    if n_add_ptv > 0:
        new_ptv_voxels = rind0_not_ptv[:n_add_ptv]
        oi['ptv'] = np.concatenate([oi['ptv'], np.array(new_ptv_voxels, dtype=int)])
        print(f"  PTV expanded: {len(ptv_set)} -> {len(oi['ptv'])} "
              f"(+{n_add_ptv} from RIND_0)")

    # Update PTV set after expansion
    ptv_set_expanded = set(oi['ptv'])

    # 2. Expand each OAR by adding PTV voxels
    for oar_key in OAR_KEYS:
        oar_idx = oi.get(oar_key, np.array([], dtype=int))
        oar_set = set(oar_idx)

        # PTV voxels not already in this OAR
        ptv_not_oar = sorted(ptv_set_expanded - oar_set)
        n_add = int(len(ptv_not_oar) * oar_expand_frac)
        if n_add > 0:
            new_voxels = ptv_not_oar[:n_add]
            oi[oar_key] = np.concatenate([oar_idx, np.array(new_voxels, dtype=int)])
            print(f"  {oar_key} expanded: {len(oar_set)} -> {len(oi[oar_key])} "
                  f"(+{n_add} PTV voxels)")

    p['patient_id'] = p['patient_id'] + '_synthetic_worst'
    return p


def contract_patient_for_best(patient, ptv_shrink_frac=0.3,
                              oar_dose_remove_frac=0.5):
    """
    Modify a patient to push penalty toward zero.

    1. Shrink PTV by removing boundary voxels (those overlapping RIND_0)
    2. Contract each OAR by removing voxels with highest D row norms
       (these are in the beam path and will receive dose > threshold)

    The remaining OAR voxels are far from the beam → receive dose ≤ threshold → penalty ≈ 0.
    """
    p = copy.deepcopy(patient)
    oi = p['organ_indices']
    D = p['D']

    if hasattr(D, 'toarray'):
        D_arr = D.toarray()
    else:
        D_arr = np.asarray(D)

    ptv_set = set(oi['ptv'])
    rind0_set = set(oi.get('rind_0', []))

    # 1. Shrink PTV: remove boundary voxels (overlap with RIND_0)
    ptv_boundary = sorted(ptv_set & rind0_set)
    n_remove_ptv = int(len(ptv_boundary) * ptv_shrink_frac)
    if n_remove_ptv > 0:
        norms = np.linalg.norm(D_arr[ptv_boundary], axis=1)
        remove_order = np.argsort(norms)[:n_remove_ptv]
        remove_set = set(np.array(ptv_boundary)[remove_order])
        new_ptv = sorted(ptv_set - remove_set)
        print(f"  PTV shrunk: {len(ptv_set)} -> {len(new_ptv)} "
              f"(-{n_remove_ptv} boundary voxels)")
        oi['ptv'] = np.array(new_ptv, dtype=int)

    # 2. Contract each OAR: remove voxels with highest D row norms
    #    These are in the beam path and will receive high dose.
    #    Keeping only low-D-norm voxels pushes penalty toward zero.
    for oar_key in OAR_KEYS:
        oar_idx = oi.get(oar_key, np.array([], dtype=int))
        if len(oar_idx) == 0:
            continue
        oar_set = set(oar_idx)
        oar_list = np.array(sorted(oar_set))

        # Sort by D row norm (descending) and remove the top fraction
        norms = np.linalg.norm(D_arr[oar_list], axis=1)
        n_remove = int(len(oar_list) * oar_dose_remove_frac)

        # Keep at least 5 voxels per OAR
        n_keep = max(5, len(oar_list) - n_remove)
        keep_indices = np.argsort(norms)[:n_keep]  # lowest D norms
        new_oar = oar_list[keep_indices]

        print(f"  {oar_key} contracted: {len(oar_set)} -> {len(new_oar)} "
              f"(-{len(oar_set) - len(new_oar)} high-dose voxels)")
        oi[oar_key] = np.sort(new_oar)

    p['patient_id'] = p['patient_id'] + '_synthetic_best'
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptv-expand-frac', type=float, default=0.5,
                        help='Fraction of RIND_0 voxels to add to worst PTV')
    parser.add_argument('--oar-expand-frac', type=float, default=0.5,
                        help='Fraction of PTV voxels to add to worst OARs')
    parser.add_argument('--ptv-shrink-frac', type=float, default=0.3,
                        help='Fraction of boundary PTV voxels to remove for best')
    parser.add_argument('--oar-dose-remove-frac', type=float, default=0.5,
                        help='Fraction of high-dose OAR voxels to remove for best')
    parser.add_argument('--n-train', type=int, default=8,
                        help='Total training size including 2 synthetic anchors')
    parser.add_argument('--seed', type=int, default=77)
    args = parser.parse_args()

    # Load full forward cache
    all_cache_file = 'results/all_patients/forward_cache_all.pkl'
    print(f"Loading {all_cache_file} ...")
    with open(all_cache_file, 'rb') as f:
        cache = pickle.load(f)

    all_patients = cache['patients']
    all_w = cache['w_solutions']
    all_obj = np.array(cache['obj_values'])

    # Filter to complete-data patients
    complete_indices = []
    for i, p in enumerate(all_patients):
        has_all = all(
            len(p['organ_indices'].get(k, np.array([], dtype=int))) > 0
            for k in OAR_KEYS
        )
        if has_all:
            complete_indices.append(i)
    print(f"  {len(complete_indices)} complete-data patients")

    # Find current best/worst among complete patients
    complete_obj = np.array([all_obj[i] for i in complete_indices])
    best_real_idx = complete_indices[int(np.argmin(complete_obj))]
    worst_real_idx = complete_indices[int(np.argmax(complete_obj))]

    print(f"\n  Real best:  {all_patients[best_real_idx]['patient_id']}  "
          f"obj={all_obj[best_real_idx]:.0f}")
    print(f"  Real worst: {all_patients[worst_real_idx]['patient_id']}  "
          f"obj={all_obj[worst_real_idx]:.0f}")

    # Create synthetic anchors
    print(f"\n--- Creating synthetic BEST anchor ---")
    best_synthetic = contract_patient_for_best(
        all_patients[best_real_idx],
        ptv_shrink_frac=args.ptv_shrink_frac,
        oar_dose_remove_frac=args.oar_dose_remove_frac,
    )

    print(f"\n--- Creating synthetic WORST anchor ---")
    worst_synthetic = expand_patient_for_worst(
        all_patients[worst_real_idx],
        ptv_expand_frac=args.ptv_expand_frac,
        oar_expand_frac=args.oar_expand_frac,
    )

    # Solve forward for synthetic anchors to get w* and obj
    print(f"\n--- Solving forward for synthetic anchors ---")
    from forward.solver import solve_forward_problem

    best_result = solve_forward_problem(
        best_synthetic['D'], best_synthetic['organ_indices'])
    worst_result = solve_forward_problem(
        worst_synthetic['D'], worst_synthetic['organ_indices'])

    print(f"  Best synthetic:  status={best_result['status']}, "
          f"obj={best_result['obj_value']:.0f}")
    print(f"  Worst synthetic: status={worst_result['status']}, "
          f"obj={worst_result['obj_value']:.0f}")

    best_synthetic['dose'] = best_result['dose']
    worst_synthetic['dose'] = worst_result['dose']

    # Select non-anchor training patients
    anchor_real_ids = {all_patients[best_real_idx]['patient_id'],
                       all_patients[worst_real_idx]['patient_id']}
    pool = [i for i in complete_indices
            if all_patients[i]['patient_id'] not in anchor_real_ids]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(pool)

    # Build caches for N=5,6,7,8
    out_dir = 'results/all_patients'
    os.makedirs(out_dir, exist_ok=True)

    for N in [5, 6, 7, 8]:
        n_others = N - 2  # 2 anchors
        selected = pool[:n_others]

        # [best_synthetic, worst_synthetic, ...others...]
        train_patients = [best_synthetic, worst_synthetic] + \
                         [all_patients[i] for i in selected]
        train_w = [best_result['w'], worst_result['w']] + \
                  [all_w[i] for i in selected]
        train_obj = [float(best_result['obj_value']),
                     float(worst_result['obj_value'])] + \
                    [float(all_obj[i]) for i in selected]

        cache_file = f'{out_dir}/forward_cache_synthetic_N{N}.pkl'
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

    # Save synthetic anchor metadata
    meta = {
        'best_real_patient': all_patients[best_real_idx]['patient_id'],
        'worst_real_patient': all_patients[worst_real_idx]['patient_id'],
        'best_synthetic_obj': float(best_result['obj_value']),
        'worst_synthetic_obj': float(worst_result['obj_value']),
        'best_real_obj': float(all_obj[best_real_idx]),
        'worst_real_obj': float(all_obj[worst_real_idx]),
        'ptv_expand_frac': args.ptv_expand_frac,
        'oar_expand_frac': args.oar_expand_frac,
        'ptv_shrink_frac': args.ptv_shrink_frac,
        'oar_dose_remove_frac': args.oar_dose_remove_frac,
    }
    meta_file = f'{out_dir}/synthetic_anchor_meta.pkl'
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"\nSaved metadata to {meta_file}")
    print("Done.")


if __name__ == '__main__':
    main()
