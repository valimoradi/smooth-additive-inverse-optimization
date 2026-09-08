"""
Data pipeline: aggregate the per-patient cohort, apply the UNIFIED 5-check validity
filter, split into train/test, and prepend the two anchors (best/worst case) after
validating them with the SAME 5-check filter.

This unifies the filtering: in the original notebook the cohort went through a 5-check
filter (check_patient_validity) while the anchors only passed a 2-check anatomy filter
during search. Here a single check_patient_validity is the sole gate for ALL data --
cohort and anchors alike. The two anchors pass it (verified), so the produced ml_data
is identical to the paper's; the code is now consistent.
"""
import os
import hashlib
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from config import (COHORT_DIR, ML_DIR, BEST_CASE_DIR, WORST_CASE_DIR,
                    STD_DEV_THRESHOLD, MIN_VOXEL_COUNT, MAX_OVERLAP_PCT, ORGAN_NAMES,
                    TEST_SET_SIZE, SPLIT_RANDOM_STATE)
from forward_model import calculate_objective


# --------------------------------------------------------------------------------------
# The single, unified validity filter (5 checks)
# --------------------------------------------------------------------------------------
def get_population_stats(organ_sets, organ_names=ORGAN_NAMES):
    stats = {}
    for organ in organ_names:
        sizes = [len(p[organ]) for p in organ_sets]
        stats[organ] = {'mean': float(np.mean(sizes)), 'std': float(np.std(sizes))}
    return stats


def check_patient_validity(patient, stats, organ_names=ORGAN_NAMES, known_hashes=None):
    """Return (is_valid, reason). The 5 checks: (1) duplicate anatomy hash, (2) empty/
    micro-organ, (3) 3-sigma organ-size outlier, (4) PTV-OAR overlap, (5) bladder>=0.5*PTV."""
    if known_hashes is None:
        known_hashes = set()
    ptv_indices = np.array(patient['ptv'], dtype=np.int32)
    ptv_set = set(ptv_indices.tolist())

    # 1. Duplicate anatomy (exact PTV signature)
    ptv_sig = hashlib.sha1(np.sort(ptv_indices).tobytes()).hexdigest()
    if ptv_sig in known_hashes:
        return False, "Duplicate Anatomy (Exact Copy)"
    known_hashes.add(ptv_sig)

    # 2. Empty / micro-organ
    for organ in organ_names:
        size = len(patient[organ])
        if size == 0:
            return False, f"Empty {organ}"
        if size < MIN_VOXEL_COUNT:
            return False, f"{organ} too small (biological noise): {size} voxels"

    # 3. Statistical outliers (3-sigma on organ size)
    for organ in organ_names:
        size = len(patient[organ])
        mean = stats[organ]['mean']; std = stats[organ]['std']
        z_score = abs(size - mean) / (std + 1e-6)
        if z_score > STD_DEV_THRESHOLD:
            return False, f"{organ} statistical outlier (Z={z_score:.1f})"

    # 4. Overlap geometry (PTV vs OARs)
    for oar in ['oar1', 'oar2', 'oar3']:
        oar_set = set(patient[oar])
        if len(ptv_set) > 0:
            overlap_pct = (len(ptv_set & oar_set) / len(ptv_set)) * 100.0
            if overlap_pct > MAX_OVERLAP_PCT:
                return False, f"Impossible Overlap: PTV intersects {oar} by {overlap_pct:.1f}%"

    # 5. Relative size: bladder (OAR1) not drastically smaller than prostate (PTV)
    if len(patient['oar1']) < len(patient['ptv']):
        if len(patient['oar1']) < 0.5 * len(patient['ptv']):
            return False, "Anatomy Error: Bladder (OAR1) significantly smaller than Prostate"

    return True, "Valid"


def clean_dataset(data, organ_names=ORGAN_NAMES, verbose=True):
    """Apply the 5-check filter to every patient in the aggregated cohort."""
    organ_sets = data['organs']
    baseline_stats = get_population_stats(organ_sets, organ_names)
    valid_indices = []
    rejection_reasons = Counter()
    known_hashes = set()
    for i, patient in enumerate(organ_sets):
        is_valid, reason = check_patient_validity(patient, baseline_stats, organ_names, known_hashes)
        if is_valid:
            valid_indices.append(i)
        else:
            rejection_reasons[reason] += 1
    idx = np.array(valid_indices)
    cleaned = {k: data[k][idx] for k in ['D', 'w', 'organs', 'obj', 'ids']}
    if verbose:
        # summarize by reason CATEGORY (overlap %, outlier, etc.) instead of dumping each
        cats = Counter()
        for r, c in rejection_reasons.items():
            cat = r.split(':')[0].split('(')[0].strip()
            cats[cat] += c
        print(f"Filter: {len(organ_sets)} -> {len(idx)} valid (dropped {len(organ_sets)-len(idx)})", flush=True)
        for cat, c in cats.most_common():
            print(f"   - {cat}: {c}", flush=True)
    return cleaned, baseline_stats


def load_cohort(cohort_dir=COHORT_DIR):
    return {
        'D': np.load(f'{cohort_dir}/D_all_successful.npy'),
        'w': np.load(f'{cohort_dir}/w_all_successful.npy'),
        'organs': np.load(f'{cohort_dir}/organ_sets_all_successful.npy', allow_pickle=True),
        'obj': np.load(f'{cohort_dir}/objectives_all_successful.npy'),
        'ids': np.load(f'{cohort_dir}/successful_original_ids.npy'),
    }


def _as_dict(o):
    return o.item() if hasattr(o, 'item') and o.ndim == 0 else o


def load_anchors():
    """Load best (min penalty) and worst (max penalty) anchors with computed objectives."""
    D_min = np.load(f'{BEST_CASE_DIR}/D_best_case.npy')
    w_min = np.load(f'{BEST_CASE_DIR}/w_opt_best_case.npy')
    org_min = _as_dict(np.load(f'{BEST_CASE_DIR}/organ_sets_best_case.npy', allow_pickle=True))
    D_max = np.load(f'{WORST_CASE_DIR}/D_worst_case.npy')
    w_max = np.load(f'{WORST_CASE_DIR}/w_opt_worst_case.npy')
    org_max = _as_dict(np.load(f'{WORST_CASE_DIR}/organ_sets_worst_case.npy', allow_pickle=True))
    obj_min = calculate_objective(D_min, org_min, w_min)
    obj_max = calculate_objective(D_max, org_max, w_max)
    return (D_min, w_min, org_min, obj_min), (D_max, w_max, org_max, obj_max)


def build_ml_data(cohort_dir=COHORT_DIR, ml_dir=ML_DIR, verbose=True):
    """Run the full data pipeline and write train/test (+anchors) to ml_dir.

    1. Load + 5-check-filter the cohort.
    2. Train/test split (deterministic: random_state, shuffle).
    3. Validate the two anchors with the SAME 5-check filter against cohort stats.
    4. Prepend anchors to the training set (best=idx0, worst=idx1; ids -1, -2).
    """
    os.makedirs(ml_dir, exist_ok=True)
    data = load_cohort(cohort_dir)
    clean_data, baseline_stats = clean_dataset(data, verbose=verbose)

    arrays = [clean_data['D'], clean_data['w'], clean_data['organs'], clean_data['obj'], clean_data['ids']]
    sp = train_test_split(*arrays, test_size=TEST_SET_SIZE, random_state=SPLIT_RANDOM_STATE, shuffle=True)
    train = {'D': sp[0], 'w': sp[2], 'organs': sp[4], 'obj': sp[6], 'ids': sp[8]}
    test = {'D': sp[1], 'w': sp[3], 'organs': sp[5], 'obj': sp[7], 'ids': sp[9]}

    # ---- UNIFIED FILTER on anchors: validate both with the same 5-check ----
    (D_min, w_min, org_min, obj_min), (D_max, w_max, org_max, obj_max) = load_anchors()
    for label, org in [("BEST", org_min), ("WORST", org_max)]:
        ok, reason = check_patient_validity(org, baseline_stats, ORGAN_NAMES, known_hashes=set())
        if verbose:
            print(f"Anchor {label}: 5-check {'PASS' if ok else 'FAIL'} ({reason})", flush=True)
        if not ok:
            raise RuntimeError(f"Anchor {label} failed the unified 5-check filter: {reason}. "
                               "Pick the next-best/worst candidate from the anchor search.")

    # Prepend anchors (best -> idx0 cost-zero baseline, worst -> idx1 max-cost baseline)
    D_to_add = np.stack([D_min, D_max], axis=0)
    w_to_add = np.stack([w_min, w_max], axis=0)
    org_to_add = np.array([org_min, org_max], dtype=object)
    obj_to_add = np.array([obj_min, obj_max])
    ids_to_add = np.array([-1, -2])

    train_aug = {
        'D': np.concatenate((D_to_add, train['D']), axis=0),
        'w': np.concatenate((w_to_add, train['w']), axis=0),
        'organs': np.concatenate((org_to_add, train['organs']), axis=0),
        'obj': np.concatenate((obj_to_add, train['obj']), axis=0),
        'ids': np.concatenate((ids_to_add, train['ids']), axis=0),
    }

    np.save(f'{ml_dir}/D_train.npy', train_aug['D'])
    np.save(f'{ml_dir}/w_train.npy', train_aug['w'])
    np.save(f'{ml_dir}/organ_sets_train.npy', train_aug['organs'], allow_pickle=True)
    np.save(f'{ml_dir}/objectives_train.npy', train_aug['obj'])
    np.save(f'{ml_dir}/original_ids_train.npy', train_aug['ids'])
    np.save(f'{ml_dir}/D_test.npy', test['D'])
    np.save(f'{ml_dir}/w_test.npy', test['w'])
    np.save(f'{ml_dir}/organ_sets_test.npy', test['organs'], allow_pickle=True)
    np.save(f'{ml_dir}/objectives_test.npy', test['obj'])
    np.save(f'{ml_dir}/original_ids_test.npy', test['ids'])
    if verbose:
        print(f"Wrote ml_data: train={len(train_aug['ids'])} (incl. 2 anchors), test={len(test['ids'])} -> {ml_dir}", flush=True)
    return train_aug, test


if __name__ == "__main__":
    build_ml_data()
