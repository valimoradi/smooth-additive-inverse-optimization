"""
Cohort generation: generate N_COHORT_PATIENTS random patients (seed 142), solve each
forward problem, and aggregate the successful ones into the cohort arrays consumed by the
data pipeline. Transcribed from notebook cells 10 (generate) and 13 (aggregate).
"""
import os
import time
import numpy as np

from config import (GRID_SHAPE, N_BEAMLETS, N_COHORT_PATIENTS, GLOBAL_SEED,
                    PATIENT_DATA_DIR, COHORT_DIR)
from forward_model import (generate_2d_anatomical_sets, generate_2d_dose_influence_matrix,
                          solve_forward_problem)

M_VOXELS = GRID_SHAPE[0] * GRID_SHAPE[1]


def generate_patients(n=N_COHORT_PATIENTS, out_dir=PATIENT_DATA_DIR, seed=GLOBAL_SEED):
    """Generate and forward-solve n patients, saving per-patient files (successful only)."""
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    all_obj = []
    t0 = time.time()
    n_ok = 0
    for i in range(n):
        organ_sets, organ_patches = generate_2d_anatomical_sets(GRID_SHAPE)
        D = generate_2d_dose_influence_matrix(GRID_SHAPE, N_BEAMLETS)
        w_opt, obj_val, status, _ = solve_forward_problem(D, organ_sets)
        if status in ("optimal", "optimal_inaccurate"):
            np.save(f'{out_dir}/D_patient_{i}.npy', D)
            np.save(f'{out_dir}/organ_sets_patient_{i}.npy', organ_sets, allow_pickle=True)
            np.save(f'{out_dir}/organ_patches_patient_{i}.npy', organ_patches, allow_pickle=True)
            np.save(f'{out_dir}/w_opt_patient_{i}.npy', w_opt)
            all_obj.append(obj_val); n_ok += 1
        if i % 100 == 0:
            print(f"patient {i} | solved {n_ok} | {time.time()-t0:.0f}s", flush=True)
    np.save(f'{out_dir}/all_objective_values.npy', np.array(all_obj))
    print(f"generation done: {n_ok}/{n} solved", flush=True)


def aggregate_cohort(n=N_COHORT_PATIENTS, in_dir=PATIENT_DATA_DIR, out_dir=COHORT_DIR):
    """Aggregate successful per-patient files into the cohort arrays (D_all_successful, ...)."""
    os.makedirs(out_dir, exist_ok=True)
    expected_D = (M_VOXELS, N_BEAMLETS)
    expected_w = (N_BEAMLETS,)
    Ds, ws, organs, ids = [], [], [], []
    for i in range(n):
        wf = f'{in_dir}/w_opt_patient_{i}.npy'; df = f'{in_dir}/D_patient_{i}.npy'
        of = f'{in_dir}/organ_sets_patient_{i}.npy'
        if not (os.path.exists(wf) and os.path.exists(df)):
            continue
        D_l = np.load(df); w_l = np.load(wf)
        if D_l.shape != expected_D or w_l.shape != expected_w:
            continue
        Ds.append(D_l); ws.append(w_l)
        organs.append(np.load(of, allow_pickle=True).item())
        ids.append(i)
    if not ids:
        print("no successful patients to aggregate", flush=True); return
    D_agg = np.stack(Ds, axis=0); w_agg = np.stack(ws, axis=0)
    obj_all = np.load(f'{in_dir}/all_objective_values.npy')
    valid_ids = [pid for pid in ids if pid < len(obj_all)]
    if len(valid_ids) != len(ids):
        keep = [k for k, pid in enumerate(ids) if pid in valid_ids]
        ids = [ids[k] for k in keep]; D_agg = D_agg[keep]; w_agg = w_agg[keep]
        organs = [organs[k] for k in keep]
    obj = obj_all[valid_ids]
    np.save(f'{out_dir}/D_all_successful.npy', D_agg)
    np.save(f'{out_dir}/w_all_successful.npy', w_agg)
    np.save(f'{out_dir}/objectives_all_successful.npy', obj)
    np.save(f'{out_dir}/successful_original_ids.npy', np.array(ids))
    np.save(f'{out_dir}/organ_sets_all_successful.npy', organs, allow_pickle=True)
    print(f"aggregated {len(ids)} patients -> {out_dir}", flush=True)


if __name__ == "__main__":
    generate_patients()
    aggregate_cohort()
