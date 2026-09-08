"""
Anchor search: scan many randomly generated anatomies (seed 142) and keep the single
best (minimum forward objective) and worst (maximum forward objective) feasible cases.
These two become the cost-normalization anchors (best -> cost 0, worst -> cost U_MAX).

A light 2-check anatomy gate (non-empty/non-micro organs, bounded PTV-OAR overlap) is
used here because population statistics for the full 5-check are unavailable during a
streaming search. The selected anchors are re-validated with the full unified 5-check
filter in data_pipeline.build_ml_data (against cohort statistics).
Transcribed from notebook cells 8 (worst) and 9 (best).
"""
import os
import time
import numpy as np

from config import (GRID_SHAPE, N_BEAMLETS, N_ANCHOR_SEARCH, GLOBAL_SEED,
                    MIN_VOXEL_COUNT, MAX_OVERLAP_PCT, ORGAN_NAMES,
                    BEST_CASE_DIR, WORST_CASE_DIR)
from forward_model import (generate_2d_anatomical_sets, generate_2d_dose_influence_matrix,
                          solve_forward_problem)


def is_anatomically_valid(organ_data, organ_names=ORGAN_NAMES):
    """Light 2-check anatomy gate used during the streaming anchor search."""
    ptv_set = set(np.array(organ_data['ptv'], dtype=np.int32).tolist())
    for organ in organ_names:
        size = len(organ_data[organ])
        if size == 0:
            return False, f"Empty {organ}"
        if size < MIN_VOXEL_COUNT:
            return False, f"{organ} too small ({size} voxels)"
    if len(ptv_set) > 0:
        for oar in ['oar1', 'oar2', 'oar3']:
            oar_set = set(organ_data[oar])
            overlap_pct = (len(ptv_set & oar_set) / len(ptv_set)) * 100.0
            if overlap_pct > MAX_OVERLAP_PCT:
                return False, f"Impossible Overlap: PTV intersects {oar} by {overlap_pct:.1f}%"
    return True, "Valid"


def _search(mode, out_dir, n_iter=N_ANCHOR_SEARCH, seed=GLOBAL_SEED):
    """mode='worst' keeps max objective, 'best' keeps min objective."""
    assert mode in ("best", "worst")
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    best_obj = -np.inf if mode == "worst" else np.inf
    best_id = -1
    t0 = time.time()
    for i in range(n_iter):
        if i % 500 == 0:
            print(f"[{mode}] iter {i} | current={best_obj:.4f} | {time.time()-t0:.0f}s", flush=True)
        organ_sets, _ = generate_2d_anatomical_sets(GRID_SHAPE)
        D = generate_2d_dose_influence_matrix(GRID_SHAPE, N_BEAMLETS)
        ok, _ = is_anatomically_valid(organ_sets)
        if not ok:
            continue
        w_opt, obj_val, status, mosek_failed = solve_forward_problem(D, organ_sets)
        if mosek_failed or status not in ("optimal", "optimal_inaccurate"):
            continue
        improve = (obj_val > best_obj) if mode == "worst" else (obj_val < best_obj)
        if improve:
            best_obj = obj_val; best_id = i
            tag = "worst_case" if mode == "worst" else "best_case"
            np.save(f'{out_dir}/D_{tag}.npy', D)
            np.save(f'{out_dir}/organ_sets_{tag}.npy', organ_sets, allow_pickle=True)
            np.save(f'{out_dir}/w_opt_{tag}.npy', w_opt)
            with open(f'{out_dir}/{tag}_info.txt', 'w') as f:
                f.write(f"Iteration ID: {i}\nObjective Value: {obj_val}\nSolver Status: {status}\n")
    print(f"[{mode}] done: obj={best_obj:.6f} at iter {best_id}", flush=True)
    return best_obj, best_id


def search_worst(**kw):
    return _search("worst", WORST_CASE_DIR, **kw)


def search_best(**kw):
    return _search("best", BEST_CASE_DIR, **kw)


if __name__ == "__main__":
    search_worst()
    search_best()
