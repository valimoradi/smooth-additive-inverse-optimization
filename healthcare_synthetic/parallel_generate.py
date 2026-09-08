"""
Parallel Stage-A generation (NON-destructive companion to run_generation.py).

Reproduces the same output-file contract that data_pipeline.load_cohort / load_anchors
expect, but parallelizes across processes by seeding PER PATIENT INDEX instead of one
sequential RNG stream. This makes generation embarrassingly parallel and fully
deterministic (patient i always uses seed BASE+i), at the cost of producing a *different*
(still valid) cohort than the old single-stream one -- acceptable here because the cohort
was deleted and we are rebuilding a fresh reproducible one.

The audited generate/solve functions in forward_model.py are used unchanged.

Thread pinning: BLAS thread env vars are set BEFORE numpy is imported and are inherited by
the spawned workers, so W parallel processes do not each spawn many BLAS threads (which
would oversubscribe the CPU). RAM: one D matrix is ~2.4 MB; workers write patients to disk
and return only scalars, so main-process memory stays small.

Usage:
  python parallel_generate.py                                  # full config counts
  python parallel_generate.py --n-anchor 40 --n-cohort 40 --workers 4   # tiny self-test
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from config import (GRID_SHAPE, N_BEAMLETS, N_ANCHOR_SEARCH, N_COHORT_PATIENTS,
                    PATIENT_DATA_DIR, COHORT_DIR, BEST_CASE_DIR, WORST_CASE_DIR)
from forward_model import (generate_2d_anatomical_sets, generate_2d_dose_influence_matrix,
                           solve_forward_problem, solve_forward_problem_misspec)
from anchor_search import is_anatomically_valid

# Select ground-truth forward: additive (default) or misspecified (SYNTH_MISSPEC env set).
# The env var is inherited by spawned workers, so this resolves consistently in every process.
_FORWARD = solve_forward_problem_misspec if os.environ.get("SYNTH_MISSPEC") else solve_forward_problem

ANCHOR_SEED_BASE = 1_000_000
COHORT_SEED_BASE = 5_000_000
M_VOXELS = GRID_SHAPE[0] * GRID_SHAPE[1]


def _gen(seed):
    """Deterministic anatomy + dose matrix for a seed (audited functions, unchanged)."""
    np.random.seed(seed)
    organ_sets, _ = generate_2d_anatomical_sets(GRID_SHAPE)
    D = generate_2d_dose_influence_matrix(GRID_SHAPE, N_BEAMLETS)
    return organ_sets, D


def _anchor_eval(seed):
    """Light 2-check anatomy gate + forward solve; return forward objective or None."""
    organ_sets, D = _gen(seed)
    ok_anat, _ = is_anatomically_valid(organ_sets)
    if not ok_anat:
        return None
    w, obj, status, mosek_failed = _FORWARD(D, organ_sets)
    if mosek_failed or status not in ("optimal", "optimal_inaccurate"):
        return None
    return float(obj)


def _cohort_gen(arg):
    """Generate+solve one cohort patient; on success write per-patient files, return (i, obj)."""
    i, seed = arg
    organ_sets, D = _gen(seed)
    w, obj, status, mosek_failed = _FORWARD(D, organ_sets)
    if mosek_failed or status not in ("optimal", "optimal_inaccurate"):
        return (i, None)
    np.save(f'{PATIENT_DATA_DIR}/D_patient_{i}.npy', D)
    np.save(f'{PATIENT_DATA_DIR}/w_opt_patient_{i}.npy', w)
    np.save(f'{PATIENT_DATA_DIR}/organ_sets_patient_{i}.npy', organ_sets, allow_pickle=True)
    return (i, float(obj))


def _save_anchor(seed, tag, out_dir):
    """Regenerate the winning anchor deterministically and write its files."""
    os.makedirs(out_dir, exist_ok=True)
    organ_sets, D = _gen(seed)
    w, obj, status, _ = _FORWARD(D, organ_sets)
    np.save(f'{out_dir}/D_{tag}.npy', D)
    np.save(f'{out_dir}/organ_sets_{tag}.npy', organ_sets, allow_pickle=True)
    np.save(f'{out_dir}/w_opt_{tag}.npy', w)
    with open(f'{out_dir}/{tag}_info.txt', 'w') as f:
        f.write(f"seed: {seed}\nObjective Value: {obj}\nSolver Status: {status}\n")
    return obj


def run_anchor_search(n_iter, workers):
    print(f"== anchor search: {n_iter} seeds x {workers} workers ==", flush=True)
    seeds = [ANCHOR_SEED_BASE + i for i in range(n_iter)]
    objs = [None] * n_iter
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, obj in enumerate(ex.map(_anchor_eval, seeds, chunksize=64)):
            objs[idx] = obj
            if idx % 2000 == 0:
                nfeas = sum(o is not None for o in objs[:idx + 1])
                print(f"  [anchor] {idx}/{n_iter} feasible={nfeas} {time.time()-t0:.0f}s", flush=True)
    feas = [(o, i) for i, o in enumerate(objs) if o is not None]
    if not feas:
        raise RuntimeError("anchor search found no feasible cases")
    best_obj, best_i = min(feas)
    worst_obj, worst_i = max(feas)
    bo = _save_anchor(ANCHOR_SEED_BASE + best_i, "best_case", BEST_CASE_DIR)
    wo = _save_anchor(ANCHOR_SEED_BASE + worst_i, "worst_case", WORST_CASE_DIR)
    print(f"  best  obj={bo:.4f} (i={best_i}) -> {BEST_CASE_DIR}", flush=True)
    print(f"  worst obj={wo:.4f} (i={worst_i}) -> {WORST_CASE_DIR}", flush=True)
    print(f"  anchor search done: {len(feas)}/{n_iter} feasible, {time.time()-t0:.0f}s", flush=True)


def run_cohort(n, workers):
    print(f"== cohort generation: {n} seeds x {workers} workers ==", flush=True)
    os.makedirs(PATIENT_DATA_DIR, exist_ok=True)
    args = [(i, COHORT_SEED_BASE + i) for i in range(n)]
    objs = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = 0
        for (i, obj) in ex.map(_cohort_gen, args, chunksize=32):
            done += 1
            if obj is not None:
                objs[i] = obj
            if done % 200 == 0:
                print(f"  [cohort] {done}/{n} solved={len(objs)} {time.time()-t0:.0f}s", flush=True)
    print(f"  generation done: {len(objs)}/{n} solved, {time.time()-t0:.0f}s", flush=True)
    _aggregate(n, objs)


def _aggregate(n, objs):
    """Stack successful per-patient files into the cohort arrays load_cohort expects.
    Objectives are saved ALIGNED to the kept patients (not the buggy positional index)."""
    os.makedirs(COHORT_DIR, exist_ok=True)
    expected_D = (M_VOXELS, N_BEAMLETS)
    Ds, ws, organs, ids, obj_list = [], [], [], [], []
    for i in range(n):
        if i not in objs:
            continue
        df = f'{PATIENT_DATA_DIR}/D_patient_{i}.npy'
        wf = f'{PATIENT_DATA_DIR}/w_opt_patient_{i}.npy'
        of = f'{PATIENT_DATA_DIR}/organ_sets_patient_{i}.npy'
        if not (os.path.exists(df) and os.path.exists(wf) and os.path.exists(of)):
            continue
        D_l = np.load(df); w_l = np.load(wf)
        if D_l.shape != expected_D or w_l.shape != (N_BEAMLETS,):
            continue
        Ds.append(D_l); ws.append(w_l)
        organs.append(np.load(of, allow_pickle=True).item())
        ids.append(i); obj_list.append(objs[i])
    if not ids:
        raise RuntimeError("no successful patients to aggregate")
    np.save(f'{COHORT_DIR}/D_all_successful.npy', np.stack(Ds, axis=0))
    np.save(f'{COHORT_DIR}/w_all_successful.npy', np.stack(ws, axis=0))
    np.save(f'{COHORT_DIR}/objectives_all_successful.npy', np.array(obj_list))
    np.save(f'{COHORT_DIR}/successful_original_ids.npy', np.array(ids))
    np.save(f'{COHORT_DIR}/organ_sets_all_successful.npy', np.array(organs, dtype=object), allow_pickle=True)
    print(f"  aggregated {len(ids)} patients -> {COHORT_DIR}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-anchor", type=int, default=N_ANCHOR_SEARCH)
    ap.add_argument("--n-cohort", type=int, default=N_COHORT_PATIENTS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--skip-anchor", action="store_true")
    ap.add_argument("--skip-cohort", action="store_true")
    args = ap.parse_args()
    print(f"cpu_count={os.cpu_count()} workers={args.workers} "
          f"n_anchor={args.n_anchor} n_cohort={args.n_cohort}", flush=True)
    t0 = time.time()
    if not args.skip_anchor:
        run_anchor_search(args.n_anchor, args.workers)
    if not args.skip_cohort:
        run_cohort(args.n_cohort, args.workers)
    print(f"== Stage A total: {time.time()-t0:.0f}s ==", flush=True)


if __name__ == "__main__":
    main()
