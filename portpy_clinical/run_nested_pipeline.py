#!/usr/bin/env python3
"""
run_nested_pipeline.py
======================
Cross-platform driver for the nested 3-stage inverse-optimization experiment.

Equivalent to run_nested_pipeline.sh, but:
  * runs on Windows / macOS / Linux (pure Python, no bash),
  * can run the independent per-N cohort chains in parallel, and
  * auto-detects the CPU and balances chains x MOSEK-threads via core_planner,
    so the machine is used fully without oversubscription.

Each N in --n-list is an independent chain: Stage 1 (minimise epsilon) -> Stage 2
(bisection for beta*) -> Stage 3 (maximise sum-delta, with auto-retry). Stages
within a chain run sequentially; different N can run concurrently. A held-out
prediction step runs once, after all chains finish.

The run is resumable: any (N, stage) whose output already exists is skipped, so
re-running after an interruption continues where it stopped.

Reproducibility note: MOSEK can return slightly different (within-tolerance)
solutions for different thread counts, because parallel reductions sum in a
different order. The default --sequential mode leaves MOSEK on its own default
thread count, matching the original run_nested_pipeline.sh behaviour. Use
--parallel only when regenerating results, not when reproducing exact cached
numbers.

Usage:
    python run_nested_pipeline.py                 # sequential (reproducible)
    python run_nested_pipeline.py --parallel      # auto: balance chains x threads
    python run_nested_pipeline.py --parallel -j 2 # cap to 2 chains at once
    python run_nested_pipeline.py --n-list 5,6    # subset of cohort sizes
    python run_nested_pipeline.py --parallel --dry-run   # print the plan only
"""
import os, sys, argparse, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core_planner import detect_cores, plan_cores

PY = sys.executable                      # run children with this same interpreter
BETA_INIT = 100.0
DEFAULT_N = [5, 6, 7, 8]


def _res_dir(N):
    return os.path.join(HERE, "results", f"nested_N{N}")


def _stage_done(N, stage):
    """Output-file existence check used to skip already-finished stages."""
    d = _res_dir(N)
    if stage == 1:
        return os.path.exists(f"{d}/epsilon_star.npy") and os.path.exists(f"{d}/outcome_data.pkl")
    if stage == 2:
        return os.path.exists(f"{d}/beta_star.npy")
    if stage == 3:
        return (os.path.exists(f"{d}/beta_used.npy")
                and os.path.exists(f"{d}/delta_func0_N{N}.npy")
                and os.path.exists(f"{d}/delta_func5_N{N}.npy"))
    return False


def _run(cmd, log_path, env):
    """Run cmd from HERE, streaming combined output to log_path. Return exit code."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=HERE, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def run_chain(N, env, dry_run=False):
    """Stage 1 -> 2 -> 3 for one cohort size N. Returns (N, ok, message)."""
    d = _res_dir(N)
    os.makedirs(d, exist_ok=True)
    stages = [
        (1, [PY, "run_stage1_nested.py", str(N), d], f"{d}/stage1.log"),
        (2, [PY, "run_stage2_nested.py", str(N), str(BETA_INIT)], f"{d}/stage2.log"),
        (3, [PY, "run_stage3_nested_auto.py", str(N), d], f"{d}/stage3.log"),
    ]
    for stage, cmd, log in stages:
        if _stage_done(N, stage):
            print(f"[N={N}] stage {stage}: already done, skipping", flush=True)
            continue
        if dry_run:
            print(f"[N={N}] stage {stage}: would run -> {' '.join(cmd)}", flush=True)
            continue
        print(f"[N={N}] stage {stage}: running ...", flush=True)
        t0 = time.time()
        rc = _run(cmd, log, env)
        if rc != 0:
            return (N, False, f"stage {stage} FAILED (rc={rc}); see {log}")
        print(f"[N={N}] stage {stage}: done ({time.time() - t0:.0f}s)", flush=True)
    return (N, True, "ok")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-list", default=",".join(map(str, DEFAULT_N)),
                    help="comma-separated cohort sizes (default: 5,6,7,8)")
    ap.add_argument("--parallel", action="store_true",
                    help="run cohort chains concurrently (default: sequential)")
    ap.add_argument("--jobs", "-j", type=int, default=0,
                    help="max chains at once (0 = auto from core count); implies --parallel")
    ap.add_argument("--reserve", type=int, default=0,
                    help="physical cores to leave free for the OS (default: 0)")
    ap.add_argument("--no-prediction", action="store_true",
                    help="skip the final held-out prediction step")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the core plan and the commands, then exit")
    args = ap.parse_args()

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    parallel = args.parallel or args.jobs != 0
    avail = max(1, detect_cores() - max(0, args.reserve))

    # --- decide chains x MOSEK-threads -----------------------------------
    # threads == 0 is the signal to leave MOSEK on its own default (all cores),
    # which we use for any single-chain run to keep behaviour reproducible.
    if not parallel:
        jobs, threads = 1, 0
    else:
        jobs, threads = plan_cores(len(n_list), reserve=args.reserve)
        if args.jobs > 0:
            jobs = max(1, min(args.jobs, len(n_list)))
            threads = max(1, avail // jobs)
        if jobs == 1:
            threads = 0

    thr_label = "MOSEK default (all cores)" if threads == 0 else f"{threads} thread(s)"
    print("=== NESTED EXPERIMENT PIPELINE ===", flush=True)
    print(f"physical cores : {detect_cores()}  (logical {os.cpu_count()})", flush=True)
    print(f"cohort sizes   : {n_list}", flush=True)
    print(f"plan           : {jobs} chain(s) in parallel x {thr_label} each", flush=True)

    # children inherit the thread cap so each MOSEK stays within its share
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if threads > 0:
        env["MSK_NUM_THREADS"] = str(threads)
    else:
        env.pop("MSK_NUM_THREADS", None)

    # --- Step 0: build nested caches (sequential, fast) ------------------
    cache_dir = os.path.join(HERE, "results", "all_patients")
    need = [N for N in n_list
            if not os.path.exists(os.path.join(cache_dir, f"forward_cache_nested_N{N}.pkl"))]
    if args.dry_run:
        print(f"[dry-run] caches missing for N={need}" if need else "[dry-run] caches present",
              flush=True)
    elif need:
        print(f"--- building nested caches (missing for N={need}) ---", flush=True)
        rc = _run([PY, "build_nested_caches.py", "--seed", "42"],
                  os.path.join(HERE, "results", "build_caches.log"), env)
        if rc != 0:
            sys.exit("Cache build FAILED; see results/build_caches.log")
    else:
        print("--- caches present; skipping build ---", flush=True)

    # --- run the chains --------------------------------------------------
    t0 = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_chain, N, env, args.dry_run): N for N in n_list}
        for fut in as_completed(futs):
            N, ok, msg = fut.result()
            print(f">>> N={N}: {'OK' if ok else 'FAIL'} -- {msg}", flush=True)
            if not ok:
                failures.append((N, msg))

    if failures:
        print(f"\n{len(failures)} chain(s) failed:", flush=True)
        for N, msg in failures:
            print(f"  N={N}: {msg}", flush=True)
        sys.exit(1)
    print(f"\nall chains done in {time.time() - t0:.0f}s", flush=True)

    # --- final prediction ------------------------------------------------
    if args.no_prediction:
        return
    if args.dry_run:
        print("[dry-run] would run prediction on held-out patients", flush=True)
        return
    pred_csv = os.path.join(HERE, "results", "prediction_nested", "prediction_errors.csv")
    if os.path.exists(pred_csv):
        print("=== prediction already done; skipping ===", flush=True)
        return
    print("=== PREDICTION on 20 held-out test patients ===", flush=True)
    rc = _run([PY, "run_prediction.py", "--experiment", "nested", "--all-N",
               "--n-test", "20", "--seed", "123"],
              os.path.join(HERE, "results", "prediction_nested.log"), env)
    if rc != 0:
        sys.exit("Prediction FAILED; see results/prediction_nested.log")
    print("Prediction done.\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
