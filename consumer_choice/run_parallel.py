#!/usr/bin/env python3
"""Multiprocess driver for the consumer experiment (both figures, all reps).

Produces the same results as ``python consumer_inverse_optimization.py`` but
solves the (utility, model, regime, size, rep) cells across a process pool --
necessary because the non-additive Smooth model is slow at large N and the
replication protocol multiplies the cell count by Config.n_reps. MOSEK is
pinned to one thread per solve so the pool does not oversubscribe cores.

Outputs land in ./figures and ./results, matching the sequential entry point:
  results_{utility}_{regime}.csv       aggregate (mean, SE per model/size)
  results_{utility}_{regime}_reps.csv  long format, one row per (rep,size,model)
  figures/<panel>.pdf                  mean curves with +-1 SE bands

Usage: python run_parallel.py [--utilities smooth nonsmooth] [--reps R]
                              [--sizes 20 40 ...]
"""
import os, sys, time, argparse, dataclasses
os.environ.setdefault("PYTHONUNBUFFERED", "1")
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
CKPT = os.path.join(RES, "_cells_checkpoint.csv")   # per-cell resume file


def _load_checkpoint():
    """Completed cells from a previous (interrupted) run: {key: (err, beta)}.

    Only successful cells are checkpointed, so failed/killed cells re-run.
    """
    done = {}
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) != 7 or parts[0] == "utility":
                    continue
                u, name, regime, size, rep, err, beta = parts
                done[(u, name, regime, int(size), int(rep))] = (
                    float(err), float(beta) if beta else None)
    return done


def _append_checkpoint(key, err, beta):
    new = not os.path.exists(CKPT)
    with open(CKPT, "a", encoding="utf-8") as fh:
        if new:
            fh.write("utility,model,regime,size,rep,RelL2,beta_star\n")
        u, name, regime, size, rep = key
        fh.write(f"{u},{name},{regime},{size},{rep},{err:.6f},"
                 f"{'' if beta is None else f'{beta:.6f}'}\n")


def cache_path(utility):
    # The historical smooth-utility cache predates the per-utility naming.
    return os.path.join(HERE, "_dataset_cache.npz" if utility == "smooth"
                        else f"_dataset_cache_{utility}.npz")


def detect_cores():
    """Number of *physical* cores to plan around.

    MOSEK interior-point solves are compute / memory-bandwidth bound, so the two
    hyperthreads on a physical core add little (~1.1-1.3x, not 2x) and running
    one MOSEK process per logical thread also doubles peak RAM. We therefore plan
    around physical cores: psutil reports them directly; without psutil we fall
    back to os.cpu_count() (logical) and halve it when it looks hyperthreaded.
    """
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        if phys:
            return phys
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2) if logical >= 4 else logical


def plan_cores(n_tasks, reserve=1):
    """Pick (n_workers, threads_per_solve) for the current machine.

    One independent solve per physical core with MOSEK single-threaded when
    tasks outnumber cores (they do by far here); otherwise spread leftover cores
    as solver threads. `reserve` keeps a core for the OS. CONSUMER_WORKERS
    overrides the worker count.
    """
    avail = max(1, detect_cores() - reserve)
    n_workers = max(1, min(n_tasks, avail))
    override = os.environ.get("CONSUMER_WORKERS")
    if override:
        n_workers = max(1, min(int(override), n_tasks))
    threads_per_solve = max(1, avail // n_workers)
    return n_workers, threads_per_solve


def _prep_cache(utility):
    """Build (or reuse) the dataset cache for one utility."""
    path = cache_path(utility)
    if os.path.exists(path):
        d = np.load(path)
        print(f"dataset[{utility}]: cached X_pool={d['Xp'].shape}", flush=True)
        return
    import consumer_inverse_optimization as C
    Xp, Pp, Xt, Pt = C.build_dataset(utility)
    Xp_pert = C.perturb_train_pool(Xp)     # drawn right after build (notebook RNG order)
    np.savez(path, Xp=Xp, Pp=Pp, Xt=Xt, Pt=Pt, Xp_pert=Xp_pert)
    print(f"dataset[{utility}]: built X_pool={Xp.shape}", flush=True)


def _worker_init(utilities, threads=1):
    import consumer_inverse_optimization as C
    params = dict(C.CFG.mosek_params); params["MSK_IPAR_NUM_THREADS"] = threads
    C.CFG = dataclasses.replace(C.CFG, mosek_params=params)
    global _C, _D
    _C = C
    _D = {u: np.load(cache_path(u)) for u in utilities}


def solve_cell(task):
    utility, model_name, regime, size, rep = task
    C, d = _C, _D[utility]
    pool = d["Xp"] if regime == "not-perturbed" else d["Xp_pert"]
    Pp, Xt, Pt = d["Pp"], d["Xt"], d["Pt"]
    mdl = next(m for m in C.MODELS if m.name == model_name)
    idx = C.rep_permutation(rep, pool.shape[0])
    Z, P = pool[idx[:size]], Pp[idx[:size]]
    t0 = time.time()
    try:
        rec = mdl.solve(Z, P)
        return (*task, C.rel_l2(Xt, mdl.predict(rec, Z, Pt)), rec.beta, "",
                time.time() - t0)
    except Exception as e:
        return (*task, float("nan"), None, f"{type(e).__name__}: {e}", time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utilities", nargs="+", default=["smooth", "kicks3"],
                        choices=["smooth", "kicks3", "nonsmooth"])
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    import consumer_inverse_optimization as C
    n_reps = args.reps if args.reps else C.CFG.n_reps
    sizes = args.sizes if args.sizes else list(C.CFG.training_sizes)
    for u in args.utilities:
        _prep_cache(u)

    # Big/slow cells first so the pool tail stays busy.
    all_tasks = [(u, m.name, r, s, rep)
                 for u in args.utilities
                 for r in ("not-perturbed", "perturbed")
                 for rep in range(n_reps)
                 for s in sizes for m in C.MODELS]
    all_tasks.sort(key=lambda t: -t[3])
    ckpt = _load_checkpoint()
    results = {(u, regime, name, size, rep): v
               for (u, name, regime, size, rep), v in ckpt.items()}
    tasks = [t for t in all_tasks if t not in ckpt]
    nproc, threads = plan_cores(len(tasks))
    print(f"solving {len(tasks)} cells ({len(all_tasks) - len(tasks)} from checkpoint; "
          f"reps={n_reps}, sizes={sizes}) on {nproc} workers x {threads} MOSEK "
          f"thread(s) [{os.cpu_count()} cores detected]...", flush=True)

    t0, ndone = time.time(), 0
    with mp.Pool(nproc, initializer=_worker_init, initargs=(args.utilities, threads)) as pool:
        for u, name, regime, size, rep, err, beta, msg, dt in \
                pool.imap_unordered(solve_cell, tasks):
            results[(u, regime, name, size, rep)] = (err, beta)
            if err == err:                      # checkpoint successes only
                _append_checkpoint((u, name, regime, size, rep), err, beta)
            ndone += 1
            tag = f"RelL2={err:6.4f}" if err == err else f"FAIL {msg}"
            print(f"  [{ndone}/{len(tasks)}] [{u}/{regime:13s}/rep={rep:2d}/N={size:3d}/"
                  f"{name:16s}] {tag} ({dt:.0f}s)", flush=True)
    print(f"all cells done in {time.time()-t0:.0f}s", flush=True)

    for u in args.utilities:
        for regime in ("not-perturbed", "perturbed"):
            errs, betas = {}, {}
            for m in C.MODELS:
                e = np.full((n_reps, len(sizes)), np.nan)
                b = np.full((n_reps, len(sizes)), np.nan)
                for rep in range(n_reps):
                    for j, s in enumerate(sizes):
                        err, beta = results.get((u, regime, m.name, s, rep), (np.nan, None))
                        e[rep, j] = err
                        if beta is not None:
                            b[rep, j] = beta
                errs[m.name], betas[m.name] = e, b
            C._write_results_csv(os.path.join(RES, f"results_{u}_{regime}.csv"),
                                 sizes, errs, betas)
            C._write_reps_csv(os.path.join(RES, f"results_{u}_{regime}_reps.csv"),
                              sizes, errs, betas)
            C.plot_panel(sizes, errs, os.path.join(FIG, C.PANELS[(u, regime)]),
                         title="")   # published panels carry no in-figure title
    C.plot_legend(os.path.join(FIG, "legend.pdf"))


if __name__ == "__main__":
    main()
