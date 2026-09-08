"""
core_planner.py
===============
CPU-topology detection and core-allocation planning for the inverse-optimization
pipeline.

The pipeline runs several independent patient-cohort chains (one per training-set
size N), and each chain calls MOSEK, which is itself multithreaded. Running every
chain at once while letting each MOSEK solve grab all cores oversubscribes the
CPU: with C cores and J parallel chains you get J x C software threads contending
for C cores, which is slower than a balanced split. This module picks that split
-- how many chains to run at once (J) and how many MOSEK threads to give each (T)
-- so that J x T stays within the physical-core budget.

Run directly to see the plan for this machine:

    python core_planner.py
"""
import os


def detect_cores() -> int:
    """Number of *physical* CPU cores to plan around.

    MOSEK's interior-point solver is compute- and memory-bandwidth-bound, so the
    second hyperthread on a physical core adds little (~1.1-1.3x, not 2x) and
    running one MOSEK process per logical thread also doubles peak memory. We
    therefore plan around physical cores. psutil reports them directly; without
    psutil we fall back to os.cpu_count() (logical) and halve it when the count
    looks hyperthreaded.
    """
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        if phys:
            return int(phys)
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2) if logical >= 4 else logical


def plan_cores(n_tasks: int, reserve: int = 0) -> tuple:
    """Split the physical cores into (n_workers, threads_per_task).

    Strategy:
      * n_workers = min(n_tasks, available cores) -- never spawn more chains than
        there is work, or than there are cores.
      * threads_per_task = available // n_workers -- hand the remaining cores to
        each task's MOSEK solve, so the whole budget is used and no two solves
        oversubscribe a core.

    `reserve` keeps that many cores free for the OS / orchestrator (0 for a batch
    run that should use the whole machine, 1 to stay responsive).

    Examples on a 16-physical-core machine (reserve=0):
        plan_cores(4)  -> (4, 4)    # 4 chains, 4 MOSEK threads each
        plan_cores(8)  -> (8, 2)
        plan_cores(20) -> (16, 1)   # more chains than cores: 1 thread each
    """
    avail = max(1, detect_cores() - max(0, reserve))
    n_workers = max(1, min(n_tasks, avail))
    threads_per_task = max(1, avail // n_workers)
    return n_workers, threads_per_task


if __name__ == "__main__":
    print(f"physical cores (planning budget) : {detect_cores()}")
    print(f"logical  cores (os.cpu_count())  : {os.cpu_count()}")
    print("plans:")
    for n in (4, 6, 8, 12, 20):
        w, t = plan_cores(n)
        print(f"  {n:2d} tasks -> {w:2d} chain(s) x {t} MOSEK thread(s)  "
              f"({w * t} cores used)")
