"""
inverse/runner.py
=================
Three-stage inverse optimization runner for per-voxel formulation.

Stage 1 — Minimize epsilon (sub-optimality gap)
Stage 2 — Bisect for beta* (minimum smoothness)
Stage 3 — Maximize sum of function values at (beta*, epsilon*)

Uses per-function normalization (paper's C7): g_k(min_z) = 0, g_k(max_z) = U_MAX.
"""

import os
import time

import cvxpy as cp
import numpy as np

from config import (
    NORMALIZATION_SETTINGS, PREFERRED_SOLVER,
    MOSEK_TOLERANCES, BETA_0, EPSILON_TOL,
    BISECT_GAP_TOL, BISECT_BETA_LOW, RESULTS_DIR,
    N_FUNCTIONS, FUNC_LABELS,
)
from .outcomes import extract_per_voxel_outcomes
from .variables import get_model_variables
from .constraints.assembler import build_constraints
from .feasibility import find_beta_bisect


def run_inverse(
    patients: list[dict],
    w_solutions: list[np.ndarray],
    norm_settings: dict = None,
    beta_0: float = BETA_0,
    epsilon_tol: float = EPSILON_TOL,
    bisect_gap_tol: float = BISECT_GAP_TOL,
    bisect_beta_low: float = BISECT_BETA_LOW,
    solver: str = PREFERRED_SOLVER,
    solver_tolerances: dict = None,
    results_dir: str = RESULTS_DIR,
    use_bisection: bool = False,
    anchor_indices: dict = None,
    max_voxels_per_oar: int = None,
) -> dict:
    """
    Run the complete 3-stage per-voxel inverse optimization.

    Returns
    -------
    dict with 'delta_funcs', 'lambda_funcs', 'Z_per_func', 'beta_star',
         'epsilon_star', 'outcome_data'
    """
    ns = norm_settings or NORMALIZATION_SETTINGS
    tol = solver_tolerances or MOSEK_TOLERANCES
    N = len(patients)

    start = time.time()
    print(f"\n{'='*60}")
    print(f"Per-Voxel Inverse Optimization  N={N} patients")
    print(f"{'='*60}")

    # Subsample voxels if requested (modifies organ_indices consistently
    # so that stationarity, CS, linking, and outcomes all see the same
    # reduced voxel set)
    if max_voxels_per_oar is not None:
        from .outcomes import subsample_patient_voxels
        patients = subsample_patient_voxels(patients, max_voxels_per_oar)
        print(f"\n  Subsampled to max {max_voxels_per_oar} voxels/OAR/patient")

    # Extract per-voxel outcomes
    print("\nExtracting per-voxel outcomes ...")
    outcome_data = extract_per_voxel_outcomes(patients, w_solutions)
    for k in range(N_FUNCTIONS):
        M_k = outcome_data['obs_per_func'][k]
        print(f"  {FUNC_LABELS[k]:20s}: {M_k:6d} observations")
    print(f"  Total: {outcome_data['total_obs']} observations")

    # Anchor patients are required to be supplied by the caller.
    # `anchor_indices` only carries scalar indices into the patient pool
    # plus a normalisation level; the inverse problem itself is solved
    # purely through the non-parametric (delta, lambda) representation.
    # No information about the parametric form of g_k (e.g. alpha * z^2)
    # is read here.
    if anchor_indices is None:
        raise ValueError(
            "anchor_indices must be supplied. The inverse method does not "
            "assume any parametric form for g_k, so anchors must be selected "
            "externally (e.g. by the caller's own data-driven criterion)."
        )
    anchors = anchor_indices
    print(f"  Anchors: best=patient {anchors['best_idx']} "
          f"(obj={anchors['best_obj']:.1f}), "
          f"worst=patient {anchors['worst_idx']} "
          f"(obj={anchors['worst_obj']:.1f})")

    # ------------------------------------------------------------------
    # Stage 1: Minimize epsilon
    # ------------------------------------------------------------------
    print(f"\n--- Stage 1: Minimize epsilon  (beta={beta_0}) ---")
    vars_s1 = get_model_variables(outcome_data, patients)
    epsilon_var = vars_s1['epsilon']

    cons_s1 = build_constraints(
        vars_s1, outcome_data, patients, w_solutions,
        beta_val=beta_0, epsilon_val=epsilon_var, norm_settings=ns,
        anchor_indices=anchors,
    )
    prob_s1 = cp.Problem(cp.Minimize(epsilon_var), cons_s1)
    _solve_or_raise(prob_s1, solver, tol, stage=1)

    epsilon_star = float(epsilon_var.value)
    epsilon_s2 = epsilon_star + epsilon_tol
    print(f"--- Stage 1 complete: epsilon*={epsilon_star:.2e} ---")

    # Free Stage 1 memory before building Stage 3
    del vars_s1, cons_s1, prob_s1, epsilon_var
    import gc; gc.collect()

    # ------------------------------------------------------------------
    # Stage 2: Bisection for beta*
    # ------------------------------------------------------------------
    if use_bisection:
        print(f"\n--- Stage 2: Bisection for beta* ---")
        try:
            beta_star = find_beta_bisect(
                outcome_data, patients, w_solutions,
                epsilon_val=epsilon_s2,
                beta_init=beta_0,
                beta_low=bisect_beta_low,
                gap_tol=bisect_gap_tol,
                norm_settings=ns,
                anchor_indices=anchors,
            )
        except RuntimeError as e:
            print(f"  Bisection failed: {e}; using beta_0={beta_0}")
            beta_star = beta_0
    else:
        beta_star = beta_0
        print(f"  Using beta={beta_star} (fixed, skipping Stage 2)")

    # ------------------------------------------------------------------
    # Stage 3: Maximize sum(delta_global) (robustification)
    # ------------------------------------------------------------------
    print(f"\n--- Stage 3: Maximize sum(delta_global)  "
          f"(beta*={beta_star:.4f}, eps={epsilon_s2:.2e}) ---")
    vars_s3 = get_model_variables(outcome_data, patients)
    cons_s3 = build_constraints(
        vars_s3, outcome_data, patients, w_solutions,
        beta_val=beta_star, epsilon_val=epsilon_s2, norm_settings=ns,
        anchor_indices=anchors,
    )

    prob_s3 = cp.Problem(cp.Maximize(cp.sum(vars_s3['delta_global'])), cons_s3)
    _solve_or_raise(prob_s3, solver, tol, stage=3)
    print("--- Stage 3 complete ---")

    elapsed = time.time() - start

    results = {
        'delta_funcs': [vars_s3['delta_funcs'][k].value for k in range(N_FUNCTIONS)],
        'lambda_funcs': [vars_s3['lambda_funcs'][k].value for k in range(N_FUNCTIONS)],
        'Z_per_func': outcome_data['Z_per_func'],
        'beta_star': beta_star,
        'epsilon_star': epsilon_star,
        'outcome_data': outcome_data,
    }

    os.makedirs(results_dir, exist_ok=True)
    _save_results(results, results_dir, N)

    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"beta*={beta_star:.4f}  epsilon*={epsilon_star:.2e}")
    _print_summary(results)

    return results


def _solve_or_raise(problem, solver, tolerances, stage):
    tol = {**tolerances, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 3600.0}
    try:
        problem.solve(solver=solver, verbose=False, mosek_params=tol)
    except cp.SolverError:
        print(f"  Stage {stage}: primary solver failed; retrying with more time")
        tol_retry = {**tol, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 7200.0}
        try:
            problem.solve(solver=solver, verbose=False, mosek_params=tol_retry)
        except cp.SolverError as exc:
            raise RuntimeError(f"Stage {stage} solver failed: {exc}")

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"Stage {stage} failed with status: {problem.status}")


def _save_results(results, results_dir, N):
    print(f"\nSaving results to {results_dir} ...")
    for k in range(N_FUNCTIONS):
        np.save(f'{results_dir}/delta_func{k}_N{N}.npy', results['delta_funcs'][k])
        np.save(f'{results_dir}/lambda_func{k}_N{N}.npy', results['lambda_funcs'][k])
        np.save(f'{results_dir}/Z_func{k}_N{N}.npy', results['Z_per_func'][k])
    np.save(f'{results_dir}/beta_star_N{N}.npy', np.array([results['beta_star']]))
    np.save(f'{results_dir}/epsilon_star_N{N}.npy', np.array([results['epsilon_star']]))
    # Save patient_voxel_map for prediction use
    import pickle
    with open(f'{results_dir}/outcome_data_N{N}.pkl', 'wb') as f:
        pickle.dump({
            'Z_per_func': results['Z_per_func'],
            'obs_per_func': results['outcome_data']['obs_per_func'],
            'patient_voxel_map': results['outcome_data']['patient_voxel_map'],
            'total_obs': results['outcome_data']['total_obs'],
        }, f)
    print("  Saved.")


def _print_summary(results):
    for k in range(N_FUNCTIONS):
        delta_k = results['delta_funcs'][k]
        lam_k = results['lambda_funcs'][k]
        z_k = results['Z_per_func'][k]
        print(f"\n  {FUNC_LABELS[k]}:")
        print(f"    Observations: {len(z_k)}")
        print(f"    Overdose range: [{z_k.min():.2f}, {z_k.max():.2f}]")
        if delta_k is not None:
            print(f"    Delta range: [{delta_k.min():.4f}, {delta_k.max():.4f}]")
        if lam_k is not None:
            print(f"    Lambda range: [{lam_k.min():.4f}, {lam_k.max():.4f}]")
