"""
inverse/feasibility.py
=======================
Feasibility probe and bisection search for the minimum feasible beta.
Updated for per-voxel formulation.
"""

import cvxpy as cp
import numpy as np

from config import (
    NORMALIZATION_SETTINGS, PREFERRED_SOLVER, MOSEK_TOLERANCES,
    BETA_0, BISECT_GAP_TOL, BISECT_BETA_LOW,
)
from .variables import get_model_variables
from .constraints.assembler import build_constraints


def is_feasible(
    beta_val: float,
    outcome_data: dict,
    patients: list[dict],
    w_solutions: list,
    epsilon_feas: float = 0.0,
    norm_settings: dict = None,
    solver: str = PREFERRED_SOLVER,
    solver_tolerances: dict = None,
    max_retries: int = 3,
    anchor_indices: dict = None,
) -> bool:
    """Return True if the model is feasible for the given (beta_val, epsilon_feas).

    Retries with relaxed solver tolerances on failure, but always tests
    the EXACT beta_val — never inflates it.
    """
    if beta_val <= 0:
        return False

    ns = norm_settings or NORMALIZATION_SETTINGS
    base_tol = solver_tolerances or MOSEK_TOLERANCES

    # Progressively relaxed tolerances for retries
    retry_tolerances = [
        {**base_tol, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 3600.0},
        {**base_tol, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 3600.0,
         'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-4,
         'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-4,
         'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-4},
        {**base_tol, 'MSK_DPAR_OPTIMIZER_MAX_TIME': 3600.0,
         'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-3,
         'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-3,
         'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-3},
    ]

    for attempt in range(min(max_retries, len(retry_tolerances))):
        try:
            variables = get_model_variables(outcome_data, patients)
            constraints = build_constraints(
                variables, outcome_data, patients, w_solutions,
                beta_val, epsilon_feas, ns, anchor_indices=anchor_indices,
            )
            prob = cp.Problem(cp.Minimize(0), constraints)
            tol = retry_tolerances[attempt]
            prob.solve(solver=solver, verbose=False, mosek_params=tol)

            if prob.status in ('optimal', 'optimal_inaccurate'):
                if attempt > 0:
                    print(f"    Feasibility: succeeded on attempt {attempt + 1}", flush=True)
                return True

            if attempt < max_retries - 1:
                print(f"    Feasibility: status='{prob.status}'; retrying with relaxed tolerances", flush=True)

        except cp.SolverError as exc:
            if attempt < max_retries - 1:
                print(f"    Feasibility: SolverError; retrying with relaxed tolerances", flush=True)
            else:
                print(f"    Feasibility: SolverError after {max_retries} attempts: {exc}", flush=True)

    return False


def find_beta_bisect(
    outcome_data: dict,
    patients: list[dict],
    w_solutions: list,
    epsilon_val: float = 0.0,
    beta_init: float = BETA_0,
    beta_low: float = BISECT_BETA_LOW,
    gap_tol: float = BISECT_GAP_TOL,
    norm_settings: dict = None,
    anchor_indices: dict = None,
) -> float:
    """Binary search for the minimum feasible beta."""
    ns = norm_settings or NORMALIZATION_SETTINGS

    beta_high = beta_init
    print(f"\nFinding feasible upper bound...", flush=True)
    while not is_feasible(beta_high, outcome_data, patients, w_solutions,
                          epsilon_val, ns, anchor_indices=anchor_indices):
        print(f"  beta={beta_high:.4f} INFEASIBLE -- doubling.", flush=True)
        beta_high *= 2.0
        if beta_high > 1e9:
            raise RuntimeError("No feasible beta found up to 1e9.")

    print(f"  Feasible upper bound: beta_high={beta_high:.4f}", flush=True)

    iteration = 1
    while beta_high - beta_low > gap_tol:
        beta_mid = 0.5 * (beta_low + beta_high)
        print(f"  Iter {iteration}: testing beta={beta_mid:.4f} ...", end=" ", flush=True)

        if is_feasible(beta_mid, outcome_data, patients, w_solutions,
                       epsilon_val, ns, anchor_indices=anchor_indices):
            beta_high = beta_mid
            print("FEASIBLE  -> update beta_high", flush=True)
        else:
            beta_low = beta_mid
            print("INFEASIBLE -> update beta_low", flush=True)

        print(f"    Interval: [{beta_low:.4f}, {beta_high:.4f}]  gap={beta_high - beta_low:.4f}", flush=True)
        iteration += 1

    print(f"\n--- Bisection complete: beta*={beta_high:.4f} ---", flush=True)
    return beta_high
