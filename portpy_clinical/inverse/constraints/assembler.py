"""
inverse/constraints/assembler.py
=================================
Assembles C5 + C6 + C7 + C8 for the per-voxel inverse model.
"""

from config import NORMALIZATION_SETTINGS
from .smoothness import build_c5_smoothness
from .monotonicity import build_c6_monotonicity
from .normalization import build_c7_normalization
from .dual import build_c8_dual


def build_constraints(
    variables: dict,
    outcome_data: dict,
    patients: list[dict],
    w_solutions: list,
    beta_val: float = 1.0,
    epsilon_val=None,
    norm_settings: dict = None,
    anchor_indices: dict = None,
) -> list:
    ns = norm_settings or NORMALIZATION_SETTINGS
    Z_per_func = outcome_data['Z_per_func']

    delta_funcs = variables['delta_funcs']
    lambda_funcs = variables['lambda_funcs']

    constraints = []

    # C5: smooth convexity per function (on raw observations)
    constraints += build_c5_smoothness(delta_funcs, lambda_funcs, Z_per_func, beta_val)

    # C6: monotonicity + convexity (lambda non-decreasing)
    constraints += build_c6_monotonicity(lambda_funcs, Z_per_func)

    # C7: anchor-based global normalization
    constraints += build_c7_normalization(variables, outcome_data, ns, anchor_indices)

    # C8: KKT stationarity + CS + linking + sub-optimality
    constraints += build_c8_dual(variables, outcome_data, patients, w_solutions, epsilon_val)

    return constraints
