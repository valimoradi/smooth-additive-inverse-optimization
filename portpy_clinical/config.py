"""
config.py
=========
Configuration for the PortPy-based per-voxel inverse optimization experiment.

Uses real prostate patient data from PortPy with 10 OARs grouped into
8 non-parametric penalty functions.
"""

import os

# ---------------------------------------------------------------------------
# PortPy data paths
# ---------------------------------------------------------------------------
PORTPY_DATA_DIR = "../portpy_prostate/prostate_data/data"
PROTOCOL_NAME   = "Prostate_26Fx"   # 26 fractions x 2.7 Gy = 70.2 Gy

# ---------------------------------------------------------------------------
# Structures (must match PortPy naming after create_opt_structures)
# ---------------------------------------------------------------------------
STRUCTURE_NAMES = {
    'ptv':     'PTV',
    'bladder': 'BLADDER',
    'rectum':  'RECTUM',
    'femur_l': 'FEMUR_L',
    'femur_r': 'FEMUR_R',
    'rind_0':  'RIND_0',
    'rind_1':  'RIND_1',
    'rind_2':  'RIND_2',
}

# OAR grouping: 6 non-parametric functions
OAR_GROUPS = {
    'bladder': {'func_idx': 0, 'label': 'Bladder'},
    'rectum':  {'func_idx': 1, 'label': 'Rectum'},
    'femur_l': {'func_idx': 2, 'label': 'Femoral Heads'},
    'femur_r': {'func_idx': 2, 'label': 'Femoral Heads'},
    'rind_0':  {'func_idx': 3, 'label': 'RIND 0-2mm'},
    'rind_1':  {'func_idx': 4, 'label': 'RIND 2-20mm'},
    'rind_2':  {'func_idx': 5, 'label': 'RIND 20-40mm'},
}

N_FUNCTIONS = 6
OAR_KEYS = ['bladder', 'rectum', 'femur_l', 'femur_r',
            'rind_0', 'rind_1', 'rind_2']

FUNC_LABELS = ['Bladder', 'Rectum', 'Femoral Heads',
               'RIND 0-2mm', 'RIND 2-20mm', 'RIND 20-40mm']

# ---------------------------------------------------------------------------
# Forward objective parameters (per-voxel penalty)
# ---------------------------------------------------------------------------
# Weights from the PortPy Prostate_26Fx optimization configuration.
# g(z_v) = alpha_k * z_v^2  where z_v = max(0, d_v - theta_k)
#
FORWARD_PARAMS = {
    'alpha_bladder': 20.0,
    'alpha_rectum':  20.0,
    'alpha_femur_l': 10.0,
    'alpha_femur_r': 10.0,
    'alpha_rind_0':  5.0,
    'alpha_rind_1':  5.0,
    'alpha_rind_2':  3.0,
}

# Dose thresholds (Gy) — literature-backed onset of clinical effects:
#   - Rectum:  15 Gy  (Wahl et al. 2016 JACMP; post-QUANTEC onset 10-13 Gy)
#   - Bladder: 15 Gy  (Wahl et al. 2016 JACMP, same threshold)
#   - Femoral heads: 5 Gy  (stochastic AVN risk, no safe dose;
#                            Kocak-Uzel et al. 2016)
#   - RIND_0:  30 Gy  (high-dose zone 0-2mm from PTV; threshold set at ~43%
#                       of prescribed dose to capture dose variation in the
#                       steep gradient region)
#   - RIND_1/2: 5 Gy  (transition/low-dose zones; dose conformity)
DOSE_THRESHOLDS = {
    'bladder':  15.0,
    'rectum':   15.0,
    'femur_l':   5.0,
    'femur_r':   5.0,
    'rind_0':   30.0,
    'rind_1':    5.0,
    'rind_2':    5.0,
}

# PTV parameters (Gy) — from PortPy Prostate_26Fx protocol
# PTV dose is enforced by HARD inequality constraints in forward/solver.py:
#   PTV_PRESCRIBED <= dose[v] <= PTV_MAX  for v in PTV.
# No PTV slack variables or soft penalty weights are used.
PTV_PRESCRIBED = 70.2    # 26 x 2.7 Gy
PTV_MAX        = 77.0    # ~110% of prescribed

# Beamlet uniformity (relaxed for real data)
BETA_1 = 0.0
BETA_2 = 1e6

# ---------------------------------------------------------------------------
# Spatial downsampling (PortPy native)
# ---------------------------------------------------------------------------
OPT_VOX_XYZ_RES_MM = [15, 15, 5]

# ---------------------------------------------------------------------------
# Inverse optimization settings
# ---------------------------------------------------------------------------
NORMALIZATION_SETTINGS = {
    'U_MAX': 100000.0,   # anchor: delta_global[worst_patient] = U_MAX
    'U_MIN': 0.0,        # anchor: delta_global[best_patient] = U_MIN
                         # Must be a non-negative scalar chosen a priori
                         # (the paper uses 0). Forward-objective-derived
                         # anchors are leakage and are no longer supported.
}
BETA_0         = 40.0    # true Lipschitz constant: 2 * max(alpha) = 2 * 20
EPSILON_TOL    = 1e-6
BISECT_GAP_TOL = 0.11
BISECT_BETA_LOW = 1.0

TRAINING_SIZES = [5, 10, 15]

# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------
PREFERRED_SOLVER = "MOSEK"

MOSEK_TOLERANCES = {
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   1e-5,
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   1e-5,
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-5,
    'MSK_DPAR_OPTIMIZER_MAX_TIME':     120.0,   # 2 min for forward solves
}

MOSEK_HIGH_PRECISION = {
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   1e-11,
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   1e-11,
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-11,
}

# MOSEK thread budget.
# By default MOSEK uses every core for each solve, which is best for a single
# sequential run. When run_nested_pipeline.py runs several cohort chains in
# parallel it exports MSK_NUM_THREADS so each chain's MOSEK stays within its
# share of the cores and the chains do not oversubscribe the CPU. Injecting the
# value here propagates it to every solve, because each stage builds its solver
# options by spreading **MOSEK_TOLERANCES (see run_stage*_nested.py,
# inverse/feasibility.py, run_prediction.py). Unset / 0 leaves MOSEK on its
# default (all cores).
_MSK_NUM_THREADS = int(os.environ.get("MSK_NUM_THREADS", "0"))
if _MSK_NUM_THREADS > 0:
    MOSEK_TOLERANCES['MSK_IPAR_NUM_THREADS'] = _MSK_NUM_THREADS
    MOSEK_HIGH_PRECISION['MSK_IPAR_NUM_THREADS'] = _MSK_NUM_THREADS

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = "./results"
NUM_QUERY_POINTS = 200
