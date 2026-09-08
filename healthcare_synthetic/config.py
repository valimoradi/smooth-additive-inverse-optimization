"""
Central configuration for the synthetic-healthcare inverse-optimization experiment
(Section 6.2, "Healthcare application" / synthetic test bed).

All constants live here so the generation, recovery, prediction, and plotting stages
stay consistent. Paths are resolved relative to DATA_ROOT, which holds the (local,
git-ignored) generated data and the experiment outputs.
"""
import os

# --------------------------------------------------------------------------------------
# Paths.  DATA_ROOT is where the large generated data and the result folders live.
# Override with the SYNTH_DATA_ROOT environment variable; defaults to this package dir.
# --------------------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "SYNTH_DATA_ROOT",
    os.path.dirname(os.path.abspath(__file__)),
)

# Generation outputs (large, local only)
WORST_CASE_DIR = os.path.join(DATA_ROOT, "worst_case_search")
BEST_CASE_DIR = os.path.join(DATA_ROOT, "best_case_search")
PATIENT_DATA_DIR = os.path.join(DATA_ROOT, "patient_data_high_std")        # per-patient files
COHORT_DIR = os.path.join(DATA_ROOT, "patient_data_clean_high_std")        # aggregated cohort
ML_DIR = os.path.join(DATA_ROOT, "ml_data_high_std")                       # train/test split (+anchors)

# Recovery parameter outputs, per model
RESULTS = {
    "add_smooth":       os.path.join(DATA_ROOT, "results_high_std"),
    "add_lp":           os.path.join(DATA_ROOT, "results_high_std_add_nonsmooth_lp"),
    "nonadd_smooth":    os.path.join(DATA_ROOT, "results_high_std_non_add_smooth"),
    "nonadd_nonsmooth": os.path.join(DATA_ROOT, "results_high_std_non_add_nonsmooth"),
    "nonadd_nonsmooth_lp": os.path.join(DATA_ROOT, "results_high_std_non_add_nonsmooth_lp"),
}

# Prediction CSV outputs, per model
PRED_OUT = {
    "add_smooth":       os.path.join(DATA_ROOT, "add_smooth_outputs"),
    "add_lp":           os.path.join(DATA_ROOT, "add_nonsmooth_lp_outputs"),
    "nonadd_smooth":    os.path.join(DATA_ROOT, "nonadd_smooth_outputs"),
    "nonadd_nonsmooth": os.path.join(DATA_ROOT, "nonadd_nonsmooth_outputs"),
    "nonadd_nonsmooth_lp": os.path.join(DATA_ROOT, "nonadd_nonsmooth_lp_outputs"),
}
PRED_CSV = {
    "add_smooth":       "add_smooth_beamlet_error_results.csv",
    "add_lp":           "add_pure_lp_beamlet_error_results.csv",
    "nonadd_smooth":    "nonadd_smooth_beamlet_error_results.csv",
    "nonadd_nonsmooth": "nonadd_nonsmooth_beamlet_error_results.csv",
    "nonadd_nonsmooth_lp": "nonadd_nonsmooth_lp_beamlet_error_results.csv",
}
PLOT_DIR = os.path.join(DATA_ROOT, "combined_plots")
FINAL_PLOT_DIR = os.path.join(DATA_ROOT, "final_comparison_plots")

# --------------------------------------------------------------------------------------
# Global reproducibility seed (anatomy Generator seeds + dose matrices via np.random)
# --------------------------------------------------------------------------------------
GLOBAL_SEED = 142

# --------------------------------------------------------------------------------------
# Geometry / generation
# --------------------------------------------------------------------------------------
GRID_SHAPE = (55, 55)
N_BEAMLETS = 100
N_COHORT_PATIENTS = 2000      # cohort generation attempts (C10)
N_ANCHOR_SEARCH = 30000       # anchor-search iterations (C8/C9)

# --------------------------------------------------------------------------------------
# Forward-model clinical parameters (ground-truth objective + feasible set X(D)).
# Objective: alpha_1*(sum o1)^2 [quad] + alpha_2*(sum o2) [linear] + alpha_exp*exp(dmax/kappa) [exp].
# --------------------------------------------------------------------------------------
PARAMS = {
    "theta_prescribed": 78.0,   # PTV min dose (hard)
    "theta_max_PTV": 81.9,      # PTV max dose (hard)
    "theta_thresh_1": 59.15,    # OAR1 (quadratic) threshold
    "theta_thresh_2": 59.15,    # OAR2 (linear) threshold
    "beta_1": 0.5,              # beamlet lower-bound fraction
    "beta_2": 1.5,              # beamlet upper-bound fraction
    "exp_scale_k": 15.0,        # OAR3 exponential scale (kappa)
    "alpha_exp": 1e-3,          # OAR3 exponential multiplier
}
ALPHA_1 = 0.001                 # OAR1 quadratic multiplier
ALPHA_2 = 1.0                   # OAR2 linear multiplier

# Misspecified ground-truth coupling strength (non-additive panel only). The misspecified
# objective is  MISSPEC_COUPLE*ALPHA_1*(sum o1 + sum o2)^2 + ALPHA_2*max(sum o1, sum o2)
# + alpha_exp*exp(dmax/kappa): a STRONG, always-on cross term (2*C*ALPHA_1*z1*z2) that
# additive models cannot fit, plus a MILD kink (max) that only slightly penalizes smooth
# models -- mirroring the consumer misspecification (dominant non-additivity, secondary
# non-smoothness, so Smooth ~ Convex-only both beat the additive models).
MISSPEC_COUPLE = 6.0

# --------------------------------------------------------------------------------------
# Data-cleaning filter (UNIFIED 5-check validity, applied to cohort AND anchors).
# --------------------------------------------------------------------------------------
STD_DEV_THRESHOLD = 3.0         # organ-size z-score outlier limit
MIN_VOXEL_COUNT = 10            # biological hard limit (micro-organ)
MAX_OVERLAP_PCT = 50.0          # PTV-OAR overlap hard limit
ORGAN_NAMES = ["ptv", "oar1", "oar2", "oar3"]

# Train/test split
TEST_SET_SIZE = 200
SPLIT_RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# Inverse-optimization (recovery) settings, per ablation model.
#   U_MAX is the C7 normalization gauge (worst anchor's pinned cost). Predictions are
#   gauge-invariant; U_MAX only sets the scale of the recovered functions.
# --------------------------------------------------------------------------------------
TRAINING_SIZES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400]

RECOVERY = {
    # Smooth additive: 3-stage (min eps -> bisect beta* -> max sum-delta), MOSEK.
    "add_smooth": {
        "U_MAX": 1000.0, "BETA_0": 2000.0, "EPSILON_TOL": 1e-6,
        "BISECT_GAP_TOL": 0.11, "BISECT_BETA_LOW": 0.01, "BISECT_BETA_INIT": 10.0,
        "STAGE3_ATTEMPTS": 16, "STAGE3_ESCALATE": 1.5, "SOLVER": "MOSEK",
        "MOSEK_TOL": 1e-6,
    },
    # Pure-LP additive (non-smooth): 2-stage, GUROBI. U_MAX matches the paper params (1000).
    "add_lp": {
        "U_MAX": 1000.0, "EPSILON_TOL": 1e-6, "SOLVER": "GUROBI",
    },
    # Non-additive smooth: 3-stage, MOSEK.
    "nonadd_smooth": {
        "U_MAX": 10000.0, "BETA_0": 2000.0, "EPSILON_TOL": 1e-6,
        "BISECT_GAP_TOL": 0.2, "BISECT_BETA_LOW": 0.01, "BISECT_BETA_INIT": 20.0,
        "SOLVER": "MOSEK", "MOSEK_TOL": 1e-5,
    },
    # Non-additive non-smooth: 2-stage with forced large beta, MOSEK (APPROXIMATION of convex-only).
    "nonadd_nonsmooth": {
        "U_MAX": 10000.0, "BETA_NON_SMOOTH": 100000.0, "EPSILON_TOL": 1e-6,
        "SOLVER": "MOSEK", "MOSEK_TOL_S1": 1e-6, "MOSEK_TOL_S2": 1e-5,
    },
    # Non-additive TRUE convex-only: no smoothness term at all (beta_inv=0), 2-stage, MOSEK.
    # Same U_MAX/tolerances as nonadd_nonsmooth so the two are directly comparable (approx vs true).
    "nonadd_nonsmooth_lp": {
        "U_MAX": 10000.0, "EPSILON_TOL": 1e-6,
        "SOLVER": "MOSEK", "MOSEK_TOL_S1": 1e-6, "MOSEK_TOL_S2": 1e-5,
    },
}

# Anchor normalization indices in the augmented training set (anchors prepended at front)
IDX_NORM_ZERO = 0   # best anchor  -> cost 0
IDX_NORM_MAX = 1    # worst anchor -> cost U_MAX
