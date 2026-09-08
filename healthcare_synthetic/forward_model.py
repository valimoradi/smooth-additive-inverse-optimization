"""
Synthetic forward model: anatomy generation, dose-influence matrix, and the
ground-truth forward planning problem (Section 6.2 synthetic test bed).

Ground-truth objective (minimized over beamlet weights w >= 0):
    alpha_1 * (sum_v max(0, d_v - theta1))^2     [OAR1, quadratic]
  + alpha_2 *  sum_v max(0, d_v - theta2)        [OAR2, linear]
  + alpha_exp * exp(d_max(OAR3) / kappa)         [OAR3, exponential]
subject to the hard PTV bounds and beamlet-fluence bounds in X(D).
Transcribed verbatim from notebook cells 2-4 (constants centralized in config.py).
"""
import numpy as np
import cvxpy as cp

from config import GRID_SHAPE, N_BEAMLETS, PARAMS, ALPHA_1, ALPHA_2, MISSPEC_COUPLE


def generate_2d_anatomical_sets(grid_shape, size_variation=0.2, jitter_scale=1.2, rng=None):
    """Generate one 2D synthetic prostate anatomy (PTV + 3 OARs) with random size/shape/jitter.

    Returns (organ_sets, organ_patches). organ_sets maps organ name -> flat voxel indices.
    """
    from scipy.ndimage import (binary_fill_holes, binary_dilation, binary_erosion,
                               gaussian_filter, label)
    import matplotlib.patches as patches

    if rng is None:
        seed = int(np.random.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)

    m_rows, m_cols = grid_shape
    organ_configs = {
        'ptv': (23, 23, 10, 8, 'ellipse'),
        'oar1': (20, 30, 12, 10, 'irregular'),
        'oar2': (20, 10, 14, 6, 'elongated'),
        'oar3': (34, 34, 7, 7, 'circular'),
    }
    organ_sets, organ_patches = {}, {}
    y_coords, x_coords = np.ogrid[:m_rows, :m_cols]
    jitter_span = max(4, int(5 * jitter_scale))

    for name, (center_r, center_c, avg_major, avg_minor, shape_type) in organ_configs.items():
        major_min = max(3, int(avg_major * (1 - size_variation)))
        major_max = int(avg_major * (1 + size_variation))
        minor_min = max(3, int(avg_minor * (1 - size_variation)))
        minor_max = int(avg_minor * (1 + size_variation))
        actual_major = rng.integers(major_min, major_max + 1)
        actual_minor = rng.integers(minor_min, minor_max + 1)
        r_jitter = rng.integers(-jitter_span, jitter_span + 1)
        c_jitter = rng.integers(-jitter_span, jitter_span + 1)
        center_r_new = max(actual_major // 2, min(m_rows - actual_major // 2, center_r + r_jitter))
        center_c_new = max(actual_minor // 2, min(m_cols - actual_minor // 2, center_c + c_jitter))

        mask = np.zeros(grid_shape, dtype=bool)
        angle = 0
        if shape_type in ('ellipse', 'circular'):
            angle = rng.uniform(-np.pi/3, np.pi/3)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = (x_coords - center_c_new) * cos_a - (y_coords - center_r_new) * sin_a
            dy = (x_coords - center_c_new) * sin_a + (y_coords - center_r_new) * cos_a
            irregularity = rng.uniform(0.75, 1.3)
            ellipse_dist = (dx / (actual_minor / 2 * irregularity))**2 + (dy / (actual_major / 2))**2
            boundary_noise = gaussian_filter(rng.normal(0, 0.3, grid_shape), sigma=rng.uniform(0.6, 1.3))
            mask = ellipse_dist <= (1 + 0.15 * boundary_noise)
        elif shape_type == 'elongated':
            angle = rng.uniform(-np.pi/3, np.pi/3)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = (x_coords - center_c_new) * cos_a - (y_coords - center_r_new) * sin_a
            dy = (x_coords - center_c_new) * sin_a + (y_coords - center_r_new) * cos_a
            elongated_major = actual_major * rng.uniform(1.0, 1.4)
            ellipse_dist = (dx / (actual_minor / 2))**2 + (dy / (elongated_major / 2))**2
            taper = 1 - 0.25 * np.abs(dy) / (elongated_major / 2)
            wiggle = 0.15 * np.sin((dy / max(1, elongated_major)) * rng.uniform(4, 7))
            mask = ellipse_dist <= (taper + wiggle)
        elif shape_type == 'irregular':
            angle = rng.uniform(-np.pi/2, np.pi/2)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = (x_coords - center_c_new) * cos_a - (y_coords - center_r_new) * sin_a
            dy = (x_coords - center_c_new) * sin_a + (y_coords - center_r_new) * cos_a
            base_dist = (dx / (actual_minor / 2))**2 + (dy / (actual_major / 2))**2
            irregularity_1 = rng.uniform(0.6, 1.4)
            mask = base_dist <= irregularity_1
            for _ in range(rng.integers(2, 5)):
                offset_r = rng.integers(-4, 5)
                offset_c = rng.integers(-4, 5)
                bulge_center_r = np.clip(center_r_new + offset_r, 0, m_rows - 1)
                bulge_center_c = np.clip(center_c_new + offset_c, 0, m_cols - 1)
                bulge_sigma = rng.uniform(actual_minor / 4, actual_minor / 2)
                bulge = np.exp(-(((y_coords - bulge_center_r)**2 + (x_coords - bulge_center_c)**2) / (2 * bulge_sigma**2)))
                mask = mask | (bulge > rng.uniform(0.2, 0.35))

        mask_float = mask.astype(float)
        for _ in range(rng.integers(1, 4)):
            bulge_center_r = rng.integers(max(0, center_r_new - actual_major), min(m_rows, center_r_new + actual_major + 1))
            bulge_center_c = rng.integers(max(0, center_c_new - actual_minor), min(m_cols, center_c_new + actual_minor + 1))
            bulge_sigma = rng.uniform(1.5, 3.5)
            bulge = np.exp(-(((y_coords - bulge_center_r)**2 + (x_coords - bulge_center_c)**2) / (2 * bulge_sigma**2)))
            mask_float += rng.uniform(0.15, 0.35) * bulge

        random_field = gaussian_filter(rng.normal(0, 1, grid_shape), sigma=rng.uniform(0.8, 1.6))
        mask_float = np.clip(mask_float + 0.2 * random_field, 0, None)
        threshold = rng.uniform(0.45, 0.6)
        mask = mask_float > threshold
        if rng.random() < 0.7:
            mask = binary_dilation(mask, iterations=rng.integers(1, 3))
        if rng.random() < 0.5:
            mask = binary_erosion(mask, iterations=1)
        mask = binary_fill_holes(mask)

        labeled_mask, num_features = label(mask)
        if num_features > 0:
            sizes = np.bincount(labeled_mask.ravel())
            if len(sizes) > 1:
                largest_component = np.argmax(sizes[1:]) + 1
                mask = (labeled_mask == largest_component)
        if np.sum(mask) < 4:
            dx = x_coords - center_c_new
            dy = y_coords - center_r_new
            mask = (dx**2 / (actual_minor / 2)**2 + dy**2 / (actual_major / 2)**2) <= 1

        organ_sets[name] = np.where(mask.ravel())[0]
        if np.sum(mask) > 0:
            rows, cols = np.where(mask)
            r_min, r_max, c_min, c_max = rows.min(), rows.max(), cols.min(), cols.max()
            if shape_type in ('ellipse', 'circular'):
                patch = patches.Ellipse((center_c_new, center_r_new), actual_minor, actual_major,
                                        angle=np.degrees(angle), linewidth=1.5, edgecolor='red',
                                        facecolor='none', label=f'{name} (Vol: {len(organ_sets[name])})')
            else:
                patch = patches.Rectangle((c_min, r_min), (c_max - c_min), (r_max - r_min),
                                          linewidth=1.5, edgecolor='red', facecolor='none',
                                          label=f'{name} (Vol: {len(organ_sets[name])})')
        else:
            patch = patches.Rectangle((center_c_new - actual_minor // 2, center_r_new - actual_major // 2),
                                      actual_minor, actual_major, linewidth=1.5, edgecolor='red',
                                      facecolor='none', label=f'{name} (Vol: {len(organ_sets[name])})')
        organ_patches[name] = patch
    return organ_sets, organ_patches


def generate_2d_dose_influence_matrix(grid_shape, n_beamlets):
    """Realistic 2D dose-influence matrix D (m_voxels x n_beamlets) with Gaussian falloff.

    NOTE: uses the legacy np.random global RNG (as in the original); seed via np.random.seed.
    """
    m_rows, m_cols = grid_shape
    m_voxels = m_rows * m_cols
    D = np.zeros((m_voxels, n_beamlets))
    x, y = np.indices(grid_shape)
    for j in range(n_beamlets):
        center_x = np.random.uniform(0, m_rows)
        center_y = np.random.uniform(0, m_cols)
        sigma_x = np.random.uniform(m_rows / 10, m_rows / 5)
        sigma_y = np.random.uniform(m_cols / 10, m_cols / 5)
        distance_sq = ((x - center_x)**2 / sigma_x**2 + (y - center_y)**2 / sigma_y**2)
        D[:, j] = np.exp(-distance_sq / 2).ravel()
    return D


def solve_forward_problem(D, organ_sets, params=PARAMS):
    """Solve the ground-truth forward planning problem for one patient.

    Returns (w_value, objective_value, status, mosek_failed).
    """
    m_voxels, n_beamlets = D.shape
    idx_ptv = organ_sets['ptv']; idx_oar1 = organ_sets['oar1']
    idx_oar2 = organ_sets['oar2']; idx_oar3 = organ_sets['oar3']

    w = cp.Variable(n_beamlets, name="w", nonneg=True)
    o_1 = cp.Variable(len(idx_oar1), name="o_1", nonneg=True)
    o_2 = cp.Variable(len(idx_oar2), name="o_2", nonneg=True)
    d_max = cp.Variable(name="d_max", nonneg=True)

    obj_oar1 = ALPHA_1 * cp.square(cp.sum(o_1))
    obj_oar2 = ALPHA_2 * cp.sum(o_2)
    obj_oar3 = params['alpha_exp'] * cp.exp(d_max / params['exp_scale_k'])
    objective = cp.Minimize(obj_oar1 + obj_oar2 + obj_oar3)

    dose = D @ w
    constraints = [
        dose[idx_ptv] >= params['theta_prescribed'],
        dose[idx_ptv] <= params['theta_max_PTV'],
        o_1 >= dose[idx_oar1] - params['theta_thresh_1'],
        o_2 >= dose[idx_oar2] - params['theta_thresh_2'],
        d_max >= dose[idx_oar3],
    ]
    sum_w = cp.sum(w)
    constraints.append(w >= (params['beta_1'] / n_beamlets) * sum_w)
    constraints.append(w <= (params['beta_2'] / n_beamlets) * sum_w)

    problem = cp.Problem(objective, constraints)
    mosek_failed = False
    try:
        problem.solve(solver='MOSEK', verbose=False)
    except (cp.error.SolverError, Exception):
        mosek_failed = True
        try:
            problem.solve(verbose=False)
        except cp.error.SolverError:
            try:
                problem.solve(solver='SCS', verbose=False)
            except cp.error.SolverError:
                if len(problem.constraints) > 1:
                    del problem.constraints[1]
                try:
                    problem.solve(solver='SCS', verbose=False)
                except cp.error.SolverError:
                    return None, None, "Relaxed_Solve_Failed", mosek_failed
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None, None, problem.status, mosek_failed
    return w.value, problem.value, problem.status, mosek_failed


def calculate_objective(D, organ_sets, w, params=PARAMS):
    """Numpy re-evaluation of the ground-truth objective for a given w (used for anchors)."""
    dose = D @ w
    diff1 = np.maximum(0, dose[organ_sets['oar1']] - params['theta_thresh_1'])
    term1 = ALPHA_1 * (np.sum(diff1) ** 2)
    diff2 = np.maximum(0, dose[organ_sets['oar2']] - params['theta_thresh_2'])
    term2 = ALPHA_2 * np.sum(diff2)
    term3 = 0.0
    if len(organ_sets['oar3']) > 0:
        d_max_val = np.max(dose[organ_sets['oar3']])
        term3 = params['alpha_exp'] * np.exp(d_max_val / params['exp_scale_k'])
    return term1 + term2 + term3


# ======================================================================================
# MISSPECIFIED ground truth: NON-additive AND NON-smooth (for the misspecification panel).
#   f(z) = C*ALPHA_1*(z1 + z2)^2  +  ALPHA_2*max(z1, z2)  +  alpha_exp*exp(z3/kappa),
# with z1 = sum o1, z2 = sum o2, z3 = d_max and C = MISSPEC_COUPLE.
# The first term is a STRONG, always-on cross-coupling (expands to a 2*C*ALPHA_1*z1*z2
# interaction) that no additive/separable model can represent -> the additive models carry
# an irreducible bias that more data cannot remove. The max(z1, z2) adds a MILD kink that
# only slightly penalizes the smooth models. Both terms are convex, so the forward problem
# stays a valid convex program. This mirrors the consumer misspecification (U = min{U1,U2}
# with a strong cross-good W-coupling): dominant non-additivity, secondary non-smoothness,
# so Smooth ~ Convex-only both beat Additive / Additive+Smooth.
# ======================================================================================
def solve_forward_problem_misspec(D, organ_sets, params=PARAMS):
    """Misspecified (non-additive, non-smooth) forward solve. Same feasible set as the
    additive forward; only the objective changes. Returns (w, obj, status, mosek_failed)."""
    m_voxels, n_beamlets = D.shape
    idx_ptv = organ_sets['ptv']; idx_oar1 = organ_sets['oar1']
    idx_oar2 = organ_sets['oar2']; idx_oar3 = organ_sets['oar3']

    w = cp.Variable(n_beamlets, name="w", nonneg=True)
    o_1 = cp.Variable(len(idx_oar1), name="o_1", nonneg=True)
    o_2 = cp.Variable(len(idx_oar2), name="o_2", nonneg=True)
    d_max = cp.Variable(name="d_max", nonneg=True)

    z1 = cp.sum(o_1); z2 = cp.sum(o_2)
    exp_term = params['alpha_exp'] * cp.exp(d_max / params['exp_scale_k'])
    coupled = MISSPEC_COUPLE * ALPHA_1 * cp.square(z1 + z2)   # non-additive (2*C*a1*z1*z2)
    kink = ALPHA_2 * cp.maximum(z1, z2)                       # mild non-smoothness (kink)
    objective = cp.Minimize(coupled + kink + exp_term)

    dose = D @ w
    constraints = [
        dose[idx_ptv] >= params['theta_prescribed'],
        dose[idx_ptv] <= params['theta_max_PTV'],
        o_1 >= dose[idx_oar1] - params['theta_thresh_1'],
        o_2 >= dose[idx_oar2] - params['theta_thresh_2'],
        d_max >= dose[idx_oar3],
    ]
    sum_w = cp.sum(w)
    constraints.append(w >= (params['beta_1'] / n_beamlets) * sum_w)
    constraints.append(w <= (params['beta_2'] / n_beamlets) * sum_w)

    problem = cp.Problem(objective, constraints)
    mosek_failed = False
    try:
        problem.solve(solver='MOSEK', verbose=False)
    except (cp.error.SolverError, Exception):
        mosek_failed = True
        try:
            problem.solve(solver='SCS', verbose=False)
        except cp.error.SolverError:
            return None, None, "Solve_Failed", mosek_failed
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None, None, problem.status, mosek_failed
    return w.value, problem.value, problem.status, mosek_failed


def calculate_objective_misspec(D, organ_sets, w, params=PARAMS):
    """Numpy re-evaluation of the misspecified objective (used for anchor scoring)."""
    dose = D @ w
    s1 = float(np.sum(np.maximum(0, dose[organ_sets['oar1']] - params['theta_thresh_1']))) if len(organ_sets['oar1']) else 0.0
    s2 = float(np.sum(np.maximum(0, dose[organ_sets['oar2']] - params['theta_thresh_2']))) if len(organ_sets['oar2']) else 0.0
    exp_t = 0.0
    if len(organ_sets['oar3']) > 0:
        exp_t = params['alpha_exp'] * np.exp(np.max(dose[organ_sets['oar3']]) / params['exp_scale_k'])
    return MISSPEC_COUPLE * ALPHA_1 * (s1 + s2) ** 2 + ALPHA_2 * max(s1, s2) + exp_t


def preprocess_outcomes(w_obs, D_obs, organ_sets_obs, params=PARAMS):
    """Observed outcome vector Z_hat (K=3): [sum OAR1 overdose, sum OAR2 overdose, max OAR3 dose]."""
    N = len(w_obs)
    Z_hat = np.zeros((N, 3))
    for i in range(N):
        d_dose = D_obs[i] @ w_obs[i]
        org_i = organ_sets_obs[i]
        v1 = org_i.get('oar1', np.array([])); v2 = org_i.get('oar2', np.array([])); v3 = org_i.get('oar3', np.array([]))
        if v1.size > 0:
            Z_hat[i, 0] = np.sum(np.maximum(0, d_dose[v1] - params['theta_thresh_1']))
        if v2.size > 0:
            Z_hat[i, 1] = np.sum(np.maximum(0, d_dose[v2] - params['theta_thresh_2']))
        if v3.size > 0:
            Z_hat[i, 2] = np.max(d_dose[v3])
    return Z_hat
