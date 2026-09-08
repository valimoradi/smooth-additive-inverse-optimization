"""
parametric_io.py
================
Parametric inverse optimization for the PortPy healthcare experiment.

Background
----------
The true forward objective is f_k(z_v) = alpha_k * z_v^2 (quadratic per-voxel
overdose penalty), with alpha_k given in config.py FORWARD_PARAMS.

A parametric IO method (Keshavarz et al. 2011) assumes a KNOWN functional
form f_k(z) = alpha_k * z^rho for a KNOWN exponent rho, and estimates the
scalar multipliers alpha_k from observed optimal decisions.

If the assumed form is correct (rho=2, quadratic), the method recovers
alpha_hat_k = alpha_k.  If the assumed form is wrong (rho=1, linear;
rho=3, cubic), no choice of alpha_hat can compensate for the structural error,
and the resulting forward prediction is catastrophically wrong.

Our nonparametric IO makes no assumption about the form of f_k.

Three estimators for alpha_k
----------------------------

1. fit_parametric_alpha  -- ORACLE UPPER BOUND (diagnostic only; NOT the paper
   method -- the paper reports method 3, the non-oracle KKT estimator)
   Uses the true alpha_k from config to compute oracle shadow prices
   mu_v^(i) = alpha_k_true * 2 * o_v^(i), then projects onto the
   gradient of the assumed form.  For rho=2 this recovers alpha_true
   exactly.  For rho != 2 it gives the best possible misspecified fit,
   making the failure purely attributable to functional-form error.

   KKT shadow-price matching:
     mu_v^(i)  = alpha_k_true * 2 * o_v^(i)     [oracle shadow price]
     g_v^(i)   = rho * (o_v^(i))^(rho-1)        [gradient of assumed form]
     alpha_hat_k = sum_{i,v} mu_v g_v / sum_{i,v} g_v^2  [OLS per OAR]

   For rho=2: alpha_hat_k = alpha_k_true  (exact)
   For rho=1: alpha_hat_k = 2 * alpha_k_true * <o^2> / <o>  (biased)
   For rho=3: alpha_hat_k = (2/3) * alpha_k_true * <o^3> / <o^4>  (biased)

2. fit_keshavarz_obj_value  -- NON-ORACLE, objective-value NNLS
   Builds F[i,k] = sum_{v in OAR_k} max(0, d_v^(i) - theta_k)^rho
   from observed w^*(i) and fits min_{alpha>=0} ||F alpha - J||^2
   where J_i is the reported optimal objective.

   FAILURE on this dataset:
   * N=5 gives 5x6 underdetermined system.
   * RIND_0 and RIND_2 have near-proportional F columns (both ring
     OARs, similar geometry), so their alpha are non-separable; NNLS
     puts mass on whichever RIND has a slightly larger column norm.

3. fit_keshavarz_kkt  -- NON-ORACLE, faithful KKT residual QP (PRIMARY)
   The faithful Keshavarz, Wang & Boyd (2011, sec. IV) estimator, with full
   dual feasibility and NO active-set tolerance.  Over ALL beamlets,
   stationarity is s = G alpha - C_lo mu_lo + C_hi mu_hi = nu >= 0, where
   C_lo = D_{lo,:}^T (PTV lower bound, mu_lo >= 0), C_hi = D_{hi,:}^T (upper
   bound, mu_hi >= 0), and nu >= 0 is the beamlet nonneg dual.  alpha is
   imputed by the convex QP
     min  sum_i sum_j ( w_j^(i) s_j^(i) )^2
     s.t. s^(i) >= 0, mu_lo, mu_hi >= 0, alpha >= 0, 1^T alpha = 1,
   i.e. the complementary-slackness residual nu_j w_j minimised over alpha
   and the sign-constrained multipliers (dual feasibility ENFORCED).  The
   w_j weighting drives s_j->0 on genuinely active beamlets and only requires
   s_j >= 0 on inactive ones, so NO active-beamlet threshold is needed;
   s >= 0 is always feasible because D >= 0 => G >= 0.  An identifiability
   diagnostic (conditioning and smallest-eigenvalue direction of the
   free-sign normal matrix) flags weakly determined alpha-combinations.

   EMPIRICALLY on the PortPy nested caches this recovers alpha at the
   correct form (rho=2) to ~0.1-0.2 relative error and predicts held-out
   patients within a few percent; the wrong forms (rho=1,3) stay
   catastrophic at every N.  n_B (binding PTV) << n_A (active beamlets),
   so each patient's stationarity is strongly informative.

The non-oracle method 3 is the primary benchmark.  Method 1 (oracle) is
retained as an upper bound that isolates functional-form error from
estimation noise; method 2 (objective-value NNLS) is retained only to
document the identification failure of the weakest estimator.

Parametric prediction (solve_forward_parametric)
--------------------------------------------------
Given fitted (alpha_k, rho), prediction solves:

  min_w  sum_k alpha_k * sum_{v in OAR_k} max(0, d_v - theta_k)^rho
  s.t.   dose[PTV] >= PTV_PRESCRIBED
         dose[PTV] <= PTV_MAX
         w >= 0

  * rho = 1: LP (linear penalty)
  * rho = 2: SOCP (quadratic, same as existing forward solver)
  * rho = 3: convex polynomial (CVXPY power cone or sum_power)
"""

import os, sys, warnings
import numpy as np
import scipy.optimize as opt
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OAR_KEYS, OAR_GROUPS, N_FUNCTIONS, FUNC_LABELS,
    FORWARD_PARAMS, DOSE_THRESHOLDS, PTV_PRESCRIBED, PTV_MAX,
    MOSEK_TOLERANCES,
)

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Group OAR voxel indices by function index (0..N_FUNCTIONS-1)
def _func_idx_for_oar(oar_key):
    return OAR_GROUPS[oar_key]['func_idx']


def _true_alpha_per_function(fp):
    """Ground-truth alpha folded to one entry per function index, for the
    verbose diagnostic only.  OARs that share a function index (femur_l and
    femur_r both map to func_idx 2 and share alpha) get one coefficient by
    last-write ASSIGNMENT -- NOT summation, which would double-count the
    femur.  This keeps all N_FUNCTIONS entries (e.g. rind_2 at func_idx 5),
    unlike the old OAR_KEYS[:N_FUNCTIONS] slice which dropped rind_2 and
    kept both femurs, printing a wrong reference of [20,20,10,10,5,5]
    instead of the correct [20,20,10,5,5,3]."""
    true_pf = np.zeros(N_FUNCTIONS)
    for k in OAR_KEYS:
        true_pf[_func_idx_for_oar(k)] = fp[f'alpha_{k}']
    return true_pf


def fit_parametric_alpha(training_patients, training_w, rho,
                         training_obj_values=None,
                         forward_params=None, dose_thresholds=None,
                         verbose=True):
    """
    Fit alpha_k (k=0..N_FUNCTIONS-1) via oracle KKT shadow-price matching.

    Oracle upper bound: the true alpha_k are used to compute the true shadow
    prices mu_v^(i) = alpha_k_true * 2 * o_v^(i).  These are projected onto
    the gradient of the assumed parametric form g_v = rho * o_v^(rho-1) via OLS
    per OAR function.  This gives the best possible parametric fit given the
    assumed exponent rho -- any non-oracle estimator can only do worse.

    Returns alpha array of shape (N_FUNCTIONS,).
    """
    fp = forward_params or FORWARD_PARAMS
    dt = dose_thresholds or DOSE_THRESHOLDS

    numerator   = np.zeros(N_FUNCTIONS)
    denominator = np.zeros(N_FUNCTIONS)

    for i, (patient, w_star) in enumerate(zip(training_patients, training_w)):
        w = np.asarray(w_star).ravel()
        D = patient['D']
        dose = np.asarray(D @ w).ravel()

        for oar_key in OAR_KEYS:
            fk = _func_idx_for_oar(oar_key)
            alpha_k_true = fp[f'alpha_{oar_key}']
            theta = dt[oar_key]
            idx = patient['organ_indices'].get(oar_key, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            o_v = np.maximum(0.0, dose[idx] - theta)
            active = o_v > 1e-10
            if not active.any():
                continue
            o = o_v[active]

            # Oracle shadow price from the true quadratic objective (rho=2)
            mu_v = alpha_k_true * 2.0 * o
            # Gradient of the assumed parametric form f_k(z) = a_k z^rho
            if rho == 1:
                g_v = np.ones_like(o)
            elif rho == 2:
                g_v = 2.0 * o
            else:
                g_v = rho * (o ** (rho - 1.0))

            numerator[fk]   += np.dot(mu_v, g_v)
            denominator[fk] += np.dot(g_v,  g_v)

        if verbose:
            print(f"    patient {i} ({patient['patient_id']}): "
                  f"active voxels processed", flush=True)

    alpha_hat = np.where(denominator > 1e-30, numerator / denominator, 0.0)

    if verbose:
        print(f"  Fitted alpha (rho={rho}): {np.round(alpha_hat, 4)}", flush=True)
        true_a = _true_alpha_per_function(fp)
        print(f"  True alpha (per function):{np.round(true_a, 4)}", flush=True)

    return alpha_hat


def fit_keshavarz_obj_value(training_patients, training_w, rho,
                             training_obj_values,
                             forward_params=None, dose_thresholds=None,
                             regularization=0.0,
                             verbose=True):
    """
    Keshavarz (2011) objective-value NNLS estimator (non-oracle).

    For each training patient i, compute
      F_k^(i) = sum_{v in OAR_k} max(0, d_v^(i) - theta_k)^rho
    using the observed optimal beamlet weights w^*(i).  Then solve
      min_{alpha >= 0}  ||F alpha - J||^2  [+ regularization * ||alpha||^2]
    where J_i is the reported optimal cost from the forward cache.

    The analyst needs only (w^*(i), J^(i)) -- no shadow prices, no
    true alpha.  This is the weakest form of Keshavarz imputation.

    Parameters
    ----------
    training_obj_values : array-like, shape (N,)
        Optimal objective values J^(i) from the forward solver cache.
    regularization : float
        Tikhonov coefficient.  0 = plain NNLS.  A small value (1e-2)
        helps when the F matrix is near-singular due to RIND collinearity.

    Returns
    -------
    alpha_hat : ndarray, shape (N_FUNCTIONS,)

    Notes
    -----
    Identification problems for the PortPy dataset:
    * N < K underdetermined at N=5 (5 observations, 6 unknowns).
    * RIND_0 and RIND_2 produce near-proportional F columns (both
      ring-shaped OARs around the same PTV); NNLS cannot separate
      their individual alpha values.
    """
    import scipy.sparse as sp

    fp = forward_params or FORWARD_PARAMS
    dt = dose_thresholds or DOSE_THRESHOLDS
    N = len(training_patients)

    F = np.zeros((N, N_FUNCTIONS))
    for i, (patient, w_star) in enumerate(zip(training_patients, training_w)):
        w = np.asarray(w_star).ravel()
        D = patient['D']
        dose = np.asarray((D @ w).todense()).ravel() if sp.issparse(D @ w) else np.asarray(D @ w).ravel()
        for oar_key in OAR_KEYS:
            fk = _func_idx_for_oar(oar_key)
            theta = dt[oar_key]
            idx = patient['organ_indices'].get(oar_key, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            o_v = np.maximum(0.0, dose[idx] - theta)
            F[i, fk] += np.sum(o_v ** rho)

    J = np.asarray(training_obj_values, dtype=float)

    if regularization > 0:
        aug = np.sqrt(regularization) * np.eye(N_FUNCTIONS)
        F_fit = np.vstack([F, aug])
        J_fit = np.concatenate([J, np.zeros(N_FUNCTIONS)])
    else:
        F_fit, J_fit = F, J

    alpha_hat, residual = opt.nnls(F_fit, J_fit)

    if verbose:
        cond = np.linalg.cond(F)
        print(f"  F matrix: {N}x{N_FUNCTIONS}, cond={cond:.2e}, "
              f"residual={residual:.4e}", flush=True)
        print(f"  Fitted alpha (rho={rho}): {np.round(alpha_hat, 4)}", flush=True)
        true_a = _true_alpha_per_function(fp)
        print(f"  True alpha (per function):{np.round(true_a, 4)}", flush=True)

    return alpha_hat


def _kkt_blocks_full(patient, w_star, rho, ptv_bind_tol, dt):
    """
    Per-patient KKT data over ALL beamlets (no active-set thresholding).

    Returns
    -------
    G   : (n_beam x K) gradient of each basis function w.r.t. every beamlet,
          G[:,k] = sum_{v in OAR_k, o_v>0} rho o_v^(rho-1) D_{v,:}.  Because
          the dose matrix D >= 0 and o_v >= 0, G >= 0 elementwise.
    C_lo: (n_beam x n_lo) = D_{lo,:}^T for PTV voxels at the LOWER bound
          (multiplier mu_lo >= 0).
    C_hi: (n_beam x n_hi) = D_{hi,:}^T for PTV voxels at the UPPER bound
          (multiplier mu_hi >= 0).
    w   : (n_beam,) observed beamlet weights = complementary-slackness weights
          for the beamlet nonneg duals.

    Stationarity over all beamlets:
        s := G alpha - C_lo mu_lo + C_hi mu_hi = nu >= 0,
    with complementary slackness nu_j w_j = 0 penalised (not thresholded), so
    no active-beamlet tolerance is required.
    """
    import scipy.sparse as sp
    w = np.asarray(w_star).ravel()
    D = patient['D']
    Dw = D @ w
    dose = np.asarray(Dw.todense()).ravel() if sp.issparse(Dw) else np.asarray(Dw).ravel()
    n_beam = D.shape[1]

    G = np.zeros((n_beam, N_FUNCTIONS))
    for oar_key in OAR_KEYS:
        fk = _func_idx_for_oar(oar_key)
        theta = dt[oar_key]
        vox_idx = patient['organ_indices'].get(oar_key, np.array([], dtype=int))
        if len(vox_idx) == 0:
            continue
        o_v = np.maximum(0.0, dose[vox_idx] - theta)
        # kink at o_v=0 excluded (for rho=1 these are subgradient-ambiguous).
        act_vox = o_v > 1e-10
        if not act_vox.any():
            continue
        o = o_v[act_vox]
        rows = vox_idx[act_vox]
        if rho == 1:
            weights = np.ones(len(o))
        elif rho == 2:
            weights = 2.0 * o
        else:
            weights = rho * (o ** (rho - 1.0))
        D_sub = D[rows, :]
        if sp.issparse(D_sub):
            D_sub = D_sub.toarray()
        G[:, fk] += D_sub.T @ weights

    ptv_idx = patient['organ_indices'].get('ptv', np.array([], dtype=int))
    if len(ptv_idx):
        d_ptv = dose[ptv_idx]
        lo = ptv_idx[np.abs(d_ptv - PTV_PRESCRIBED) <= ptv_bind_tol]
        hi = ptv_idx[np.abs(d_ptv - PTV_MAX)       <= ptv_bind_tol]
    else:
        lo = hi = np.array([], dtype=int)

    def _cols(rows):
        if len(rows) == 0:
            return np.zeros((n_beam, 0))
        sub = D[rows, :]
        if sp.issparse(sub):
            sub = sub.toarray()
        return sub.T                         # (n_beam x len(rows)) = D_{rows,:}^T

    return G, _cols(lo), _cols(hi), w


def _identifiability_M(blocks):
    """
    Free-sign, complementary-slackness-weighted normal matrix, for
    identifiability geometry ONLY (conditioning + unidentified direction).
    For each patient, weight the gradient by the beamlet weights, diag(w) G,
    project off the range of the binding-constraint columns diag(w) C, and
    accumulate the Gram matrix.  The point estimate uses the sign-constrained
    comp-slack QP, NOT this relaxation; this is reported so the user can see
    which alpha-combinations are weakly determined.
    """
    M = np.zeros((N_FUNCTIONS, N_FUNCTIONS))
    leverage = []
    for (G, C_lo, C_hi, w) in blocks:
        WG = w[:, None] * G
        GtG = WG.T @ WG
        C = np.hstack([C_lo, C_hi]) if (C_lo.shape[1] + C_hi.shape[1]) else None
        if C is not None and C.shape[1] > 0:
            WC = w[:, None] * C
            U, sv, _ = np.linalg.svd(WC, full_matrices=False)
            r = int(np.sum(sv > 1e-9 * sv[0])) if len(sv) else 0
            if r > 0:
                Ur = U[:, :r]
                H = Ur.T @ WG
                GtG = GtG - H.T @ H
        Mi = 0.5 * (GtG + GtG.T)
        M += Mi
        leverage.append(np.trace(Mi))
    return M, np.array(leverage)


def fit_keshavarz_kkt(training_patients, training_w, rho,
                      forward_params=None, dose_thresholds=None,
                      ptv_bind_tol=0.10, regularization=1e-8,
                      return_diagnostics=False, verbose=True):
    """
    Keshavarz, Wang & Boyd (2011) KKT residual estimator (non-oracle), fully
    faithful: dual feasibility enforced and NO active-set tolerance.

    Forward objective sum_k alpha_k phi_k(w),
    phi_k(w) = sum_{v in OAR_k} max(0, d_v - theta_k)^rho, d = D w, under hard
    PTV bounds (lower P, upper M) and w >= 0.  Given observed optima w^*(i),
    alpha is imputed by minimising the KKT residual jointly over alpha and ALL
    multipliers, with complementary slackness for the beamlet nonneg duals
    PENALISED rather than thresholded:

        min   sum_i sum_j ( w_j^(i) * s_j^(i) )^2
        s.t.  s^(i) = G^(i) alpha - C_lo^(i) mu_lo^(i) + C_hi^(i) mu_hi^(i) >= 0
              mu_lo^(i) >= 0,  mu_hi^(i) >= 0,  alpha >= 0,  1^T alpha = 1.

    Here s_j^(i) = nu_j^(i) >= 0 is the beamlet nonneg dual and (w_j s_j)^2 is
    the squared complementary-slackness residual nu_j w_j, which drives s_j->0
    on genuinely active beamlets (large w_j) while only requiring s_j >= 0 on
    inactive ones.  This removes the active-beamlet threshold entirely.  The
    constraint s >= 0 is always feasible (D >= 0 => G >= 0 => s = G alpha >= 0
    at mu = 0), so the QP is well posed for every rho, including the
    misspecified ones.  The PTV duals mu_lo, mu_hi >= 0 enforce dual
    feasibility (lower-bound voxels push dose up, upper-bound voxels push it
    down): the earlier free-sign relaxation is NOT used for the estimate.

    Parameters
    ----------
    ptv_bind_tol : float
        Dose tolerance (Gy) for classifying a PTV voxel as binding.  The
        estimate is insensitive to this over roughly [0.05, 1.0] Gy.
    regularization : float
        Tiny ridge on alpha to keep the QP strictly convex.
    return_diagnostics : bool
        Also return conditioning, the unidentified alpha-direction, and
        per-patient leverage from the identifiability normal matrix.

    Returns
    -------
    alpha_hat : ndarray (K,) on the simplex (prediction is scale-invariant in
        alpha), or (alpha_hat, diagnostics) if return_diagnostics=True.

    Notes
    -----
    For rho=1 the objective is piecewise linear and nondifferentiable at
    d_v = theta_k; voxels within 1e-10 of the threshold are treated as
    inactive, i.e. the natural subgradient g_v in {0, 1} is taken.  This is a
    subgradient KKT residual, reported as such.
    """
    fp = forward_params or FORWARD_PARAMS
    dt = dose_thresholds or DOSE_THRESHOLDS

    blocks = [b for b in (_kkt_blocks_full(p, w, rho, ptv_bind_tol, dt)
                          for p, w in zip(training_patients, training_w))
              if np.linalg.norm(b[0]) > 0]
    if not blocks:
        if verbose:
            print("  no usable patient blocks -> uniform alpha", flush=True)
        out = np.ones(N_FUNCTIONS) / N_FUNCTIONS
        return (out, {}) if return_diagnostics else out

    # ---- faithful Keshavarz comp-slack QP (the estimate) ------------------
    # Builder for the comp-slack QP at a given global rescale.  Dividing every
    # G, C_lo, C_hi by gs and every w by wsc (and the ridge by gs^2 wsc^2)
    # multiplies the whole objective and each ">= 0" row by positive constants,
    # so the minimiser over (alpha-simplex, mu >= 0) is mathematically
    # UNCHANGED -- only the magnitudes the solver sees change.
    def _build(gs, wsc):
        a = cp.Variable(N_FUNCTIONS, nonneg=True)
        cs = [cp.sum(a) == 1.0]
        tms = []
        for (G, C_lo, C_hi, w) in blocks:
            s = (G / gs) @ a
            if C_lo.shape[1]:
                mu_lo = cp.Variable(C_lo.shape[1], nonneg=True)
                s = s - (C_lo / gs) @ mu_lo
            if C_hi.shape[1]:
                mu_hi = cp.Variable(C_hi.shape[1], nonneg=True)
                s = s + (C_hi / gs) @ mu_hi
            cs.append(s >= 0)                                    # nu = s >= 0 (dual feas.)
            tms.append(cp.sum_squares(cp.multiply(w / wsc, s)))  # comp-slack residual
        objective = sum(tms) + (regularization / (gs ** 2 * wsc ** 2)) * cp.sum_squares(a)
        return cp.Problem(cp.Minimize(objective), cs), a

    def _absmax(arr):
        return float(np.abs(arr).max()) if arr.size else 0.0
    gscale = max((max(_absmax(G), _absmax(C_lo), _absmax(C_hi))
                  for (G, C_lo, C_hi, w) in blocks), default=0.0) or 1.0
    wscale = max((_absmax(w) for (G, C_lo, C_hi, w) in blocks), default=0.0) or 1.0

    # Two-stage solve.
    #   Stage 1 -- UNSCALED problem on CLARABEL.  Reproduces the published
    #     rho=1,2,3 table byte-for-byte: CLARABEL returns 'optimal' and stage 2
    #     is never reached.
    #   Stage 2 -- only if stage 1 is non-optimal, which happens for steep forms
    #     (rho>=4): the gradient weights rho*o^(rho-1) make the G/C dynamic range
    #     ~1e7, and BOTH solvers misreport the unscaled QP infeasible.  We then
    #     solve the ESTIMATE-PRESERVING rescaled QP on MOSEK, whose equilibration
    #     handles the wide range.  CLARABEL is deliberately NOT used on the
    #     rescaled problem -- it converges to a wrong vertex there (verified: it
    #     shifts the well-identified rho=2 estimate by ~23%), whereas MOSEK on
    #     the rescaled problem recovers rho=2 to ~1e-3 and solves rho=4.
    attempts = [(1.0, 1.0, 'CLARABEL')]
    if 'MOSEK' in cp.installed_solvers():
        attempts.append((gscale, wscale, 'MOSEK'))
    prob, alpha, used = None, None, (1.0, 1.0)
    last_status = None
    for gs, wsc, slv in attempts:
        prob_i, alpha_i = _build(gs, wsc)
        try:
            prob_i.solve(solver=getattr(cp, slv), verbose=False)
        except Exception as e:
            last_status = f'exception({slv}): {e}'
            continue
        last_status = f'{slv}(gs={gs:.3g},wsc={wsc:.3g}):{prob_i.status}'
        if prob_i.status in ('optimal', 'optimal_inaccurate') and alpha_i.value is not None:
            prob, alpha, used = prob_i, alpha_i, (gs, wsc)
            break

    if prob is None or alpha is None or alpha.value is None:
        warnings.warn(f"Keshavarz KKT QP non-optimal (last={last_status}); "
                      f"returning uniform alpha (degenerate fallback, not a "
                      f"valid estimate).", RuntimeWarning)
        out = np.ones(N_FUNCTIONS) / N_FUNCTIONS
        return (out, {}) if return_diagnostics else out

    a = np.maximum(0.0, alpha.value)
    alpha_hat = a / a.sum() if a.sum() > 1e-12 else np.ones(N_FUNCTIONS) / N_FUNCTIONS
    # Report the residual on the ORIGINAL objective scale (undo the stage-2
    # rescale) so it is comparable across rho regardless of which stage solved.
    resid = float(prob.value) * (used[0] ** 2 * used[1] ** 2)

    # ---- identifiability geometry (free-sign normal matrix) ---------------
    M, leverage = _identifiability_M(blocks)
    eigval, eigvec = np.linalg.eigh(M)
    cond = float(eigval[-1] / eigval[0]) if eigval[0] > 1e-30 else np.inf
    unident_dir = eigvec[:, 0]                           # least-determined direction

    if verbose:
        true_a = _true_alpha_per_function(fp)
        print(f"  patients used={len(blocks)}  comp-slack resid={resid:.4e}", flush=True)
        print(f"  alpha_hat (sign-constrained): {np.round(alpha_hat, 4)}", flush=True)
        print(f"  true alpha/sum (per function):{np.round(true_a/true_a.sum(),4)}", flush=True)
        print(f"  cond(M)={cond:.2e}  per-patient leverage shares (N entries)="
              f"{np.round(leverage/leverage.sum(),3)}", flush=True)
        print(f"  unidentified direction (|smallest eigvec|, per function): "
              f"{np.round(np.abs(unident_dir),3)}", flush=True)

    if return_diagnostics:
        diag = dict(M=M, eigvals=eigval, cond=cond, unident_dir=unident_dir,
                    leverage=leverage, resid=resid, n_patients=len(blocks))
        return alpha_hat, diag
    return alpha_hat


def solve_forward_parametric(D, organ_indices, alpha_k, rho,
                              forward_params=None, dose_thresholds=None,
                              solver=None, mosek_params=None,
                              verbose=False):
    """
    Solve the parametric forward problem:

      min_w  sum_k alpha_k * sum_{v in OAR_k} max(0, d_v - theta_k)^rho
      s.t.   d[PTV] >= PTV_PRESCRIBED
             d[PTV] <= PTV_MAX
             w >= 0

    rho=1: LP
    rho=2: SOCP (quadratic, same structure as existing forward solver)
    rho=3: convex polynomial (CVXPY power cone)

    Returns beamlet-weight vector w (or None on failure).
    """
    fp = forward_params or FORWARD_PARAMS
    dt = dose_thresholds or DOSE_THRESHOLDS
    mp = mosek_params or MOSEK_TOLERANCES

    n_voxels, n_beamlets = D.shape
    w = cp.Variable(n_beamlets, nonneg=True)
    dose = D @ w

    constraints = []
    obj_terms = []

    # PTV hard bounds
    ptv_idx = organ_indices.get('ptv', np.array([], dtype=int))
    if len(ptv_idx) > 0:
        constraints.append(dose[ptv_idx] >= PTV_PRESCRIBED)
        constraints.append(dose[ptv_idx] <= PTV_MAX)

    # Map function index -> list of (voxel indices, threshold).  alpha_k is the
    # fitted per-function coefficient; OARs sharing a func_idx share it.
    func_parts = {}    # func_idx -> list of (idx_array, theta)
    for oar_key in OAR_KEYS:
        idx = organ_indices.get(oar_key, np.array([], dtype=int))
        if len(idx) == 0:
            continue
        fk = _func_idx_for_oar(oar_key)
        func_parts.setdefault(fk, []).append((idx, dt[oar_key]))

    for fk, parts in func_parts.items():
        a_val = float(alpha_k[fk])
        if a_val <= 0:
            continue
        for (idx, theta) in parts:
            if rho == 1:
                # Linear: sum_v max(0, d_v - theta)
                o_v = cp.Variable(len(idx), nonneg=True)
                constraints.append(o_v >= dose[idx] - theta)
                obj_terms.append(a_val * cp.sum(o_v))
            elif rho == 2:
                # Quadratic: sum_v max(0, d_v - theta)^2
                o_v = cp.Variable(len(idx), nonneg=True)
                constraints.append(o_v >= dose[idx] - theta)
                obj_terms.append(a_val * cp.sum_squares(o_v))
            elif rho == 3:
                # Cubic: sum_v max(0, d_v - theta)^3
                # Use cp.power(o_v, 3) which is DCP for o_v >= 0
                o_v = cp.Variable(len(idx), nonneg=True)
                constraints.append(o_v >= dose[idx] - theta)
                obj_terms.append(a_val * cp.sum(cp.power(o_v, 3)))
            else:
                # General power: sum_v max(0, d_v - theta)^rho, rho > 1
                o_v = cp.Variable(len(idx), nonneg=True)
                constraints.append(o_v >= dose[idx] - theta)
                obj_terms.append(a_val * cp.sum(cp.power(o_v, rho)))

    if not obj_terms:
        return None

    # Solver selection: prefer MOSEK if available; otherwise fall back to an
    # open-source conic solver so a MOSEK-less public user still reproduces the
    # table.  rho>=2 needs SOC/power cones (CLARABEL); rho=1 is an LP (CLARABEL
    # or ECOS).  Pass solver= explicitly to override.
    installed = cp.installed_solvers()
    if solver is None:
        solver = 'MOSEK' if 'MOSEK' in installed else ('CLARABEL'
                 if 'CLARABEL' in installed else 'ECOS')
    elif solver not in installed:
        warnings.warn(f"requested solver {solver!r} not installed; "
                      f"available: {installed}", RuntimeWarning)

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)
    try:
        if solver == 'MOSEK':
            prob.solve(solver=cp.MOSEK, verbose=verbose, mosek_params=mp)
        else:
            prob.solve(solver=getattr(cp, solver), verbose=verbose)
    except cp.error.SolverError as e:
        warnings.warn(f"forward solve failed with solver {solver}: {e}",
                      RuntimeWarning)
        return None
    except Exception as e:
        if verbose:
            print(f"    Solver error: {e}")
        return None

    if prob.status in ('optimal', 'optimal_inaccurate'):
        return w.value
    warnings.warn(f"forward solve non-optimal (status={prob.status}, "
                  f"solver={solver})", RuntimeWarning)
    return None


def rel_error_beamlets(w_pred, w_true):
    """Norm-based relative error: ||w_pred - w_true|| / ||w_true||.
    Matches the MeanRelError metric used in run_prediction.py."""
    diff = np.asarray(w_pred) - np.asarray(w_true)
    norm_true = np.linalg.norm(w_true)
    if norm_true < 1e-12:
        return np.nan
    return np.linalg.norm(diff) / norm_true


def rel_error_dose(w_pred, w_true, D):
    """Norm-based relative error in dose space: ||D w_pred - D w_true|| / ||D w_true||.
    Same Rel-L2 as rel_error_beamlets but on the voxel dose d = D w."""
    d_true = np.asarray(D @ np.asarray(w_true)).ravel()
    d_pred = np.asarray(D @ np.asarray(w_pred)).ravel()
    norm_true = np.linalg.norm(d_true)
    if norm_true < 1e-12:
        return np.nan
    return np.linalg.norm(d_pred - d_true) / norm_true
