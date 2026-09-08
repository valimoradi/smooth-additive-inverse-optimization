"""
Inverse-optimization recovery for the four ablation models. The recovered structural
functions are parameterized by (delta, lambda) at the observed outcomes Z_hat, with the
KKT/sub-optimality constraints (C5 convexity+smoothness, C6 monotonicity, C7 normalization,
C8 sub-optimality, D1-D4 duality) from the paper.

Models:
  add_smooth        3-stage (min eps @ BETA_0 -> bisect beta* -> max sum-delta), MOSEK, additive
  add_lp            2-stage pure LP (min eps -> max sum-delta), GUROBI, additive non-smooth
  nonadd_smooth     3-stage, MOSEK, non-additive (single joint hull)
  nonadd_nonsmooth  2-stage with forced large beta, MOSEK, non-additive non-smooth

Transcribed from notebook cells 22/27/29/31 (audited verbatim); constants in config.py.
"""
import os
import time
import numpy as np
import cvxpy as cp

from config import PARAMS, RECOVERY, RESULTS, ML_DIR, IDX_NORM_ZERO, IDX_NORM_MAX
from forward_model import preprocess_outcomes


def load_data(data_dir, N):
    D = np.load(f'{data_dir}/D_train.npy', allow_pickle=True)
    w = np.load(f'{data_dir}/w_train.npy', allow_pickle=True)
    o = np.load(f'{data_dir}/organ_sets_train.npy', allow_pickle=True)
    return D[:N], w[:N], o[:N]


# ======================================================================================
# ADDITIVE models (delta is (N,K) per-component; C5 over sorted adjacent pairs per outcome)
# ======================================================================================
def _vars_additive(N, K, D_obs, organ_sets_obs):
    v = {"delta": cp.Variable((N, K), nonneg=True), "delta_global": cp.Variable(N),
         "lambdas": cp.Variable((N, K)), "epsilon": cp.Variable(nonneg=True),
         "mu_1": [], "mu_2": [], "nu_1": [], "nu_2": [], "pi": [], "sigma_1": [], "sigma_2": []}
    for i in range(N):
        n_b = D_obs[i].shape[1]; org = organ_sets_obs[i]
        v["mu_1"].append(cp.Variable(len(org.get('ptv', [])), nonneg=True))
        v["mu_2"].append(cp.Variable(len(org.get('ptv', [])), nonneg=True))
        v["nu_1"].append(cp.Variable(len(org.get('oar1', [])), nonneg=True))
        v["nu_2"].append(cp.Variable(len(org.get('oar2', [])), nonneg=True))
        v["pi"].append(cp.Variable(len(org.get('oar3', [])), nonneg=True))
        v["sigma_1"].append(cp.Variable(n_b, nonneg=True))
        v["sigma_2"].append(cp.Variable(n_b, nonneg=True))
    return v


def _cons_additive(V, Z_hat, D_obs, organ_sets_obs, params, beta_val, epsilon_val, U_MAX, smooth):
    """smooth=True -> quadratic smoothness term (beta); smooth=False -> pure LP (no q_term)."""
    delta = V["delta"]; delta_global = V["delta_global"]; lambdas = V["lambdas"]
    N, K = Z_hat.shape
    cons = []
    beta_inv = 1.0 / (2.0 * beta_val) if smooth else 0.0
    for k in range(K):
        idx_ord = np.argsort(Z_hat[:, k])
        i_idx = idx_ord[:-1]; j_idx = idx_ord[1:]
        dz = Z_hat[j_idx, k] - Z_hat[i_idx, k]
        di = delta[i_idx, k]; dj = delta[j_idx, k]
        li = lambdas[i_idx, k]; lj = lambdas[j_idx, k]
        if smooth:
            q = beta_inv * cp.square(li - lj)
            cons.append(dj - di - cp.multiply(li, dz) >= q)
            cons.append(di - dj - cp.multiply(lj, -dz) >= q)
        else:
            cons.append(dj - di - cp.multiply(li, dz) >= 0)
            cons.append(di - dj - cp.multiply(lj, -dz) >= 0)
    cons.append(lambdas >= 0)
    cons.append(delta_global == cp.sum(delta, axis=1))
    if IDX_NORM_ZERO < N:
        cons.append(delta_global[IDX_NORM_ZERO] == 0)
    if IDX_NORM_MAX < N:
        cons.append(delta_global[IDX_NORM_MAX] == U_MAX)
    for i in range(N):
        lam_i = lambdas[i, :]; z_hat_i = Z_hat[i, :]; D_i = D_obs[i]; org_i = organ_sets_obs[i]
        n_b = D_i.shape[1]
        mu1 = V["mu_1"][i]; mu2 = V["mu_2"][i]; nu1 = V["nu_1"][i]; nu2 = V["nu_2"][i]
        pi = V["pi"][i]; s1 = V["sigma_1"][i]; s2 = V["sigma_2"][i]
        vp = org_i.get('ptv', np.array([])); v1 = org_i.get('oar1', np.array([]))
        v2 = org_i.get('oar2', np.array([])); v3 = org_i.get('oar3', np.array([]))
        ot = []
        if vp.size > 0:
            ot.append(cp.sum(mu1 * params['theta_prescribed'] - mu2 * params['theta_max_PTV']))
        if v1.size > 0:
            ot.append(-cp.sum(nu1 * params['theta_thresh_1']))
        if v2.size > 0:
            ot.append(-cp.sum(nu2 * params['theta_thresh_2']))
        cons.append(lam_i @ z_hat_i - sum(ot) <= epsilon_val)
        if v1.size > 0:
            cons.append(nu1 <= lam_i[0])
        if v2.size > 0:
            cons.append(nu2 <= lam_i[1])
        if v3.size > 0:
            cons.append(cp.sum(pi) <= lam_i[2])
        grad = -s1 + s2
        grad += (params['beta_1'] / n_b) * cp.sum(s1)
        grad -= (params['beta_2'] / n_b) * cp.sum(s2)
        if vp.size > 0:
            grad -= (mu1 - mu2) @ D_i[vp, :]
        if v1.size > 0:
            grad += nu1 @ D_i[v1, :]
        if v2.size > 0:
            grad += nu2 @ D_i[v2, :]
        if v3.size > 0:
            grad += pi @ D_i[v3, :]
        cons.append(grad >= 0)
    return cons


# ======================================================================================
# NON-ADDITIVE models (delta_global only; C5 over all i!=j pairs jointly)
# ======================================================================================
def _vars_nonadd(N, K, D_obs, organ_sets_obs):
    v = {"delta_global": cp.Variable(N, nonneg=True), "lambdas": cp.Variable((N, K)),
         "epsilon": cp.Variable(nonneg=True),
         "mu_1": [], "mu_2": [], "nu_1": [], "nu_2": [], "pi": [], "sigma_1": [], "sigma_2": []}
    for i in range(N):
        n_b = D_obs[i].shape[1]; org = organ_sets_obs[i]
        v["mu_1"].append(cp.Variable(len(org.get('ptv', [])), nonneg=True))
        v["mu_2"].append(cp.Variable(len(org.get('ptv', [])), nonneg=True))
        v["nu_1"].append(cp.Variable(len(org.get('oar1', [])), nonneg=True))
        v["nu_2"].append(cp.Variable(len(org.get('oar2', [])), nonneg=True))
        v["pi"].append(cp.Variable(len(org.get('oar3', [])), nonneg=True))
        v["sigma_1"].append(cp.Variable(n_b, nonneg=True))
        v["sigma_2"].append(cp.Variable(n_b, nonneg=True))
    return v


def _cons_nonadd(V, Z_hat, D_obs, organ_sets_obs, params, beta_val, epsilon_val, U_MAX, smooth):
    """smooth=True -> joint quadratic smoothness term (1/2beta); smooth=False -> pure LP (no q_term)."""
    delta_global = V["delta_global"]; lambdas = V["lambdas"]
    N, K = Z_hat.shape
    cons = []
    beta_inv = 1.0 / (2.0 * beta_val) if smooth else 0.0
    idx_i_full = np.repeat(np.arange(N), N); idx_j_full = np.tile(np.arange(N), N)
    mask = idx_i_full != idx_j_full
    idx_i = idx_i_full[mask]; idx_j = idx_j_full[mask]
    Z_diff = Z_hat[idx_j] - Z_hat[idx_i]
    linear_term = cp.sum(cp.multiply(lambdas[idx_i], Z_diff), axis=1)
    lam_diff = lambdas[idx_j] - lambdas[idx_i]
    q_term = beta_inv * cp.sum(cp.square(lam_diff), axis=1)
    cons.append(delta_global[idx_j] - delta_global[idx_i] - linear_term >= q_term)
    cons.append(lambdas >= 0)
    if IDX_NORM_ZERO < N:
        cons.append(delta_global[IDX_NORM_ZERO] == 0)
    if IDX_NORM_MAX < N:
        cons.append(delta_global[IDX_NORM_MAX] == U_MAX)
    for i in range(N):
        lam_i = lambdas[i, :]; z_hat_i = Z_hat[i, :]; D_i = D_obs[i]; n_b = D_i.shape[1]
        mu1 = V["mu_1"][i]; mu2 = V["mu_2"][i]; nu1 = V["nu_1"][i]; nu2 = V["nu_2"][i]
        pi = V["pi"][i]; s1 = V["sigma_1"][i]; s2 = V["sigma_2"][i]
        org_i = organ_sets_obs[i]
        h_i = 0
        if mu1.size > 0:
            h_i += cp.sum(mu1) * params['theta_prescribed']
        if mu2.size > 0:
            h_i -= cp.sum(mu2) * params['theta_max_PTV']
        if nu1.size > 0:
            h_i -= cp.sum(nu1) * params['theta_thresh_1']
        if nu2.size > 0:
            h_i -= cp.sum(nu2) * params['theta_thresh_2']
        cons.append(lam_i @ z_hat_i - h_i <= epsilon_val)
        if nu1.size > 0:
            cons.append(nu1 <= lam_i[0])
        if nu2.size > 0:
            cons.append(nu2 <= lam_i[1])
        if pi.size > 0:
            cons.append(cp.sum(pi) <= lam_i[2])
        tD_p = (mu1 - mu2) @ D_i[org_i.get('ptv', []), :] if mu1.size > 0 else 0
        tD_1 = nu1 @ D_i[org_i.get('oar1', []), :] if nu1.size > 0 else 0
        tD_2 = nu2 @ D_i[org_i.get('oar2', []), :] if nu2.size > 0 else 0
        tD_3 = pi @ D_i[org_i.get('oar3', []), :] if pi.size > 0 else 0
        ts1 = (params['beta_1'] / n_b) * cp.sum(s1)
        ts2 = (params['beta_2'] / n_b) * cp.sum(s2)
        LHS = (-tD_p + tD_1 + tD_2 + tD_3 - s1 + ts1 + s2 - ts2)
        cons.append(LHS >= 0)
    return cons


# ======================================================================================
# Per-model runners
# ======================================================================================
def _run_add_smooth(N, Z, D, O, cfg):
    tol = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': cfg["MOSEK_TOL"] for k in ("PFEAS", "DFEAS", "REL_GAP")}
    # Looser fallback: MOSEK 11 throws on some small (e.g. N=20) stage-3 problems at 1e-6;
    # retrying the SAME beta at 1e-5 solves it at the natural bisection beta* (no escalation),
    # while N>=40 still solves at 1e-6 first so cached params reproduce exactly.
    tol_fb = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': 1e-5 for k in ("PFEAS", "DFEAS", "REL_GAP")}

    def feasible(beta, eps, retries=3):
        if beta <= 0:
            return False
        b = beta
        for _ in range(retries):
            try:
                V = _vars_additive(N, 3, D, O)
                c = _cons_additive(V, Z, D, O, PARAMS, b, eps, cfg["U_MAX"], smooth=True)
                p = cp.Problem(cp.Minimize(0), c)
                p.solve(solver="MOSEK", verbose=False, mosek_params=tol)
                if p.status in {"optimal", "optimal_inaccurate"}:
                    return True
                b *= 1.1
            except cp.SolverError:
                b *= 1.1
        return False

    # Stage 1
    V1 = _vars_additive(N, 3, D, O)
    c1 = _cons_additive(V1, Z, D, O, PARAMS, cfg["BETA_0"], V1["epsilon"], cfg["U_MAX"], smooth=True)
    p1 = cp.Problem(cp.Minimize(V1["epsilon"]), c1)
    p1.solve(solver="MOSEK", verbose=False, mosek_params=tol)
    if p1.status != "optimal":
        raise RuntimeError(f"Stage 1 failed: {p1.status}")
    eps_star = V1["epsilon"].value
    eps2 = eps_star + cfg["EPSILON_TOL"]
    # Stage 2: bisection
    bh = cfg["BISECT_BETA_INIT"]; bl = cfg["BISECT_BETA_LOW"]
    while not feasible(bh, eps2):
        bh *= 2.0
        if bh > 1e9:
            raise RuntimeError("infeasible even at large beta")
    while bh - bl > cfg["BISECT_GAP_TOL"]:
        bm = 0.5 * (bl + bh)
        if feasible(bm, eps2):
            bh = bm
        else:
            bl = bm
    beta_star = bh
    # Stage 3: maximize sum-delta. Try 1e-6 then 1e-5 at each beta; escalate only if both fail.
    beta3 = beta_star
    for _ in range(cfg["STAGE3_ATTEMPTS"]):
        for mtol in (tol, tol_fb):
            try:
                V3 = _vars_additive(N, 3, D, O)
                c3 = _cons_additive(V3, Z, D, O, PARAMS, beta3, eps2, cfg["U_MAX"], smooth=True)
                p3 = cp.Problem(cp.Maximize(cp.sum(V3["delta_global"])), c3)
                p3.solve(solver="MOSEK", verbose=False, mosek_params=mtol)
                if p3.status in ["optimal", "optimal_inaccurate"]:
                    return {"final_delta_global": V3["delta_global"].value, "final_delta_comp": V3["delta"].value,
                            "final_lambda": V3["lambdas"].value, "Z_hat_obs": Z,
                            "beta_star": np.array([beta3]), "epsilon_star": np.array([eps_star])}, "_v3"
            except (cp.SolverError, Exception):
                pass
        beta3 *= cfg["STAGE3_ESCALATE"]
    raise RuntimeError("Stage 3 failed after all escalation attempts")


def _run_add_lp(N, Z, D, O, cfg):
    V1 = _vars_additive(N, 3, D, O)
    c1 = _cons_additive(V1, Z, D, O, PARAMS, 1.0, V1["epsilon"], cfg["U_MAX"], smooth=False)
    p1 = cp.Problem(cp.Minimize(V1["epsilon"]), c1)
    p1.solve(solver="GUROBI", verbose=False)
    if p1.status != "optimal":
        raise RuntimeError(f"Stage 1 failed: {p1.status}")
    eps_star = V1["epsilon"].value
    eps2 = eps_star + cfg["EPSILON_TOL"]
    V2 = _vars_additive(N, 3, D, O)
    c2 = _cons_additive(V2, Z, D, O, PARAMS, 1.0, eps2, cfg["U_MAX"], smooth=False)
    p2 = cp.Problem(cp.Maximize(cp.sum(V2["delta_global"])), c2)
    p2.solve(solver="GUROBI", verbose=False)
    if p2.status != "optimal":
        raise RuntimeError(f"Stage 2 failed: {p2.status}")
    return {"final_delta_global": V2["delta_global"].value, "final_delta_comp": V2["delta"].value,
            "final_lambda": V2["lambdas"].value, "Z_hat_obs": Z,
            "epsilon_star": np.array([eps_star])}, "_lp"


def _run_nonadd_smooth(N, Z, D, O, cfg):
    def feasible(beta, eps):
        if beta <= 0:
            return False
        try:
            V = _vars_nonadd(N, 3, D, O)
            c = _cons_nonadd(V, Z, D, O, PARAMS, beta, eps, cfg["U_MAX"], smooth=True)
            p = cp.Problem(cp.Minimize(0), c)
            p.solve(solver="MOSEK", verbose=False)
            return p.status in {"optimal", "optimal_inaccurate"}
        except Exception:
            return False

    V1 = _vars_nonadd(N, 3, D, O)
    c1 = _cons_nonadd(V1, Z, D, O, PARAMS, cfg["BETA_0"], V1["epsilon"], cfg["U_MAX"], smooth=True)
    p1 = cp.Problem(cp.Minimize(V1["epsilon"]), c1)
    p1.solve(solver="MOSEK", verbose=False)
    if p1.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage 1 failed: {p1.status}")
    eps_star = p1.value
    eps2 = eps_star + cfg["EPSILON_TOL"]
    bh = cfg["BISECT_BETA_INIT"]; bl = cfg["BISECT_BETA_LOW"]
    while not feasible(bh, eps2):
        bh *= 2.0
        if bh > 1e9:
            raise RuntimeError("infeasible even at large beta")
    while bh - bl > cfg["BISECT_GAP_TOL"]:
        bm = 0.5 * (bl + bh)
        if feasible(bm, eps2):
            bh = bm
        else:
            bl = bm
    beta_star = bh
    tol = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': cfg["MOSEK_TOL"] for k in ("PFEAS", "DFEAS", "REL_GAP")}
    V3 = _vars_nonadd(N, 3, D, O)
    c3 = _cons_nonadd(V3, Z, D, O, PARAMS, beta_star, eps2, cfg["U_MAX"], smooth=True)
    p3 = cp.Problem(cp.Maximize(cp.sum(V3["delta_global"])), c3)
    try:
        p3.solve(solver="MOSEK", verbose=False, mosek_params=tol)
    except Exception:
        p3.solve(solver="MOSEK", verbose=True)
    if p3.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage 3 failed: {p3.status}")
    return {"final_delta_global": V3["delta_global"].value, "final_lambda": V3["lambdas"].value,
            "Z_hat_obs": Z, "beta_star": np.array([beta_star]), "epsilon_star": np.array([eps_star])}, "_vS"


def _run_nonadd_nonsmooth(N, Z, D, O, cfg):
    beta = cfg["BETA_NON_SMOOTH"]
    tol1 = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': cfg["MOSEK_TOL_S1"] for k in ("PFEAS", "DFEAS", "REL_GAP")}
    tol2 = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': cfg["MOSEK_TOL_S2"] for k in ("PFEAS", "DFEAS", "REL_GAP")}
    V1 = _vars_nonadd(N, 3, D, O)
    c1 = _cons_nonadd(V1, Z, D, O, PARAMS, beta, V1["epsilon"], cfg["U_MAX"], smooth=True)
    p1 = cp.Problem(cp.Minimize(V1["epsilon"]), c1)
    p1.solve(solver="MOSEK", verbose=False, mosek_params=tol1)
    if p1.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage 1 failed: {p1.status}")
    eps_star = p1.value
    eps2 = eps_star + cfg["EPSILON_TOL"]
    V2 = _vars_nonadd(N, 3, D, O)
    c2 = _cons_nonadd(V2, Z, D, O, PARAMS, beta, eps2, cfg["U_MAX"], smooth=True)
    p2 = cp.Problem(cp.Maximize(cp.sum(V2["delta_global"])), c2)
    p2.solve(solver="MOSEK", verbose=False, mosek_params=tol2)
    if p2.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage 2 failed: {p2.status}")
    return {"final_delta_global": V2["delta_global"].value, "final_lambda": V2["lambdas"].value,
            "Z_hat_obs": Z, "beta_star": np.array([beta]), "epsilon_star": np.array([eps_star])}, "_vNS"


def _run_nonadd_nonsmooth_lp(N, Z, D, O, cfg):
    """TRUE non-additive convex-only: no smoothness term at all (beta_inv=0), 2-stage, MOSEK.
    Companion to _run_nonadd_nonsmooth (large-beta approximation) for the approx-vs-true check.
    Everything except the smoothness term is held identical to the approximation so the two
    differ only in beta_inv (5e-6 vs 0)."""
    tol1 = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': cfg["MOSEK_TOL_S1"] for k in ("PFEAS", "DFEAS", "REL_GAP")}
    tol2 = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': cfg["MOSEK_TOL_S2"] for k in ("PFEAS", "DFEAS", "REL_GAP")}
    V1 = _vars_nonadd(N, 3, D, O)
    c1 = _cons_nonadd(V1, Z, D, O, PARAMS, 1.0, V1["epsilon"], cfg["U_MAX"], smooth=False)
    p1 = cp.Problem(cp.Minimize(V1["epsilon"]), c1)
    p1.solve(solver="MOSEK", verbose=False, mosek_params=tol1)
    if p1.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage 1 failed: {p1.status}")
    eps_star = p1.value
    eps2 = eps_star + cfg["EPSILON_TOL"]
    V2 = _vars_nonadd(N, 3, D, O)
    c2 = _cons_nonadd(V2, Z, D, O, PARAMS, 1.0, eps2, cfg["U_MAX"], smooth=False)
    p2 = cp.Problem(cp.Maximize(cp.sum(V2["delta_global"])), c2)
    p2.solve(solver="MOSEK", verbose=False, mosek_params=tol2)
    if p2.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage 2 failed: {p2.status}")
    return {"final_delta_global": V2["delta_global"].value, "final_lambda": V2["lambdas"].value,
            "Z_hat_obs": Z, "epsilon_star": np.array([eps_star])}, "_vNSLP"


_RUNNERS = {"add_smooth": _run_add_smooth, "add_lp": _run_add_lp,
            "nonadd_smooth": _run_nonadd_smooth, "nonadd_nonsmooth": _run_nonadd_nonsmooth,
            "nonadd_nonsmooth_lp": _run_nonadd_nonsmooth_lp}


def run_recovery(model, sizes, ml_dir=ML_DIR, out_dir=None, perturb=0.0):
    """Recover parameters for `model` across `sizes`, saving .npy files per size.
    perturb>0: multiply each training plan w by (1 +/- perturb) elementwise (fixed per-patient
    seed; the two anchors at idx 0,1 are left clean so the gauge is unaffected), emulating
    near-optimal (noisy) observations -- the consumer perturbed regime. The test set stays clean."""
    cfg = RECOVERY[model]
    out_dir = out_dir or RESULTS[model]
    os.makedirs(out_dir, exist_ok=True)
    for N in sizes:
        t0 = time.time()
        D, w, O = load_data(ml_dir, N)
        N = len(D)
        if N <= 1:
            continue
        if perturb and perturb > 0:
            w = list(w)
            for i in range(2, N):  # skip the two anchors (idx 0,1) so the normalization gauge stays clean
                r = np.random.default_rng(777_000 + i).uniform(-perturb, perturb, size=np.shape(w[i]))
                w[i] = np.asarray(w[i]) * (1.0 + r)
        Z = preprocess_outcomes(w, D, O, PARAMS)
        try:
            arrays, suffix = _RUNNERS[model](N, Z, D, O, cfg)
        except Exception as e:
            print(f"[{model}] N={N} FAILED: {e}", flush=True)
            continue
        for stem, arr in arrays.items():
            np.save(f'{out_dir}/{stem}_N{N}{suffix}.npy', arr)
        beta_msg = f" beta*={float(arrays.get('beta_star', [float('nan')])[0]):.4f}" if 'beta_star' in arrays else ""
        print(f"[{model}] N={N}: eps*={float(arrays['epsilon_star'][0]):.8f}{beta_msg} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    import sys
    from config import TRAINING_SIZES
    model = sys.argv[1] if len(sys.argv) > 1 else "add_smooth"
    sizes = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else TRAINING_SIZES
    run_recovery(model, sizes)
