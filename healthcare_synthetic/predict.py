"""
Prediction: for each held-out test patient, solve the imputed forward problem under the
recovered structural functions and compare the predicted beamlet weights to ground truth.
Writes a per-model CSV of (training_size, MAE, MAPE, RMSE, cosine-similarity).

The imputed forward problem uses the perspective (SOCP) form for the smooth models and
the max-of-affines (LP) form for the additive non-smooth model. Transcribed from notebook
cells 26/28/30/32 (audited verbatim); the same 3 borderline-infeasible test patients are
skipped at every N (clinical feasibility is objective-independent).
"""
import os
import time
import numpy as np
import cvxpy as cp
import pandas as pd

from config import PARAMS, RESULTS, ML_DIR, PRED_OUT, PRED_CSV


def _load_test(ml_dir=ML_DIR):
    D = np.load(f'{ml_dir}/D_test.npy', allow_pickle=True)
    w = np.load(f'{ml_dir}/w_test.npy', allow_pickle=True)
    o = np.load(f'{ml_dir}/organ_sets_test.npy', allow_pickle=True)
    return D, w, o


def _feasible_block(constraints, D_patient, w, o1, o2, d_max, organ_sets_patient, p):
    """Shared clinical feasibility constraints X(D) mapping w -> outcomes."""
    idx_ptv = organ_sets_patient.get('ptv', np.array([]))
    idx_oar1 = organ_sets_patient.get('oar1', np.array([]))
    idx_oar2 = organ_sets_patient.get('oar2', np.array([]))
    idx_oar3 = organ_sets_patient.get('oar3', np.array([]))
    dose = D_patient @ w
    if idx_ptv.size > 0:
        constraints.append(dose[idx_ptv] >= p['theta_prescribed'])
        constraints.append(dose[idx_ptv] <= p['theta_max_PTV'])
    if idx_oar1.size > 0:
        constraints.append(o1 >= dose[idx_oar1] - p['theta_thresh_1'])
    if idx_oar2.size > 0:
        constraints.append(o2 >= dose[idx_oar2] - p['theta_thresh_2'])
    if idx_oar3.size > 0:
        constraints.append(d_max >= dose[idx_oar3])
    n_b = D_patient.shape[1]
    sum_w = cp.sum(w)
    constraints.append(w >= (p['beta_1'] / n_b) * sum_w)
    constraints.append(w <= (p['beta_2'] / n_b) * sum_w)


def _solve_additive_socp(D_patient, org, p, lp):
    """add_smooth: perspective SOCP with per-(j,k) alpha/v/t."""
    delta_p, lambda_p, z_hat_p, beta_p = lp['delta'], lp['lambda'], lp['z_hat'], lp['beta']
    N, K = delta_p.shape
    n_b = D_patient.shape[1]
    n1 = len(org.get('oar1', [])); n2 = len(org.get('oar2', []))
    w = cp.Variable(n_b, nonneg=True); o1 = cp.Variable(n1, nonneg=True)
    o2 = cp.Variable(n2, nonneg=True); d_max = cp.Variable(nonneg=True)
    alpha = cp.Variable((N, K), nonneg=True); v = cp.Variable((N, K)); t = cp.Variable((N, K), nonneg=True)
    obj = cp.Minimize(cp.sum((beta_p / 2.0) * t + cp.multiply(v, lambda_p)
                             - cp.multiply(alpha, cp.multiply(lambda_p, z_hat_p))
                             + cp.multiply(alpha, delta_p)))
    cons = [cp.sum(alpha, axis=0) == 1]
    quad = v - cp.multiply(alpha, z_hat_p)
    for j in range(N):
        for k in range(K):
            cons.append(cp.quad_over_lin(quad[j, k], alpha[j, k]) <= t[j, k])
    cons.append(cp.sum(o1) <= cp.sum(v[:, 0]))
    cons.append(cp.sum(o2) <= cp.sum(v[:, 1]))
    cons.append(d_max <= cp.sum(v[:, 2]))
    _feasible_block(cons, D_patient, w, o1, o2, d_max, org, p)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="MOSEK", verbose=False)
    except Exception:
        return None
    return w.value if prob.status in ["optimal", "optimal_inaccurate"] else None


def _solve_additive_lp(D_patient, org, p, lp):
    """add_lp: additive max-of-affines LP (gamma per outcome)."""
    delta_p, lambda_p, z_hat_p = lp['delta'], lp['lambda'], lp['z_hat']
    N = delta_p.shape[0]; n_b = D_patient.shape[1]
    n1 = len(org.get('oar1', [])); n2 = len(org.get('oar2', []))
    w = cp.Variable(n_b, nonneg=True); o1 = cp.Variable(n1, nonneg=True)
    o2 = cp.Variable(n2, nonneg=True); d_max = cp.Variable(nonneg=True)
    g1, g2, g3 = cp.Variable(), cp.Variable(), cp.Variable()
    z0, z1, z2 = cp.sum(o1), cp.sum(o2), d_max
    cons = []
    for i in range(N):
        cons.append(g1 >= delta_p[i, 0] + lambda_p[i, 0] * (z0 - z_hat_p[i, 0]))
    for i in range(N):
        cons.append(g2 >= delta_p[i, 1] + lambda_p[i, 1] * (z1 - z_hat_p[i, 1]))
    for i in range(N):
        cons.append(g3 >= delta_p[i, 2] + lambda_p[i, 2] * (z2 - z_hat_p[i, 2]))
    # add_lp requires all four structures present (matches notebook)
    if org.get('ptv', np.array([])).size == 0 or org.get('oar1', np.array([])).size == 0 \
       or org.get('oar2', np.array([])).size == 0 or org.get('oar3', np.array([])).size == 0:
        return None
    _feasible_block(cons, D_patient, w, o1, o2, d_max, org, p)
    prob = cp.Problem(cp.Minimize(g1 + g2 + g3), cons)
    try:
        prob.solve(solver="GUROBI", verbose=False)
    except Exception:
        return None
    return w.value if prob.status in ["optimal", "optimal_inaccurate"] else None


def _solve_nonadd_socp(D_patient, org, p, lp):
    """nonadd_smooth / nonadd_nonsmooth: perspective SOCP with single (N,) alpha vector."""
    delta_p, lambda_p, z_hat_p, beta_p = lp['delta_global'], lp['lambda'], lp['z_hat'], lp['beta']
    N, K = lambda_p.shape
    n_b = D_patient.shape[1]
    n1 = len(org.get('oar1', [])); n2 = len(org.get('oar2', []))
    w = cp.Variable(n_b, nonneg=True); o1 = cp.Variable(n1, nonneg=True)
    o2 = cp.Variable(n2, nonneg=True); d_max = cp.Variable(nonneg=True)
    alpha = cp.Variable(N, nonneg=True); t = cp.Variable(N, nonneg=True); v = cp.Variable((N, K))
    lam_dot_zhat = np.sum(lambda_p * z_hat_p, axis=1)
    obj = cp.Minimize((beta_p / 2.0) * cp.sum(t) + cp.sum(cp.multiply(v, lambda_p))
                      - alpha @ lam_dot_zhat + alpha @ delta_p)
    cons = [cp.sum(alpha) == 1]
    for j in range(N):
        cons.append(cp.quad_over_lin(v[j, :] - alpha[j] * z_hat_p[j, :], alpha[j]) <= t[j])
    v_sum = cp.sum(v, axis=0)
    cons.append(cp.sum(o1) <= v_sum[0]); cons.append(cp.sum(o2) <= v_sum[1]); cons.append(d_max <= v_sum[2])
    _feasible_block(cons, D_patient, w, o1, o2, d_max, org, p)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="MOSEK", verbose=False)
    except Exception:
        return None
    return w.value if prob.status in ["optimal", "optimal_inaccurate"] else None


def _solve_nonadd_lp(D_patient, org, p, lp):
    """nonadd_nonsmooth_lp: TRUE convex-only reconstruction via a joint (multivariate)
    max-of-affines LP. f(z) = max_i [ delta_global_i + lambda_i . (z - z_hat_i) ] over the
    K=3 outcome vector z = (sum o1, sum o2, d_max). No perspective/beta term (pure LP);
    this is the non-additive analog of _solve_additive_lp."""
    delta_g, lambda_p, z_hat_p = lp['delta_global'], lp['lambda'], lp['z_hat']
    N = lambda_p.shape[0]; n_b = D_patient.shape[1]
    n1 = len(org.get('oar1', [])); n2 = len(org.get('oar2', []))
    w = cp.Variable(n_b, nonneg=True); o1 = cp.Variable(n1, nonneg=True)
    o2 = cp.Variable(n2, nonneg=True); d_max = cp.Variable(nonneg=True)
    g = cp.Variable()
    z0, z1, z2 = cp.sum(o1), cp.sum(o2), d_max
    cons = []
    for i in range(N):
        cons.append(g >= delta_g[i]
                    + lambda_p[i, 0] * (z0 - z_hat_p[i, 0])
                    + lambda_p[i, 1] * (z1 - z_hat_p[i, 1])
                    + lambda_p[i, 2] * (z2 - z_hat_p[i, 2]))
    _feasible_block(cons, D_patient, w, o1, o2, d_max, org, p)
    prob = cp.Problem(cp.Minimize(g), cons)
    try:
        prob.solve(solver="MOSEK", verbose=False)
    except Exception:
        return None
    return w.value if prob.status in ["optimal", "optimal_inaccurate"] else None


def _load_params(model, n, results_dir):
    if model == "add_smooth":
        return {'delta': np.load(f'{results_dir}/final_delta_comp_N{n}_v3.npy'),
                'lambda': np.load(f'{results_dir}/final_lambda_N{n}_v3.npy'),
                'z_hat': np.load(f'{results_dir}/Z_hat_obs_N{n}_v3.npy'),
                'beta': np.load(f'{results_dir}/beta_star_N{n}_v3.npy')[0]}
    if model == "add_lp":
        return {'delta': np.load(f'{results_dir}/final_delta_comp_N{n}_lp.npy'),
                'lambda': np.load(f'{results_dir}/final_lambda_N{n}_lp.npy'),
                'z_hat': np.load(f'{results_dir}/Z_hat_obs_N{n}_lp.npy')}
    if model == "nonadd_nonsmooth_lp":
        return {'delta_global': np.load(f'{results_dir}/final_delta_global_N{n}_vNSLP.npy'),
                'lambda': np.load(f'{results_dir}/final_lambda_N{n}_vNSLP.npy'),
                'z_hat': np.load(f'{results_dir}/Z_hat_obs_N{n}_vNSLP.npy')}
    suffix = "_vS" if model == "nonadd_smooth" else "_vNS"
    return {'delta_global': np.load(f'{results_dir}/final_delta_global_N{n}{suffix}.npy'),
            'lambda': np.load(f'{results_dir}/final_lambda_N{n}{suffix}.npy'),
            'z_hat': np.load(f'{results_dir}/Z_hat_obs_N{n}{suffix}.npy'),
            'beta': np.load(f'{results_dir}/beta_star_N{n}{suffix}.npy')[0]}


_SOLVERS = {"add_smooth": _solve_additive_socp, "add_lp": _solve_additive_lp,
            "nonadd_smooth": _solve_nonadd_socp, "nonadd_nonsmooth": _solve_nonadd_socp,
            "nonadd_nonsmooth_lp": _solve_nonadd_lp}


def run_prediction(model, sizes, ml_dir=ML_DIR, results_dir=None, out_dir=None):
    """Evaluate `model` over `sizes`, writing the beamlet-error CSV."""
    results_dir = results_dir or RESULTS[model]
    out_dir = out_dir or PRED_OUT[model]
    os.makedirs(out_dir, exist_ok=True)
    solve = _SOLVERS[model]
    D_test, w_test, org_test = _load_test(ml_dir)
    total = len(D_test)
    rows = []
    for n in sizes:
        try:
            lp = _load_params(model, n, results_dir)
        except FileNotFoundError:
            print(f"[{model}] N={n}: params missing, skipping", flush=True)
            continue
        pred, gt = [], []
        t0 = time.time()
        for i in range(total):
            wv = solve(D_test[i], org_test[i], PARAMS, lp)
            if wv is not None:
                pred.append(wv); gt.append(w_test[i])
        wp = np.array(pred); wt = np.array(gt)
        abs_err = np.abs(wp - wt)
        mae = float(np.mean(abs_err))
        mape = float(np.mean(100 * abs_err / (np.abs(wt) + 1e-6)))
        rmse = float(np.sqrt(np.mean((wp - wt) ** 2)))
        # PRIMARY metric (unified across the paper): relative L2 error of the
        # predicted beamlet vector vs the observed beamlets, averaged over test
        # patients. Robust to zero beamlets, unlike a per-beamlet percentage error.
        rel_l2 = float(np.mean([np.linalg.norm(wp[i] - wt[i]) / (np.linalg.norm(wt[i]) + 1e-12)
                                for i in range(len(wp))]))
        cos = float(np.mean([np.dot(wp[i], wt[i]) / ((np.linalg.norm(wp[i]) + 1e-6) * (np.linalg.norm(wt[i]) + 1e-6))
                             for i in range(len(wp))]))
        row = {'training_size': n, 'test_RelL2_w': rel_l2, 'test_MAE_w': mae,
               'test_MAPE_w': mape, 'test_RMSE_w': rmse, 'test_AvgCosSim_w': cos}
        if model == "add_lp":
            row.update({'patients_solved': len(pred), 'patients_total': total, 'success_rate': len(pred) / total})
        rows.append(row)
        print(f"[{model}] N={n}: RelL2_w={rel_l2:.4f} cos={cos:.4f} solved={len(pred)}/{total} ({time.time()-t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, PRED_CSV[model])
    df.to_csv(csv_path, index=False)
    print(f"[{model}] wrote {csv_path}", flush=True)
    return df


if __name__ == "__main__":
    import sys
    from config import TRAINING_SIZES
    model = sys.argv[1] if len(sys.argv) > 1 else "add_smooth"
    sizes = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else TRAINING_SIZES
    run_prediction(model, sizes)
