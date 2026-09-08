"""
run_prediction.py
=================
Evaluate recovered penalty functions by predicting beamlet weights for
test patients and comparing with the true forward-optimal w*.

The prediction problem uses the Proposition 1 conjugate in the SOC
perspective form given in the appendix.

The smoothness budget beta is the Stage 2 bisection result
(beta_star.npy), or beta_used.npy when Stage 3 escalated beta for
solver feasibility. It is read per N inside main() and threaded into
solve_prediction. Proposition 1 characterises f only at the beta the
recovery was solved under; evaluating at any other beta gives a
different function.

For each experiment (independent or nested, each N):
  1. Load recovered delta_k(z), lambda_k(z) for each function k AND
     the beta the recovery was solved under (load_recovered_beta).
  2. Apply the max-of-affines envelope reduction (compute_upper_envelope)
     to shrink the per-anchor sets. This is exact for beta=infty and a
     bounded-error (max 5e-3) approximation for the smooth conjugate.
  3. For each test patient, solve the prediction SOCP at the loaded beta:
       min sum_k sum_{v in OAR_k} f_k(z_v)
       s.t. z_v >= d_v - theta_k, z_v >= 0, PTV constraints, w >= 0,
            per-voxel simplex + perspective + rotated-SOC reformulation
            of f_k from paper appendix.
  4. Compare w_pred with w* (forward solution).

Usage:
  python run_prediction.py --experiment independent --N 8 --n-test 20 --seed 123
  python run_prediction.py --experiment nested --N 8 --n-test 20 --seed 123
"""

import os, sys, argparse, pickle, time, threading
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OAR_KEYS, OAR_GROUPS, N_FUNCTIONS, FUNC_LABELS,
    FORWARD_PARAMS, DOSE_THRESHOLDS, PTV_PRESCRIBED, PTV_MAX,
    MOSEK_TOLERANCES,
)
from forward.solver import solve_forward_problem


def get_experiment_dir(experiment_type, N):
    """Return the results directory for the given experiment."""
    if experiment_type == 'independent':
        return f'results/exp_N{N}_bisection'
    elif experiment_type == 'nested':
        return f'results/nested_N{N}'
    elif experiment_type == 'nested_oracle_anchor':
        return f'results/nested_N{N}_oracle_anchor'
    elif experiment_type == 'nested_ratio_anchor':
        return f'results/nested_N{N}_ratio_anchor'
    elif experiment_type == 'nested_fullleak_anchor':
        return f'results/nested_N{N}_fullleak_anchor'
    elif experiment_type.startswith('synthetic_'):
        return f'results/{experiment_type}_N{N}'
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def compute_upper_envelope(z, delta, lam):
    """
    Compute the exact upper envelope of tangent lines.

    Each tangent line is: y = lam_j * x + (delta_j - lam_j * z_j)
    The upper envelope max_j { delta_j + lam_j * (x - z_j) } is a
    piecewise linear convex function. Only lines that appear on the
    envelope are needed — dominated lines are removed.

    This is mathematically exact (no approximation).

    Returns indices of non-dominated lines in the input arrays.
    """
    # Each line: y = a*x + b  where a=lam_j, b=delta_j - lam_j*z_j
    a = lam.copy()
    b = delta - lam * z

    n = len(a)
    if n <= 2:
        return np.arange(n)

    # Sort by slope (ascending)
    order = np.argsort(a)
    a = a[order]
    b = b[order]

    # Graham scan for upper envelope of lines
    stack = [0]
    for i in range(1, n):
        while len(stack) >= 2:
            j = stack[-1]
            k = stack[-2]
            # Check if line j is dominated at the intersection of k and i
            denom = a[i] - a[k]
            if abs(denom) > 1e-15:
                x_int = (b[k] - b[i]) / denom
                y_j = a[j] * x_int + b[j]
                y_env = a[i] * x_int + b[i]
                if y_j <= y_env + 1e-12:
                    stack.pop()
                else:
                    break
            else:
                if b[i] >= b[k]:
                    stack.pop()
                else:
                    break
        stack.append(i)

    return order[np.array(stack)]


def load_recovered_functions(res_dir, N):
    """
    Load recovered delta, lambda, z_hat for each function.
    Computes the exact upper envelope to remove dominated tangent lines.

    Returns dict with keys 0..K-1, each containing:
      'z_hat': z values (envelope tangent points)
      'delta': delta values (recovered function values)
      'lam':   lambda values (recovered subgradients)
    """
    funcs = {}
    for k in range(N_FUNCTIONS):
        z = np.load(f'{res_dir}/Z_func{k}_N{N}.npy')
        delta = np.load(f'{res_dir}/delta_func{k}_N{N}.npy')
        lam = np.load(f'{res_dir}/lambda_func{k}_N{N}.npy')

        n_total = len(z)

        # Compute exact upper envelope (no approximation)
        env_idx = compute_upper_envelope(z, delta, lam)
        z = z[env_idx]
        delta = delta[env_idx]
        lam = lam[env_idx]

        # Sort by z for clean ordering
        order = np.argsort(z)
        z = z[order]
        delta = delta[order]
        lam = lam[order]

        funcs[k] = {'z_hat': z, 'delta': delta, 'lam': lam}
        print(f"    func {k} ({FUNC_LABELS[k]}): {n_total} obs -> "
              f"{len(z)} envelope lines")

    return funcs


def load_recovered_beta(res_dir):
    """
    Read the smoothness parameter beta that was used at recovery time.

    Proposition 1 of the paper describes the recovered f_k as the Fenchel
    conjugate of a beta-smooth function g_k, where beta is the smoothness
    budget reported by Stage 2's bisection (saved as beta_star.npy). The
    recovered tangent triples (z_j, delta_j, lambda_j) are tangents to
    f_{beta^*}, NOT to f at any other smoothness level. The SOC perspective
    form used by solve_prediction (paper appendix line 3697-3715) is itself
    parameterised by beta: the (beta/2) sum aux term in the objective
    encodes the quadratic penalty whose Lipschitz-gradient constant is
    exactly beta. Plugging the recovered (delta, lambda) into the SOC at a
    different beta therefore evaluates a different function from the one
    Proposition 1 analyses, with no formal guarantee.

    Preference order: beta_used.npy (Stage 3's actually-solved beta, which
    equals beta_star except in the rare case Stage 3 had to escalate beta
    for MOSEK feasibility); fall back to beta_star.npy if beta_used.npy
    does not exist. Raise if neither is found -- the recovery must have
    saved one, otherwise prediction has no defensible beta to use.
    """
    used = os.path.join(res_dir, 'beta_used.npy')
    star = os.path.join(res_dir, 'beta_star.npy')
    if os.path.exists(used):
        beta = float(np.load(used)[0])
        print(f"    beta_used (from Stage 3) = {beta:.4f}")
        return beta
    if os.path.exists(star):
        beta = float(np.load(star)[0])
        print(f"    beta_star (from Stage 2) = {beta:.4f}  "
              f"(beta_used.npy not present)")
        return beta
    raise FileNotFoundError(
        f"Neither beta_used.npy nor beta_star.npy in {res_dir}. "
        f"Cannot run prediction without the beta from Stage 2/3 because "
        f"the recovered f_k is the beta*-smooth conjugate; using any other "
        f"beta evaluates a different function (see Proposition 1).")


def select_test_patients(all_cache_file, training_ids, n_test, seed,
                         fixed_test_ids=None):
    """
    Select n_test patients with complete data, not in any training set.
    If fixed_test_ids is given, use that exact list instead (and ignore
    n_test/seed). This guarantees identical test sets across different N
    values, which is required when comparing MAPE as we add more training
    patients.
    Returns (patients, w_solutions, obj_values) for test patients.
    """
    with open(all_cache_file, 'rb') as f:
        cache = pickle.load(f)

    all_patients = cache['patients']
    all_w = cache['w_solutions']
    all_obj = cache['obj_values']

    if fixed_test_ids is not None:
        id_to_idx = {p['patient_id']: i for i, p in enumerate(all_patients)}
        missing = [pid for pid in fixed_test_ids if pid not in id_to_idx]
        if missing:
            raise RuntimeError(f"Fixed test IDs not in cache: {missing}")
        selected = [id_to_idx[pid] for pid in fixed_test_ids]
    else:
        # Filter: complete data + not in training
        candidates = []
        for i, p in enumerate(all_patients):
            if p['patient_id'] in training_ids:
                continue
            has_all = all(
                len(p['organ_indices'].get(k, np.array([], dtype=int))) > 0
                for k in OAR_KEYS
            )
            if has_all:
                candidates.append(i)

        rng = np.random.default_rng(seed)
        rng.shuffle(candidates)
        selected = candidates[:n_test]

    patients = [all_patients[i] for i in selected]
    w_sols = [all_w[i] for i in selected]
    obj_vals = [float(all_obj[i]) for i in selected]

    return patients, w_sols, obj_vals


def get_all_training_ids(experiment_type='nested'):
    """Collect patient IDs from all training caches."""
    ids = set()
    if experiment_type.startswith('synthetic_'):
        for N in [5, 6, 7, 8]:
            pattern = f'results/all_patients/forward_cache_synthetic_N{N}.pkl'
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    c = pickle.load(f)
                for p in c['patients']:
                    ids.add(p['patient_id'])
    else:
        for N in [5, 6, 7, 8]:
            for pattern in [f'results/all_patients/forward_cache_N{N}.pkl',
                            f'results/all_patients/forward_cache_nested_N{N}.pkl']:
                if os.path.exists(pattern):
                    with open(pattern, 'rb') as f:
                        c = pickle.load(f)
                    for p in c['patients']:
                        ids.add(p['patient_id'])
    return ids


def solve_prediction(D, organ_indices, recovered_funcs, beta_val,
                     verbose=False, mosek_tol=1e-5, timeout=600.0):
    """
    Solve the prediction problem using the recovered penalty functions.

    Proposition 1, in the SOC perspective form given in the appendix.
    The recovered f_k(z) is the Fenchel conjugate of a
    1-strongly-convex function:

        f_k(z) = inf_{y >= z} g_k^*(y)
        g_k(lambda) = max_j {(1/(2 beta))(lambda - lambda_j)^2
                              + lambda * z_j - delta_j}

    Encoded with the simplex + perspective + rotated-SOC reformulation
    from paper appendix line 3697-3715:

        f_k(z_v) = min_{alpha, v_p}
                      sum_j[ (beta/2)(v_p_j - alpha_j*z_j)^2 / alpha_j
                              + lambda_j v_p_j
                              - alpha_j lambda_j z_j
                              + alpha_j delta_j ]
        s.t.    sum_j alpha_j = 1,  alpha_j >= 0   (simplex)
                sum_j v_p_j     >= z_v             (monotone envelope)

    The quadratic perspective term is encoded via a rotated SOC:
        (v_p_j - alpha_j z_j)^2 <= alpha_j * aux_j
        ⇔  ||2 * (v_p_j - alpha_j z_j), alpha_j - aux_j||_2
             <= alpha_j + aux_j


    Problem:
      min  sum_k sum_{v in OAR_k} f_k(z_v)
      s.t. z_v >= dose_v - theta_k,  z_v >= 0,
           PTV_PRESCRIBED <= dose[ptv] <= PTV_MAX,
           w >= 0,
           per-voxel simplex + perspective + rotated SOC as above.

    Parameters
    ----------
    D, organ_indices, recovered_funcs : as before
    beta_val : smoothness parameter beta. MUST be the same beta at which
        Stage 3 recovered (delta_j, lambda_j) -- this is beta_used.npy (or
        beta_star.npy from Stage 2's bisection if Stage 3 did not need to
        escalate). The recovered tangents are tangents to f_{beta*}, and
        the (beta/2) sum aux term in the SOC objective evaluates the
        beta-smooth conjugate at the beta passed here. Passing any other
        beta evaluates a different function with no Proposition 1
        guarantee. There is no defensible default -- caller must read
        beta from disk via load_recovered_beta() and pass it explicitly.
    """
    n_voxels, n_beamlets = D.shape
    w = cp.Variable(n_beamlets, nonneg=True)
    dose = D @ w

    constraints = []
    obj_terms = []

    # PTV constraints
    idx_ptv = organ_indices['ptv']
    if len(idx_ptv) > 0:
        constraints.append(dose[idx_ptv] >= PTV_PRESCRIBED)
        constraints.append(dose[idx_ptv] <= PTV_MAX)

    BETA = float(beta_val)

    for oar_key in OAR_KEYS:
        idx = organ_indices.get(oar_key, np.array([], dtype=int))
        if len(idx) == 0:
            continue

        func_idx = OAR_GROUPS[oar_key]['func_idx']
        rf = recovered_funcs[func_idx]
        theta = DOSE_THRESHOLDS[oar_key]

        n_v = len(idx)
        M = len(rf['z_hat'])
        z_anchors = np.asarray(rf['z_hat'], dtype=float)
        delta_anchors = np.asarray(rf['delta'], dtype=float)
        lambda_anchors = np.asarray(rf['lam'], dtype=float)

        # Overdose variable z_v = max(0, dose_v - theta)
        z_v = cp.Variable(n_v, nonneg=True)
        constraints.append(z_v >= dose[idx] - theta)

        # Per-voxel simplex + perspective + rotated SOC
        alpha = cp.Variable((n_v, M), nonneg=True)
        v_persp = cp.Variable((n_v, M))
        aux = cp.Variable((n_v, M), nonneg=True)

        # Simplex per voxel: sum_j alpha[v_idx, j] = 1
        constraints.append(cp.sum(alpha, axis=1) == 1)
        # Monotone envelope per voxel: sum_j v_p[v_idx, j] >= z_v[v_idx]
        constraints.append(cp.sum(v_persp, axis=1) >= z_v)

        # Rotated SOC per (v, j): (v_p_{v,j} - alpha_{v,j} z_j)^2
        #                         <= alpha_{v,j} * aux_{v,j}
        # Vectorize over voxels for each anchor j -- M loop iters.
        for j in range(M):
            zj = z_anchors[j]
            u_j = v_persp[:, j] - alpha[:, j] * zj    # shape (n_v,)
            s_j = alpha[:, j]
            t_j = aux[:, j]
            # rotated SOC: u^2 <= s*t  <=>  ||2u, s-t||_2 <= s + t
            constraints.append(
                cp.SOC(s_j + t_j,
                       cp.vstack([2 * u_j,
                                  s_j - t_j]))
            )

        # Objective contribution from this OAR (paper eq. line 3699-3707)
        # sum_v [ (beta/2) sum_j aux_{v,j}
        #         + sum_j lambda_j * v_p_{v,j}
        #         - sum_j alpha_{v,j} * lambda_j * z_j
        #         + sum_j alpha_{v,j} * delta_j ]
        obj_terms.append(
            (BETA / 2.0) * cp.sum(aux)
            + cp.sum(cp.multiply(v_persp, lambda_anchors[None, :]))
            - cp.sum(cp.multiply(alpha,
                                 (lambda_anchors * z_anchors)[None, :]))
            + cp.sum(cp.multiply(alpha, delta_anchors[None, :]))
        )

    if not obj_terms:
        return None

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)

    # MSK_DPAR_OPTIMIZER_MAX_TIME only limits MOSEK's internal loop; CVXPY
    # canonicalization runs before MOSEK starts and can hang on large patients.
    # Use a daemon thread with a wall-clock timeout to guard both phases.
    tol = {**MOSEK_TOLERANCES,
           'MSK_DPAR_OPTIMIZER_MAX_TIME': max(timeout * 0.8, 480.0),
           'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   mosek_tol,
           'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   mosek_tol,
           'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': mosek_tol}
    _exc = [None]

    def _solve():
        try:
            prob.solve(solver='MOSEK', verbose=verbose, mosek_params=tol)
        except Exception as e:
            _exc[0] = e

    t = threading.Thread(target=_solve, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return None  # timed out — skip patient
    if _exc[0] is not None:
        if isinstance(_exc[0], (cp.SolverError, MemoryError)):
            return None
        raise _exc[0]

    if prob.status in ('optimal', 'optimal_inaccurate'):
        return w.value
    return None


def compute_errors(w_pred_list, w_true_list):
    """Compute prediction error metrics. Handles variable-length w vectors."""
    # Per-patient metrics (w vectors may have different lengths across patients)
    mae_list = []
    rmse_list = []
    rel_errors = []
    cos_sims = []

    for i in range(len(w_pred_list)):
        wp = w_pred_list[i]
        wt = w_true_list[i]

        diff = wp - wt
        mae_list.append(np.mean(np.abs(diff)))
        rmse_list.append(np.sqrt(np.mean(diff ** 2)))

        norm_true = np.linalg.norm(wt)
        norm_pred = np.linalg.norm(wp)
        if norm_true > 1e-10:
            rel_errors.append(np.linalg.norm(diff) / norm_true)
        if norm_true > 1e-10 and norm_pred > 1e-10:
            cos_sims.append(np.dot(wp, wt) / (norm_pred * norm_true))

    return {
        'MAE': np.mean(mae_list) if mae_list else np.nan,
        'RMSE': np.mean(rmse_list) if rmse_list else np.nan,
        'MeanRelError': np.mean(rel_errors) if rel_errors else np.nan,
        'MeanCosSim': np.mean(cos_sims) if cos_sims else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        choices=['independent', 'nested', 'nested_oracle_anchor',
                                 'nested_ratio_anchor', 'nested_fullleak_anchor',
                                 'synthetic_umin1', 'synthetic_ratio',
                                 'synthetic_umin0'],
                        default='nested')
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--all-N', action='store_true',
                        help='Run for all N=5,6,7,8')
    parser.add_argument('--N-list', type=str, default=None,
                        help='Comma-separated list of N values, overrides --N/--all-N')
    parser.add_argument('--test-ids-file', type=str, default=None,
                        help='Path to text file with one patient ID per line; '
                             'pins the test set across all N (for valid trend comparison)')
    parser.add_argument('--patient-indices', type=str, default=None,
                        help='Comma-separated 0-based patient indices to run '
                             '(e.g. "7,13,15"). Skips all others. '
                             'Does not overwrite prediction_errors.csv.')
    parser.add_argument('--mosek-tol', type=float, default=1e-5,
                        help='MOSEK feasibility/gap tolerance (default 1e-5; '
                             'try 1e-4 for hard patients).')
    parser.add_argument('--timeout', type=float, default=600.0,
                        help='Per-patient wall-clock timeout in seconds (default 600).')
    parser.add_argument('--no-csv', action='store_true',
                        help='Do not overwrite prediction_errors.csv '
                             '(use with --patient-indices).')
    args = parser.parse_args()

    if args.N_list:
        N_values = [int(x) for x in args.N_list.split(',')]
    elif args.all_N:
        N_values = [5, 6, 7, 8]
    else:
        N_values = [args.N]

    fixed_test_ids = None
    if args.test_ids_file:
        with open(args.test_ids_file, 'r') as f:
            fixed_test_ids = [line.strip() for line in f if line.strip()]
        print(f"Using fixed test set of {len(fixed_test_ids)} patients "
              f"from {args.test_ids_file}")

    patient_indices = None
    if args.patient_indices:
        patient_indices = set(int(x) for x in args.patient_indices.split(','))
        print(f"Targeting patient indices: {sorted(patient_indices)}")

    # Collect all training patient IDs
    training_ids = get_all_training_ids(args.experiment)
    print(f"Total training patients across all experiments: {len(training_ids)}")

    # Select test patients (same set for all N)
    test_patients, test_w, test_obj = select_test_patients(
        'results/all_patients/forward_cache_all.pkl',
        training_ids, args.n_test, args.seed,
        fixed_test_ids=fixed_test_ids,
    )
    print(f"Selected {len(test_patients)} test patients:")
    for i, p in enumerate(test_patients):
        print(f"  [{i}] {p['patient_id']}  obj={test_obj[i]:.2f}")

    all_results = []

    for N in N_values:
        res_dir = get_experiment_dir(args.experiment, N)
        print(f"\n{'='*60}")
        print(f"Evaluating {args.experiment} N={N} from {res_dir}")
        print(f"{'='*60}")

        # Load recovered functions AND the beta they were recovered under.
        # Proposition 1 makes f_k the Fenchel conjugate of a beta-smooth
        # function; the recovered (delta_j, lambda_j) at this N are tangents
        # to f_{beta*} where beta* is the Stage 2 bisection result. The SOC
        # form used by solve_prediction is parameterised by beta, so we MUST
        # evaluate it at the same beta the recovery used -- otherwise the
        # prediction problem is over a different function with no formal
        # guarantee. There is no defensible hardcoded default.
        try:
            recovered = load_recovered_functions(res_dir, N)
            beta_recovered = load_recovered_beta(res_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        w_pred_list = []
        w_true_list = []
        dose_records = []  # per-patient (d_true, d_pred, organ_indices, patient_id)

        for i, (patient, w_star) in enumerate(zip(test_patients, test_w)):
            if patient_indices is not None and i not in patient_indices:
                continue
            t0 = time.time()
            print(f"  Patient {i+1}/{len(test_patients)}: "
                  f"{patient['patient_id']} ...", end=" ", flush=True)

            try:
                w_pred = solve_prediction(
                    patient['D'], patient['organ_indices'],
                    recovered, beta_val=beta_recovered, verbose=False,
                    mosek_tol=args.mosek_tol, timeout=args.timeout
                )
            except Exception as e:
                print(f"CRASH ({type(e).__name__}: {str(e)[:80]})")
                w_pred = None

            elapsed = time.time() - t0
            if w_pred is not None:
                d_true = patient['D'] @ w_star
                d_pred = patient['D'] @ w_pred
                dose_records.append({
                    'patient_id': patient['patient_id'],
                    'd_true': d_true,
                    'd_pred': d_pred,
                    'w_true': w_star,
                    'w_pred': w_pred,
                    'organ_indices': patient['organ_indices'],
                })

            if w_pred is not None:
                w_pred_list.append(w_pred)
                w_true_list.append(w_star)
                rel_err = np.linalg.norm(w_pred - w_star) / max(np.linalg.norm(w_star), 1e-10)
                print(f"OK ({elapsed:.1f}s, rel_err={rel_err:.4f})")
            else:
                print(f"FAILED ({elapsed:.1f}s)")

        if not w_pred_list:
            print("  No patients solved successfully!")
            continue

        errors = compute_errors(w_pred_list, w_true_list)
        errors['N'] = N
        errors['experiment'] = args.experiment
        errors['n_solved'] = len(w_pred_list)
        all_results.append(errors)

        print(f"\n  Results for N={N}:")
        print(f"    Solved: {len(w_pred_list)}/{len(test_patients)}")
        print(f"    MAE:          {errors['MAE']:.6f}")
        print(f"    RMSE:         {errors['RMSE']:.6f}")
        print(f"    Mean Rel Err: {errors['MeanRelError']:.4f}")
        print(f"    Mean Cos Sim: {errors['MeanCosSim']:.6f}")

        # Persist per-patient dose arrays for scatter plots
        dose_dir = f'results/prediction_{args.experiment}/doses_N{N}'
        os.makedirs(dose_dir, exist_ok=True)
        for rec in dose_records:
            pid = rec['patient_id']
            np.savez(f'{dose_dir}/{pid}.npz',
                     d_true=rec['d_true'], d_pred=rec['d_pred'],
                     w_true=rec['w_true'], w_pred=rec['w_pred'],
                     ptv_idx=rec['organ_indices'].get('ptv', np.array([], dtype=int)))

    # Save results
    out_dir = f'results/prediction_{args.experiment}'
    os.makedirs(out_dir, exist_ok=True)

    if all_results and not args.no_csv:
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_path = f'{out_dir}/prediction_errors.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
        print(df.to_string())
    elif all_results and args.no_csv:
        import pandas as pd
        df = pd.DataFrame(all_results)
        print("\nResults (--no-csv: not written to disk):")
        print(df.to_string())

        # Plot error vs N if multiple N values
        if len(all_results) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].plot(df['N'], df['MeanRelError'], 'o-', linewidth=2)
            axes[0].set_xlabel('Training size N')
            axes[0].set_ylabel('Mean Relative Error')
            axes[0].set_title('Relative Error vs Training Size')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(df['N'], df['MeanCosSim'], 's-', linewidth=2, color='green')
            axes[1].set_xlabel('Training size N')
            axes[1].set_ylabel('Mean Cosine Similarity')
            axes[1].set_title('Cosine Similarity vs Training Size')
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(df['N'], df['RMSE'], 'D-', linewidth=2, color='red')
            axes[2].set_xlabel('Training size N')
            axes[2].set_ylabel('RMSE')
            axes[2].set_title('RMSE vs Training Size')
            axes[2].grid(True, alpha=0.3)

            plt.suptitle(f'Prediction Error ({args.experiment.title()} Experiment)',
                         fontsize=14)
            plt.tight_layout()
            plot_path = f'{out_dir}/error_vs_N.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved error plot to {plot_path}")


if __name__ == '__main__':
    main()
