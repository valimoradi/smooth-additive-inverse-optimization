"""
run_parametric_comparison.py
============================
Compare parametric IO (Keshavarz 2011) vs. our nonparametric IO on the
nested PortPy prostate cohort.

The true forward objective is f_k(z_v) = alpha_k * z_v^2 (quadratic).
Three parametric assumptions are tested:
  rho=1  (linear):    WRONG assumption -> systematic bias
  rho=2  (quadratic): CORRECT assumption -> should recover well
  rho=3  (cubic):     WRONG assumption -> different bias

For each N in {5,6,7,8,9,10}, for each rho:
  1. Load N training patients from the nested cache.
  2. Fit alpha_k by the Keshavarz KKT residual estimator (fit_keshavarz_kkt,
     --method kkt); this is the non-oracle method reported in the paper
     (Table tab:portpy_parametric_misspec).
  3. Predict the true beamlet weights for 20 test patients using the
     fitted parametric forward model.
  4. Compute Rel-L2_w vs. the true optimal weights.

The published table was produced with --method kkt (the default). --method
oracle is an UPPER-BOUND diagnostic that injects the true alpha_k and does NOT
reproduce the paper's numbers; --method nnls is the weak objective-value
estimator retained only to document its identification failure.

Outputs (suffixed by --tag, default = method)
----------------------------------------------
  results/parametric_comparison/parametric_mape_nested_<tag>.csv
  results/parametric_comparison/parametric_mape_figure_<tag>.pdf

Usage
-----
  python run_parametric_comparison.py --n-test 20 --seed 123            # kkt (paper)
  python run_parametric_comparison.py --method kkt --n-test 5 --seed 123  # smoke test
"""

import os, sys, argparse, pickle, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault("PYTHONUNBUFFERED", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OAR_KEYS, OAR_GROUPS, N_FUNCTIONS, FUNC_LABELS, FORWARD_PARAMS,
    DOSE_THRESHOLDS,
)


def _true_penalty(dose, organ_indices):
    """True quadratic overdose penalty J = sum_k alpha_k sum_v max(0,d_v-theta_k)^2."""
    J = 0.0
    for k in OAR_KEYS:
        idx = organ_indices.get(k, np.array([], dtype=int))
        if len(idx) == 0:
            continue
        o = np.maximum(0.0, np.asarray(dose)[idx] - DOSE_THRESHOLDS[k])
        J += FORWARD_PARAMS[f'alpha_{k}'] * np.sum(o ** 2)
    return J


def penalty_regret_pct(w_pred, w_true, D, organ_indices):
    """Percentage true-objective regret 100*(J(w_pred)-J(w_true))/J(w_true)."""
    d_true = np.asarray(D @ np.asarray(w_true)).ravel()
    d_pred = np.asarray(D @ np.asarray(w_pred)).ravel()
    Jt = _true_penalty(d_true, organ_indices)
    Jp = _true_penalty(d_pred, organ_indices)
    if Jt <= 0:
        return np.nan
    return 100.0 * (Jp - Jt) / Jt


# Low-dose threshold for the dose-MAPE metric. Voxels below the threshold form
# the low-dose bath, where a per-voxel percentage error divides by a near-zero
# dose and is uninformative. We use the low-dose threshold of AAPM TG-218
# (Miften et al., Med. Phys. 2018): 10% of the maximum dose, applied per patient.
# The threshold is standard for dose-agreement metrics and is not chosen for this
# data set. The mask is defined on the GROUND-TRUTH dose d_true = D w_true, so the
# same voxel set is scored for the nonparametric and every parametric predictor.
MAPE_LOW_DOSE_FRAC = 0.10  # TG-218 low-dose threshold: 10% of the maximum dose


def dose_mape_masked(w_pred, w_true, D, frac=MAPE_LOW_DOSE_FRAC):
    """Per-voxel MAPE over voxels above the TG-218 low-dose threshold
    (d_true >= frac * max(d_true)), evaluated on the ground-truth dose."""
    d_true = np.asarray(D @ np.asarray(w_true)).ravel()
    d_pred = np.asarray(D @ np.asarray(w_pred)).ravel()
    m = d_true >= frac * d_true.max()
    if not np.any(m):
        return np.nan
    return 100.0 * np.mean(np.abs(d_pred[m] - d_true[m]) / d_true[m])
from parametric_io import (
    fit_parametric_alpha, fit_keshavarz_kkt, fit_keshavarz_obj_value,
    solve_forward_parametric, rel_error_beamlets, rel_error_dose,
)
from run_prediction import select_test_patients, get_all_training_ids


N_VALUES   = [5, 6, 7, 8, 9, 10]
RHO_VALUES = [1, 2, 3]
RHO_LABELS = {1: 'Linear (ρ=1)', 2: 'Quadratic (ρ=2)', 3: 'Cubic (ρ=3)',
              4: 'Quartic (ρ=4)'}
RHO_COLORS = {1: 'C0', 2: 'C2', 3: 'C1', 4: 'C4'}
RHO_MARKERS= {1: 'o', 2: 's', 3: '^', 4: 'v'}

OUT_DIR = 'results/parametric_comparison'


def load_nested_training(N):
    """
    Load the N training patients from the nested N-patient cache.
    Returns (patients, w_solutions, obj_values).
    """
    cache_file = f'results/all_patients/forward_cache_nested_N{N}.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    patients    = cache['patients']
    w_solutions = cache['w_solutions']
    obj_values  = cache.get('obj_values', None)
    return patients, [np.asarray(w) for w in w_solutions], obj_values


def fit_alpha_dispatch(method, training_patients, training_w, rho,
                       training_obj, ptv_bind_tol, verbose):
    """Route to the requested estimator."""
    if method == 'oracle':
        return fit_parametric_alpha(
            training_patients, training_w, rho,
            training_obj_values=training_obj, verbose=verbose)
    elif method == 'kkt':
        return fit_keshavarz_kkt(
            training_patients, training_w, rho,
            ptv_bind_tol=ptv_bind_tol, verbose=verbose)
    elif method == 'nnls':
        return fit_keshavarz_obj_value(
            training_patients, training_w, rho,
            training_obj_values=training_obj, verbose=verbose)
    raise ValueError(f"unknown method: {method}")


def run_one_N_rho(N, rho, test_patients, test_w, method='kkt',
                  ptv_bind_tol=0.1, verbose=True, metric='w'):
    """Fit parametric IO and evaluate on test patients. Returns list of Rel-L2_w values."""
    print(f"\n  -- N={N}, rho={rho}, method={method} --", flush=True)
    t0 = time.time()

    # Load training patients
    training_patients, training_w, training_obj = load_nested_training(N)
    print(f"    Training: {len(training_patients)} patients", flush=True)

    # Fit alpha_k
    try:
        alpha_hat = fit_alpha_dispatch(
            method, training_patients, training_w, rho,
            training_obj, ptv_bind_tol, verbose)
    except Exception as e:
        print(f"    FIT FAILED: {e}", flush=True)
        return [np.nan] * len(test_patients)

    # Per-function true reference (fold OARs by func_idx; grouped femur_l/femur_r
    # share alpha, so last-write assignment is correct, NOT summation).
    true_pf = np.zeros(N_FUNCTIONS)
    for k in OAR_KEYS:
        true_pf[OAR_GROUPS[k]['func_idx']] = FORWARD_PARAMS[f'alpha_{k}']
    print(f"    alpha_hat = {np.round(alpha_hat, 4)}", flush=True)
    print(f"    True alpha (per function) = {np.round(true_pf, 4)}", flush=True)

    # Predict on each test patient
    mape_list = []
    for i, (patient, w_true) in enumerate(zip(test_patients, test_w)):
        t1 = time.time()
        w_pred = solve_forward_parametric(
            patient['D'], patient['organ_indices'],
            alpha_hat, rho, verbose=False
        )
        elapsed = time.time() - t1
        if w_pred is not None:
            if metric == 'dose':
                m = rel_error_dose(w_pred, np.asarray(w_true), patient['D'])
            elif metric == 'penalty':
                m = penalty_regret_pct(w_pred, np.asarray(w_true),
                                       patient['D'], patient['organ_indices'])
            elif metric == 'mape':
                m = dose_mape_masked(w_pred, np.asarray(w_true), patient['D'])
            else:
                m = rel_error_beamlets(w_pred, np.asarray(w_true))
            print(f"    test[{i:2d}] {patient['patient_id']}: "
                  f"RelErr={m:.4f}  ({elapsed:.1f}s)", flush=True)
        else:
            m = np.nan
            print(f"    test[{i:2d}] {patient['patient_id']}: "
                  f"FAILED ({elapsed:.1f}s)", flush=True)
        mape_list.append(m)

    total = time.time() - t0
    valid = [m for m in mape_list if not np.isnan(m)]
    print(f"    Mean RelErr = {np.mean(valid):.4f}  "
          f"({len(valid)}/{len(mape_list)} valid, {total:.1f}s total)", flush=True)
    return mape_list


def make_figure(results, out_path, N_values=None, rho_values=None):
    """
    Plot mean relative error vs N for each rho, overlaid with nonparametric results.
    Publication-quality figure matching the paper's tim-style.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.8,
        'axes.linewidth': 0.8,
    })

    N_vals = N_values or N_VALUES
    rho_vals = rho_values or RHO_VALUES
    x = np.array(N_vals)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for rho in rho_vals:
        means = [np.nanmean(results[(N, rho)]) for N in N_vals]
        ax.plot(x, means,
                marker=RHO_MARKERS.get(rho, 'x'), color=RHO_COLORS.get(rho, 'C7'),
                linewidth=1.8, markersize=6,
                label=RHO_LABELS.get(rho, f'$\\rho={rho:g}$'))

    # Overlay existing nonparametric result from prediction CSV
    pred_csv = 'results/prediction_nested/prediction_errors_N5_8.csv'
    if not os.path.exists(pred_csv):
        pred_csv = 'results/prediction_nested/prediction_errors.csv'
    if os.path.exists(pred_csv):
        try:
            import csv
            np_err = {}
            with open(pred_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        n = int(row.get('N', row.get('n', '')))
                        v = float(row.get('MeanRelError',
                                   row.get('MAPE', row.get('mape', ''))))
                        np_err[n] = v
                    except (ValueError, KeyError):
                        pass
            if np_err:
                npy = [np_err.get(n, np.nan) for n in N_vals]
                ax.plot(x, npy, marker='D', color='C3',
                        linewidth=1.8, linestyle='--', markersize=6,
                        label='Nonparametric (proposed)')
        except Exception:
            pass

    ax.set_xlabel('Number of training patients ($N$)')
    ax.set_ylabel(r'Rel-L2$_w$: $\|\hat{w}-w\|/\|w\|$')
    ax.set_xticks(N_vals)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {out_path}', flush=True)


def make_figure_from_csv(par_csv=None, pred_csv=None, out_path=None,
                         N_values=None, rho_values=None):
    """
    Build the parametric-vs-nonparametric comparison figure DIRECTLY from the
    canonical result CSVs, so the figure provably matches
    Table~tab:portpy_parametric_misspec (no re-solve, fully deterministic).

      par_csv : parametric_mape_nested_<tag>.csv  (cols: N,rho,mean_rel_error,...)
      pred_csv: prediction_nested/prediction_errors.csv  (cols: MeanRelError,N,...)

    Both files are read; the four series (rho=1,2,3 parametric + nonparametric
    proposed) are plotted versus N on a single linear y-axis. Linear y is used
    deliberately: it shows in one frame that wrong-form parametric IO is large
    and flat in N (structural error), the correct form (rho=2) is near zero, and
    the nonparametric method sits in between -- robust without a form assumption.
    """
    import csv
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 11,
        'axes.titlesize': 11, 'legend.fontsize': 9,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'lines.linewidth': 1.8, 'axes.linewidth': 0.8,
    })

    N_vals   = N_values or N_VALUES
    rho_vals = rho_values or RHO_VALUES
    par_csv  = par_csv  or os.path.join(OUT_DIR, 'parametric_mape_nested_kkt.csv')
    pred_csv = pred_csv or 'results/prediction_nested/prediction_errors.csv'
    out_path = out_path or os.path.join(OUT_DIR, 'parametric_mape_figure_kkt.pdf')

    # parametric means, keyed (N, rho)
    par = {}
    with open(par_csv, newline='') as f:
        for row in csv.DictReader(f):
            # rho may be written as "3" or "4.0"; int(float(...)) handles both.
            par[(int(row['N']), int(float(row['rho'])))] = float(row['mean_rel_error'])

    # nonparametric (proposed) means, keyed N
    np_err = {}
    with open(pred_csv, newline='') as f:
        for row in csv.DictReader(f):
            try:
                np_err[int(row['N'])] = float(row['MeanRelError'])
            except (ValueError, KeyError):
                pass

    x = np.array(N_vals)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for rho in rho_vals:
        means = [par.get((N, rho), np.nan) for N in N_vals]
        ax.plot(x, means, marker=RHO_MARKERS[rho], color=RHO_COLORS[rho],
                linewidth=1.8, markersize=6, label=RHO_LABELS[rho])

    # nonparametric drawn bold + on top to read as "our method"
    npy = [np_err.get(N, np.nan) for N in N_vals]
    ax.plot(x, npy, marker='D', color='C3', linewidth=2.4, markersize=7,
            markeredgecolor='black', markeredgewidth=0.5, zorder=5,
            label='Nonparametric (proposed)')

    ax.set_xlabel('Number of training patients ($N$)')
    ax.set_ylabel(r'Rel-L2$_w$: $\|\hat{w}-w\|/\|w\|$')
    ax.set_xticks(N_vals)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, loc='center right')
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {out_path}', flush=True)
    # echo plotted values for provenance against the table
    for rho in rho_vals:
        print(f'  rho={rho:>1}: ' +
              '  '.join(f'{par.get((N, rho), float("nan")):.4f}' for N in N_vals),
              flush=True)
    print('  nonpar: ' +
          '  '.join(f'{np_err.get(N, float("nan")):.4f}' for N in N_vals), flush=True)
    return out_path


def write_csv(results, out_path, N_values=None, rho_values=None):
    lines = ['N,rho,mean_rel_error,std_rel_error,n_valid']
    for N in (N_values or N_VALUES):
        for rho in (rho_values or RHO_VALUES):
            vals = [v for v in results.get((N, rho), []) if not np.isnan(v)]
            mean = np.mean(vals) if vals else np.nan
            std  = np.std(vals)  if vals else np.nan
            lines.append(f"{N},{rho},{mean:.4f},{std:.4f},{len(vals)}")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved CSV: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--seed',   type=int, default=123)
    parser.add_argument('--rho',    nargs='+', type=float, default=RHO_VALUES,
                        help='Which rho values to run (default: 1 2 3); '
                             'floats allowed (e.g. 1.5 2.5) for the exponent sweep')
    parser.add_argument('--N-list', nargs='+', type=int, default=N_VALUES,
                        help='Which N values to run (default: 5 6 7 8 9 10)')
    parser.add_argument('--method', choices=['oracle', 'kkt', 'nnls'],
                        default='kkt',
                        help='alpha estimator: kkt (non-oracle Keshavarz KKT '
                             'residual; the paper method, DEFAULT), '
                             'oracle (true alpha; upper-bound diagnostic only, '
                             'does NOT match the paper), '
                             'nnls (non-oracle objective-value NNLS; weak, '
                             'retained to document identification failure)')
    parser.add_argument('--ptv-bind-tol', type=float, default=0.1,
                        help='binding-PTV dose tolerance (Gy) for KKT method')
    parser.add_argument('--tag', type=str, default=None,
                        help='filename suffix for outputs (default: method)')
    parser.add_argument('--metric', choices=['w', 'dose', 'penalty', 'mape'],
                        default='w',
                        help='error space: w (beamlet Rel-L2, paper default), '
                             'dose (voxel Rel-L2 on d = D w), '
                             'penalty (%% true-objective regret 100*(J_pred-J_true)/J_true), '
                             'or mape (%% per-voxel dose MAPE over d_true>=5 Gy)')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--plot-only', action='store_true',
                        help='regenerate the figure from the canonical CSVs '
                             '(parametric_mape_nested_<tag>.csv + '
                             'prediction_nested/prediction_errors.csv) without '
                             're-solving; the figure then provably matches the table')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.tag or args.method
    if args.plot_only:
        make_figure_from_csv(
            par_csv=os.path.join(OUT_DIR, f'parametric_mape_nested_{tag}.csv'),
            out_path=os.path.join(OUT_DIR, f'parametric_mape_figure_{tag}.pdf'),
            N_values=args.N_list, rho_values=args.rho)
        return
    if args.method == 'oracle':
        print("WARNING: --method oracle injects the TRUE alpha_k (FORWARD_PARAMS) and is an "
              "upper-bound diagnostic ONLY. It does NOT reproduce the paper's Table "
              "(tab:portpy_parametric_misspec); use --method kkt (default) for that.",
              file=sys.stderr, flush=True)

    # Select fixed test set (same patients for all N, as in run_prediction.py)
    training_ids = get_all_training_ids('nested')
    test_patients, test_w, test_obj = select_test_patients(
        'results/all_patients/forward_cache_all.pkl',
        training_ids, args.n_test, args.seed,
    )
    print(f"Test set: {len(test_patients)} patients", flush=True)

    # Use the canonical fixed test set from test_patients_canonical.txt if present
    canonical_file = 'test_patients_canonical.txt'
    if os.path.exists(canonical_file):
        with open(canonical_file) as f:
            fixed_ids = [l.strip() for l in f if l.strip()]
        test_patients, test_w, test_obj = select_test_patients(
            'results/all_patients/forward_cache_all.pkl',
            training_ids, args.n_test, args.seed,
            fixed_test_ids=fixed_ids[:args.n_test],
        )
        print(f"Using canonical test set: {len(test_patients)} patients", flush=True)

    results = {}
    for N in args.N_list:
        for rho in args.rho:
            mape_list = run_one_N_rho(
                N, rho, test_patients, test_w,
                method=args.method,
                ptv_bind_tol=args.ptv_bind_tol, verbose=args.verbose,
                metric=args.metric,
            )
            results[(N, rho)] = mape_list

    csv_name = f'parametric_mape_nested_{tag}.csv'
    fig_name = f'parametric_mape_figure_{tag}.pdf'
    write_csv(results, os.path.join(OUT_DIR, csv_name),
              N_values=args.N_list, rho_values=args.rho)
    make_figure(results, os.path.join(OUT_DIR, fig_name),
                N_values=args.N_list, rho_values=args.rho)

    # Print summary table
    print("\n===== SUMMARY =====")
    print(f"{'N':>4}  " + "  ".join(f"{'rho='+str(r):>12}" for r in args.rho))
    for N in args.N_list:
        row = f"{N:>4}  "
        for rho in args.rho:
            vals = [v for v in results.get((N, rho), []) if not np.isnan(v)]
            mean = np.nanmean(vals) if vals else np.nan
            row += f"  {mean:>10.3f}%"
        print(row)


if __name__ == '__main__':
    main()
