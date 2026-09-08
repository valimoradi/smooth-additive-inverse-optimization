"""
Paper figures for the synthetic healthcare experiment:
  - combined_mape_vs_training_size_OR_style.png : test-set MAPE vs N for the 4 ablation models
  - true_vs_retrieved_1x3_large.png             : recovered vs ground-truth structural functions
Transcribed from notebook cells 34 and 58.
"""
import os
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (PRED_OUT, PRED_CSV, RESULTS, PLOT_DIR, FINAL_PLOT_DIR,
                    ALPHA_1, PARAMS)

_OR_STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.5, "lines.markersize": 6, "axes.linewidth": 1.0, "mathtext.fontset": "cm",
}

# ---- Combined MAPE figure (cell 34) ----
# Capped at N=200: out-of-sample error plateaus by ~N=60, so N=300,400 added range
# without information. The paper figure stops at 200 in step (see tim-style Fig 4).
COMBINED_SIZES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
# Legend taxonomy, colors, and markers are kept identical to the consumer-choice
# figures (consumer_inverse_optimization.py MODELS) so the 2x2 model family reads
# the same across every figure in the paper:
#   Convex only=C0/o, Additive=C2/s, Smooth=C1/D, Additive+Smooth=C3/^ (all solid).
MODEL_PLOT = [
    ("nonadd_nonsmooth_lp", "Convex only",  "C0", "o", "-"),
    ("add_lp",           "Additive",        "C2", "s", "-"),
    ("nonadd_smooth",    "Smooth",          "C1", "D", "-"),
    ("add_smooth",       "Additive+Smooth", "C3", "^", "-"),
]


def plot_combined_mape(out_dir=PLOT_DIR, pred_out=None):
    plt.rcParams.update(_OR_STYLE)
    os.makedirs(out_dir, exist_ok=True)
    pred_out = pred_out or PRED_OUT
    plt.figure(figsize=(8, 6))
    for key, label, color, marker, ls in MODEL_PLOT:
        path = os.path.join(pred_out[key], PRED_CSV[key])
        if not os.path.exists(path):
            print(f"  missing {path}", flush=True); continue
        df = pd.read_csv(path)
        df = df[df['training_size'].isin(COMBINED_SIZES)].sort_values('training_size')
        plt.plot(df['training_size'], df['test_RelL2_w'], label=label, color=color,
                 marker=marker, linestyle=ls, linewidth=1.8)
    plt.title("Beamlet Prediction Accuracy vs. Training Size")
    plt.xlabel(r"Training Set Size ($N$)")
    plt.ylabel(r"Relative $L_2$ error of predicted beamlets")
    plt.xticks(sorted(COMBINED_SIZES))
    plt.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
    plt.grid(False); plt.tight_layout()
    fname = os.path.join(out_dir, "combined_mape_vs_training_size_OR_style.png")
    plt.savefig(fname, dpi=300); plt.close()
    print(f"wrote {fname}", flush=True)
    return fname


# ---- True vs retrieved structural functions (cell 58) ----
SIZES_TO_PLOT = [60, 120, 200]
OAR_NAMES = ["Quadratic", "Linear", "Exponential"]
X_CAPS = [2000, 1000, 200]
NUM_QUERY_POINTS = 50
STYLES = [
    {'n': 60,  'color': 'tab:red',   'ls': '-.', 'label': 'N=60'},
    {'n': 120, 'color': 'tab:green', 'ls': '--', 'label': 'N=120'},
    {'n': 200, 'color': 'tab:blue',  'ls': '-',  'label': 'N=200'},
]
_FINE_TOL = {f'MSK_DPAR_INTPNT_CO_TOL_{k}': 1e-11 for k in ("REL_GAP", "PFEAS", "DFEAS")}


def _ground_truth(z, k):
    if k == 0:
        return ALPHA_1 * (z ** 2)
    if k == 1:
        return 1.0 * z
    return PARAMS['alpha_exp'] * np.exp(z / PARAMS['exp_scale_k'])


def _load_add_smooth_params(n, results_dir):
    return {'delta': np.load(f'{results_dir}/final_delta_comp_N{n}_v3.npy'),
            'lambda': np.load(f'{results_dir}/final_lambda_N{n}_v3.npy'),
            'z_hat': np.load(f'{results_dir}/Z_hat_obs_N{n}_v3.npy'),
            'beta': np.load(f'{results_dir}/beta_star_N{n}_v3.npy')[0]}


def _eval_structural(z_query, k, params):
    d_k = params['delta'][:, k]; l_k = params['lambda'][:, k]; z_h = params['z_hat'][:, k]
    beta = params['beta']; n = len(d_k)
    alpha = cp.Variable(n, nonneg=True); v = cp.Variable(n); t = cp.Variable(n, nonneg=True)
    obj = cp.Minimize((0.5 * beta) * cp.sum(t) + cp.sum(cp.multiply(v, l_k))
                      - cp.sum(cp.multiply(alpha, l_k * z_h)) + cp.sum(cp.multiply(alpha, d_k)))
    cons = [cp.sum(alpha) == 1, cp.sum(v) >= z_query]
    for j in range(n):
        cons.append(cp.quad_over_lin(v[j] - alpha[j] * z_h[j], alpha[j]) <= t[j])
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.MOSEK, mosek_params=_FINE_TOL)
        return prob.value
    except Exception:
        try:
            prob.solve(solver=cp.MOSEK)
            return prob.value
        except Exception:
            return None


def plot_true_vs_retrieved(out_dir=FINAL_PLOT_DIR, results_dir=None):
    # The recovered objective is identified only up to a positive scale: argmin f is
    # invariant under f -> a*f (a>0), and U_max (=1000, fixed) is an arbitrary gauge
    # that only sets that scale (predictions are gauge-invariant; see config note).
    # The two-point normalization pins the TOTAL penalty to U_max at a fixed anchor, so
    # raw per-component magnitudes are not individually identified and depend on which
    # anchor set was used: N=120 and N=200 share the same anchor and their raw curves
    # nearly coincide, while the N=60 run used a different (nearer) anchor and sits
    # higher -- so the apparent spread across curves is an anchor/gauge effect, not a
    # trend in N. Plotting raw magnitudes on a shared autoscaled axis conflates this with
    # accuracy. To compare shapes honestly we put every curve on ONE axis in true-f units,
    # rescaling each by the single positive factor that matches the true value at the
    # largest queried outcome (scale-only, shape-free).
    plt.rcParams.update({**_OR_STYLE, "font.size": 14, "axes.labelsize": 16,
                         "axes.titlesize": 18, "legend.fontsize": 14, "lines.linewidth": 2.5})
    os.makedirs(out_dir, exist_ok=True)
    results_dir = results_dir or RESULTS["add_smooth"]
    # Rug: observed training outcomes from the largest available cohort.
    rug_params = None
    for style in sorted(STYLES, key=lambda s: -s['n']):
        try:
            rug_params = _load_add_smooth_params(style['n'], results_dir); break
        except FileNotFoundError:
            continue
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    plt.subplots_adjust(wspace=0.30, left=0.07, right=0.98, bottom=0.15, top=0.85)
    for idx, oar_name in enumerate(OAR_NAMES):
        ax = axes[idx]
        z_range = np.linspace(0, X_CAPS[idx], NUM_QUERY_POINTS)
        ax.plot(z_range, _ground_truth(z_range, idx), color='black', linestyle='-',
                linewidth=3.0, label='True Forward', zorder=5)
        for style in STYLES:
            try:
                params = _load_add_smooth_params(style['n'], results_dir)
            except FileNotFoundError:
                continue
            f_ret, valid_z = [], []
            for zq in z_range:
                val = _eval_structural(zq, idx, params)
                if val is not None:
                    f_ret.append(val); valid_z.append(zq)
            if len(valid_z) >= 2 and abs(f_ret[-1]) > 0:
                scale = _ground_truth(valid_z[-1], idx) / f_ret[-1]  # scale-free alignment at z_max
                ax.plot(valid_z, [scale * v for v in f_ret], color=style['color'],
                        linestyle=style['ls'], linewidth=2.0, label=f"Retrieved ({style['label']})")
        if rug_params is not None:
            rug = rug_params['z_hat'][:, idx]
            ax.plot(rug, np.zeros_like(rug), marker='|', linestyle='None',
                    color='0.35', markersize=10, markeredgewidth=0.8, alpha=0.7)
        ax.set_title(oar_name, fontsize=18, fontweight='bold', pad=12)
        ax.set_xlabel(r"Outcome ($z$)")
        ax.set_ylabel(r"Penalty value $f(z)$  (retrieved rescaled to true at $z_{\max}$)", color='black')
        ax.set_xlim(left=0, right=X_CAPS[idx])
        ax.set_ylim(bottom=0)
    legend_elements = [plt.Line2D([0], [0], color='black', lw=3, label='True Forward Function')]
    for s in STYLES:
        legend_elements.append(plt.Line2D([0], [0], color=s['color'], linestyle=s['ls'], lw=2.0,
                                          label=f"Retrieved ({s['label']})"))
    legend_elements.append(plt.Line2D([0], [0], color='0.35', marker='|', linestyle='None',
                                      markersize=10, label='Training outcomes (rug)'))
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=5, frameon=False)
    fname = os.path.join(out_dir, "true_vs_retrieved_1x3_large.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"wrote {fname}", flush=True)
    return fname


if __name__ == "__main__":
    plot_combined_mape()
    plot_true_vs_retrieved()
