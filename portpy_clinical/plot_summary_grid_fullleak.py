"""
Exact-anchor (full-leakage) counterpart of Figure 5 (summary_grid_nested).

Identical layout, evaluation and conventions to plot_summary_grid_nested.py
(rows = training size N, columns = the six OAR groups; recovered f_k on a dense
grid via the Proposition 1 conjugate, ground-truth alpha_k z^2 on a twin axis,
RIND_0 restricted to its observation support with the data-free gap shaded).
The ONLY change is the source directory: results/nested_N{N}_fullleak_anchor,
i.e. both anchors pinned to the exact forward objectives
    U_min = obj(i_b),  U_max = obj(i_w)
instead of the deployable U_min=0. This is the "what Figure 5 would look like
with the anchors known" diagnostic. Oracle only; not deployable.

Output: results/plots_nested/summary_grid_fullleak.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recovery.structural_function import evaluate_function_curve_fast

LABELS = ['Bladder', 'Rectum', 'Femoral Heads',
          'RIND 0-2mm', 'RIND 2-20mm', 'RIND 20-40mm']
TRUE_ALPHAS = [20.0, 20.0, 10.0, 5.0, 5.0, 3.0]
N_VALUES = [3, 4, 5, 6]
RESTRICT_X_COLS = {3}
BETA = 40.0
N_QUERY = 3000

ROOT = os.path.abspath(os.path.dirname(__file__))
RUN_DIR = os.environ.get('FL_RUN_DIR', 'nested_N{N}_fullleak_anchor')
DATA_FMT = os.path.join(ROOT, 'results', RUN_DIR, '{name}_func{k}_N{N}.npy')
OUT_DIR = os.path.join(ROOT, 'results', 'plots_nested')
OUT = os.path.join(OUT_DIR, os.environ.get('FL_OUT', 'summary_grid_fullleak.png'))


def load_params_for_N(N):
    delta_funcs, lambda_funcs, z_funcs = [], [], []
    for k in range(6):
        zp = DATA_FMT.format(N=N, name='Z', k=k)
        dp = DATA_FMT.format(N=N, name='delta', k=k)
        lp = DATA_FMT.format(N=N, name='lambda', k=k)
        if not (os.path.exists(zp) and os.path.exists(dp)
                and os.path.exists(lp)):
            z_funcs.append(np.array([], dtype=float))
            delta_funcs.append(np.array([], dtype=float))
            lambda_funcs.append(np.array([], dtype=float))
            continue
        z = np.load(zp).astype(float)
        d = np.load(dp).astype(float)
        l = np.load(lp).astype(float)
        o = np.argsort(z)
        z_funcs.append(z[o])
        delta_funcs.append(d[o])
        lambda_funcs.append(l[o])
    base = os.path.join(ROOT, 'results', RUN_DIR.format(N=N))
    bu_path = os.path.join(base, 'beta_used.npy')
    bs_path = os.path.join(base, 'beta_star.npy')
    if os.path.exists(bu_path):
        beta = float(np.load(bu_path))
    elif os.path.exists(bs_path):
        beta = float(np.load(bs_path))
    else:
        beta = BETA
    return {
        'delta_funcs': delta_funcs,
        'lambda_funcs': lambda_funcs,
        'Z_per_func': z_funcs,
        'beta': beta,
    }


fig, axes = plt.subplots(len(N_VALUES), 6, figsize=(22, 3 * len(N_VALUES)))

for row, N in enumerate(N_VALUES):
    params = load_params_for_N(N)
    print(f'N={N}: beta={params["beta"]:.4f}; evaluating f_k on dense grid ...',
          flush=True)
    for col in range(6):
        ax = axes[row, col]
        z = params['Z_per_func'][col]
        d = params['delta_funcs'][col]
        a_true = TRUE_ALPHAS[col]

        if len(z) == 0:
            ax.text(0.5, 0.5, 'MISSING', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{LABELS[col]}, N={N}', fontsize=9)
            continue

        if col in RESTRICT_X_COLS:
            mask = d > d.max() * 1e-6
            if mask.sum() > 5:
                x_lo = max(float(np.quantile(z[mask], 0.005)) - 1.0, 0.0)
                x_hi = float(np.quantile(z[mask], 0.999)) + 1.0
            else:
                x_lo, x_hi = float(z.min()), float(z.max())
        else:
            x_lo, x_hi = 0.0, max(float(z.max()), 1e-6)

        z_grid = np.linspace(x_lo, x_hi, N_QUERY)
        f_grid = evaluate_function_curve_fast(z_grid, col, params)
        ax.plot(z_grid, f_grid, color='tab:blue', linewidth=1.2,
                label=r'$\hat f_k(z)$', zorder=3)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel('z (Gy)', fontsize=8)
        ax.set_ylabel(r'$\hat f_k$', color='tab:blue', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=7)
        ax.set_title(f'{LABELS[col]}, N={N}', fontsize=9)
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        z_truth = np.linspace(x_lo, x_hi, 400)
        ax2.plot(z_truth, a_true * z_truth ** 2,
                 color='tab:red', linewidth=1.4, linestyle='--',
                 label=r'truth $\alpha_k z^2$', zorder=3)
        ax2.set_ylabel(r'$\alpha_k z^2$', color='tab:red', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=7)

        # (No data-free-gap shading: the exact-anchor recovery tracks the truth
        # across the RIND_0 gap region at every N, so the gap does not explain
        # the U_min=0 failure -- the anchor does. Shading it would misattribute.)

        if row == 0 and col == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc='upper left', fontsize=6, framealpha=0.9)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT, dpi=130)
plt.close()
print(f'Saved {OUT}')
