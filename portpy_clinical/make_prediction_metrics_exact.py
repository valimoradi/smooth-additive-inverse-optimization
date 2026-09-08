"""
make_3panel_figure.py  (now 2x2: four complementary error spaces)
=================================================================
Defensible §6.3 figure: prediction quality of the recovered objective vs. training
size N, shown in FOUR complementary error spaces, log-y, with the misspecified and
correctly-specified parametric baselines overlaid in each panel.

Panels (all log-y; lower is better):
  (a) True-objective regret   100*(J(w_pred)-J(w_true))/J(w_true), J the true
      forward penalty sum_k alpha_k sum_v max(0,d_v-theta_k)^2       -- decision quality (%).
  (b) Dose-space Rel-L2        100*||d_pred-d_true|| / ||d_true||     -- physical (%).
  (c) High-dose dose-MAPE      100*mean_v |d_pred-d_true|/d_true over the TG-218
      low-dose threshold d_true >= 10% of max(d_true)                 -- clinical (%).
  (d) Beamlet Rel-L2_w         ||w_pred-w_true|| / ||w_true||         -- decision vector
      (the metric reported in Sections 6.1-6.2; a fraction, not a percent).

The mask in (c) is the low-dose threshold of AAPM TG-218 (Miften et al., Med. Phys.
2018): voxels below 10% of the maximum dose are excluded, the standard cut for
dose-agreement metrics, which removes the low-dose bath where a per-voxel
percentage error divides by a near-zero dose. The SAME ground-truth mask
(d_true >= 0.1*max d_true, per patient) is applied to every predictor, like-for-like.

Series in every panel:
  - Nonparametric (proposed)  : recomputed from doses_N*/*.npz (mean over test patients)
  - Parametric rho=1 (linear, misspecified)   -> structural error, flat-high in N
  - Parametric rho=3 (cubic,  misspecified)    -> structural error, flat-high in N
  - Parametric rho=2 (quadratic, correct form) -> near-zero floor (attainable only if rho known)

Story the log axis carries: in ALL FOUR spaces the proposed method drops from its
N=3 value to a low floor by N=4 and holds it, sitting well below every misspecified
parametric form and approaching the correct-form floor -- without committing to a
functional form. The beamlet panel (d) shows the ~0.2 decision-vector error of
Sections 6.1-6.2 is small next to the misspecified forms (0.6-1.1), not large.

Parametric means read from the canonical CSVs (provably match the numbers in text):
  regret : parametric_mape_nested_penalty_kkt.csv   (mean_rel_error in %)
  dose   : parametric_mape_nested_dose_kkt.csv       (mean_rel_error as fraction -> x100)
  mape   : parametric_mape_nested_mape_kkt.csv       (mean_rel_error in %)
  w      : parametric_mape_nested_kkt.csv            (mean_rel_error as fraction)
Nonparametric beamlet Rel-L2_w is taken from doses_N*/*.npz where w is saved
(N=3,4,8) and from prediction_nested/prediction_errors.csv (MeanRelError) otherwise;
the two agree exactly where both exist (N=8: 0.2023).
"""
import os, glob, pickle, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from config import FORWARD_PARAMS, DOSE_THRESHOLDS, OAR_KEYS

OUT_DIR   = 'results/parametric_comparison'
CACHE     = 'results/all_patients/forward_cache_all.pkl'
DOSES_DIR = os.environ.get('FL_DOSES_DIR', 'results/prediction_nested_fullleak_anchor')
PRED_CSV  = os.path.join(DOSES_DIR, 'prediction_errors.csv')
N_VALUES  = [3, 4, 5, 6]
MAPE_LOW_DOSE_FRAC = 0.10  # TG-218 low-dose threshold: 10% of the maximum dose
# Linear y-axis (from 0): the message is the proposed method's rapid descent to a
# low floor as patients are added, and its gap to the flat misspecified baselines.
# On a log axis that descent reads as flat; the correct-form rho=2 sits near 0.

PAR_CSV = {
    'regret': os.path.join(OUT_DIR, 'parametric_mape_nested_penalty_kkt.csv'),
    'dose':   os.path.join(OUT_DIR, 'parametric_mape_nested_dose_kkt.csv'),
    'mape':   os.path.join(OUT_DIR, 'parametric_mape_nested_mape_kkt.csv'),
    'w':      os.path.join(OUT_DIR, 'parametric_mape_nested_kkt.csv'),
}
# CSV mean_rel_error unit -> plotted-unit multiplier
PAR_SCALE = {'regret': 1.0, 'dose': 100.0, 'mape': 1.0, 'w': 1.0}


def true_penalty(dose, organ_indices):
    J = 0.0
    for k in OAR_KEYS:
        idx = organ_indices.get(k, np.array([], dtype=int))
        if len(idx) == 0:
            continue
        o = np.maximum(0.0, dose[idx] - DOSE_THRESHOLDS[k])
        J += FORWARD_PARAMS[f'alpha_{k}'] * np.sum(o ** 2)
    return J


def _csv_beamlet_relL2():
    """Nonparametric beamlet Rel-L2_w per N from prediction_errors.csv (MeanRelError)."""
    out = {}
    if os.path.exists(PRED_CSV):
        with open(PRED_CSV, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    out[int(row['N'])] = float(row['MeanRelError'])
                except (ValueError, KeyError):
                    pass
    return out


def nonparametric_by_N():
    """Mean error per N for all four metrics, from saved doses (own provenance)."""
    cache = pickle.load(open(CACHE, 'rb'))
    oi = {p['patient_id']: p['organ_indices'] for p in cache['patients']}
    csv_w = _csv_beamlet_relL2()
    out = {'regret': {}, 'dose': {}, 'mape': {}, 'w': {}}
    for N in N_VALUES:
        files = sorted(glob.glob(f'{DOSES_DIR}/doses_N{N}/Prostate_Patient_*.npz'))
        reg, dl2, mape, wl2 = [], [], [], []
        for f in files:
            a = np.load(f)
            dt = a['d_true'].astype(float)
            dp = a['d_pred'].astype(float)
            pid = os.path.basename(f).replace('.npz', '')
            nt = np.linalg.norm(dt)
            if nt > 0:
                dl2.append(100.0 * np.linalg.norm(dp - dt) / nt)
            m = dt >= MAPE_LOW_DOSE_FRAC * dt.max()
            if np.any(m):
                mape.append(100.0 * np.mean(np.abs(dp[m] - dt[m]) / dt[m]))
            if 'w_true' in a.files:
                wt = a['w_true'].astype(float); wp = a['w_pred'].astype(float)
                nw = np.linalg.norm(wt)
                if nw > 0:
                    wl2.append(np.linalg.norm(wp - wt) / nw)
            if pid in oi:
                Jt = true_penalty(dt, oi[pid])
                Jp = true_penalty(dp, oi[pid])
                if Jt > 0:
                    reg.append(100.0 * (Jp - Jt) / Jt)
        if reg:
            out['regret'][N] = float(np.mean(reg))
        if dl2:
            out['dose'][N] = float(np.mean(dl2))
        if mape:
            out['mape'][N] = float(np.mean(mape))
        if wl2:
            out['w'][N] = float(np.mean(wl2))
        elif N in csv_w:
            out['w'][N] = csv_w[N]
    return out


def parametric_by_N(metric):
    """{rho: {N: mean error}} from the canonical CSV for this metric."""
    par, path, scale = {}, PAR_CSV[metric], PAR_SCALE[metric]
    if not os.path.exists(path):
        return par
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            v = row['mean_rel_error']
            if v in ('', 'nan'):
                continue
            rho = int(float(row['rho']))
            if rho not in (1, 2, 3):   # ignore stray rho=4 rows from old sweeps
                continue
            N = int(row['N'])
            par.setdefault(rho, {})[N] = float(v) * scale
    return par


def main():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 10.5,
        'axes.titlesize': 11, 'legend.fontsize': 10,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'lines.linewidth': 1.8, 'axes.linewidth': 0.8,
    })

    nonpar = nonparametric_by_N()
    Ns = N_VALUES
    x = np.array(Ns)

    par_styles = {
        1: ('C0', 'o', r'Parametric $\rho{=}1$ (linear, misspecified)'),
        3: ('C1', '^', r'Parametric $\rho{=}3$ (cubic, misspecified)'),
        2: ('C2', 's', r'Parametric $\rho{=}2$ (quadratic, correct form)'),
    }
    panels = [
        ('regret', '(a) True-objective regret',                 r'Mean regret (\%)'),
        ('dose',   r'(b) Dose-space Rel-$L_2$',                  r'Mean Rel-$L_2$ dose error (\%)'),
        ('mape',   r'(c) High-dose MAPE ($d\geq 10\%\,d_{\max}$)', r'Mean dose MAPE (\%)'),
        ('w',      r'(d) Beamlet Rel-$L_2$ (decision vector)',   r'Mean $\|\hat w-w\|/\|w\|$'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    axes = axes.ravel()
    handles, labels = [], []
    for ax, (metric, title, ylab) in zip(axes, panels):
        par = parametric_by_N(metric)
        for rho in (1, 3, 2):
            if rho not in par:
                continue
            c, mk, lab = par_styles[rho]
            y = [par[rho].get(N, np.nan) for N in Ns]
            h, = ax.plot(x, y, marker=mk, color=c, markersize=6,
                         linewidth=1.7, alpha=0.9, label=lab)
            if lab not in labels:
                handles.append(h); labels.append(lab)
        yn = [nonpar[metric].get(N, np.nan) for N in Ns]
        h, = ax.plot(x, yn, marker='D', color='C3', linewidth=2.8, markersize=7.5,
                     markeredgecolor='black', markeredgewidth=0.5, zorder=5,
                     label='Nonparametric (proposed)')
        if 'Nonparametric (proposed)' not in labels:
            handles.append(h); labels.append('Nonparametric (proposed)')

        ax.set_ylim(bottom=0)
        ax.set_title(title, pad=6)
        ax.set_xlabel('Number of training patients ($N$)')
        ax.set_ylabel(ylab)
        ax.set_xticks(Ns)
        ax.grid(True, which='both', linestyle=':', alpha=0.5, linewidth=0.7)
        ax.spines[['top', 'right']].set_visible(False)

    order = [labels.index('Nonparametric (proposed)')] + \
            [i for i, l in enumerate(labels) if l != 'Nonparametric (proposed)']
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=2.0, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out_pdf = os.path.join(OUT_DIR, 'prediction_metrics_exact.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_pdf}')

    for metric, _, _ in panels:
        par = parametric_by_N(metric)
        print(f'\n[{metric}]')
        print('  nonpar :', {N: round(nonpar[metric].get(N, float("nan")), 4) for N in Ns})
        for rho in (1, 2, 3):
            if rho in par:
                print(f'  rho={rho}  :', {N: round(par[rho].get(N, float("nan")), 4) for N in Ns})


if __name__ == '__main__':
    main()
