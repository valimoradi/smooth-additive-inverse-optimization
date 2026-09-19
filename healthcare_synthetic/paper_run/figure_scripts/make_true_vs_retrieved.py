"""Regenerate Fig 3 (true-vs-retrieved) from CANONICAL 142 results.

Presentation: single axis in true-f units, each scale-free curve rescaled to true at the largest
observed outcome, one rug row per training size.

Revised 2026-07-29. The first version drew a single rug taken from the N=60 cohort only, so the
reader could not tell which outcomes belonged to which training size, and it used hand-picked
panel caps [2000, 1000, 200]. Those caps hid 19 of 60 quadratic and 8 of 60 linear outcomes, and
on the exponential panel they ran 32% of the panel width past the last observation (z=136.7) and
placed the rescaling anchor inside that empty region. Both are fixed here: the panel range and the
anchor now come from the observed outcomes, and each training size gets its own rug row.

Revised 2026-09-13 (NEW-24, B12). The evaluator is the monotone envelope of Appendix B.4, i.e. the
perspective form with sum(v) >= z (it was sum(v) == z; on the published parameters the two agree to
about 1e-9 of f at the cap). A query counts only at MOSEK status 'optimal' (one retry at 1e-5,
otherwise the script stops). Input folder and output file can be overridden:
    python make_true_vs_retrieved.py [RESULTS_DIR [OUT_PNG]]
or with the environment variables FIG3_RESULTS_DIR / FIG3_OUT. The defaults are the same files as
before (142 copy/results_high_std and figures/true_vs_retrieved_1x3_large.png), located relative to
this script instead of the hard-coded d:/ drive.
"""
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
import numpy as np, cvxpy as cp, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RD = os.path.normpath(os.path.join(HERE, "..", "results_high_std"))
_DEFAULT_OUT = os.path.join(HERE, "true_vs_retrieved_1x3_large.png")
RD = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIG3_RESULTS_DIR", _DEFAULT_RD)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FIG3_OUT", _DEFAULT_OUT)
print("results:", RD, "\noutput:", OUT)
NAMES = ["Quadratic", "Linear", "Exponential"]
STYLES = [{'n': 20, 'c': 'tab:red',   'ls': '-.', 'lab': 'N=20'},
          {'n': 40, 'c': 'tab:green', 'ls': '--', 'lab': 'N=40'},
          {'n': 60, 'c': 'tab:blue',  'ls': '-',  'lab': 'N=60'}]
GT = lambda z, k: (0.001*z**2) if k == 0 else (1.0*z if k == 1 else 1e-3*np.exp(z/15.0))
ONE_THREAD = {'MSK_IPAR_NUM_THREADS': 1}
RETRY_TOL = {f'MSK_DPAR_INTPNT_CO_TOL_{s}': 1e-5 for s in ('PFEAS', 'DFEAS', 'REL_GAP')}


def load(n):
    return {'delta': np.load(f'{RD}/final_delta_comp_N{n}_v3.npy'),
            'lambda': np.load(f'{RD}/final_lambda_N{n}_v3.npy'),
            'z_hat': np.load(f'{RD}/Z_hat_obs_N{n}_v3.npy'),
            'beta': float(np.load(f'{RD}/beta_star_N{n}_v3.npy')[0])}


_EV = {}


def ev(zq, k, p):
    """f_k(zq): monotone envelope (B.4), min of the perspective form subject to sum(v) >= zq."""
    key = (k, id(p))
    if key not in _EV:
        dk, lk, zh, beta = p['delta'][:, k], p['lambda'][:, k], p['z_hat'][:, k], p['beta']
        n = len(dk)
        z = cp.Parameter()
        a = cp.Variable(n, nonneg=True); v = cp.Variable(n); t = cp.Variable(n, nonneg=True)
        ob = cp.Minimize(0.5*beta*cp.sum(t) + cp.sum(cp.multiply(v, lk))
                         - cp.sum(cp.multiply(a, lk*zh)) + cp.sum(cp.multiply(a, dk)))
        co = [cp.sum(a) == 1, cp.sum(v) >= z] + \
             [cp.quad_over_lin(v[j] - a[j]*zh[j], a[j]) <= t[j] for j in range(n)]
        _EV[key] = (cp.Problem(ob, co), z)
    prob, z = _EV[key]
    z.value = float(zq)
    statuses = []
    for extra in ({}, RETRY_TOL):
        try:
            prob.solve(solver=cp.MOSEK, mosek_params={**ONE_THREAD, **extra})
            st = prob.status
        except cp.error.SolverError as ex:
            st = f"exception:{type(ex).__name__}"
        statuses.append(st)
        if st == 'optimal':
            if extra:
                print(f"  [retry at 1e-5 tolerances at z={zq:.6f}, k={k}: {statuses}]")
            return prob.value
    raise RuntimeError(f"evaluator failed at z={zq}, k={k}: {statuses}")


# Panel range = largest observed outcome, except that a single outcome more than three times the
# next largest is left off the panel and reported in the caption. This fires on the quadratic
# component only (one patient at z=26143 against 4810 for the next), where drawing to the full
# support would crowd 59 of 60 outcomes into the left fifth of the panel. The cohorts are nested,
# so the cap is the same for all three training sizes and lies inside the observed support of each.
P = {s['n']: load(s['n']) for s in STYLES}
CAPS, DROPPED = [], []
for k in range(3):
    v = np.sort(P[60]['z_hat'][:, k])
    cap = v[-2] if v[-1] > 3*v[-2] else v[-1]
    CAPS.append(float(cap))
    DROPPED.append(int((P[60]['z_hat'][:, k] > cap).sum()))
print("caps:", [f"{c:.1f}" for c in CAPS], "off-panel at N=60:", DROPPED)

plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.labelsize": 16,
                     "axes.titlesize": 18, "legend.fontsize": 14, "xtick.direction": "in",
                     "ytick.direction": "in", "mathtext.fontset": "cm"})
fig = plt.figure(figsize=(20, 7.6))
gs = GridSpec(2, 3, figure=fig, height_ratios=[6, 1], hspace=0.06, wspace=0.26,
              left=0.06, right=0.985, bottom=0.13, top=0.83)

for k in range(3):
    ax = fig.add_subplot(gs[0, k])
    axr = fig.add_subplot(gs[1, k], sharex=ax)
    cap = CAPS[k]
    zr = np.linspace(0, cap, 60)
    ax.plot(zr, GT(zr, k), color='black', lw=3, label='True Forward', zorder=5)
    for s in STYLES:
        p = P[s['n']]
        zmin = p['z_hat'][:, k].min()   # query only where observed (no extrapolation below z_min)
        vz = list(np.linspace(zmin, cap, 60))
        fr = [ev(zq, k, p) for zq in vz]
        if len(vz) >= 2 and abs(fr[-1]) > 0:
            sc = GT(vz[-1], k)/fr[-1]   # anchor at the largest observed outcome on the panel
            ax.plot(vz, [sc*x for x in fr], color=s['c'], ls=s['ls'], lw=2.0,
                    label=f"Retrieved ({s['lab']})")
            vz, fr = np.array(vz), sc*np.array(fr)
            g = GT(vz, k); d = fr - g; i = int(np.argmax(np.abs(d)))
            print(f"  {NAMES[k]:11s} N={s['n']:2d}  RelL2={np.sqrt(np.sum(d**2)/np.sum(g**2)):.3f}  "
                  f"max gap {d[i]:+.3g} at z={vz[i]:.1f} ({d[i]/g[-1]*100:+.0f}% of f* at the cap)")
    ax.set_title(NAMES[k], fontweight='bold', pad=12)
    ax.set_ylabel(r"Penalty value $f(z)$")
    ax.set_xlim(0, cap); ax.set_ylim(bottom=0)
    ax.tick_params(labelbottom=False)

    # one rug row per training size, colour-matched to its curve
    for j, s in enumerate(STYLES):
        rug = P[s['n']]['z_hat'][:, k]
        rug = rug[rug <= cap]
        axr.plot(rug, np.full_like(rug, 2-j), marker='|', ls='None',
                 color=s['c'], ms=11, mew=1.0, alpha=0.85)
    axr.set_ylim(-0.7, 2.7); axr.set_yticks([2, 1, 0])
    axr.set_yticklabels([s['lab'] for s in STYLES], fontsize=11)
    axr.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        axr.spines[sp].set_visible(False)
    axr.set_xlabel(r"Outcome ($z$)")

h = [plt.Line2D([0], [0], color='black', lw=3, label='True Forward Function')]
for s in STYLES:
    h.append(plt.Line2D([0], [0], color=s['c'], ls=s['ls'], lw=2.0,
                        label=f"Retrieved ({s['lab']})"))
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print("wrote", OUT)
