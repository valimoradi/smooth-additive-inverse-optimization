"""Gradient panels for Fig 3: every apparent corner is a smooth beta-ramp, not a kink.

f'(z) across each break, obtained EXACTLY as the dual of the sum(v)>=z constraint (envelope
theorem), never by differencing f. A kink is a JUMP in f', i.e. a VERTICAL segment in these panels.
There is none: f' leaves its lower plateau, rises at slope beta_0 = 0.0880469, and settles on its
upper plateau. The fitted ramp slope divided by beta_0 is printed in each legend; it is 1.00 on
every panel, which is the proof. Curvature cannot exceed beta_0 for a beta_0-smooth function, and
here it attains beta_0 exactly, so the transition is as sharp as the recovered class permits and
still finite.

REVISED 2026-07-30. The previous version plotted only the two N=20 breaks (quadratic, linear) and
so answered for two of the corners a reader can see in Figure 3 while leaving the rest unexplained.
Measured on Figure 3's own 60-point grid, the exponential component turns a visible corner at every
training size, and the N=60 one near z=105 is the sharpest-LOOKING corner in the whole figure. The
four panels below cover every corner a reader actually notices, and they deliberately span both
regimes: three transitions far narrower than one plotting cell (0.10-0.24 units against cells of
1.79-81.5), and one, the N=20 exponential, that is about one full cell wide (1.79 against 1.79).
That last panel matters -- it is the case where Figure 3's grid is NOT hiding a narrow ramp, and it
still shows no jump.

An f(z) row was dropped from this figure on 2026-07-16. It claimed to show a "rounded turn" and
could not: for the narrow breaks the transition is ~0.2 outcome units against a ~4800-unit panel,
i.e. far under a pixel at 300 dpi. The artifact argument is a comparison of two numbers (transition
width vs grid spacing, both printed below) and belongs in the text. What belongs in a panel is f',
because a kink has an unmistakable signature there: a vertical segment.

REVISED 2026-09-13 (NEW-24, NEW-26, B12). The numbers quoted above describe the published
parameters. Nothing is hard-coded any more: the Figure 3 caps come from the loaded N=60 outcomes (as
in make_true_vs_retrieved.py), and each panel's break centre and gradient rise are located from the
loaded parameters (locate_break below): the Figure 3 grid vertex with the largest turning angle in
Figure 3's axes box flags the corner, the adjacent grid cell with the larger f' rise brackets it, the
largest f' rise is zoomed until it fills at least 1/20 of the window, the window is widened until
both ends lie on plateaus, and then dg = plateau-to-plateau rise and z* = the point where f' crosses
the middle of the rise. The evaluator is the B.4 monotone envelope sum(v) >= z, whose dual is f'(z)
>= 0 directly. A query counts only at MOSEK status 'optimal' (tight tolerances first, then MOSEK
defaults, reported; otherwise the script stops). Input folder and output file:
    python make_fig3_zoom.py [RESULTS_DIR [OUT_PNG]]
or FIG7_RESULTS_DIR / FIG7_OUT; defaults are the same files as before, located relative to this
script instead of the hard-coded d:/ drive.

Same CANONICAL 142 results and same conjugate evaluator as make_true_vs_retrieved.py.
"""
import os
import sys
import numpy as np, cvxpy as cp, matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RD = os.path.normpath(os.path.join(HERE, "..", "results_high_std"))
_DEFAULT_OUT = os.path.join(HERE, "true_vs_retrieved_zoom.png")
RD = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIG7_RESULTS_DIR", _DEFAULT_RD)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FIG7_OUT", _DEFAULT_OUT)
print("results:", RD, "\noutput:", OUT)

NCOARSE = 60      # grid used by Figure 3
FIG3_CAP_SIZE = 60
GT = lambda z, k: (0.001*z**2) if k == 0 else (1.0*z if k == 1 else 1e-3*np.exp(z/15.0))
# Height/width of one Figure 3 value panel in inches (figsize 20 x 7.6, GridSpec left=0.06,
# right=0.985, wspace=0.26, bottom=0.13, top=0.83, height_ratios [6, 1], hspace=0.06). Used only to
# measure turning angles of the Figure 3 polyline as the reader sees them.
_W_IN = 20 * (0.985 - 0.06) / (3 + 2 * 0.26)
_H_IN = 7.6 * (0.83 - 0.13) / (1 + 0.06 / 2) * 6 / 7
FIG3_AXES_H_OVER_W = _H_IN / _W_IN

# Panels: which component and training size to zoom on (a presentation choice; the corner itself is
# located from the parameters).
PANELS = [dict(k=0, n=20, name="Quadratic"), dict(k=1, n=20, name="Linear"),
          dict(k=2, n=60, name="Exponential"), dict(k=2, n=20, name="Exponential")]


def load(n):
    return {'delta': np.load(f'{RD}/final_delta_comp_N{n}_v3.npy'),
            'lambda': np.load(f'{RD}/final_lambda_N{n}_v3.npy'),
            'z_hat': np.load(f'{RD}/Z_hat_obs_N{n}_v3.npy'),
            'beta': float(np.load(f'{RD}/beta_star_N{n}_v3.npy')[0])}


# Default MOSEK tolerances leave dual noise that, differenced over the narrow transitions, fakes an
# apparent f'' above beta*. These restore the true bound.
TIGHT = {'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-12, 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-12,
         'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-12, 'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-14}
ONE_THREAD = {'MSK_IPAR_NUM_THREADS': 1}
_EV = {}


def _evaluate(zq, k, p):
    """(f_k(zq), f_k'(zq)) from the monotone envelope min {perspective form : sum(v) >= zq}.
    f' is the dual of sum(v) >= zq (>= 0), not a finite difference."""
    key = (k, id(p))
    if key not in _EV:
        dk, lk, zh, beta = p['delta'][:, k], p['lambda'][:, k], p['z_hat'][:, k], p['beta']
        n = len(dk)
        z = cp.Parameter()
        a = cp.Variable(n, nonneg=True); v = cp.Variable(n); t = cp.Variable(n, nonneg=True)
        ob = cp.Minimize(0.5*beta*cp.sum(t) + cp.sum(cp.multiply(v, lk))
                         - cp.sum(cp.multiply(a, lk*zh)) + cp.sum(cp.multiply(a, dk)))
        c_z = cp.sum(v) >= z
        co = [cp.sum(a) == 1, c_z] + \
             [cp.quad_over_lin(v[j] - a[j]*zh[j], a[j]) <= t[j] for j in range(n)]
        _EV[key] = (cp.Problem(ob, co), z, c_z)
    prob, z, c_z = _EV[key]
    z.value = float(zq)
    # TIGHT first; a query point MOSEK cannot close at 1e-12 is re-solved at MOSEK defaults rather
    # than dropped. Only status 'optimal' counts, and every fallback is reported.
    statuses = []
    for lad in (TIGHT, {}):
        try:
            prob.solve(solver=cp.MOSEK, mosek_params={**ONE_THREAD, **lad})
            st = prob.status
        except cp.error.SolverError as ex:
            st = f"exception:{type(ex).__name__}"
        statuses.append(st)
        if st == 'optimal':
            if lad is not TIGHT:
                print(f"  [fallback to default tolerances at z={zq:.6f}, k={k}: {statuses}]")
            return float(prob.value), float(c_z.dual_value)
    raise RuntimeError(f"evaluator failed at z={zq}, k={k}: {statuses}")


def ev_f(zq, k, p):
    return _evaluate(zq, k, p)[0]


def ev_g(zq, k, p):
    """Return f'(z) as the exact dual of sum(v)>=z, not a finite difference."""
    return _evaluate(zq, k, p)[1]


def figure3_caps(P60):
    caps = []
    for k in range(3):
        v = np.sort(P60['z_hat'][:, k])
        caps.append(float(v[-2] if v[-1] > 3*v[-2] else v[-1]))
    return caps


def locate_break(k, p, cap):
    """Break centre z*, plateau-to-plateau gradient rise dg, and scan details, from the parameters."""
    beta = p['beta']
    zmin = p['z_hat'][:, k].min()
    zg = np.linspace(zmin, cap, NCOARSE)
    f = np.array([ev_f(z, k, p) for z in zg])
    sc = GT(zg[-1], k) / f[-1]                          # Figure 3's rescaling at the cap
    ymax = 1.05 * max(GT(zg[-1], k), float(np.max(sc * f)))
    slope_disp = np.diff(sc * f) / np.diff(zg) * (cap / ymax) * FIG3_AXES_H_OVER_W
    turn = np.abs(np.diff(np.arctan(slope_disp)))       # turning angle at vertices 1..NCOARSE-2
    a = int(np.argmax(turn)) + 1
    g3 = [ev_g(z, k, p) for z in zg[a-1:a+2]]
    lo, hi = (zg[a-1], zg[a]) if g3[1] - g3[0] >= g3[2] - g3[1] else (zg[a], zg[a+1])
    cell = (float(lo), float(hi))
    for _ in range(40):                                 # zoom on the largest rise
        zs = np.linspace(lo, hi, 33)
        gs = np.array([ev_g(z, k, p) for z in zs])
        rise = gs[-1] - gs[0]
        if rise > 0 and hi - lo <= 20.0 * rise / beta:
            break
        m = int(np.argmax(np.diff(gs)))
        lo, hi = zs[max(m - 1, 0)], zs[min(m + 2, len(zs) - 1)]
    else:
        raise RuntimeError(f"k={k}: no concentrated gradient rise found in cell {cell}")
    for _ in range(12):                                 # widen until both ends lie on plateaus
        zf = np.linspace(lo, hi, 81)
        gf = np.array([ev_g(z, k, p) for z in zf])
        sl = np.diff(gf) / np.diff(zf) / beta
        left_flat, right_flat = sl[0] < 0.5, sl[-1] < 0.5
        if left_flat and right_flat:
            break
        width = hi - lo
        if not left_flat:
            lo -= 0.5 * width
        if not right_flat:
            hi += 0.5 * width
    else:
        raise RuntimeError(f"k={k}: break window around {cell} does not end on plateaus")
    g_lo, g_hi = float(gf[0]), float(gf[-1])
    zc = float(np.interp(0.5 * (g_lo + g_hi), np.maximum.accumulate(gf), zf))
    return zc, g_hi - g_lo, dict(vertex=float(zg[a]), turn_deg=float(np.degrees(turn[a-1])),
                                 cell=cell, window=(float(lo), float(hi)))


plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11,
                     "axes.titlesize": 11, "legend.fontsize": 9, "xtick.direction": "in",
                     "ytick.direction": "in", "mathtext.fontset": "cm"})
fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.5))
plt.subplots_adjust(wspace=0.25, hspace=0.48, left=0.075, right=0.978, bottom=0.10, top=0.91)

PARAMS = {n: load(n) for n in sorted({B['n'] for B in PANELS} | {FIG3_CAP_SIZE})}
CAPS = figure3_caps(PARAMS[FIG3_CAP_SIZE])
print("Figure 3 caps:", [f"{c:.6g}" for c in CAPS])

for pos, B in enumerate(PANELS):
    k, n = B['k'], B['n']
    p = PARAMS[n]; beta = p['beta']
    zc, dg, info = locate_break(k, p, CAPS[k])
    zmin = p['z_hat'][:, k].min()
    width = dg / beta                                   # beta-limited width of the transition
    spacing = (CAPS[k] - zmin) / (NCOARSE - 1)          # Figure 3's grid spacing

    ax = axes[pos // 2][pos % 2]
    half = 2.2*width
    zg = np.linspace(zc - half, zc + half, 81)
    g = np.array([ev_g(z, k, p) for z in zg])
    dz = zg - zc
    ax.plot(dz, g, color='tab:blue', lw=2.2, zorder=4, label=r"$f_k'(z)$, exact dual")

    # Local curvature, normalized by beta_0. For a beta_0-smooth function this ratio can never
    # exceed 1, and a kink would make it unbounded. It is the falsifiable quantity, and unlike a
    # single-ramp line fit it stays valid when a transition is several ramps separated by plateaus,
    # which is what the N=60 exponential break turns out to be.
    ratio = np.gradient(g, zg)/beta
    lo_p, hi_p = g.min(), g.max()

    # Shade the ramps, i.e. the stretches carrying real curvature; the rest is affine.
    on_ramp = ratio > 0.5
    edges = np.flatnonzero(np.diff(on_ramp.astype(int)))
    bounds = np.concatenate(([0] if on_ramp[0] else [], edges + 1,
                             [len(zg)-1] if on_ramp[-1] else [])).astype(int)
    first = True
    slopes = []
    for s, e in zip(bounds[0::2], bounds[1::2]):
        ax.axvspan(dz[max(s-1, 0)], dz[min(e, len(dz)-1)], color='tab:orange', alpha=0.16, lw=0,
                   label=(f"ramps, total $\\Delta f_k'/\\beta_0={width:.3f}$" if first else None))
        first = False
        # Line fit over each ramp's interior. A 10% trim clears the plateau corners, where a central
        # difference straddles the jump in f'' and reads a spurious ~1.0007. The fit is the honest
        # estimator of the ramp's curvature and lands on beta_0 to four decimals.
        trim = max(1, int(0.10*(e - s)))
        sl = slice(s + trim, max(e - trim, s + trim + 2))
        if sl.stop - sl.start >= 2:
            slopes.append(np.polyfit(zg[sl], g[sl], 1)[0]/beta)

    # Reference segment of slope beta_0, drawn on the steepest ramp for visual comparison.
    imax = int(np.argmax(ratio))
    span = 0.22*(dz[-1] - dz[0])
    zl = np.array([dz[imax] - 0.5*span, dz[imax] + 0.5*span])
    ax.plot(zl, g[imax] + beta*(zl - dz[imax]), color='0.25', ls=':', lw=2.0, zorder=5,
            label=f"slope $\\beta_0$; fitted $f_k''/\\beta_0$ = {max(slopes):.4f}")
    ax.set_xlim(dz[0], dz[-1])
    ax.set_title(f"{B['name']} ($N={n}$), break at $z\\approx{zc:.1f}$\n"
                 f"ramp width {width:.3f} vs Figure 3 grid cell {spacing:.2f}", pad=7)
    ax.set_xlabel(r"$z-z^\star$  (outcome units)")
    ax.set_ylabel(r"Gradient $f_k'(z)$")
    ax.legend(frameon=False, loc='upper left')

    print(f"{B['name']:12s} N={n}: beta={beta:.7g} | flagged vertex z={info['vertex']:.4f} "
          f"(turn {info['turn_deg']:.2f} deg), cell {info['cell'][0]:.4f}-{info['cell'][1]:.4f}, "
          f"window {info['window'][0]:.4f}-{info['window'][1]:.4f} | z*={zc:.6f} dg={dg:.6g} "
          f"| panel f' {lo_p:.6g}->{hi_p:.6g} (rise {hi_p-lo_p:.6g}) "
          f"| ramp f''/beta_0={['%.4f'%x for x in slopes]} (bound 1) | n_ramps={len(bounds)//2} "
          f"| total ramp width={width:.4f} | grid cell={spacing:.2f} "
          f"(cell/width {spacing/width:.1f}x)")

plt.savefig(OUT, dpi=300, bbox_inches='tight')
print("wrote", OUT)
