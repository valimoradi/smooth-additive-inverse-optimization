"""
_dedup_anchors.py
=================
Remove tangent lines that are indistinguishable from their neighbours at the
solver's own resolution.

Motivation
----------
The recovered penalty is represented as an upper envelope of tangents
    f(z) = max_j { lambda_j z + b_j },  b_j = delta_j - lambda_j z_j.
The prediction SOCP allocates three dense (n_v, M) variable blocks and n_v * M
rotated cones per OAR, so memory is LINEAR in the anchor count M.

Adjacent tangents are separated by a protective margin (1/2 beta) (dlam)^2.
On this data that margin is ~5e-9 while MOSEK's own tolerance on delta is
~1e-5 * max|delta| ~ 1e-1 -- seven orders of magnitude larger. Lines separated
by less than the solver can resolve carry no information; keeping them only
multiplies the cone count.

What this does
--------------
Iteratively removes an envelope line when doing so lowers f(z) by at most `eps`
anywhere. For a max-of-affine function the loss from dropping an interior line j
is exactly the vertical distance from j down to the crossing of its two
neighbours, evaluated at that crossing -- the height of the little triangle j
carves above them. Lines are removed cheapest-first and the cumulative bound is
tracked, so the returned envelope satisfies

    0 <= f_original(z) - f_dedup(z) <= total_bound   for all z

This is an exact bound, not a heuristic: it is a controlled simplification whose
worst-case error is reported, unlike thin_envelope's target-count Douglas-Peucker.
Endpoints (smallest and largest slope) are never removed, so behaviour at z=0 and
in the high-dose tail is preserved exactly.
"""
import numpy as np


def _triangle_heights(a, b):
    """Height by which each interior line rises above its neighbours' crossing.

    a: slopes (sorted ascending), b: intercepts.  Returns array of len(a) with
    np.inf at the two endpoints (never removable).
    """
    n = len(a)
    h = np.full(n, np.inf)
    if n < 3:
        return h
    al, bl = a[:-2], b[:-2]      # left neighbour
    am, bm = a[1:-1], b[1:-1]    # candidate
    ar, br = a[2:], b[2:]        # right neighbour
    denom = ar - al
    ok = np.abs(denom) > 0
    x = np.where(ok, (bl - br) / np.where(ok, denom, 1.0), np.nan)
    y_neigh = al * x + bl        # value of the neighbours' crossing
    y_mid = am * x + bm          # value of the candidate there
    hh = y_mid - y_neigh         # >= 0 when the candidate is above
    hh = np.where(np.isfinite(hh), hh, np.inf)
    h[1:-1] = np.maximum(hh, 0.0)
    return h


def dedup_envelope(z, delta, lam, eps):
    """Drop lines whose removal lowers f by <= eps in total.

    Returns (z, delta, lam, info) with info = dict(n_in, n_out, bound).
    """
    z = np.asarray(z, float)
    delta = np.asarray(delta, float)
    lam = np.asarray(lam, float)
    n_in = len(z)
    if n_in <= 2 or eps <= 0:
        return z, delta, lam, dict(n_in=n_in, n_out=n_in, bound=0.0)

    order = np.argsort(lam, kind='stable')
    z, delta, lam = z[order], delta[order], lam[order]
    a = lam.copy()
    b = delta - lam * z
    alive = np.ones(len(a), bool)
    budget = float(eps)
    used = 0.0

    while True:
        idx = np.flatnonzero(alive)
        if len(idx) <= 2:
            break
        h = _triangle_heights(a[idx], b[idx])
        j = int(np.argmin(h))
        if not np.isfinite(h[j]) or h[j] > budget - used:
            break
        used += float(h[j])
        alive[idx[j]] = False

    keep = np.flatnonzero(alive)
    return (z[keep], delta[keep], lam[keep],
            dict(n_in=n_in, n_out=len(keep), bound=used))


def dedup_recovered(recovered, eps_rel, verbose=True):
    """Apply dedup_envelope to every component of a recovered-functions dict.

    eps_rel is relative to max|delta| of that component, so it scales with the
    component's own magnitude. Pass the solver tolerance (e.g. 1e-5) to remove
    only what the solver cannot resolve.
    """
    out = {}
    tot_in = tot_out = 0
    for k, rf in recovered.items():
        scale = max(float(np.max(np.abs(rf['delta']))), 1e-30)
        eps = eps_rel * scale
        zz, dd, ll, info = dedup_envelope(rf['z_hat'], rf['delta'], rf['lam'], eps)
        out[k] = {'z_hat': zz, 'delta': dd, 'lam': ll}
        tot_in += info['n_in']
        tot_out += info['n_out']
        if verbose:
            print("    dedup func %d: %d -> %d lines (eps=%.4g, max f-drop=%.4g)"
                  % (k, info['n_in'], info['n_out'], eps, info['bound']), flush=True)
    if verbose:
        print("    dedup TOTAL: %d -> %d anchors (%.2fx reduction)"
              % (tot_in, tot_out, tot_in / max(tot_out, 1)), flush=True)
    return out
