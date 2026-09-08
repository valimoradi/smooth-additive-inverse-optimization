#!/usr/bin/env python3
"""
Consumer-choice inverse optimization: smooth and additive nonparametric models.
================================================================================

Reproduces the consumer-behavior study (Section 6.1) of "Learning Convex
Objectives from Decisions: Smooth and Additive Nonparametric Inverse
Optimization". A consumer chooses bundles ``x`` by solving the forward problem

    min_x   p^T x - U(x),    x >= 0,

for prices ``p`` and a concave utility ``U``. We observe optimal bundles and
impute a convex cost ``f = -U`` without a parametric form, then predict
out-of-sample bundles and report the relative L2 error of the predicted bundle
versus the training-set size.

Four models (a 2x2 ablation)
----------------------------
Each model imposes a different subset of structural assumptions on ``f``:

    +------------------+------------------+------------------+
    |                  | no smoothness    | beta-smooth      |
    +------------------+------------------+------------------+
    | no additivity    | Convex only      | Smooth           |
    |                  | (Li 2019 base)   |                  |
    +------------------+------------------+------------------+
    | additive         | Additive         | Additive+Smooth  |
    +------------------+------------------+------------------+

PREDICTION (the subtle part)
----------------------------
For every model we predict by *resolving the forward problem with the learned
objective*. The recovered objective is the Fenchel-conjugate reconstruction
from the recovered values ``delta`` and gradients ``lambda``:

    f(z) = max_{lambda in C} { lambda^T z
                               - max_j [ (1/2beta) ||lambda - lambda_j||^2
                                         + lambda^T z_j - delta_j ] }.

When the model assumes NO smoothness (beta -> infinity), the (1/2beta) term
vanishes and the recovered gradients ``lambda_j`` drop out: the reconstruction
depends only on the observed points ``z_j`` and recovered values ``delta_j``.
The prediction is therefore regime-dependent:

  * non-smooth models (Convex only, Additive)  -> value-based forward resolve
    (``predict_convex_combination``; uses ``delta`` only, ``lambda`` discarded).
  * smooth models (Smooth, Additive+Smooth)     -> perspective forward resolve
    (``predict_perspective``; uses ``delta``, ``lambda`` and ``beta``).

Additivity is imposed in the inverse SOLVE (per-coordinate concavity), not by
changing the prediction machinery. This keeps the ablation clean: additivity
acts through the recovered values, smoothness through the reconstruction.

UNPERTURBED vs PERTURBED
------------------------
Both panels are generated from a *single* dataset. The unperturbed panel feeds
the optimal training bundles to the inverse solvers; the perturbed panel feeds
the same bundles after a +-5% multiplicative perturbation of the training
inputs only. The held-out test set is the clean optimum in both cases, so the
two panels isolate exactly how prediction error grows when the observed
decisions are noisy.

REPLICATIONS
------------
Each curve point is a nested prefix of a shuffled train pool. A single ordering
makes the whole learning curve inherit that one draw's luck: the smallest
feasible smoothness beta* is a sample statistic, and a prefix that happens to
stay feasible at a smaller beta over-smooths in its own favour (the source of
the non-monotone kink in the original single-ordering figure). The reported
curves therefore average ``Config.n_reps`` independent orderings of the pool
(anchors fixed; rep 0 is the original ordering), with epsilon* and beta*
re-estimated from scratch in every (rep, N) cell, and the figures show the
mean with a +-1 SE band. The estimation method itself is untouched.

Configuration matches the generating notebooks (``additive smooth non para
inverse - Log- optimal values used`` and its ``- perturbed`` sibling): 5
products, gamma = 40, prices ~U[8,20], a ~U[1,1000], b ~U[2,60], 600 random
scenarios (200 test, the rest train pool) plus two synthetic anchors. The RNG
draw order (prices, then a, then b) is preserved so the dataset reproduces the
published figures bit-for-bit.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import cvxpy as cp

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt


# ======================================================================
# Configuration
# ======================================================================
@dataclass(frozen=True)
class Config:
    """All experiment parameters in one place."""

    # Problem size and ground-truth utility parameters.
    # Canonical config reproducing the §6.1 LOG figures (see the generating
    # notebooks): 5 products, gamma=40, prices U[8,20], 600 random scenarios,
    # 200 test, the rest in the train pool. Draw order is prices -> a -> b.
    num_products: int = 5
    gamma: float = 40.0                      # dimensional conditioning of log utility
    seed: int = 42

    price_lo: float = 8.0                    # prices ~ U[price_lo, price_hi]
    price_hi: float = 20.0
    a_lo: float = 1.0                        # a ~ U[a_lo, a_hi]
    a_hi: float = 1000.0
    b_lo: float = 2.0                        # b ~ U[b_lo, b_hi]
    b_hi: float = 60.0
    interaction_multiplier: float = 300.0    # non-smooth misspecification only

    # Data split: test = first TEST_SIZE random rows, train pool = the rest
    # (plus the two anchors), matching the notebook.
    n_random_total: int = 600
    test_size: int = 200                     # held-out rows (anchors never here)
    perturb_lo: float = -0.05                # perturbation ~ U[1+lo, 1+hi]
    perturb_hi: float = 0.05

    # Learning curve x-axis. The paper trains up to N=200; the figures report
    # exactly this grid.
    training_sizes: tuple = tuple(range(20, 220, 20))   # 20, 40, ..., 200

    # Replications of the training-pool ORDER. Each curve point is one nested
    # prefix of a shuffled train pool; a single ordering makes the curve inherit
    # that draw's luck (e.g. a sample that stays feasible at a smaller beta* and
    # temporarily over-smooths in its favour). We therefore repeat the whole
    # nested sweep for n_reps independent orderings -- beta*, epsilon* and all
    # stages re-estimated from scratch per (rep, N) -- and report mean +- SE.
    # Rep 0 is the identity ordering (the original seed-42 pipeline).
    n_reps: int = 20
    rep_seed_base: int = 1000                # permutation seed for rep r is base + r

    # Inverse-model bounds / anchors.
    delta_anchor_stage1: float = 1000.0      # delta[1] in the epsilon-min stage
    delta_anchor_stage2: float = 2000.0      # delta[1] in the sum-delta-max stage
    beta_init: float = 1000.0                # large initial beta for the smooth bisection

    # Solver tolerances. 1e-6 matches the generating notebooks; tighter (1e-8)
    # makes MOSEK declare the epsilon->0 problem non-optimal on *exact*
    # (unperturbed) data, which sits precisely on the optimality boundary.
    mosek_params: dict = field(default_factory=lambda: {
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1.0e-6,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1.0e-6,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1.0e-6,
        "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_FREE",
    })


CFG = Config()

_OPTIMAL = ("optimal", "optimal_inaccurate")


# ======================================================================
# Solver wrapper (MOSEK first, ECOS fallback)
# ======================================================================
def _solve(prob: cp.Problem, use_params: bool = True) -> bool:
    """Solve in place; return True on (near-)optimal status.

    Layered fallback: MOSEK at high precision, then MOSEK at default tolerances,
    then ECOS. Exact (unperturbed) data at large N can make MOSEK return a
    non-optimal *status* (not an exception) under very tight tolerances, so we
    must retry on bad status, not only on exceptions.
    """
    attempts = []
    if use_params:
        attempts.append(dict(solver=cp.MOSEK, verbose=False, mosek_params=CFG.mosek_params))
    attempts.append(dict(solver=cp.MOSEK, verbose=False))   # MOSEK default tolerances
    attempts.append(dict(solver=cp.ECOS, verbose=False))
    for kw in attempts:
        try:
            prob.solve(**kw)
        except Exception:
            continue
        if prob.status in _OPTIMAL:
            return True
    return prob.status in _OPTIMAL


# ======================================================================
# Forward model (ground-truth data generation)
# ======================================================================
def _forward_smooth(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """argmin_x  p^T x - gamma * sum_i log(a_i x_i + b_i),  x >= 0."""
    x = cp.Variable(CFG.num_products, nonneg=True)
    util = cp.sum(cp.log(cp.multiply(a, x) + b))
    prob = cp.Problem(cp.Minimize(p @ x - CFG.gamma * util))
    _solve(prob)
    return x.value


def _forward_nonsmooth(p, a, b, c, d, W) -> np.ndarray:
    """Misspecification: U(x) = min(U1, U2) with a W interaction term."""
    x = cp.Variable(CFG.num_products, nonneg=True)
    mult = CFG.interaction_multiplier
    util1 = cp.sum([cp.log(a[i] * x[i] + b[i] + mult * (W[i, :] @ x))
                    for i in range(CFG.num_products)])
    util2 = cp.sum([cp.log(c[i] * x[i] + d[i] + mult * (W[i, :] @ x))
                    for i in range(CFG.num_products)])
    u = cp.Variable()
    prob = cp.Problem(cp.Minimize(p @ x - CFG.gamma * u),
                      [u <= util1, u <= util2])
    _solve(prob)
    return x.value


def _forward_kicks3(p, a, b, c, d, e, f, W) -> np.ndarray:
    """Fig-2 misspecification: U(x) = min(U1, U2, U3), shared interaction W.

    Each U_k is a sum of logs with its own slope/intercept set; the three sets
    are permutations of one another with hand-tweaked slopes, so the minimum
    introduces kinks without changing the overall scale. Faithful to the
    generating notebook (``... non smooth function 3 kicks-m-v3``)."""
    n = CFG.num_products
    mult = CFG.interaction_multiplier
    x = cp.Variable(n, nonneg=True)
    utils = []
    for s0, s1 in ((a, b), (c, d), (e, f)):
        utils.append(cp.sum([cp.log(s0[i] * x[i] + s1[i] + mult * (W[i, :] @ x))
                             for i in range(n)]))
    u = cp.Variable()
    prob = cp.Problem(cp.Minimize(p @ x - CFG.gamma * u),
                      [u <= t for t in utils])
    _solve(prob)
    return x.value


def build_dataset_kicks3():
    """Dataset for the misspecification figure (min-of-three-logs ground truth).

    Verbatim port of the generating notebook's data cell (seed 50): draw order
    is prices -> a -> b -> permutation_for_c -> permutation_for_e (redrawn while
    equal) -> the seven W entries -> the train/test permutation. The asymmetric
    last W assignment (W[4,2] left at 0 while W[2,4] is drawn) reproduces the
    notebook exactly, RNG stream included. Anchors are the synthetic p=1000 ->
    x=0 and p=0 -> x=1000 rows, fixed at pool rows 0 and 1.
    """
    n = CFG.num_products
    np.random.seed(50)

    prices_rand = np.random.uniform(CFG.price_lo, CFG.price_hi,
                                    size=(CFG.n_random_total, n))
    a = np.random.uniform(700.0, 1000.0, size=n)
    b = np.random.uniform(CFG.b_lo, CFG.b_hi, size=n)

    permutation_for_c = np.random.permutation(n)
    permutation_for_e = np.random.permutation(n)
    while np.array_equal(permutation_for_c, permutation_for_e):
        permutation_for_e = np.random.permutation(n)
    c = a[permutation_for_c].copy(); d = b[permutation_for_c].copy()
    e = a[permutation_for_e].copy(); f = b[permutation_for_e].copy()
    c[1] *= 1.5; c[3] /= 1.5
    e[1] /= 1.5; e[3] *= 1.5

    W = np.zeros((n, n))
    W[0, 1] = np.random.uniform(0.1, 0.3);  W[1, 0] = W[0, 1]
    W[0, 2] = -np.random.uniform(0.1, 0.5); W[2, 0] = W[0, 2]
    W[0, 3] = np.random.uniform(0.1, 0.5);  W[3, 0] = W[0, 3]
    W[0, 4] = -np.random.uniform(0.1, 0.3); W[4, 0] = W[0, 4]
    W[1, 2] = np.random.uniform(0.1, 0.5);  W[2, 1] = W[1, 2]
    W[1, 3] = np.random.uniform(0.1, 0.5);  W[3, 1] = W[1, 3]
    W[2, 3] = -np.random.uniform(0.1, 0.3); W[3, 2] = W[2, 3]
    W[2, 4] = np.random.uniform(0.1, 0.5)
    W[4, 2] = W[4, 2]     # notebook artifact: W stays asymmetric here (kept verbatim)

    X = np.zeros_like(prices_rand)
    for i in range(CFG.n_random_total):
        sol = _forward_kicks3(prices_rand[i], a, b, c, d, e, f, W)
        X[i] = np.zeros(n) if sol is None else sol

    p_lo, x_lo = np.full(n, 1000.0), np.full(n, 0.0)
    p_hi, x_hi = np.full(n, 0.0), np.full(n, 1000.0)
    prices = np.vstack([p_lo, p_hi, prices_rand])
    X_true = np.vstack([x_lo, x_hi, X])

    all_idx = np.arange(2, CFG.n_random_total + 2)
    perm = np.random.permutation(all_idx)
    te = perm[:CFG.test_size]
    tr = np.concatenate(([0, 1], perm[CFG.test_size:]))
    return X_true[tr], prices[tr], X_true[te], prices[te]


def build_dataset(utility: str):
    """Build (X_pool, P_pool, X_test, P_test) for the requested ground truth.

    The train pool is ``[lower_anchor, upper_anchor, *random]`` (rows 0 and 1
    are the synthetic anchors p=1000 -> x=0 and p=0 -> x=1000). Anchors never
    appear in the test set.

    Note: we use NumPy's *legacy* global RNG (``np.random.seed`` /
    ``np.random.uniform``) in this exact draw order -- prices first, then a, then
    b -- to reproduce the published pipeline bit-for-bit; ``default_rng`` or a
    different draw order would yield a different stream.
    """
    if utility == "kicks3":
        return build_dataset_kicks3()

    n = CFG.num_products
    np.random.seed(CFG.seed)

    # Draw order matches the notebooks: prices first, then the utility params.
    prices = np.random.uniform(CFG.price_lo, CFG.price_hi, size=(CFG.n_random_total, n))
    a = np.random.uniform(CFG.a_lo, CFG.a_hi, size=n)
    b = np.random.uniform(CFG.b_lo, CFG.b_hi, size=n)

    c = d = W = None
    if utility == "nonsmooth":
        c = np.random.uniform(CFG.a_lo, CFG.a_hi, size=n)
        d = np.random.uniform(CFG.b_lo, CFG.b_hi, size=n)
        W = np.random.uniform(0.0, 1.0, size=(n, n))
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)

    X = np.zeros_like(prices)
    for i in range(CFG.n_random_total):
        sol = (_forward_smooth(prices[i], a, b) if utility == "smooth"
               else _forward_nonsmooth(prices[i], a, b, c, d, W))
        X[i] = np.zeros(n) if sol is None else sol

    # Synthetic anchors.
    p_lo, x_lo = np.full(n, 1000.0), np.full(n, 0.0)
    p_hi, x_hi = np.full(n, 0.0), np.full(n, 1000.0)

    perm = np.random.permutation(CFG.n_random_total)
    te = perm[:CFG.test_size]                 # test = first TEST_SIZE random rows
    tr = perm[CFG.test_size:]                 # train pool = all remaining random rows

    X_pool = np.vstack([x_lo, x_hi, X[tr]])
    P_pool = np.vstack([p_lo, p_hi, prices[tr]])
    return X_pool, P_pool, X[te], prices[te]


def rep_permutation(rep: int, n_pool: int) -> np.ndarray:
    """Row order of the train pool for replication ``rep``.

    Rep 0 is the identity (the published ordering). Reps >= 1 shuffle the random
    rows with an independent RNG while keeping the two synthetic anchors fixed
    at rows 0 and 1 (the inverse models pin delta[0] and delta[1], so every
    training prefix must contain both anchors). Nested prefixes of the permuted
    pool give the nested training sets of that replication.
    """
    if rep == 0:
        return np.arange(n_pool)
    rng = np.random.default_rng(CFG.rep_seed_base + rep)
    return np.concatenate([[0, 1], 2 + rng.permutation(n_pool - 2)])


def perturb_train_pool(X_pool: np.ndarray) -> np.ndarray:
    """Multiply every training bundle by (1 + U[lo, hi]).

    This is the only difference between the unperturbed and perturbed panels:
    the inverse solvers see optimal bundles in one case and +-5%-noisy bundles
    in the other, while the test set stays the clean optimum. The whole pool is
    perturbed (the zero anchor is a no-op; the x=1000 anchor moves by <=5%),
    matching the notebook's ``X_pool = X_pool_original * (1 + U[-.05, .05])``.

    Drawn from the legacy global RNG immediately after ``build_dataset`` so the
    perturbation stream matches the published pipeline.
    """
    factor = np.random.uniform(1.0 + CFG.perturb_lo, 1.0 + CFG.perturb_hi, size=X_pool.shape)
    return X_pool * factor


# ======================================================================
# Recovered objective (uniform container)
# ======================================================================
@dataclass
class Recovered:
    """Parameters of the imputed objective returned by every solver.

    ``delta`` is the 1-D vector of recovered function values (the global value
    for the additive models). ``lambdas`` are the recovered subgradients;
    ``beta`` is the smoothness. ``beta is None`` flags a non-smooth model, in
    which case ``lambdas`` is not used at prediction time (see module docstring).
    """

    delta: np.ndarray
    lambdas: Optional[np.ndarray] = None
    beta: Optional[float] = None
    epsilon: float = 0.0


# ======================================================================
# Inverse solvers (one per model). Two stages: minimise epsilon, then
# maximise sum(delta) at the fixed epsilon*. Smooth models add a beta search.
# ======================================================================
# On exact (unperturbed) data the epsilon-min problem occasionally fails (the
# optimum sits on the boundary at epsilon ~ 0). When it does, we fall back to a
# short sweep of small fixed epsilons -- the true value is ~0 there -- and take
# the first one that lets Stage 2 solve. The normal path (min-eps succeeds) is
# tried first, so the recovered objective is unchanged when there is no failure.
_EPS_FALLBACK = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def solve_convex_only(Z: np.ndarray, P: np.ndarray) -> Recovered:
    """Nonparametric convex model (Li 2019 baseline): no additivity, no smoothness."""
    m, n = Z.shape

    def build(delta, lamb, eps_expr, anchor):
        cons = [delta >= 0, delta <= anchor, delta[0] == 0, delta[1] == anchor,
                P >= lamb,
                cp.sum(cp.multiply(P - lamb, Z), axis=1) <= eps_expr]
        for j in range(m):
            cons.append(delta[j] + (Z - Z[j]) @ lamb[j] >= delta)
        return cons

    # Stage 1: minimise epsilon.
    d1 = cp.Variable(m)
    l1 = cp.Variable((m, n), nonneg=True)
    eps = cp.Variable(nonneg=True)
    eps_star = (float(eps.value)
                if _solve(cp.Problem(cp.Minimize(eps),
                                     build(d1, l1, eps, CFG.delta_anchor_stage1)))
                else None)

    # Stage 2: maximise sum(delta), trying epsilon* first then the fallback sweep.
    for eps_try in ([eps_star] if eps_star is not None else []) + list(_EPS_FALLBACK):
        d2 = cp.Variable(m)
        l2 = cp.Variable((m, n), nonneg=True)
        if _solve(cp.Problem(cp.Maximize(cp.sum(d2)),
                             build(d2, l2, eps_try, CFG.delta_anchor_stage2))) \
                and d2.value is not None:
            return Recovered(d2.value, l2.value, None, eps_try)
    if eps_star is not None and d1.value is not None:
        return Recovered(d1.value, l1.value, None, eps_star)
    raise RuntimeError("convex-only solve failed (Stage-1 and epsilon sweep)")


def solve_additive(Z: np.ndarray, P: np.ndarray) -> Recovered:
    """Additive convex model: per-coordinate concavity, no smoothness."""
    m, n = Z.shape
    order = [np.argsort(Z[:, k]) for k in range(n)]

    def build(delta, delta_g, lamb, eps_expr, anchor):
        cons = [delta >= 0, delta <= anchor, delta_g >= 0, delta_g <= anchor,
                delta_g == cp.sum(delta, axis=1),
                delta_g[0] == 0, delta_g[1] == anchor,
                P - lamb >= 0,
                cp.sum(cp.multiply(P - lamb, Z), axis=1) <= eps_expr]
        for k in range(n):
            cur, nxt = order[k][:-1], order[k][1:]
            dz = Z[cur, k] - Z[nxt, k]
            cons.append(delta[cur, k] + cp.multiply(lamb[cur, k], -dz) >= delta[nxt, k])
            cons.append(delta[nxt, k] + cp.multiply(lamb[nxt, k], dz) >= delta[cur, k])
        return cons

    # Stage 1: minimise epsilon (ill-conditioned on exact data -> may fail).
    d1 = cp.Variable((m, n))
    dg1 = cp.Variable(m)
    l1 = cp.Variable((m, n), nonneg=True)
    eps = cp.Variable(nonneg=True)
    eps_star = (float(eps.value)
                if _solve(cp.Problem(cp.Minimize(eps),
                                     build(d1, dg1, l1, eps, CFG.delta_anchor_stage1)))
                else None)

    # Stage 2: maximise sum(delta), epsilon* first then the fallback sweep.
    for eps_try in ([eps_star] if eps_star is not None else []) + list(_EPS_FALLBACK):
        d2 = cp.Variable((m, n))
        dg2 = cp.Variable(m)
        l2 = cp.Variable((m, n), nonneg=True)
        if _solve(cp.Problem(cp.Maximize(cp.sum(dg2)),
                             build(d2, dg2, l2, eps_try + 1e-6, CFG.delta_anchor_stage2))) \
                and dg2.value is not None:
            return Recovered(dg2.value, l2.value, None, eps_try)
    if eps_star is not None and dg1.value is not None:
        return Recovered(dg1.value, l1.value, None, eps_star)
    raise RuntimeError("additive solve failed (Stage-1 and epsilon sweep)")


def _smooth_constraints(Z, P, beta_inv, eps_expr, delta, lamb, lam_opt):
    """General (non-additive) smooth inverse constraints (pairwise)."""
    m = Z.shape[0]
    cons = [delta >= 0, delta <= CFG.delta_anchor_stage2,
            delta[0] == 0, delta[1] == CFG.delta_anchor_stage2,
            lamb >= 0, lam_opt >= 0, P - lam_opt >= 0, P - lamb >= 0,
            cp.sum(cp.multiply(P - lamb, Z), axis=1) <= eps_expr]
    for j in range(m):
        grad = (Z - Z[j]) @ lamb[j]
        penalty = beta_inv * cp.sum(cp.square(lamb - lamb[j]), axis=1)
        cons.append((delta[j] - delta) + grad >= penalty)
    return cons


def _additive_smooth_constraints(Z, P, beta_inv, eps_expr, delta, delta_g, lamb, lam_opt):
    """Additive smooth inverse constraints (per-coordinate, sorted neighbours)."""
    m, n = Z.shape
    cons = [delta >= 0, delta_g >= 0, delta_g <= CFG.delta_anchor_stage2,
            delta_g[0] == 0, delta_g[1] == CFG.delta_anchor_stage2,
            delta_g == cp.sum(delta, axis=1),
            lamb >= 0, lam_opt >= 0, P - lam_opt >= 0, P - lamb >= 0,
            cp.sum(cp.multiply(P - lamb, Z), axis=1) <= eps_expr]
    for k in range(n):
        idx = np.argsort(Z[:, k])
        zk = Z[idx, k]
        dz = zk[:-1] - zk[1:]
        dk = delta[idx, k]
        lk = lamb[idx, k]
        rhs = beta_inv * cp.square(lk[:-1] - lk[1:])
        cons.append(-dk[:-1] + dk[1:] + cp.multiply(lk[1:], dz) >= rhs)
        cons.append(-dk[1:] + dk[:-1] - cp.multiply(lk[:-1], dz) >= rhs)
    return cons


def _feasible_at_beta(Z, P, eps_val, beta_val, additive: bool) -> bool:
    if beta_val <= 1e-4:
        return False
    m, n = Z.shape
    beta_inv = 1.0 / (2.0 * beta_val)
    delta = cp.Variable((m, n)) if additive else cp.Variable(m)
    lamb = cp.Variable((m, n))
    lam_opt = cp.Variable((m, n))
    if additive:
        delta_g = cp.Variable(m)
        cons = _additive_smooth_constraints(Z, P, beta_inv, eps_val, delta, delta_g, lamb, lam_opt)
    else:
        cons = _smooth_constraints(Z, P, beta_inv, eps_val, delta, lamb, lam_opt)
    return _solve(cp.Problem(cp.Minimize(0), cons))


def _min_epsilon(Z, P, additive: bool) -> float:
    m, n = Z.shape
    beta_inv = 1.0 / (2.0 * CFG.beta_init)
    delta = cp.Variable((m, n)) if additive else cp.Variable(m)
    lamb = cp.Variable((m, n))
    lam_opt = cp.Variable((m, n))
    eps = cp.Variable(nonneg=True)
    if additive:
        delta_g = cp.Variable(m)
        cons = _additive_smooth_constraints(Z, P, beta_inv, eps, delta, delta_g, lamb, lam_opt)
    else:
        cons = _smooth_constraints(Z, P, beta_inv, eps, delta, lamb, lam_opt)
    if _solve(cp.Problem(cp.Minimize(eps), cons)):
        return max(float(eps.value), 1e-6) * 1.05
    return 1.0


def _beta_bisection(Z, P, eps_val, additive: bool, gap_tol: float) -> float:
    high = CFG.beta_init
    if not _feasible_at_beta(Z, P, eps_val, high, additive):
        for _ in range(5):
            high *= 2
            if _feasible_at_beta(Z, P, eps_val, high, additive):
                break
        else:
            return CFG.beta_init
    low = 0.0
    for _ in range(15):
        if high - low < gap_tol:
            break
        mid = 0.5 * (low + high)
        if _feasible_at_beta(Z, P, eps_val, mid, additive):
            high = mid
        else:
            low = mid
    return high


# Stage-3 beta escalation: the bisected beta* sits on the feasibility boundary,
# where the final maximize-sum-delta solve is occasionally ill-conditioned. We
# nudge beta upward by small factors until the solve is clean. This does not
# reintroduce the approximation: beta stays at the smallest value that solves
# reliably, not a huge constant.
_BETA_ESCALATION = (1.0, 1.2, 1.44, 1.728, 2.0)


def solve_smooth(Z: np.ndarray, P: np.ndarray) -> Recovered:
    """Smooth convex model: beta-smooth, no additivity."""
    m, n = Z.shape
    eps_val = _min_epsilon(Z, P, additive=False)
    beta0 = _beta_bisection(Z, P, eps_val, additive=False, gap_tol=2.0)
    for factor in _BETA_ESCALATION:
        beta = beta0 * factor
        delta = cp.Variable(m)
        lamb = cp.Variable((m, n))
        lam_opt = cp.Variable((m, n))
        cons = _smooth_constraints(Z, P, 1.0 / (2.0 * beta), eps_val, delta, lamb, lam_opt)
        if _solve(cp.Problem(cp.Maximize(cp.sum(delta)), cons)) and delta.value is not None:
            return Recovered(delta.value, lamb.value, beta, eps_val)
    raise RuntimeError("smooth Stage-3 solve failed after beta escalation")


def solve_additive_smooth(Z: np.ndarray, P: np.ndarray) -> Recovered:
    """Additive smooth model: beta-smooth and additive."""
    m, n = Z.shape
    eps_val = _min_epsilon(Z, P, additive=True)
    beta0 = _beta_bisection(Z, P, eps_val, additive=True, gap_tol=1.0)
    for factor in _BETA_ESCALATION:
        beta = beta0 * factor
        delta = cp.Variable((m, n))
        delta_g = cp.Variable(m)
        lamb = cp.Variable((m, n))
        lam_opt = cp.Variable((m, n))
        cons = _additive_smooth_constraints(Z, P, 1.0 / (2.0 * beta), eps_val,
                                            delta, delta_g, lamb, lam_opt)
        if _solve(cp.Problem(cp.Maximize(cp.sum(delta_g)), cons)) and delta_g.value is not None:
            return Recovered(delta_g.value, lamb.value, beta, eps_val)
    raise RuntimeError("additive-smooth Stage-3 solve failed after beta escalation")


# ======================================================================
# Prediction: resolve the forward problem with the learned objective.
# ======================================================================
def predict_convex_combination(Z: np.ndarray, delta: np.ndarray,
                               P_test: np.ndarray) -> np.ndarray:
    """Value-based forward resolve for the NON-smooth models.

    Solves, per test price p,  min_z p^T z - delta^T alpha  over convex
    combinations  z = sum_j alpha_j z_j  of observed bundles. Uses the recovered
    values ``delta`` only; the gradients ``lambda`` are absent from the
    non-smooth reconstruction (beta -> infinity limit) and are not used here.
    """
    m, n = Z.shape
    z = cp.Variable(n, nonneg=True)
    alpha = cp.Variable(m, nonneg=True)
    p = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(p @ z - delta @ alpha),
                      [alpha @ Z == z, cp.sum(alpha) == 1])
    out = np.full((P_test.shape[0], n), np.nan)
    for i in range(P_test.shape[0]):
        p.value = P_test[i]
        if _solve(prob, use_params=True):
            out[i] = z.value
    return out


def predict_perspective(Z: np.ndarray, delta: np.ndarray, lambdas: np.ndarray,
                        beta: float, P_test: np.ndarray) -> np.ndarray:
    """Perspective forward resolve for the SMOOTH models.

    Solves the conjugate (perspective) forward problem implied by the recovered
    (delta, lambda, beta); uses all three. Single convex-combination weight
    ``mu`` over observations.
    """
    m, n = Z.shape
    beta_inv = 1.0 / beta
    const = -0.5 * beta_inv * np.sum(lambdas ** 2, axis=1) - delta
    x = cp.Variable(n, nonneg=True)
    mu = cp.Variable(m, nonneg=True)
    p = cp.Parameter(n)
    v = x - (mu @ Z) - (mu @ lambdas) * beta_inv
    obj = cp.Minimize(p @ x + (beta / 2.0) * cp.sum_squares(cp.pos(-v)) + mu @ const)
    prob = cp.Problem(obj, [cp.sum(mu) == 1])
    out = np.full((P_test.shape[0], n), np.nan)
    for i in range(P_test.shape[0]):
        p.value = P_test[i]
        if _solve(prob):
            out[i] = x.value
    return out


# ======================================================================
# Model registry (ties solver + predictor together; encodes the ablation)
# ======================================================================
@dataclass(frozen=True)
class Model:
    name: str
    additive: bool
    smooth: bool
    solve: callable
    marker: str
    color: str

    def predict(self, rec: Recovered, Z: np.ndarray, P_test: np.ndarray) -> np.ndarray:
        # Regime-dependent prediction: smoothness => perspective form (uses
        # lambda, beta); otherwise value-based forward resolve (delta only).
        if rec.beta is None:
            return predict_convex_combination(Z, rec.delta, P_test)
        return predict_perspective(Z, rec.delta, rec.lambdas, rec.beta, P_test)


MODELS = (
    Model("Convex only",     additive=False, smooth=False, solve=solve_convex_only,    marker="o", color="C0"),
    Model("Additive",        additive=True,  smooth=False, solve=solve_additive,       marker="s", color="C2"),
    Model("Smooth",          additive=False, smooth=True,  solve=solve_smooth,         marker="D", color="C1"),
    Model("Additive+Smooth", additive=True,  smooth=True,  solve=solve_additive_smooth, marker="^", color="C3"),
)


# ======================================================================
# Evaluation
# ======================================================================
def percentage_mae(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """100 * mean(|x_pred - x_true| / (|x_true| + 1e-6)) over flattened bundles."""
    yt = np.asarray(x_true, float).ravel()
    yp = np.asarray(x_pred, float).ravel()
    return 100.0 * np.nanmean(np.abs(yp - yt) / (np.abs(yt) + 1e-6))


def rel_l2(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Relative L2 error of the predicted bundle, averaged over test points:
    mean_t ||x_pred_t - x_true_t|| / ||x_true_t||. This is the unified
    prediction-error metric used across the paper (decision-vector relative L2);
    robust to near-zero coordinates, unlike a per-coordinate percentage error."""
    xt = np.asarray(x_true, float); xp = np.asarray(x_pred, float)
    num = np.linalg.norm(xp - xt, axis=1)
    den = np.linalg.norm(xt, axis=1) + 1e-12
    return float(np.nanmean(num / den))


# ======================================================================
# Experiment driver
# ======================================================================
def run_panel(utility: str, regime: str, X_pool, P_pool, X_test, P_test,
              training_sizes=None, n_reps=None):
    """Sweep training size for one (utility, regime) across n_reps orderings.

    Returns (sizes, errs, betas) where errs[model] and betas[model] are
    (n_reps, n_sizes) arrays; every (rep, size) cell re-runs the full
    estimation (epsilon*, beta* bisection, all stages) from scratch.
    """
    if training_sizes is None:
        training_sizes = CFG.training_sizes
    if n_reps is None:
        n_reps = CFG.n_reps
    pool = X_pool if regime == "not-perturbed" else perturb_train_pool(X_pool)
    sizes = [min(int(s), pool.shape[0]) for s in training_sizes]
    errs = {mdl.name: np.full((n_reps, len(sizes)), np.nan) for mdl in MODELS}
    betas = {mdl.name: np.full((n_reps, len(sizes)), np.nan) for mdl in MODELS}

    for rep in range(n_reps):
        idx = rep_permutation(rep, pool.shape[0])
        for j, size in enumerate(sizes):
            Z_tr, P_tr = pool[idx[:size]], P_pool[idx[:size]]
            for mdl in MODELS:
                t0 = time.time()
                beta_star = None
                try:
                    rec = mdl.solve(Z_tr, P_tr)
                    beta_star = rec.beta
                    err = rel_l2(X_test, mdl.predict(rec, Z_tr, P_test))
                except Exception as exc:  # noqa: BLE001 - record and continue
                    err = np.nan
                    print(f"    [{utility}/{regime}/rep={rep}/N={size}/{mdl.name}] "
                          f"FAILED: {type(exc).__name__}: {exc}", flush=True)
                errs[mdl.name][rep, j] = err
                if beta_star is not None:
                    betas[mdl.name][rep, j] = beta_star
                # beta* is the smallest feasible smoothness for the smooth models
                # (no approximation); "no-beta" for the convex-only / additive models.
                bstr = f"beta*={beta_star:8.1f}" if beta_star is not None else "no-beta       "
                print(f"  [{utility}/{regime}/rep={rep}/N={size:3d}/{mdl.name:16s}] "
                      f"RelL2={err:7.4f}  {bstr}  ({time.time() - t0:.1f}s)", flush=True)
    return sizes, errs, betas


# ======================================================================
# Plotting
# ======================================================================
BAND_SE = 2.0        # band half-width in standard errors (paper: "two standard errors")
BAND_ALPHA = 0.25    # 2026-09-05: was +-1 SE at 0.18, invisible under the line at print size


def plot_panel(sizes, errs, out_path: str, title: str = "") -> None:
    """One panel: per-model mean rel-L2 across replications with a +-BAND_SE band.

    ``errs[model]`` is an (n_reps, n_sizes) array; a 1-D array (single rep,
    the legacy layout) is accepted and plotted without a band. ``title`` is a
    debugging aid only; the published panels are written with title="".
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.asarray(sizes, float)
    for mdl in MODELS:
        ys = np.atleast_2d(np.asarray(errs[mdl.name], float))
        mean = np.nanmean(ys, axis=0)
        mask = ~np.isnan(mean)
        if not mask.any():
            continue
        if ys.shape[0] > 1:
            n_ok = np.sum(~np.isnan(ys), axis=0)
            se = np.nanstd(ys, axis=0) / np.sqrt(np.maximum(n_ok, 1))
            lo, hi = mean - BAND_SE * se, mean + BAND_SE * se
            ax.fill_between(xs[mask], lo[mask], hi[mask],
                            color=mdl.color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(xs[mask], mean[mask], marker=mdl.marker, color=mdl.color,
                linewidth=1.8, label=mdl.name)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(r"Test relative $L_2$ error")
    ax.set_ylim(bottom=0)              # linear y, no log scale
    ax.grid(True, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def plot_legend(out_path: str) -> None:
    fig_tmp, ax_tmp = plt.subplots()
    handles = [ax_tmp.plot([0, 1], [0, 1], marker=m.marker, color=m.color,
                           linewidth=1.8, label=m.name)[0] for m in MODELS]
    plt.close(fig_tmp)
    fig = plt.figure(figsize=(10, 0.5))
    ax = fig.add_subplot(111)
    ax.legend(handles, [m.name for m in MODELS], loc="center",
              ncol=len(MODELS), frameon=True, edgecolor="black", mode="expand")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# ======================================================================
# Main
# ======================================================================
# Panel-title display names (the utility keys are internal; "kicks3" renders as
# the paper's "non-smooth" label).
UTILITY_DISPLAY = {"smooth": "smooth", "kicks3": "non-smooth",
                   "nonsmooth": "non-smooth (2min legacy)"}

PANELS = {
    ("smooth", "not-perturbed"): "smooth-additive-not-perturbed.pdf",
    ("smooth", "perturbed"): "smooth-additive-perturbed.pdf",
    # Fig 2 (misspecification) is the min-of-three-logs ground truth from the
    # 3-kicks-m-v3 notebook; "kicks3" owns the published panel filenames.
    ("kicks3", "not-perturbed"): "non-smooth-additive-not-perturbed.pdf",
    ("kicks3", "perturbed"): "non-smooth-additive-perturbed.pdf",
    # Legacy min-of-two variant (kept runnable, not a paper figure).
    ("nonsmooth", "not-perturbed"): "legacy-nonsmooth-2min-not-perturbed.pdf",
    ("nonsmooth", "perturbed"): "legacy-nonsmooth-2min-perturbed.pdf",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="consumer_figures",
                        help="output directory for the PDF panels")
    parser.add_argument("--utilities", nargs="+", default=["smooth", "kicks3"],
                        choices=["smooth", "kicks3", "nonsmooth"])
    parser.add_argument("--test-subset", type=int, default=None,
                        help="evaluate on the first K test points only (smoke test)")
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="override the training-size grid (default: 20..200 step 20)")
    parser.add_argument("--reps", type=int, default=None,
                        help="replications of the training-pool order (default: CFG.n_reps)")
    args = parser.parse_args()
    training_sizes = args.sizes if args.sizes else CFG.training_sizes
    n_reps = args.reps if args.reps else CFG.n_reps

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Config: {CFG.num_products} products, gamma={CFG.gamma}, "
          f"prices~U[{CFG.price_lo},{CFG.price_hi}], N={CFG.n_random_total} "
          f"({CFG.n_random_total - CFG.test_size} train pool / {CFG.test_size} test), "
          f"sizes={list(CFG.training_sizes)}", flush=True)

    for utility in args.utilities:
        print(f"\n{'=' * 70}\nBuilding dataset: {utility}\n{'=' * 70}", flush=True)
        X_pool, P_pool, X_test, P_test = build_dataset(utility)
        if args.test_subset is not None:
            X_test, P_test = X_test[:args.test_subset], P_test[:args.test_subset]

        for regime in ("not-perturbed", "perturbed"):
            key = (utility, regime)
            if key not in PANELS:
                continue
            print(f"\n--- Panel: {utility} / {regime} ---", flush=True)
            sizes, errs, betas = run_panel(utility, regime, X_pool, P_pool,
                                           X_test, P_test, training_sizes, n_reps)
            plot_panel(sizes, errs, os.path.join(args.out_dir, PANELS[key]),
                       title=f"{UTILITY_DISPLAY.get(utility, utility)} / {regime}")
            _write_results_csv(os.path.join(args.out_dir, f"results_{utility}_{regime}.csv"),
                               sizes, errs, betas)
            _write_reps_csv(os.path.join(args.out_dir, f"results_{utility}_{regime}_reps.csv"),
                            sizes, errs, betas)

    plot_legend(os.path.join(args.out_dir, "legend.pdf"))
    print("\nDone.", flush=True)


def _write_results_csv(path, sizes, errs, betas) -> None:
    """Aggregate CSV: per-size mean and SE of the rel-L2 across replications.

    ``errs``/``betas`` values are (n_reps, n_sizes) arrays. The companion
    per-replication file is written by ``_write_reps_csv``.
    """
    cols = ["training_size"]
    for mdl in MODELS:
        cols += [f"{mdl.name}_RelL2_mean", f"{mdl.name}_RelL2_se"]
    lines = [",".join(cols)]
    for j, s in enumerate(sizes):
        row = [str(s)]
        for mdl in MODELS:
            ys = np.atleast_2d(np.asarray(errs[mdl.name], float))[:, j]
            n_ok = int(np.sum(~np.isnan(ys)))
            mean = np.nanmean(ys) if n_ok else np.nan
            se = (np.nanstd(ys) / np.sqrt(n_ok)) if n_ok else np.nan
            row.append("" if mean != mean else f"{mean:.4f}")
            row.append("" if se != se else f"{se:.4f}")
        lines.append(",".join(row))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved {path}", flush=True)


def _write_reps_csv(path, sizes, errs, betas) -> None:
    """Long-format per-replication CSV: one row per (rep, size, model)."""
    lines = ["rep,training_size,model,RelL2,beta_star"]
    for mdl in MODELS:
        ys = np.atleast_2d(np.asarray(errs[mdl.name], float))
        bs = np.atleast_2d(np.asarray(betas[mdl.name], float))
        for rep in range(ys.shape[0]):
            for j, s in enumerate(sizes):
                v, b = ys[rep, j], bs[rep, j]
                lines.append(f"{rep},{s},{mdl.name},"
                             f"{'' if v != v else f'{v:.4f}'},"
                             f"{'' if b != b else f'{b:.4f}'}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
