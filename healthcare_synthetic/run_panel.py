"""
Run one PANEL of the consumer-style 4-class comparison: recovery + prediction for the four
model classes on a given regime, then plot the combined Rel-L2-vs-N figure for that panel.

The four classes (true-LP convex-only is used, NOT the large-beta approximation, and it is
evaluated with the correct max-of-affines predictor):
  nonadd_nonsmooth_lp = Convex only,  add_lp = Additive,
  nonadd_smooth = Smooth,             add_smooth = Additive+Smooth

Regimes are separated by an output-dir suffix so they coexist:
  --suffix ""      unperturbed (clean training)
  --suffix _pert   perturbed   (--perturb 0.05)

Usage:
  python run_panel.py --suffix "" --perturb 0.0                 # correct-spec, unperturbed
  python run_panel.py --suffix _pert --perturb 0.05            # correct-spec, perturbed
  python run_panel.py --suffix "" --sizes 20,40,60 --no-plot   # quick check
"""
import os
import sys
import time
import argparse

import recovery
import predict
import plots
from config import TRAINING_SIZES, RESULTS, PRED_OUT

# fast LP classes first so a slow class (nonadd_smooth at large N) doesn't gate early results
MODELS = ["nonadd_nonsmooth_lp", "add_lp", "add_smooth", "nonadd_smooth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in TRAINING_SIZES))
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot-dir", default=None)
    args = ap.parse_args()
    models = args.models.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]

    for m in models:
        rdir = RESULTS[m] + args.suffix
        pdir = PRED_OUT[m] + args.suffix
        t0 = time.time()
        print(f"\n### [{args.suffix or 'unpert'}] {m}: recovery (perturb={args.perturb}) -> {rdir}", flush=True)
        recovery.run_recovery(m, sizes, out_dir=rdir, perturb=args.perturb)
        print(f"### [{args.suffix or 'unpert'}] {m}: prediction -> {pdir}", flush=True)
        predict.run_prediction(m, sizes, results_dir=rdir, out_dir=pdir)
        print(f"### [{args.suffix or 'unpert'}] {m}: done in {time.time()-t0:.0f}s", flush=True)

    if not args.no_plot:
        pred_out = {m: PRED_OUT[m] + args.suffix for m in PRED_OUT}
        out_dir = args.plot_dir or (os.path.join(os.path.dirname(RESULTS["add_smooth"]),
                                                 f"panel_plots{args.suffix or '_unpert'}"))
        fname = plots.plot_combined_mape(out_dir=out_dir, pred_out=pred_out)
        print(f"panel figure -> {fname}", flush=True)


if __name__ == "__main__":
    main()
