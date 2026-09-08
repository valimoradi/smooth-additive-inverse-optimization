"""
Approx-vs-true validation for the non-additive convex-only class.

Runs, on the SAME fresh cohort, both:
  - nonadd_nonsmooth      : large-beta (1e5) APPROXIMATION of convex-only (residual smoothness)
  - nonadd_nonsmooth_lp   : TRUE convex-only LP (no smoothness term at all, beta_inv=0)

and prints a side-by-side table of out-of-sample Rel-L2_w vs training size. If the two
coincide, the large-beta approximation is validated; if they diverge, the approximation
was materially distorting the convex-only curve.

Assumes Stage A (parallel_generate.py) has produced the cohort + anchors.

Usage:
  python run_convexonly_compare.py                 # full TRAINING_SIZES
  python run_convexonly_compare.py 20,40,60        # subset (quick check)
  python run_convexonly_compare.py 20,40,60 --skip-data   # reuse existing ml_data split
"""
import os
import sys
import numpy as np
import pandas as pd

import data_pipeline
import recovery
import predict
from config import TRAINING_SIZES, PRED_OUT, PRED_CSV

MODELS = ["nonadd_nonsmooth", "nonadd_nonsmooth_lp"]
LABEL = {"nonadd_nonsmooth": "approx (beta=1e5)", "nonadd_nonsmooth_lp": "true LP"}


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip_data = "--skip-data" in sys.argv
    sizes = [int(x) for x in argv[0].split(",")] if argv else TRAINING_SIZES

    if not skip_data:
        print("== build_ml_data (fresh cohort) ==", flush=True)
        data_pipeline.build_ml_data()

    for m in MODELS:
        print(f"== recovery: {m} ==", flush=True)
        recovery.run_recovery(m, sizes)
        print(f"== prediction: {m} ==", flush=True)
        predict.run_prediction(m, sizes)

    # ---- side-by-side comparison ----
    dfs = {}
    for m in MODELS:
        path = os.path.join(PRED_OUT[m], PRED_CSV[m])
        dfs[m] = pd.read_csv(path).set_index("training_size")["test_RelL2_w"]
    tab = pd.DataFrame({LABEL[m]: dfs[m] for m in MODELS})
    tab["abs_diff"] = (tab[LABEL["nonadd_nonsmooth"]] - tab[LABEL["nonadd_nonsmooth_lp"]]).abs()
    tab["rel_diff_%"] = 100 * tab["abs_diff"] / tab[LABEL["nonadd_nonsmooth_lp"]].abs()
    print("\n=== Convex-only: approximation vs true LP (out-of-sample Rel-L2_w) ===", flush=True)
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"\nmax abs diff = {tab['abs_diff'].max():.4f} | "
          f"max rel diff = {tab['rel_diff_%'].max():.2f}%", flush=True)
    out = os.path.join(os.path.dirname(PRED_OUT["nonadd_nonsmooth_lp"]), "convexonly_approx_vs_true.csv")
    tab.to_csv(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
