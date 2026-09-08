"""
Stage B: the experiment pipeline that produces the paper figures from the (cached or
freshly generated) cohort.

  1. data_pipeline.build_ml_data   : unified 5-check filter + split + anchor augmentation
  2. recovery.run_recovery         : inverse-optimization for the 4 ablation models
  3. predict.run_prediction        : held-out beamlet-error CSVs for the 4 models
  4. plots                         : combined-MAPE + true-vs-retrieved figures

Usage:
  python run_experiment.py                 # full pipeline, all training sizes
  python run_experiment.py --skip-data     # reuse existing ml_data
  python run_experiment.py --models add_smooth --sizes 20,40,60
"""
import argparse

from config import TRAINING_SIZES, RESULTS
import data_pipeline
import recovery
import predict
import plots

MODELS = ["add_smooth", "add_lp", "nonadd_smooth", "nonadd_nonsmooth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true", help="reuse existing ml_data split")
    ap.add_argument("--skip-recovery", action="store_true", help="reuse existing recovered params")
    ap.add_argument("--skip-predict", action="store_true", help="reuse existing prediction CSVs")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in TRAINING_SIZES))
    args = ap.parse_args()
    models = args.models.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]

    if not args.skip_data:
        print("== Stage 1: data pipeline (unified filter + split + anchors) ==", flush=True)
        data_pipeline.build_ml_data()

    if not args.skip_recovery:
        print("== Stage 2: recovery ==", flush=True)
        for m in models:
            recovery.run_recovery(m, sizes)

    if not args.skip_predict:
        print("== Stage 3: prediction ==", flush=True)
        for m in models:
            predict.run_prediction(m, sizes)

    print("== Stage 4: plots ==", flush=True)
    plots.plot_combined_mape()
    plots.plot_true_vs_retrieved()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
