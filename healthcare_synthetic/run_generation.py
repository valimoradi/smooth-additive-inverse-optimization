"""
Stage A (optional, heavy, local): regenerate the synthetic data from scratch.
  1. anchor search (best/worst case)   -> best_case_search/, worst_case_search/
  2. cohort generation + aggregation   -> patient_data_high_std/, patient_data_clean_high_std/

This is expensive (tens of thousands of forward solves) and the produced data is kept
LOCAL (git-ignored). Skipped automatically if outputs already exist unless --force.
"""
import os
import argparse

from config import BEST_CASE_DIR, WORST_CASE_DIR, COHORT_DIR
import anchor_search
import generate_cohort


def _exists(path):
    return os.path.exists(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    args = ap.parse_args()

    if args.force or not _exists(f"{WORST_CASE_DIR}/D_worst_case.npy"):
        anchor_search.search_worst()
    else:
        print("worst-case anchor exists; skipping (use --force to regenerate)", flush=True)
    if args.force or not _exists(f"{BEST_CASE_DIR}/D_best_case.npy"):
        anchor_search.search_best()
    else:
        print("best-case anchor exists; skipping", flush=True)

    if args.force or not _exists(f"{COHORT_DIR}/D_all_successful.npy"):
        generate_cohort.generate_patients()
        generate_cohort.aggregate_cohort()
    else:
        print("cohort exists; skipping", flush=True)


if __name__ == "__main__":
    main()
