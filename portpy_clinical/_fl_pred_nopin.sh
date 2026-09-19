#!/usr/bin/env bash
# No-pin prediction sweep, N=6,5,4,3.
#
# SERIAL, ONE MOSEK THREAD -- matching _fl_pred_sweep.sh, whose header records
# the reason: "per-thread factorization workspace otherwise exceeds 61.6GB RAM".
# A previous attempt at MOSEK_THREADS=6 committed ~64GB per worker and was
# killed by the Windows Resource-Exhaustion-Detector (Event 2004, 17:09:04).
# Do not raise the thread count on this machine.
#
# One process per (N,patient) so a crash resumes; the driver skips any patient
# whose .npz already exists. Order is small->big by w-norm.
# Reads   results/nested_N{N}_fullleak_nopin        (recovered delta/lambda)
# Writes  results/prediction_nested_fullleak_nopin/doses_N{N}/
# The pinned dirs (*_fullleak_anchor) are never touched.
cd "$(dirname "$0")"

ORDER="8 14 11 10 15 0 19 6 1 4 9 2 5 16 13 18 12 17 7 3"

for N in 6 5 4 3; do
  D=results/prediction_nested_fullleak_nopin/doses_N${N}
  mkdir -p "$D"
  LOG=results/_fl_pred_nopin_N${N}.log
  echo "######## N=$N START $(date '+%m-%d %H:%M:%S')  serial, 1 thread ########" >> $LOG
  for idx in $ORDER; do
    FL_N=$N \
    FL_RES_DIR=results/nested_N${N}_fullleak_nopin \
    FL_DOSE_DIR=$D \
    MOSEK_THREADS=1 MSK_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    python3.11.exe _run_fullleak_pred_parallel.py "$idx" >> $LOG 2>&1
    echo "---- N=$N idx=$idx exit=$? done=$(ls $D | wc -l)/20 $(date '+%H:%M:%S')" >> $LOG
  done
  echo "######## N=$N FINISHED $(date '+%m-%d %H:%M:%S')  files=$(ls $D | wc -l)/20 ########" >> $LOG
done
echo "######## NO-PIN PREDICTION ALL COHORTS DONE $(date '+%m-%d %H:%M:%S') ########" >> results/_fl_pred_nopin_N6.log
