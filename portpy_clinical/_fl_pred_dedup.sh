#!/usr/bin/env bash
# No-pin prediction sweep with exact anchor dedup, cohorts N=5,4,3.
#
# Why dedup: removing the per-voxel pin inflated the anchor count ~1.7x, which
# put N=5 (7761 anchors) past available memory -- 12 of 20 patients segfaulted.
# _dedup_anchors.py merges envelope lines whose removal lowers f(z) by at most
# 1e-5 relative, i.e. below MOSEK's own tolerance on delta. Validated on N=5
# Prostate_Patient_72: 7761 -> 2487 anchors, beamlet error 0.089036 -> 0.089366
# (3.3e-04), runtime 2968s -> 1074s.
#
# ALL 20 patients are re-run per cohort into fresh directories. The existing
# partial non-dedup results are left in place as a cross-check; they are NOT
# mixed with these, because a cohort mean must come from uniformly-treated solves.
#
# N=6 is deliberately NOT included here -- it already has 20/20 at full
# resolution and is being left alone for now.
#
# Serial, one MOSEK thread (per-thread factorization workspace otherwise
# exceeds 61.6GB RAM -- see _fl_pred_sweep.sh header).
cd "$(dirname "$0")"

ORDER="8 14 11 10 15 0 19 6 1 4 9 2 5 16 13 18 12 17 7 3"

for N in 5 4 3; do
  D=results/prediction_nested_fullleak_nopin_dedup/doses_N${N}
  mkdir -p "$D"
  LOG=results/_fl_pred_dedup_N${N}.log
  echo "######## N=$N START $(date '+%m-%d %H:%M:%S')  dedup=1e-5, serial, 1 thread ########" >> $LOG
  for idx in $ORDER; do
    FL_N=$N \
    FL_RES_DIR=results/nested_N${N}_fullleak_nopin \
    FL_DOSE_DIR=$D \
    FL_DEDUP=1e-5 \
    MOSEK_THREADS=1 MSK_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    python3.11.exe _run_fullleak_pred_parallel.py "$idx" >> $LOG 2>&1
    echo "---- N=$N idx=$idx exit=$? done=$(ls $D | wc -l)/20 $(date '+%H:%M:%S')" >> $LOG
  done
  echo "######## N=$N FINISHED $(date '+%m-%d %H:%M:%S')  files=$(ls $D | wc -l)/20 ########" >> $LOG
done
echo "######## DEDUP SWEEP N=5,4,3 ALL DONE $(date '+%m-%d %H:%M:%S') ########" >> results/_fl_pred_dedup_N5.log
