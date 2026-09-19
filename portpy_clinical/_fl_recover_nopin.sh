#!/usr/bin/env bash
# Full-leak (exact-objective anchors) recovery for N=3,4,5,6 with the
# per-function pin C7'c REMOVED from inverse/constraints/normalization.py.
# Results -> results/nested_N{N}_fullleak_nopin (the *_fullleak_anchor dirs
# are left intact as the record of the pinned run).
cd "$(dirname "$0")"
unset MSK_NUM_THREADS
export PYTHONUNBUFFERED=1
PY=python3.11.exe
read U_MIN_ANCHOR U_MAX_ANCHOR < <($PY -c "import pickle; c=pickle.load(open('results/all_patients/forward_cache_nested_N5.pkl','rb')); ov=c['obj_values']; print(repr(float(ov[0])), repr(float(ov[1])))")
export U_MIN_ANCHOR U_MAX_ANCHOR
echo "NOPIN full-leak anchors: U_MIN=$U_MIN_ANCHOR  U_MAX=$U_MAX_ANCHOR"

retry() {
  local desc="$1"; shift
  local max="$1"; shift
  local a=1
  while [ "$a" -le "$max" ]; do
    echo "======== $desc (attempt $a/$max) @ $(date +%H:%M:%S) ========"
    "$@"
    local rc=$?
    if [ "$rc" -eq 0 ]; then echo "======== $desc OK @ $(date +%H:%M:%S) ========"; return 0; fi
    echo "======== $desc FAILED rc=$rc -- retrying ========"
    a=$((a+1)); sleep 5
  done
  echo "======== $desc GAVE UP after $max ========"
  return 1
}

for N in 5 3 4 6; do
  D=results/nested_N${N}_fullleak_nopin
  mkdir -p "$D"
  retry "N=$N Stage1" 3 $PY run_stage1_nested.py "$N" "$D"        || { echo "SKIP N=$N (stage1)"; continue; }
  retry "N=$N Stage2" 8 $PY run_stage2_nested.py "$N" 100.0 "$D"  || { echo "SKIP N=$N (stage2)"; continue; }
  retry "N=$N Stage3" 3 $PY run_stage3_nested_auto.py "$N" "$D"   || { echo "SKIP N=$N (stage3)"; continue; }
  echo "######## N=$N RECOVERY COMPLETE @ $(date +%H:%M:%S) ########"
done
echo "######## NOPIN RECOVERY N=3,4,5,6 DONE @ $(date +%H:%M:%S) ########"
