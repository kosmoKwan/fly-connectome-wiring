#!/bin/bash
# Round 3 queue (pre-registration: NAS round3_20260929/预注册_第三轮_20260929.md). Order: smoke(MANC) -> A -> C -> D2 -> B -> F.
set -uo pipefail
cd /root/fly-connectome
mkdir -p logs results/round3 graphs/round3
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PY=.venv/bin/python; LOG=logs/round3.log
stage(){ echo "$(date -Is) STAGE $*" | tee -a logs/round3.stage >> $LOG; }
run_scan(){ R3_LABEL=$2 R3_KIND=$3 R3_NODES=$4 R3_GRAPHS=$5 R3_GRID=std $PY src/round3_scan.py $1 >> $LOG 2>&1 || echo "FAILED scan $1 $2" | tee -a $LOG; }
run_conf(){ R3_SUFFIX=$2 R3_KIND=$3 R3_NODES=$4 R3_GRAPHS=$5 R3_SETS=$6 R3_AT_REAL=${7:-0} $PY src/round3_pairs.py $1 >> $LOG 2>&1 || echo "FAILED conf $1 $2" | tee -a $LOG; }
waitflag(){ while [ ! -f "graphs/round3/$1" ] && [ ! -f "graphs/round3/$2" ]; do sleep 60; done; }
echo "$(date -Is) round3 queue start pid $$" >> $LOG
stage smoke_manc50
R3_LABEL=smoke_tanh_manc50 R3_GRAPHS=real R3_KIND=tanh_rect R3_GRID=std $PY src/round3_scan.py manc50 >> $LOG 2>&1
R3_GRAPHS=real R3_SETS=round3/nullscan_sets_smoke_tanh_manc50.parquet R3_KIND=tanh_rect R3_SUFFIX=smoke_tanh $PY src/round3_pairs.py manc50 >> $LOG 2>&1
if ! $PY src/r3_smokecheck.py manc50 >> $LOG 2>&1; then echo "$(date -Is) SMOKE FAIL manc50 - queue aborted" | tee -a $LOG logs/round3.stage; exit 1; fi
FAMS="er role degree leg hl leg_hl"
for TAG in theta50 manc50; do
  stage A_relu_sat $TAG
  GL="real"; for f in $FAMS; do for s in 0 1 2 3 4; do GL="$GL,nulls/$TAG/${f}_s$s"; done; done
  run_scan $TAG relu_$TAG relu_sat $TAG "$GL"
  run_conf $TAG relu relu_sat $TAG "$GL" round3/nullscan_sets_relu_$TAG.parquet
done
stage C_wait; waitflag rescue_ready rescue_skip
if [ -f graphs/round3/rescue_ready ]; then for TAG in theta50 manc50; do
  stage C_rescue $TAG; GL=$(cat graphs/round3/rescue_${TAG}_list.txt)
  run_scan $TAG rescue_$TAG tanh_rect $TAG "$GL"
  run_conf $TAG rescue tanh_rect $TAG "$GL" round3/nullscan_sets_rescue_$TAG.parquet
done; fi
stage D2_wait; waitflag cap_ready cap_infeasible
if [ -f graphs/round3/cap_ready ]; then for TAG in theta50 manc50; do
  stage D2_cap $TAG; GL=$(cat graphs/round3/cap_${TAG}_list.txt)
  ARCH=$([ $TAG = theta50 ] && echo nullscan_sets_stat_theta50_part1.parquet || echo nullscan_sets_stat_manc50.parquet)
  run_conf $TAG cap tanh_rect $TAG "$GL" $ARCH 1
done; fi
for TAG in theta50 manc50; do
  stage B_n25 $TAG
  GL=""; for f in hl degree leg_hl; do for s in $(seq 5 24); do GL="$GL,nulls/${TAG}_n100/${f}_s$s"; done; done; GL=${GL#,}
  run_scan $TAG n25_$TAG tanh_rect $TAG "$GL"
  run_conf $TAG n25 tanh_rect $TAG "$GL" round3/nullscan_sets_n25_$TAG.parquet
done
stage F_wait; waitflag signerr_ready signerr_skip
if [ -f graphs/round3/signerr_ready ]; then while read TAG NTAG FAM1 FAM2; do
  [ -z "$TAG" ] && continue
  stage F $TAG $NTAG
  GL="real"; for f in $FAM1 $FAM2; do for s in 0 1 2 3 4; do GL="$GL,nulls/$TAG/${f}_s$s"; done; done
  run_scan $TAG $NTAG tanh_rect $NTAG "$GL"
  run_conf $TAG $NTAG tanh_rect $NTAG "$GL" round3/nullscan_sets_$NTAG.parquet
done < graphs/round3/signerr_variants.txt; fi
stage ALL_DONE
date -Is > logs/round3.done
