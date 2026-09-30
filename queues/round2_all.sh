#!/usr/bin/env bash
# Round 2 (pre-registration round2_20260927/预注册_第二轮_20260927.md + amendment 1), one locked queue:
#   experiment 1: pair-resolved 60-s re-run (both connectomes)
#   experiment 2: dose-matched strength-controlled surgery (25-s scans of X_hi/W, then 60-s confirmation)
#   experiment 3: 300-s trajectories (MANC persistence)
set -uo pipefail
cd /root/fly-connectome
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p results/round2
exec >> logs/round2_all.log 2>&1 </dev/null
echo "$(date -Is) round2 queue start pid $$"
echo $$ > logs/round2_all.pid
echo "exp1" > logs/round2.stage
for T in theta50 manc50; do
  .venv/bin/python -u src/round2_pairs.py $T || echo "$(date -Is) exp1 $T FAILED rc=$?"
done
echo "exp2_scan" > logs/round2.stage
[ -s results/nullscan_sets_stat_r2_theta50.parquet ] || NULL_LEVELS= SURG="surgery_r2_theta50/mn_r2_cross_hi_s0,surgery_r2_theta50/mn_r2_cross_hi_s1,surgery_r2_theta50/mn_r2_cross_hi_s2,surgery_r2_theta50/mn_r2_cross_hi_s3,surgery_r2_theta50/mn_r2_cross_hi_s4,surgery_r2_theta50/mn_r2_within_s0,surgery_r2_theta50/mn_r2_within_s1,surgery_r2_theta50/mn_r2_within_s2,surgery_r2_theta50/mn_r2_within_s3,surgery_r2_theta50/mn_r2_within_s4" LABEL_PREFIX=stat_r2_ .venv/bin/python -u src/stationary_scan.py theta50 > logs/round2_scan_theta50.log 2>&1 || echo "$(date -Is) exp2 scan theta50 FAILED rc=$?"
[ -s results/nullscan_sets_stat_r2_manc50.parquet ] || NULL_LEVELS= SURG="surgery_r2_manc50/mn_r2_cross_hi_s0,surgery_r2_manc50/mn_r2_cross_hi_s1,surgery_r2_manc50/mn_r2_cross_hi_s2,surgery_r2_manc50/mn_r2_cross_hi_s3,surgery_r2_manc50/mn_r2_cross_hi_s4,surgery_r2_manc50/mn_r2_within_s0,surgery_r2_manc50/mn_r2_within_s1,surgery_r2_manc50/mn_r2_within_s2,surgery_r2_manc50/mn_r2_within_s3,surgery_r2_manc50/mn_r2_within_s4" LABEL_PREFIX=stat_r2_ .venv/bin/python -u src/stationary_scan.py manc50 > logs/round2_scan_manc50.log 2>&1 || echo "$(date -Is) exp2 scan manc50 FAILED rc=$?"
echo "exp2_confirm" > logs/round2.stage
for T in theta50 manc50; do
  R2_JOBSET=surg2 .venv/bin/python -u src/round2_pairs.py $T || echo "$(date -Is) exp2 confirm $T FAILED rc=$?"
done
echo "exp3" > logs/round2.stage
.venv/bin/python -u src/round2_long.py || echo "$(date -Is) exp3 FAILED rc=$?"
echo "done" > logs/round2.stage
date -Is > logs/round2_all.done
echo "$(date -Is) round2 queue end"
