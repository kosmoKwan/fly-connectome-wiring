#!/usr/bin/env bash
# smoke tests: installs scripts, runs the real MaleCNS graph (60 s) and MANC real stream A (60 s, chunked), compares with the archive
set -uo pipefail
W=/mnt/c/Users/Administrator/work/fly-connectome/round2
cd /root/fly-connectome
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p results/round2
cp $W/round2_common.py $W/round2_pairs.py $W/round2_long.py $W/round2_smokecheck.py src/
rm -f results/round2/smoke_pairs_*
( time R2_SMOKE=1 .venv/bin/python -u src/round2_pairs.py theta50 ) 2>&1 | grep -v -i "proxy" | tail -8
[ -f results/round2/smoke_long_scalar.parquet ] || ( time R2_SMOKE=1 .venv/bin/python -u src/round2_long.py ) 2>&1 | grep -v -i "proxy" | tail -8
.venv/bin/python src/round2_smokecheck.py
