#!/bin/bash
set -uo pipefail
W=/mnt/c/Users/Administrator/work/fly-connectome/round3
cd /root/fly-connectome
cp $W/round3_scan.py $W/round3_pairs.py $W/r3_smokecheck.py src/
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
TAG=${1:-theta50}
R3_LABEL=smoke_tanh_$TAG R3_GRAPHS=real R3_KIND=tanh_rect R3_GRID=std .venv/bin/python src/round3_scan.py $TAG 2>&1 | grep -v -i -E "proxy|Failed to allocate" | tail -3
R3_GRAPHS=real R3_SETS=round3/nullscan_sets_smoke_tanh_$TAG.parquet R3_KIND=tanh_rect R3_SUFFIX=smoke_tanh .venv/bin/python src/round3_pairs.py $TAG 2>&1 | grep -v -i -E "proxy|Failed to allocate" | tail -3
.venv/bin/python src/r3_smokecheck.py $TAG
