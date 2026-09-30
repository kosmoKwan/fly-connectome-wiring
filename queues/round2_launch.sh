#!/usr/bin/env bash
# started from Windows (WMI); installs round-2 scripts and graphs into the WSL project and runs one locked queue
set -uo pipefail
W=/mnt/c/Users/Administrator/work/fly-connectome/round2
cd /root/fly-connectome
cp $W/round2_common.py $W/round2_pairs.py $W/round2_long.py src/
cp $W/round2_all.sh $W/round2_status.sh .
for T in theta50 manc50; do mkdir -p graphs/surgery_r2_$T && cp -n $W/graphs_r2/surgery_r2_$T/*.parquet graphs/surgery_r2_$T/; done
cp -n $W/graphs_r2/surgery_r2_report.json graphs/ 2>/dev/null
exec flock -n /root/fly-connectome/.round2.lock bash /root/fly-connectome/round2_all.sh
