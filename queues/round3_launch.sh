#!/bin/bash
W=/mnt/c/Users/Administrator/work/fly-connectome/round3
cd /root/fly-connectome
cp $W/round3_scan.py $W/round3_pairs.py $W/r3_smokecheck.py src/
exec flock -n /root/fly-connectome/.round3.lock bash $W/round3_queue.sh
