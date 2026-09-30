#!/usr/bin/env bash
set -euo pipefail
cd /root/fly-connectome
exec >> logs/heldout_stat_queue.log 2>&1 </dev/null

echo "$(date -Is) validation queue started"
while [[ ! -s results/nullscan_sets_stat_part2_theta50.parquet || ! -s results/nullscan_sets_stat_manc50.parquet ]]; do
    sleep 60
done
echo "$(date -Is) both stationary scans wrote final per-setting files"

.venv/bin/python - <<'PY'
from pathlib import Path
import polars as pl
base = Path('/root/fly-connectome/results')
cases = {
    'theta50': (['nullscan_sets_stat_theta50_part1.parquet', 'nullscan_sets_stat_part2_theta50.parquet'], 61),
    'manc50': (['nullscan_sets_stat_manc50.parquet'], 51),
}
for tag, (files, expected) in cases.items():
    frames = [pl.read_parquet(base / name) for name in files]
    rows = pl.concat(frames, how='diagonal_relaxed')
    counts = rows.group_by('graph', 'seed').len()
    assert counts.height == expected, (tag, counts.height, expected)
    assert counts['len'].min() == counts['len'].max() == 132, (tag, counts.select('len').describe())
    print(tag, 'validated', expected, 'graphs and 132 settings per graph', flush=True)
PY

echo "$(date -Is) beginning 60-s fresh-seed confirmation: theta50"
.venv/bin/python -u src/heldout_stat_resume.py theta50 nullscan_sets_stat_theta50_part1.parquet,nullscan_sets_stat_part2_theta50.parquet
echo "$(date -Is) theta50 confirmation complete"

echo "$(date -Is) beginning 60-s fresh-seed confirmation: manc50"
.venv/bin/python -u src/heldout_stat_resume.py manc50 nullscan_sets_stat_manc50.parquet
echo "$(date -Is) manc50 confirmation complete"

printf 'done %s\n' "$(date -Is)" > logs/heldout_stat.done
