#!/usr/bin/env bash
# Regenerate the whole figure set from the archived data (read-only). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/figvenv/bin/python}
$PY sim_examples.py          # CPU example traces (labelled as example simulations)
$PY struct_example.py        # per-neuron influences for Fig. 3a (checked against the archive)
$PY check_numbers.py         # manuscript v1 numbers -> ../qa/numbers_check.csv
$PY check_round2.py          # round-2 results recomputed from the parquet files, checked against analysis/ -> ../qa/round2_check.csv
$PY check_round3.py          # round-3 results recomputed from the parquet files, checked against analysis/ -> ../qa/round3_check.csv
for f in fig2 fig3 fig4 fig5 fig6 ed1 ed2 ed3 ed4 ed5 ed6 ed7 fig1; do $PY $f.py; done   # clipping + alignment gates run before every export
S=${SKILL_SCRIPTS:-$HOME/.claude/skills/nature-figure/scripts}
if [ -d "$S" ]; then
  for f in ../Fig*.pdf ../EDFig*.pdf; do b=$(basename "$f" .pdf); case "$b" in EDFig*) n=${b:0:6};; *) n=${b:0:4};; esac
    $PY "$S/audit_figure_collisions.py" "$f" --json-out ../qa/$n.collision-audit.json --overlay-pdf ../qa/$n.collision-audit.pdf || true
    $PY "$S/audit_pdf_text.py" "$f" --min-pt 5 > ../qa/$n.text-audit.txt || true
  done
fi
$PY make_docs.py             # README, captions, figure_numbers.csv, phone PDF
