#!/usr/bin/env bash
# Rebuild the Nature Communications version (v9_NC) in English and Chinese (faithful translation): legends from captions_draft.md, SI tables
# from the analysis tables, number check (English vs data), NC length check, number check (Chinese vs English), PDFs.
# The two PDFs go to <experiment root>/paper_最新版/, so that the current paper is one level below the root.
# Run after `bash ../figures_v3/Claude_v8第三轮_20260930/src/run_all.sh` whenever the figures or their data change.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/figvenv/bin/python}
OUT="../../paper_最新版"
$PY src/captions_md2tex.py
$PY src/make_si_tables.py
$PY src/check_numbers.py          # writes numbers_check.md; exits non-zero if any number does not match its source
$PY src/nc_limits.py              # exits non-zero if the title, abstract, main text or a legend exceeds the NC guide
$PY src/check_zh.py               # copies the references; Chinese editions: numbers equal to (translation) or contained in (plain) the English
tectonic -X compile manuscript.tex --keep-logs
tectonic -X compile manuscript_zh.tex --keep-logs
mkdir -p "$OUT"
mv -f manuscript.pdf "$OUT/英文版_果蝇连接组论文_NC投稿v9.pdf"
mv -f manuscript_zh.pdf "$OUT/中文版_果蝇连接组论文_NC投稿v9.pdf"
# NC wants the main text and the Supplementary Information as separate files; the combined PDF is kept
$PY src/split_si.py "$OUT/英文版_果蝇连接组论文_NC投稿v9.pdf" "$OUT/英文版_正文_NC投稿v9.pdf" "$OUT/英文版_补充材料_NC投稿v9.pdf"
echo "PDFs -> $(cd "$OUT" && pwd)"
