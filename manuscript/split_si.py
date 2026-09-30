"""Split the English PDF into the main manuscript and the Supplementary Information, which Nature Communications asks
for as separate files. The combined PDF is kept; the split is exact, so every number, label and cross-reference is the
one resolved in the combined build.

The Supplementary Information starts on the first page labelled S1 (its page numbering restarts there) and must begin with
its own title block. Internal links are rebuilt as page links inside each file; links that would point into the other file
(for example, from "Supplementary Fig. 2" in the main text) are dropped, and URI links are kept.

usage: python split_si.py combined.pdf main.pdf si.pdf
"""
import sys
from pathlib import Path

import pymupdf

TITLE = "How much of fly walking is written in the wiring?"
AUTHORS = "Isabel Guan, Yuntian Zhao, Dingyuan Zhang, Shipeng Lyu, I-Ming Chen"

src_path, main_path, si_path = (Path(a) for a in sys.argv[1:4])
src = pymupdf.open(src_path)


def is_si(label):                                  # xdvipdfmx writes the label prefix "S" as a UTF-16 hex string
    return label.startswith("S") or label.upper().startswith("<FEFF0053>")


labels = [p.get_label() for p in src]
k = next(i for i, lab in enumerate(labels) if is_si(lab))
if not all(is_si(lab) for lab in labels[k:]) or any(is_si(lab) for lab in labels[:k]):
    raise SystemExit(f"unexpected page labels: {labels}")
_first = src[k].get_text().replace("\ufb01", "fi").replace("\ufb02", "fl")    # undo ligatures; drop the line numbers
if not " ".join(w for w in _first.split() if not w.isdigit()).startswith("Supplementary Information " + TITLE):
    raise SystemExit("the first SI page does not start with the Supplementary Information title block")


def part(a, b, out, title, prefix):
    d = pymupdf.open()
    d.insert_pdf(src, from_page=a, to_page=b, links=False, annots=False)
    kept = dropped = 0
    for i in range(a, b + 1):
        page = d[i - a]
        for ln in src[i].get_links():
            if ln["kind"] == pymupdf.LINK_URI:
                page.insert_link({"kind": pymupdf.LINK_URI, "from": ln["from"], "uri": ln["uri"]})
                kept += 1
            elif ln["kind"] in (pymupdf.LINK_GOTO, pymupdf.LINK_NAMED) and a <= ln.get("page", -1) <= b:
                page.insert_link({"kind": pymupdf.LINK_GOTO, "from": ln["from"], "page": ln["page"] - a,
                                  "to": ln.get("to") or pymupdf.Point(0, 0), "zoom": 0})
                kept += 1
            else:
                dropped += 1
    d.set_page_labels([{"startpage": 0, "prefix": prefix, "style": "D", "firstpagenum": 1}])
    d.set_metadata({"title": title, "author": AUTHORS, "creator": "LaTeX (tectonic); split by src/split_si.py"})
    d.save(out, garbage=3, deflate=True)
    return len(d), kept, dropped


m = part(0, k - 1, main_path, TITLE, "")
s = part(k, len(src) - 1, si_path, "Supplementary Information: " + TITLE, "S")
print(f"split {src_path.name}: main text {m[0]} pages ({m[1]} links kept, {m[2]} into the SI dropped); "
      f"Supplementary Information {s[0]} pages ({s[1]} kept, {s[2]} into the main text dropped)")
