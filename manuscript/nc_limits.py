"""Check manuscript.tex against the Nature Communications length guide for Articles (initial submission).

Title <= 15 words; abstract <= 200 words; main text (Introduction, Results and Discussion; not Abstract, Methods,
References or legends) <= 5,000 words; each figure legend <= 350 words. Methods are "typically less than 3,000 words"
and are reported, not enforced. Words are counted after removing LaTeX commands, citations, labels, inline
mathematics (one word each), displayed equations and figure environments, so counts are close to, not identical
with, a word processor's.

usage: python nc_limits.py     exit status 1 if a limit is exceeded
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "manuscript.tex"
CAP = HERE.parent / "generated" / "captions.tex"


def plain(t):
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\begin\{(equation|figure|table)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " x ", t)
    t = re.sub(r"\\(citep|citet|cite|ref|label|href|input|pending)\{[^}]*\}(\{[^}]*\})?", " ", t)
    t = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", t)
    return re.sub(r"[{}~]", " ", t)


def words(t):
    return len(re.findall(r"[A-Za-z0-9][^\s]*", plain(t)))


tex = TEX.read_text(encoding="utf-8")
body = tex.split("\\begin{document}", 1)[1]


def between(a, b):
    i = body.index(a) + len(a)
    return body[i:body.index(b, i)]


title = re.search(r"\\title\{\\textbf\{(.*?)\}\}", tex).group(1)
n = {"Title": words(title),
     "Abstract": words(between("\\begin{abstract}", "\\end{abstract}")),
     "Introduction": words(between("\\section*{Introduction}", "\\section*{Results}")),
     "Results": words(between("\\section*{Results}", "\\section*{Discussion}")),
     "Discussion": words(between("\\section*{Discussion}", "\\section*{Methods}")),
     "Methods": words(between("\\section*{Methods}", "\\section*{Data availability}"))}
n["Main text"] = n["Introduction"] + n["Results"] + n["Discussion"]
limit = {"Title": 15, "Abstract": 200, "Main text": 5000}

caps = {}
for m in re.finditer(r"^\\newcommand\{\\cap(Title|Body)([A-Z]+)\}\{(.*)\}$", CAP.read_text(encoding="utf-8"), flags=re.M):
    caps[m.group(2)] = caps.get(m.group(2), 0) + words(m.group(3))
names = {k: (f"Supplementary Fig. {ord(k[2]) - 64}" if k.startswith("ED") else f"Fig. {ord(k) - 64}") for k in caps}

bad = []
print(f"{'part':28s} {'words':>6s}  limit")
for k in ["Title", "Abstract", "Introduction", "Results", "Discussion", "Main text", "Methods"]:
    lim = limit.get(k)
    note = f"<= {lim}" if lim else ("typically < 3000" if k == "Methods" else "")
    print(f"{k:28s} {n[k]:6d}  {note}")
    if lim and n[k] > lim:
        bad.append(k)
for k in sorted(caps, key=lambda k: (k.startswith("ED"), k)):
    print(f"{names[k] + ' legend':28s} {caps[k]:6d}  <= 350")
    if caps[k] > 350:
        bad.append(names[k])
if n["Methods"] >= 3000:
    print("note: Methods above the typical 3,000 words")
print("NC length guide:", "met" if not bad else "EXCEEDED: " + ", ".join(bad))
sys.exit(1 if bad else 0)
