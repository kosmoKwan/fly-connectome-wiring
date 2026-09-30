"""Keep the Chinese translation (manuscript_zh.tex) in step with the English manuscript (manuscript.tex).

1. Copies the reference list of manuscript.tex verbatim to ../generated/bib.tex (the Chinese version keeps the
   English references).
2. Compares the numbers of the two versions: both are cut into the same sequence of parts (abstract, then every
   section and subsection, including the Supplementary Notes), and every part must contain exactly the same numeric
   tokens with the same multiplicities; likewise every figure legend (generated/captions.tex against
   zh/captions_zh.tex). The English numbers are themselves checked against the data by check_numbers.py, so a
   translation that passes carries no number that is not in the English text, in the same part.
   Cross-references (\\ref), citations, month numbers of Chinese dates and names such as DNg100, 13A or T1R are
   not counted, as in check_numbers.py.

3. Checks the plain-language edition (manuscript_zhp.tex, "说人话"版) in the same parts, but as a subset: it may leave
   numbers out, but every number it prints must occur (at least as often) in the same part or legend of the English text.

usage: python check_zh.py     exit status 1 if a part or legend differs, or if the parts do not correspond
"""
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
EN, ZH = V / "manuscript.tex", V / "manuscript_zh.tex"
CAP_EN, CAP_ZH = V / "generated" / "captions.tex", V / "zh" / "captions_zh.tex"
BIB = V / "generated" / "bib.tex"
TOK = re.compile(r"(?<![A-Za-z\\{.\d])\d+(?:,\d{3})*(?:\.\d+)?")

en = EN.read_text(encoding="utf-8")
bib = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", en, flags=re.S).group(0)
BIB.write_text("% Copied from manuscript.tex by src/check_zh.py -- do not edit by hand.\n" + bib + "\n", encoding="utf-8")


def expand(tex):
    return re.sub(r"\\input\{(zhp?/[^}]+)\}", lambda m: (V / m.group(1)).read_text(encoding="utf-8"), tex)


def norm(t):
    t = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in t.splitlines())
    t = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", t, flags=re.S)
    t = re.sub(r"\\(ref|label|citep|input|figpage|mainfig|includegraphics|setcounter|captionsetup|href|renewcommand|vspace)"
               r"\*?(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", " ", t)
    t = re.sub(r"\\parbox\{[^}]*\}", " ", t)
    t = re.sub(r"\\\\\[\d+pt\]", " ", t)
    t = re.sub(r"(Supplementary (Note|Table)|补充说明|补充表)~\d+", " ", t)
    t = re.sub(r"(\d{4})\s*年\s*\d{1,2}\s*月\s*(\d{1,2})\s*日", r"\1 \2", t)     # English dates name the month
    t = re.sub(r"\d{1,2}\s*月\s*(\d{1,2})\s*日", r"\1", t)
    t = re.sub(r"(?<![A-Za-z0-9])(DNg100|DNxl058|RTX 5090|v1\.0|13A|13B|01A|T[123][LR]?)(?![A-Za-z0-9])", " ", t)
    return t.replace("$", " ").replace("\\,", " ").replace("~", " ")


def parts(tex):
    body = tex.split("\\begin{document}", 1)[1]
    body = body[body.index("\\begin{abstract}"):]
    out = []
    for p in re.split(r"(?=\\section\*\{|\\subsection\*\{)", body):
        m = re.match(r"\\(?:sub)?section\*\{([^}]*)\}", p)
        out.append((m.group(1) if m else "Abstract", Counter(TOK.findall(norm(p)))))
    return out


def legends(path):
    d = {}
    for m in re.finditer(r"^\\newcommand\{\\(cap(?:Title|Body)[A-Z]+)\}\{(.*)\}$", path.read_text(encoding="utf-8"), flags=re.M):
        d[m.group(1)] = Counter(TOK.findall(norm(m.group(2))))
    return d


def diff(a, b):
    return f"only in English {sorted((a - b).elements())}; only in Chinese {sorted((b - a).elements())}"


bad = []
pe, le = parts(en), legends(CAP_EN)
EDITIONS = [("faithful translation", V / "manuscript_zh.tex", CAP_ZH, "equal")]   # v8: the plain-language edition of v7 is not continued
for label, tex_path, cap_path, mode in EDITIONS:
    pz, lz = parts(expand(tex_path.read_text(encoding="utf-8"))), legends(cap_path)
    nb = len(bad)
    if len(pe) != len(pz):
        bad.append(f"{label}: parts do not correspond: {len(pe)} English, {len(pz)} Chinese\n  EN: {[t for t, _ in pe]}\n  ZH: {[t for t, _ in pz]}")
    else:
        for (te, ce), (tz, cz) in zip(pe, pz):
            if (mode == "equal" and ce != cz) or (mode == "subset" and cz - ce):
                bad.append(f"{label}, {te} / {tz}: {diff(ce, cz) if mode == 'equal' else 'numbers not in the English part ' + str(sorted((cz - ce).elements()))}")
    if set(le) != set(lz):
        bad.append(f"{label}: legend macros differ: {sorted(set(le) ^ set(lz))}")
    for k in sorted(set(le) & set(lz)):
        if (mode == "equal" and le[k] != lz[k]) or (mode == "subset" and lz[k] - le[k]):
            bad.append(f"{label}, legend {k}: {diff(le[k], lz[k]) if mode == 'equal' else 'numbers not in the English legend ' + str(sorted((lz[k] - le[k]).elements()))}")
    n_tok = sum(sum(c.values()) for _, c in pz) + sum(sum(c.values()) for c in lz.values())
    print(f"Chinese {label}: {len(pz)} parts and {len(lz)} legend macros compared ({'same numbers as' if mode == 'equal' else 'numbers contained in'} "
          f"the English text), {n_tok} numeric tokens; " + ("all match." if len(bad) == nb else f"{len(bad) - nb} MISMATCH(ES)."))
for b in bad:
    print("  " + b)
sys.exit(1 if bad else 0)
