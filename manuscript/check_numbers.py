"""Recompute every number in the NC version (v9_NC) from the archived data (read-only) and write ../numbers_check.md.

Each check records: where the number appears, what it states, the printed value, the value recomputed here,
the source file and the computation. Numbers from the earlier short-window protocol (Supplementary Note 1) are
not recomputed: they are checked against the lab notebook in which they were recorded (paper/draft_v0.md).
A final audit extracts every numeric token from manuscript.tex and lists any token not covered by a check.
The supplementary tables and figure legends are generated directly from data tables and are not re-checked here.

usage: python check_numbers.py      exit status 1 if any check fails or any numeric token is uncovered
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = V2.parents[1]
FH = ROOT / "final_heldout_20260927_121507_B9ue3i"
G, H = FH / "graphs", FH / "results"
REC = ROOT / "claude_audit_20260927/recalc/out"
ST = ROOT / "record_20260927_0202/results"
AUD = ROOT / "exploratory_analysis_20260927_060636/tables"
R2A, R2R = ROOT / "round2_20260927/analysis", ROOT / "round2_20260927/results"
R2G = ROOT / "round2_20260927/experiment2_graphs/surgery_r2_report.json"
NB = ROOT / "paper/draft_v0.md"
PRE = ROOT / "round2_20260927/预注册_第二轮_20260927.md"
SRC = FH / "src"
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]
SEC = ["degree_recip", "leg_hl_recip"]
W4 = ["w15_25", "w25_35", "w35_45", "w50_60"]

rows = []


def rel(p):
    return str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)


def same(printed, value):
    """printed: a number as printed (commas, %, minus sign allowed); value: float. Compare at printed precision."""
    s = printed.replace(",", "").replace("\u2212", "-").strip()
    if s.endswith("%"):
        s, value = s[:-1], value * 100
    dec = len(s.split(".")[1]) if "." in s else 0
    return abs(round(float(value), dec) - float(s)) < 10 ** (-dec) / 2 + 1e-12


def chk(section, statement, printed, computed, source, method):
    """computed: float, int, bool, str, or list of floats matched element-wise to '/'-separated printed values."""
    if isinstance(computed, bool):
        ok, shown = computed, "true" if computed else "FALSE"
    elif isinstance(computed, str):
        ok, shown = printed == computed, computed
    elif isinstance(computed, (list, tuple)):
        parts = [p.strip() for p in re.split(r"\s*(?:/|–|--)\s*", printed)]
        ok = len(parts) == len(computed) and all(same(p, v) for p, v in zip(parts, computed))
        shown = " / ".join(f"{v:.6g}" for v in computed)
    else:
        ok, shown = same(printed, computed), f"{computed:.6g}"
    rows.append({"section": section, "statement": statement, "printed": printed, "computed": shown, "ok": ok,
                 "source": source, "method": method})


def nb_section(title_start):
    """Text of a lab-notebook section: up to the next heading of the same or a higher level."""
    text = NB.read_text(encoding="utf-8")
    i = text.index(title_start)
    stops = ["\n## "] if title_start.startswith("## ") else ["\n## ", "\n### "]
    ends = [x for x in (text.find(t, i + 1) for t in stops) if x > 0]
    return text[i:min(ends) if ends else len(text)]


def nb(section, statement, printed, heading, needles):
    """Early-protocol number: check that the recorded values appear in the lab-notebook section."""
    sec = nb_section(heading)
    ok = all(n in sec for n in needles)
    rows.append({"section": section, "statement": statement, "printed": printed,
                 "computed": "recorded: " + "; ".join(needles), "ok": ok, "source": "paper/draft_v0.md, " + heading[:60],
                 "method": "early 5-s protocol; transcribed from the lab notebook, not recomputed"})


def code(section, statement, printed, path, pattern):
    ok = re.search(pattern, Path(path).read_text(encoding="utf-8")) is not None
    rows.append({"section": section, "statement": statement, "printed": printed, "computed": f"code matches /{pattern}/",
                 "ok": ok, "source": rel(path), "method": "constant in the simulation/analysis code"})


# ------------------------------------------------------------------------------------------ data
nodes = {t: pl.read_parquet(G / f"nodes_{t}.parquet") for t in ("theta50", "manc50")}
edges = {t: pl.read_parquet(G / f"edges_{t}.parquet") for t in ("theta50", "manc50")}
mnmap = {t: pl.read_parquet(G / f"mn_joint_map_{t}.parquet") for t in ("theta50", "manc50")}
held = {t: pl.read_parquet(H / f"heldout_stat_{t}.parquet") for t in ("theta50", "manc50")}
scan = {"theta50": pl.concat([pl.read_parquet(H / "nullscan_sets_stat_theta50_part1.parquet"),
                              pl.read_parquet(H / "nullscan_sets_stat_part2_theta50.parquet")], how="diagonal_relaxed"),
        "manc50": pl.read_parquet(H / "nullscan_sets_stat_manc50.parquet")}
levels = pl.read_csv(REC / "claim_levels_3_23_on_heldout.csv")
surg = pl.read_csv(REC / "surgery_conditions_3_26_on_heldout.csv")
r2lev = pl.read_csv(R2A / "exp1_claim_levels.csv")
r2c6 = pl.read_csv(R2A / "exp1_C6_pair_identity.csv")
r2cons = pl.read_csv(R2A / "exp1_consistency.csv")
audit = pl.read_csv(AUD / "intervention_edge_family_medians.csv")
r2rep = json.loads(R2G.read_text())


def net_means(t, rule="own", window="w15_25", col="gmean"):
    d = held[t].filter((pl.col("rule") == rule) & (pl.col("mode") == "stochastic") & (pl.col("window") == window))
    return d.group_by("graph", "gseed").agg(pl.col(col).drop_nans().mean().alias("v"))


def real_runs(t, mode="stochastic", window=None):
    d = held[t].filter((pl.col("graph") == "real") & (pl.col("rule") == "own") & (pl.col("mode") == mode))
    return d if window is None else d.filter(pl.col("window") == window)


FAST = (pl.col("freq") >= 3) & (pl.col("n_gated_pairs") <= 3)
CON = {"theta50": "MaleCNS", "manc50": "MANC"}

# ------------------------------------------------------------------------------------------ Results 1 / Methods: networks
for t, sec in (("theta50", "Results 1"), ("manc50", "Results 1")):
    n, e = nodes[t], edges[t]
    src = f"{rel(G)}/nodes_{t}.parquet, edges_{t}.parquet"
    chk(sec, f"{CON[t]} neurons", f"{n.height:,}", n.height, src, "rows of the node table")
    for role in ("IN", "DN", "MN", "SN"):
        k = n.filter(pl.col("role") == role).height
        chk(sec, f"{CON[t]} {role}", f"{k:,}", k, src, f"role == {role}")
    chk(sec, f"{CON[t]} connections", f"{e.height:,}", e.height, src, "rows of the edge table")
    chk(sec, f"{CON[t]} minimum synapses per connection", "5", float(e["weight"].min()), src, "min(weight)")
    chk(sec, f"{CON[t]} DNg100 neurons", "2", n.filter(pl.col("type") == "DNg100").height, src, "type == DNg100")
printed_nodes = {"theta50": ("10,173", "8,702", "424", "381", "666", "462,650"), "manc50": ("10,440", "8,951", "498", "396", "595", "634,207")}

for t in ("theta50", "manc50"):
    m = mnmap[t]; mp = m.filter(pl.col("joint").is_not_null())
    src = f"{rel(G)}/mn_joint_map_{t}.parquet"
    chk("Results 1 / Methods", f"{CON[t]} motor neurons assigned to a joint", "302", mp.height, src, "joint not null")
    k = mp.filter(pl.col("approx")).height
    chk("Results 1 / Methods", f"{CON[t]} approximately assigned", str(k), k, src, "approx flag")
    chk("Results 1 / Discussion", f"{CON[t]} approximate fraction 'about a quarter'", "0.25" if t == "theta50" else "0.27",
        k / mp.height, src, "approx / mapped (text: about a quarter)")
    pairs = mp.group_by("leg", "joint").agg(pl.col("direction").n_unique().alias("nd")).filter(pl.col("nd") == 2)
    chk("Results 1 / Methods", f"{CON[t]} antagonistic pairs", "20", pairs.height, src, "leg x joint with both directions")
    per = dict(pairs.group_by("joint").len().rows())
    chk("Results 1", f"{CON[t]} pairs per joint ThC/CTr/FTi/TiTa", "6/6/6/2", [per["ThC"], per["CTr"], per["FTi"], per["TiTa"]], src, "count per joint")
    ex = mp.filter(~pl.col("approx")).group_by("leg", "joint").agg(pl.col("direction").n_unique().alias("nd")).filter(pl.col("nd") == 2)
    pe = dict(ex.group_by("joint").len().rows())
    chk("Results 5 / Methods", f"{CON[t]} exactly assigned pairs total and CTr/FTi/ThC/TiTa", "16/6/6/2/2",
        [ex.height, pe["CTr"], pe["FTi"], pe["ThC"], pe["TiTa"]], src, "pairs from non-approximate motor neurons")
    chk("Results 5 / Methods", f"{CON[t]} pairs outside ThC", "14", pairs.filter(pl.col("joint") != "ThC").height, src, "pairs with joint != ThC")
chk("Methods", "MaleCNS motor neurons (all)", "381", mnmap["theta50"].height, f"{rel(G)}/mn_joint_map_theta50.parquet", "rows")
chk("Methods", "MANC motor neurons (all)", "396", mnmap["manc50"].height, f"{rel(G)}/mn_joint_map_manc50.parquet", "rows")
thc = mnmap["theta50"].filter(pl.col("joint") == "ThC")
t23 = thc.filter(pl.col("leg").str.starts_with("T2") | pl.col("leg").str.starts_with("T3"))
only_sar = all(set(g["type"].to_list()) == {"Sternal anterior rotator MN"} for _, g in t23.filter(pl.col("direction") > 0).group_by("leg"))
chk("Results 5", "T2/T3 ThC: one pool only sternal anterior rotator MNs (4 of 6 ThC pairs)", "4",
    float(t23["leg"].n_unique()) if only_sar else -1.0, f"{rel(G)}/mn_joint_map_theta50.parquet", "legs T2/T3 whose +1 ThC pool is only 'Sternal anterior rotator MN'")
approx_types = set(mnmap["theta50"].filter(pl.col("approx"))["type"].to_list())
chk("Methods", "approximate types: sternal rotators (ThC) and long-tendon-muscle (TiTa)", "ok",
    "ok" if approx_types == {"Sternal anterior rotator MN", "Sternal posterior rotator MN", "ltm MN", "ltm1-tibia MN", "ltm2-femur MN"} else str(approx_types),
    f"{rel(G)}/mn_joint_map_theta50.parquet", "set of approximate types")

code("Methods", "MANC DNg100 = DNxl058", "DNxl058", SRC / "build_graph_manc.py", r"DNg100 = MANC type DNxl058")
code("Methods", "extraction: >=0.5 of VNC synapses in leg neuropils, >=20 leg synapses, >=5 synapses", "0.5/20/5",
     SRC / "build_graph_manc.py", r"MIN_SYN, MIN_LEG = 5, 20")
code("Methods", "MANC extraction threshold theta = 0.5", "0.5", SRC / "build_graph_manc.py", r"else 0\.5")
nb("Methods", "MaleCNS synapse confidence >= 0.5; theta = 0.5; >= 20 leg synapses; >= 5 synapses", "0.5/20/5",
   "## 2. Methods", ["minconf 0.5", "≥ 20 leg synapses", "≥ 5 synapses", "main θ = 0.5"])

# ------------------------------------------------------------------------------------------ model and readout constants
OL = SRC / "openloop.py"
code("Methods", "tau = 20 ms, dt = 1 ms", "20 ms / 1 ms", OL, r"dt=1e-3, tau=0\.02")
code("Methods", "noise s.d. 0.01 per step", "0.01", OL, r"noise=0\.01")
code("Methods", "initial rates U[0, 0.01)", "0.01", OL, r"r0 = 0\.01 \* jax\.random\.uniform")
code("Methods", "rectified tanh", "max(tanh x, 0)", OL, r"jnp\.maximum\(jnp\.tanh\(x\), 0\.0\)")
code("Methods", "grid: 11 gains 1e-4..10^-1.5; beta {0.5,1,2}; d {0.5,1,2,4}", "11/0.5,1,2/0.5,1,2,4", OL,
     r"np\.logspace\(-4, -1\.5, 11\); betas = \[0\.5, 1\.0, 2\.0\]; drives = \[0\.5, 1\.0, 2\.0, 4\.0\]")
code("Methods", "record at 500 Hz", "500 Hz", OL, r"fs=500\.0")
code("Methods", "oscillating: s.d. > 0.01", "0.01", OL, r"osc = sd > 0\.01")
code("Methods", "rhythmicity band 0.5-50 Hz", "0.5–50 Hz", OL, r"band = \(f >= 0\.5\) & \(f <= 50\)")
code("Methods", "gate: s.d. > 1e-4; ratio >= 0.2; min >= 0.001", "1e-4/0.2/0.001", OL,
     r"if se > 1e-4 and sf > 1e-4:[\s\S]*ratio >= 0\.2 and min\(se, sf\) >= 0\.001")
code("Methods", "eligible: >20% oscillating, 1 <= freq <= 50 Hz", "20%/1–50 Hz", SRC / "stationary_scan.py",
     r"> 0\.2\) & \(pl\.col\(f\"freq_\{w\}\"\) >= 1\.0\) & \(pl\.col\(f\"freq_\{w\}\"\) <= 50\.0\)")
nsets = {t: scan[t].select("g", "beta", "drive").unique().height for t in scan}
chk("Results 1 / Methods", "parameter settings per network", "132", float(min(nsets.values())), f"{rel(H)}/nullscan_sets_stat_*", "unique (g, beta, drive)")
code("Methods", "rewired: ten attempted exchanges per edge", "10", SRC / "nulls.py", r"swaps_per_edge: int = 10")
code("Methods", "rewired: er/role place edges at random; weights permuted", "random placement", SRC / "nulls.py",
     r"if level == \"er\":[\s\S]*_random_in_blocks[\s\S]*w2 = rng\.permutation\(w\)[\s\S]*if level == \"role\":[\s\S]*_random_in_blocks")
code("Methods", "reassignment: 20 attempts per edge, five networks, seeds", "20/5", SRC / "surgery.py", r"def swap_mn_inputs\(e, post_key, seed, swaps=20\)")
code("Methods", "strength-matched: sign x decile blocks", "decile", SRC / "surgery_strength.py", r"np\.quantile\(w\[sel\], np\.linspace\(0, 1, 11\)\[1:-1\]\)")
code("Methods", "reassignment presynaptic blocks = role x segment x side x hemilineage", "leg_hl groups", SRC / "surgery_strength.py", r"pre_group = node_groups\(nodes, \"leg_hl\"\)")

# ------------------------------------------------------------------------------------------ confirmation design
for t in ("theta50", "manc50"):
    r = real_runs(t)
    chk("Results 1 / Methods", f"{CON[t]} real: stochastic runs / noise-free runs / windows", "20/3/4",
        [r["run"].n_unique(), real_runs(t, "deterministic")["run"].n_unique(), r["window"].n_unique()], f"{rel(H)}/heldout_stat_{t}.parquet", "unique runs and windows")
fz = {t: real_runs(t).select("g", "beta", "drive").unique().row(0) for t in held}
chk("Methods", "MaleCNS frozen setting log10 g / beta / d", "-2.5/0.5/0.5", [np.log10(fz["theta50"][0]), fz["theta50"][1], fz["theta50"][2]], f"{rel(H)}/heldout_stat_theta50.parquet", "real network, rule own")
chk("Methods", "MANC frozen setting log10 g / beta / d", "-1.75/1/4", [np.log10(fz["manc50"][0]), fz["manc50"][1], fz["manc50"][2]], f"{rel(H)}/heldout_stat_manc50.parquet", "real network, rule own")
chk("Discussion", "MANC drive = highest grid value", "4", float(max(scan["manc50"]["drive"].unique())), f"{rel(H)}/nullscan_sets_stat_manc50.parquet", "max drive of the grid (real frozen d = 4)")
und = pl.read_csv(REC / "graph_window_summary_theta50.csv").filter(pl.col("n_undef") > 0)
chk("Methods", "undefined runs: networks affected", "1", float(und.select("graph", "gseed").unique().height), rel(REC / "graph_window_summary_theta50.csv"), "networks with n_undef > 0 (MaleCNS; none in MANC)")
chk("Methods", "undefined runs per window / defined-run mean / zero-filled mean", "9/0.064/0.035",
    [float(und["n_undef"].min()), und.filter(pl.col("window") == "w15_25")["mean_finite"][0], und.filter(pl.col("window") == "w15_25")["mean_zero"][0]],
    rel(REC / "graph_window_summary_theta50.csv"), "leg_hl_recip s2, 15-25 s (9 of 20 in every window)")
chk("Methods", "MANC: no undefined runs", "0", float(pl.read_csv(REC / "graph_window_summary_manc50.csv")["n_undef"].sum()), rel(REC / "graph_window_summary_manc50.csv"), "sum n_undef")

# ------------------------------------------------------------------------------------------ Results 2: scan
def scan_best(t):
    s = scan[t].filter(pl.col("rhythmic_stat") & ~pl.col("graph").str.starts_with("mn_"))
    return s.group_by("graph", "seed").agg(pl.col("gmean_stat").max().alias("bestS"), pl.col("rhythm_stat").max().alias("maxR"))


for t, nnull, prS, prMax, prR in (("theta50", 40, "0.330", "0.139", ("19", "0.96", "0.70")), ("manc50", 30, "0.215", "0.106", ("8", None, None))):
    b = scan_best(t); real = b.filter(pl.col("graph") == "real").row(0, named=True); nul = b.filter(pl.col("graph") != "real")
    src = f"{rel(H)}/nullscan_sets_stat_{t}*"
    chk("Results 2", f"{CON[t]} scan: rewired networks", str(nnull), float(nul.height), src, "main + secondary families (no reassignment)")
    chk("Abstract / Results 2", f"{CON[t]} scan: real best S", prS, real["bestS"], src, "max S over eligible settings")
    chk("Results 2", f"{CON[t]} scan: max best S of rewired", prMax, nul["bestS"].max(), src, "max over rewired networks")
    chk("Results 2", f"{CON[t]} scan: rewired networks with max rhythmicity above real", prR[0], float((nul["maxR"] > real["maxR"]).sum()), src, "count maxR(rewired) > maxR(real), eligible settings")
    if prR[1]:
        chk("Results 2", "MaleCNS scan: rewired max rhythmicity / real max", "0.96/0.70", [nul["maxR"].max(), real["maxR"]], src, "max rhythmicity over eligible settings")
    chk("Figure 1 legend", f"{CON[t]} scan: eligible settings of the real network", "27" if t == "theta50" else str(scan[t].filter((pl.col("graph") == "real") & pl.col("rhythmic_stat")).height),
        float(scan[t].filter((pl.col("graph") == "real") & pl.col("rhythmic_stat")).height), src, "eligible settings (legend of Fig. 1d, MaleCNS)")

# ------------------------------------------------------------------------------------------ Results 2: confirmation
nm = {}
for t in ("theta50", "manc50"):
    for w in W4:
        d = net_means(t, window=w)
        nm[(t, w)] = d
for t, pr in (("theta50", {"w15_25": ("0.312", "0.144"), "w50_60": ("0.312", "0.147")}), ("manc50", {"w15_25": ("0.173", "0.084"), "w50_60": ("0.128", "0.080")})):
    for w, (pR, pM) in pr.items():
        d = nm[(t, w)]
        real = d.filter(pl.col("graph") == "real")["v"][0]
        rew = d.filter(pl.col("graph").is_in(MAIN + SEC))["v"].max()
        src = f"{rel(H)}/heldout_stat_{t}.parquet"
        chk("Results 2", f"{CON[t]} {w}: real S", pR, real, src, "mean of 20 stochastic runs (defined runs)")
        chk("Results 2", f"{CON[t]} {w}: strongest rewired S", pM, rew, src, "max network mean over rewired networks (main + secondary)")
dW = nm[("theta50", "w15_25")]
chk("Abstract", "MaleCNS primary window: 0.31 vs at most 0.14", "0.31/0.14", [dW.filter(pl.col("graph") == "real")["v"][0], dW.filter(pl.col("graph").is_in(MAIN + SEC))["v"].max()], f"{rel(H)}/heldout_stat_theta50.parquet", "as above, 2 decimals")
dW = nm[("manc50", "w15_25")]
chk("Abstract", "MANC primary window: 0.17 vs at most 0.08", "0.17/0.08", [dW.filter(pl.col("graph") == "real")["v"][0], dW.filter(pl.col("graph").is_in(MAIN))["v"].max()], f"{rel(H)}/heldout_stat_manc50.parquet", "as above, 2 decimals")
for t, lev in (("theta50", "STRONG"), ("manc50", "MODERATE")):
    got = levels.filter((pl.col("connectome") == t) & pl.col("family").str.starts_with("__LEVEL__"))["family"].to_list()
    chk("Abstract / Results 2", f"{CON[t]} level in all four windows", lev, lev if all(g.endswith(lev) for g in got) and len(got) == 4 else str(got), rel(REC / "claim_levels_3_23_on_heldout.csv"), "section 3.23 rule, six main families")
    p = levels.filter((pl.col("connectome") == t) & pl.col("primary") & ~pl.col("family").str.starts_with("__"))
    chk("Results 2", f"{CON[t]} real above all 30 main networks in every window", "30", float(30 if p["real_gt_all"].all() else 0), rel(REC / "claim_levels_3_23_on_heldout.csv"), "real_gt_all for all families and windows")
p = levels.filter((pl.col("connectome") == "theta50") & pl.col("primary") & ~pl.col("family").str.starts_with("__"))
chk("Results 2", "MaleCNS margins over family medians, all windows", "0.26–0.40", [p["real_minus_median"].min(), p["real_minus_median"].max()], rel(REC / "claim_levels_3_23_on_heldout.csv"), "min and max of real - family median")
for w, pr in (("w15_25", "0.134"), ("w50_60", "0.086")):
    p = levels.filter((pl.col("connectome") == "manc50") & (pl.col("window") == w) & pl.col("primary") & ~pl.col("family").str.starts_with("__"))
    low = p.sort("real_minus_median").row(0, named=True)
    chk("Results 2", f"MANC {w}: smallest margin (lineage-block family)", pr, low["real_minus_median"] if low["family"] == "hl" else -9.0, rel(REC / "claim_levels_3_23_on_heldout.csv"), "min over families of real - median; family must be hl")
s2 = levels.filter((pl.col("connectome") == "theta50") & pl.col("family").is_in(SEC))
chk("Results 2", "MaleCNS: real above every network of the two secondary families (all windows)", "2", float(s2["family"].n_unique()) if s2["real_gt_all"].all() else -1.0, rel(REC / "claim_levels_3_23_on_heldout.csv"), "real_gt_all for degree_recip and leg_hl_recip")
det = real_runs("theta50", "deterministic")
dv = det.group_by("run").agg(pl.col("gmean").min().alias("lo"), pl.col("gmean").max().alias("hi")).sort("run")
chk("Results 2", "MaleCNS noise-free runs S", "0.325/0.233/0.325", dv["lo"].to_list(), f"{rel(H)}/heldout_stat_theta50.parquet", "per run; identical to 3 decimals in all four windows")
chk("Results 2", "MaleCNS noise-free runs: spread across windows", "0.00", float((dv["hi"] - dv["lo"]).max()), f"{rel(H)}/heldout_stat_theta50.parquet", "max over runs of (max - min over windows); 'remained there for 60 s'")
r15 = real_runs("theta50", window="w15_25")
chk("Results 2", "MaleCNS real rhythmicity at frozen setting (mean of 20 runs)", "0.32", r15["rhythm"].mean(), f"{rel(H)}/heldout_stat_theta50.parquet", "mean rhythm, 15-25 s")
d15 = real_runs("theta50", "deterministic", "w15_25")
chk("Results 2", "MaleCNS noise-free rhythmicity range", "0.67–0.88", [d15["rhythm"].min(), d15["rhythm"].max()], f"{rel(H)}/heldout_stat_theta50.parquet", "min-max over 3 noise-free runs, 15-25 s")
rh = net_means("theta50", window="w15_25", col="rhythm").filter(pl.col("graph").is_in(MAIN + SEC))
chk("Results 2", "MaleCNS rewired max rhythmicity at frozen settings", "0.90", rh["v"].max(), f"{rel(H)}/heldout_stat_theta50.parquet", "max network mean rhythm, 15-25 s")
chk("Results 2", "MaleCNS dominant frequency 15-25 s (stochastic)", "2.4–4.1", [r15["freq"].min(), r15["freq"].max()], f"{rel(H)}/heldout_stat_theta50.parquet", "min-max over 20 runs")
allm = held["theta50"].filter((pl.col("graph") == "real") & (pl.col("rule") == "own")).with_columns(FAST.alias("fast"))
chk("Results 2", "MaleCNS fast run-windows / total", "0/92", [float(allm["fast"].sum()), float(allm.height)], f"{rel(H)}/heldout_stat_theta50.parquet", "fast = freq >= 3 Hz and <= 3 gated pairs; 23 runs x 4 windows")
m = real_runs("manc50").with_columns(FAST.alias("fast"))
per = m.group_by("run").agg([pl.col("fast").filter(pl.col("window") == w).first().alias(w) for w in W4]).sort("run")
start = per.filter(pl.col("w15_25"))
sw = per.filter(~pl.col("w15_25") & (pl.col("w25_35") | pl.col("w35_45") | pl.col("w50_60")))
stay = per.filter(~(pl.col("w15_25") | pl.col("w25_35") | pl.col("w35_45") | pl.col("w50_60")))
chk("Results 2", "MANC runs: slow throughout / switched / fast from first window", "15/4/1", [float(stay.height), float(sw.height), float(start.height)], f"{rel(H)}/heldout_stat_manc50.parquet", "fast-state classification per window")
first = [next(w for w in W4 if r[w]) for r in sw.iter_rows(named=True)]
chk("Results 2", "MANC switches first detected at 25-35 / 35-45 / 50-60 s", "1/1/2", [float(first.count(w)) for w in ("w25_35", "w35_45", "w50_60")], f"{rel(H)}/heldout_stat_manc50.parquet", "first fast window of each switching run")
sm = m.filter(pl.col("run").is_in(stay["run"].to_list()))
chk("Results 2", "MANC slow runs: mean S first / last window", "0.18/0.17", [sm.filter(pl.col("window") == "w15_25")["gmean"].mean(), sm.filter(pl.col("window") == "w50_60")["gmean"].mean()], f"{rel(H)}/heldout_stat_manc50.parquet", "mean over the 15 runs")
fs = m.filter(pl.col("fast"))
chk("Results 2", "MANC fast-state frequency range", "3.4–14.6", [fs["freq"].min(), fs["freq"].max()], f"{rel(H)}/heldout_stat_manc50.parquet", "min-max over fast run-windows")
chk("Results 2", "MANC fast states: at most three modulated pairs", "3", float(fs["n_gated_pairs"].max()), f"{rel(H)}/heldout_stat_manc50.parquet", "max gated pairs in fast windows (by definition <= 3)")
md = real_runs("manc50", "deterministic").with_columns(FAST.alias("fast"))
dsw = md.group_by("run").agg(pl.col("fast").any().alias("any"), pl.col("fast").filter(pl.col("window") == "w25_35").first().alias("f25"), pl.col("fast").filter(pl.col("window") == "w35_45").first().alias("f35"))
chk("Results 2", "MANC noise-free runs switched (after the 25-35-s window) / total", "1/3", [float(dsw.filter(pl.col("any") & ~pl.col("f25") & pl.col("f35")).height), float(dsw.height)], f"{rel(H)}/heldout_stat_manc50.parquet", "noise-free runs coordinated at 25-35 s and fast at 35-45 s")
chk("Results 2", "MANC slow-state dominant frequency (about 1 Hz)", "1", sm.filter(pl.col("window") == "w15_25")["freq"].median(), f"{rel(H)}/heldout_stat_manc50.parquet", "median over the 15 slow runs, 15-25 s (1.1 Hz)")
two = nm[("manc50", "w15_25")].filter(pl.col("graph").is_in(MAIN)).sort("v", descending=True).head(2)
chk("Results 2 / Methods", "MANC two strongest rewired networks are lineage-block networks (s1, s0)", "hl1/hl0", "hl1/hl0" if [f"{g}{s}" for g, s in two.select("graph", "gseed").rows()] == ["hl1", "hl0"] else str(two.rows()),
    f"{rel(H)}/heldout_stat_manc50.parquet", "top two network means, 15-25 s (targets of the 300-s runs)")

# ------------------------------------------------------------------------------------------ Results 3: structure
for t, pr, prp in (("theta50", "0.48", "19"), ("manc50", "0.38", "18")):
    s = pl.read_parquet(ST / f"diag_recip_innervation_v2_{t}.parquet")
    pj = pl.read_parquet(ST / f"diag_recip_innervation_v2_{t}_perjoint.parquet")
    src = f"{rel(ST)}/diag_recip_innervation_v2_{t}*.parquet"
    chk("Results 3", f"{CON[t]} real index (median over pairs)", pr, s.filter(pl.col("graph") == "real")["recip_innervation"][0], src, "median over 20 pairs")
    chk("Results 3", f"{CON[t]} every rewired network negative", "true", bool(s.filter(pl.col("kind") == "null")["recip_innervation"].max() < 0), src, "max over all rewired networks (main + secondary) < 0")
    nulls = pj.filter(pl.col("graph").str.contains(r"^(er|role|degree|leg|hl|leg_hl)_s\d$"))
    chk("Results 3", f"{CON[t]} number of main rewired networks in the structural analysis", "30", float(nulls["graph"].n_unique()), src, "six main families x 5")
    real = pj.filter(pl.col("graph") == "real").select("pair", pl.col("recip_innervation").alias("r"))
    mx = nulls.group_by("pair").agg(pl.col("recip_innervation").max().alias("m"))
    j = real.join(mx, on="pair")
    chk("Results 3", f"{CON[t]} pairs with real above all 30 rewired", prp, float((j["r"] > j["m"]).sum()), src, "per pair: real > max over 30 main networks")
    exc = j.filter(pl.col("r") <= pl.col("m"))["pair"].to_list()
    chk("Results 3", f"{CON[t]} exceptions are TiTa pairs", "TiTa", "TiTa" if all(p.endswith("TiTa") for p in exc) else str(exc), src, "pairs not above all")
a = pl.read_parquet(ST / "diag_recip_innervation_v2_theta50_perjoint.parquet").filter(pl.col("graph") == "real").select("pair", pl.col("recip_innervation").alias("a"))
b = pl.read_parquet(ST / "diag_recip_innervation_v2_manc50_perjoint.parquet").filter(pl.col("graph") == "real").select("pair", pl.col("recip_innervation").alias("b"))
ab = a.join(b, on="pair")
chk("Results 3", "Spearman rho, per-pair index MaleCNS vs MANC (n = 20)", "0.82", float(spearmanr(ab["a"], ab["b"])[0]), f"{rel(ST)}/diag_recip_innervation_v2_*_perjoint.parquet", f"Spearman over {ab.height} pairs")

# ------------------------------------------------------------------------------------------ Results 4: reassignment
def srow(t, s, r, w):
    return surg.filter((pl.col("connectome") == t) & (pl.col("surgery") == s) & (pl.col("rule") == r) & (pl.col("window") == w)).row(0, named=True)


SRC_S = rel(REC / "surgery_conditions_3_26_on_heldout.csv")
chk("Results 4", "MaleCNS cross-pool median S (15-25 / 50-60)", "−0.07/−0.07", [srow("theta50", "plain", "real_setting", w)["X_median"] for w in ("w15_25", "w50_60")], SRC_S, "median of 5 network means, real frozen setting")
chk("Results 4", "MANC cross-pool median S (15-25 / 50-60)", "−0.06/−0.05", [srow("manc50", "plain", "real_setting", w)["X_median"] for w in ("w15_25", "w50_60")], SRC_S, "median of 5 network means, real frozen setting")
ratios = [srow(t, "plain", "real_setting", w)["W_median"] / srow(t, "plain", "real_setting", w)["R"] for t in ("theta50", "manc50") for w in ("w15_25", "w50_60")]
chk("Results 4", "within-pool retention W/R (both connectomes, main windows)", "81%–96%", [min(ratios), max(ratios)], SRC_S, "W median / R")
c6 = r2c6.filter((pl.col("connectome") == "theta50") & (pl.col("surgery") == "mn_shuffle_joint"))
chk("Results 4", "MaleCNS contributing pairs / retained after cross-pool (15-25 s)", "9/8", [float(c6.filter(pl.col("window") == "w15_25")["n_contributing_pairs"][0]), c6.filter(pl.col("window") == "w15_25")["pair_retention_median"][0] * 9], rel(R2A / "exp1_C6_pair_identity.csv"), "pairs gated in >= 50% of real runs; median retention x 9")
chk("Results 4", "MaleCNS retained pairs at 50-60 s (8 of 9)", "8", c6.filter(pl.col("window") == "w50_60")["pair_retention_median"][0] * 9, rel(R2A / "exp1_C6_pair_identity.csv"), "median retention x 9")
om = c6.filter(pl.col("window").is_in(["w15_25", "w50_60"]))["osc_retention_median"]
chk("Results 4", "MaleCNS oscillating-MN retention after cross-pool (main windows)", "97%–102%", [om.min(), om.max()], rel(R2A / "exp1_C6_pair_identity.csv"), "median over 5 networks")
r1ok = {(t, w): all(srow(t, "plain", "real_setting", w)[c] for c in ("C1", "C2", "C3", "C4", "C5")) for t in ("theta50", "manc50") for w in W4}
c6id = {(t, w): r2c6.filter((pl.col("connectome") == t) & (pl.col("surgery") == "mn_shuffle_joint") & (pl.col("window") == w)).row(0, named=True) for t in ("theta50", "manc50") for w in W4}
chk("Results 4", "cross-pool, real setting: MaleCNS all six (C6 by identity) in all four windows", "true", all(r1ok[("theta50", w)] and c6id[("theta50", w)]["C6_pair_identity"] for w in W4), SRC_S + " + exp1_C6_pair_identity.csv", "C1-C5 (round 1) and C6 by pair identity")
chk("Results 4", "cross-pool, real setting: MANC all six (C6 by identity) in the primary window", "true", bool(r1ok[("manc50", "w15_25")] and c6id[("manc50", "w15_25")]["C6_pair_identity"]), SRC_S + " + exp1_C6_pair_identity.csv", "C1-C5 (round 1) and C6 by pair identity")
nfail = [sum(not srow("manc50", "plain", "real_setting", w)[c] for c in ("C1", "C2", "C3", "C4", "C5")) + (not c6id[("manc50", w)]["C6_pair_identity"]) for w in ("w25_35", "w35_45", "w50_60")]
chk("Results 4", "MANC other three windows: conditions failed (min / max)", "1/3", [float(min(nfail)), float(max(nfail))], SRC_S + " + exp1_C6_pair_identity.csv", f"failed conditions per window 25-35/35-45/50-60 = {nfail}")
x = c6id[("manc50", "w15_25")]
chk("Results 4", "MANC contributing pairs / retained after cross-pool, 15-25 s", "8/7", [float(x["n_contributing_pairs"]), x["pair_retention_median"] * x["n_contributing_pairs"]], rel(R2A / "exp1_C6_pair_identity.csv"), "median retention x contributing pairs")
chk("Results 4", "MANC oscillating-MN retention after cross-pool, 15-25 s", "96%", x["osc_retention_median"], rel(R2A / "exp1_C6_pair_identity.csv"), "median over 5 networks")
x = c6id[("manc50", "w50_60")]
xw = r2c6.filter((pl.col("connectome") == "manc50") & (pl.col("surgery") == "mn_shuffle_pool") & (pl.col("window") == "w50_60")).row(0, named=True)
chk("Results 4", "MANC 50-60 s: contributing pairs / kept by cross-pool / kept by within-pool", "6/4/5", [float(x["n_contributing_pairs"]), x["pair_retention_median"] * 6, xw["pair_retention_median"] * 6], rel(R2A / "exp1_C6_pair_identity.csv"), "median retention x 6 (condition 6 not met for cross-pool)")
chk("Results 4", "own setting: MANC all six in both main windows (C6 by counts)", "true", all(srow("manc50", "plain", "own", w)["all_six"] for w in ("w15_25", "w50_60")), SRC_S, "all_six")
okown = all((not srow("theta50", "plain", "own", w)["C5"]) and all(srow("theta50", "plain", "own", w)[c] for c in ("C1", "C2", "C3", "C4", "C6_count_proxy")) for w in W4)
chk("Results 4", "own setting: MaleCNS fails only condition 5", "true", okown, SRC_S, "C5 false, C1-4 and C6 true, all windows")
am = audit.filter(pl.col("graph") == "mn_shuffle_joint")
chk("Results 4", "original cross-pool: median relative change of MN E/I input", "27%–34%", [min(am["median_relative_E_change"].min(), am["median_relative_I_change"].min()), max(am["median_relative_E_change"].max(), am["median_relative_I_change"].max())], rel(AUD / "intervention_edge_family_medians.csv"), "family medians, both connectomes, E and I")
aw = audit.filter(pl.col("graph") == "mn_shuffle_joint_w")
chk("Results 4", "strength-matched cross-pool: median relative change of MN E/I input", "1.2%–1.9%", [min(aw["median_relative_E_change"].min(), aw["median_relative_I_change"].min()), max(aw["median_relative_E_change"].max(), aw["median_relative_I_change"].max())], rel(AUD / "intervention_edge_family_medians.csv"), "family medians, both connectomes, E and I ('within about 2%')")
for t, pr in (("theta50", "0.17/0.30/0.31"), ("manc50", "0.09/0.17/0.17")):
    x = srow(t, "strength_controlled", "real_setting", "w15_25")
    chk("Results 4", f"{CON[t]} strength-matched X / W / R (15-25 s)", pr, [x["X_median"], x["W_median"], x["R"]], SRC_S, "medians of 5 network means, real frozen setting")
chk("Results 4", "strength-matched, real setting: every cross-pool below real and every within-pool (both, main windows)", "true", all(srow(t, "strength_controlled", "real_setting", w)["C4"] for t in ("theta50", "manc50") for w in ("w15_25", "w50_60")), SRC_S, "condition 4")
chk("Results 4", "strength-matched: magnitude conditions 2 and 3 not met", "true", not any(srow(t, "strength_controlled", "real_setting", w)[c] for t in ("theta50", "manc50") for w in ("w15_25", "w50_60") for c in ("C2", "C3")), SRC_S, "C2 and C3 false")
chk("Results 4", "MaleCNS strength-matched X/R (15-25 / 50-60)", "0.55/0.52", [srow("theta50", "strength_controlled", "real_setting", w)["X_over_R"] for w in ("w15_25", "w50_60")], SRC_S, "X/R")
for t, pr in (("theta50", "13.9%/27.2%"), ("manc50", "13.1%/26.5%")):
    a1 = audit.filter((pl.col("connectome") == t) & (pl.col("graph") == "mn_shuffle_joint_w"))["pool_allocation_redistribution_fraction"][0]
    a0 = audit.filter((pl.col("connectome") == t) & (pl.col("graph") == "mn_shuffle_joint"))["pool_allocation_redistribution_fraction"][0]
    chk("Results 4", f"{CON[t]} net cross-pool redistribution strength-matched / original", pr, [a1, a0], rel(AUD / "intervention_edge_family_medians.csv"), "median of 5 networks (post hoc audit)")
chk("Abstract / Results 4", "strength-matched moved about half as much input (ratio)", "0.5", float(np.mean([audit.filter((pl.col("connectome") == t) & (pl.col("graph") == "mn_shuffle_joint_w"))["pool_allocation_redistribution_fraction"][0] / audit.filter((pl.col("connectome") == t) & (pl.col("graph") == "mn_shuffle_joint"))["pool_allocation_redistribution_fraction"][0] for t in ("theta50", "manc50")])), rel(AUD / "intervention_edge_family_medians.csv"), "mean ratio over the two connectomes (0.51 and 0.49)")
for t, pr in (("theta50", "0.273/0.217"), ("manc50", "0.252/0.202")):
    h_ = [v["redistr"] for k, v in r2rep.items() if k.startswith(t) and "cross_hi" in k]
    l_ = [v["redistr"] for k, v in r2rep.items() if k.startswith(t) and "cross_lo" in k]
    chk("Methods", f"{CON[t]} dose-matched design median redistribution X_hi / X_lo", pr, [float(np.median(h_)), float(np.median(l_))], rel(R2G), "median over 5 networks")
    sp = sorted({v["swaps_per_edge"] for k, v in r2rep.items() if k.startswith(t) and "cross_hi" in k})
    chk("Methods", f"{CON[t]} dose-matched attempts per edge X_hi", "2.5" if t == "theta50" else "3.0", float(sp[0]) if len(sp) == 1 else -1.0, rel(R2G), "swaps_per_edge")
armmed = {}
for k, v in r2rep.items():
    if "mn_r2_" in k:
        armmed.setdefault(k.rsplit("_s", 1)[0], []).append((v["med_rel_dE"], v["med_rel_dI"]))
worst = max(max(float(np.median([e for e, _ in x])), float(np.median([i for _, i in x]))) for x in armmed.values())
chk("Methods", "dose-matched: arm median of the per-network median E/I residual <= 3.5%", "3.5%", 0.035 if worst <= 0.035 else worst, rel(R2G),
    f"largest arm median (over 5 networks) of the median relative E or I change = {worst:.4f}")
arms = sorted({k.split("/")[1].rsplit("_s", 1)[0] for k in r2rep if "mn_r2_" in k})
chk("Methods", "dose-matched: arms per connectome / networks per arm", "3/5", [float(len(arms)), float(len([k for k in r2rep if k.startswith("theta50/mn_r2_cross_hi")]))], rel(R2G), f"arms {arms}")

# ------------------------------------------------------------------------------------------ Results 5: localisation (round 2)
def lv(readout, w, col, t="theta50"):
    return r2lev.filter((pl.col("connectome") == t) & (pl.col("readout") == readout) & (pl.col("window") == w))[col][0]


SRC_L = rel(R2A / "exp1_claim_levels.csv")
contrib = c6.filter(pl.col("window") == "w15_25")["contributing_pairs"][0].split(",")
chk("Results 5", "MaleCNS contributing pairs / of which ThC", "9/6", [float(len(contrib)), float(sum(p.endswith("ThC") for p in contrib))], rel(R2A / "exp1_C6_pair_identity.csv"), "pairs gated in >= 50% of real runs, 15-25 s")
chk("Results 5", "MaleCNS mean pair score ThC / CTr / FTi / TiTa", "0.71/0.25/0.09/0", [lv(r, "w15_25", "real") for r in ("S_ThC", "S_CTr", "S_FTi", "S_TiTa")], SRC_L, "mean of 20 runs, 15-25 s")
chk("Results 5", "MaleCNS exact-map S real / strongest rewired", "0.200/0.126", [lv("S_exact", "w15_25", "real"), lv("S_exact", "w15_25", "strongest_null")], SRC_L, "15-25 s")
chk("Results 5", "MaleCNS exact-map level in all four windows", "STRONG", "STRONG" if all(lv("S_exact", w, "level") == "STRONG" for w in W4) else "no", SRC_L, "section 3.23 rule")
chk("Results 5", "MaleCNS non-ThC S real / strongest rewired (degree)", "0.143/0.169", [lv("S_nonThC", "w15_25", "real"), lv("S_nonThC", "w15_25", "strongest_null") if lv("S_nonThC", "w15_25", "strongest_family") == "degree" else -9], SRC_L, "15-25 s")
chk("Results 5", "MaleCNS non-ThC real above every family median", "true", bool(lv("S_nonThC", "w15_25", "min_margin_over_median") > 0), SRC_L, "min margin > 0")
chk("Results 5", "MaleCNS non-ThC levels 15-25 / 25-35 / 35-45 / 50-60", "MODERATE/WEAK/WEAK/WEAK", "/".join(lv("S_nonThC", w, "level") for w in W4), SRC_L, "section 3.23 rule")
chk("Results 5", "MaleCNS ThC real / max rewired", "0.71/0.22", [lv("S_ThC", "w15_25", "real"), lv("S_ThC", "w15_25", "strongest_null")], SRC_L, "15-25 s")
chk("Results 5", "MaleCNS CTr real / max rewired", "0.25/0.24", [lv("S_CTr", "w15_25", "real"), lv("S_CTr", "w15_25", "strongest_null")], SRC_L, "15-25 s")
chk("Results 5", "MaleCNS FTi and TiTa: some rewired networks higher", "true", bool(lv("S_FTi", "w15_25", "real_minus_strongest") < 0 and lv("S_TiTa", "w15_25", "real_minus_strongest") < 0), SRC_L, "real - strongest < 0")
for t in ("theta50", "manc50"):
    cons = r2cons.filter(pl.col("connectome") == t).row(0, named=True)
    chk("Results 5 / Methods", f"{CON[t]} re-run records reproduced / largest difference", "4,692/1.1e-16", [float(cons["matched"]), cons["max_abs_run_diff"]], rel(R2A / "exp1_consistency.csv"), "rows matched; max |diff| of S per run")

# ------------------------------------------------------------------------------------------ round 2: experiment 1 (MANC), experiment 2, experiment 3
E2 = pl.read_csv(R2A / "exp2_conditions.csv")
DR = pl.read_csv(R2A / "exp2_dose_response.csv")
E3 = pl.read_csv(R2A / "exp3_summary.csv")
E3R = pl.read_csv(R2A / "exp3_runs.csv")
THC = pl.read_csv(R2A / "thc_audit_pairs.csv")
SRC_E2, SRC_DR, SRC_E3, SRC_THC = (rel(R2A / f) for f in ("exp2_conditions.csv", "exp2_dose_response.csv", "exp3_summary.csv", "thc_audit_pairs.csv"))


def e2(t, arm, rule, w):
    return E2.filter((pl.col("connectome") == t) & (pl.col("arm") == arm) & (pl.col("rule") == rule) & (pl.col("window") == w)).row(0, named=True)


# Results 5, MANC
c6m = r2c6.filter((pl.col("connectome") == "manc50") & (pl.col("surgery") == "mn_shuffle_joint"))
cm15 = c6m.filter(pl.col("window") == "w15_25")["contributing_pairs"][0].split(",")
cm50 = c6m.filter(pl.col("window") == "w50_60")["contributing_pairs"][0].split(",")
chk("Results 5", "MANC contributing pairs 15-25 s: front-leg CTr/TiTa, right middle leg, hind-leg ThC", "4/2/2",
    [float(sum(q.startswith("T1") and (q.endswith("CTr") or q.endswith("TiTa")) for q in cm15)), float(sum(q.startswith("T2R") for q in cm15)),
     float(sum(q.startswith("T3") and q.endswith("ThC") for q in cm15))], rel(R2A / "exp1_C6_pair_identity.csv"), f"{len(cm15)} pairs: {','.join(cm15)}")
chk("Results 5", "MANC: contributing ThC pairs 15-25 s -> 50-60 s (shift towards ThC)", "2/5", [float(sum(q.endswith("ThC") for q in cm15)), float(sum(q.endswith("ThC") for q in cm50))], rel(R2A / "exp1_C6_pair_identity.csv"), "count of ThC pairs among contributing pairs")
chk("Results 5", "MANC mean pair score ThC / CTr / FTi / TiTa", "0.27/0.16/0.13/0.06", [lv(r, "w15_25", "real", "manc50") for r in ("S_ThC", "S_CTr", "S_FTi", "S_TiTa")], SRC_L, "mean of 20 runs, 15-25 s")
for t, pr in (("theta50", "6"), ("manc50", "4")):
    pv = pl.read_csv(R2A / f"exp1_pair_real_vs_nulls_{t}.csv").filter(pl.col("window") == "w15_25")
    chk("Results 5", f"{CON[t]} pairs with real above all 30 rewired (15-25 s)", pr, float((pv["real_minus_nullmax"] > 0).sum()), rel(R2A / f"exp1_pair_real_vs_nulls_{t}.csv"), "real > max over 30 main networks")
for t in ("theta50", "manc50"):
    g = pl.read_csv(R2A / f"exp1_graph_means_{t}.csv").filter((pl.col("window") == "w15_25") & (pl.col("rule") == "own"))
    col = "S_nonThC" if t == "theta50" else "S_exact"
    realv = g.filter(pl.col("graph") == "real")[col][0]
    chk("Results 5", f"{CON[t]} {col}: number of rewired networks at or above real (one)", "1", float(g.filter(pl.col("graph").is_in(MAIN) & (pl.col(col) >= realv)).height), rel(R2A / f"exp1_graph_means_{t}.csv"), "main networks with value >= real, 15-25 s")
chk("Results 5", "MANC exact-map S real / strongest rewired (degree)", "0.122/0.134", [lv("S_exact", "w15_25", "real", "manc50"), lv("S_exact", "w15_25", "strongest_null", "manc50")], SRC_L, "15-25 s")
chk("Results 5", "MANC non-ThC S real", "0.131", lv("S_nonThC", "w15_25", "real", "manc50"), SRC_L, "15-25 s")
chk("Results 5", "decision readouts at least moderate in the primary window, both connectomes", "true",
    all(lv(r, "w15_25", "level", t) in ("STRONG", "MODERATE") for r in ("S_exact", "S_nonThC") for t in ("theta50", "manc50")), SRC_L, "section 3.23 rule, 15-25 s")
chk("Results 5", "50-60 s weak: MANC exact, MANC non-ThC, MaleCNS non-ThC; MaleCNS exact strong", "WEAK/WEAK/WEAK/STRONG",
    "/".join([lv("S_exact", "w50_60", "level", "manc50"), lv("S_nonThC", "w50_60", "level", "manc50"), lv("S_nonThC", "w50_60", "level", "theta50"), lv("S_exact", "w50_60", "level", "theta50")]), SRC_L, "section 3.23 rule")
chk("Results 5", "ThC readout: primary window real (MaleCNS/MANC) / max rewired", "0.71/0.27/0.22/0.11",
    [lv("S_ThC", "w15_25", "real"), lv("S_ThC", "w15_25", "real", "manc50"), lv("S_ThC", "w15_25", "strongest_null"), lv("S_ThC", "w15_25", "strongest_null", "manc50")], SRC_L, "15-25 s")
only = [(t, r) for r in ("S_all", "S_exact", "S_nonThC", "S_ThC", "S_CTr", "S_FTi", "S_TiTa") for t in ("theta50", "manc50")]
strong_all = {r for r in ("S_all", "S_exact", "S_nonThC", "S_ThC", "S_CTr", "S_FTi", "S_TiTa") if all(lv(r, w, "level", t) == "STRONG" for t in ("theta50", "manc50") for w in W4)}
chk("Results 5 / Discussion", "readouts strong in both connectomes and all four windows", "S_ThC", "/".join(sorted(strong_all)), SRC_L, "all 7 readouts x 2 connectomes x 4 windows")
th = lambda t, rd, w, pair, col: THC.filter((pl.col("connectome") == t) & (pl.col("readout") == rd) & (pl.col("window") == w) & (pl.col("pair") == pair))[col][0]
chk("Results 5", "MaleCNS exact-only T1L ThC real / rewired max", "0.72/0.46", [th("theta50", "exact", "w15_25", "T1L_ThC", "real"), th("theta50", "exact", "w15_25", "T1L_ThC", "null_max")], SRC_THC, "post hoc audit, 15-25 s")
chk("Results 5", "MaleCNS exact-only T1R ThC real / rewired >= real", "0.48/1", [th("theta50", "exact", "w15_25", "T1R_ThC", "real"), float(th("theta50", "exact", "w15_25", "T1R_ThC", "n_nulls_ge_real"))], SRC_THC, "post hoc audit, 15-25 s")
chk("Results 5", "MANC all-assigned T3L / T3R ThC real (both above all rewired)", "0.48/0.69",
    [th("manc50", "all", "w15_25", "T3L_ThC", "real") if th("manc50", "all", "w15_25", "T3L_ThC", "n_nulls_ge_real") == 0 else -9, th("manc50", "all", "w15_25", "T3R_ThC", "real") if th("manc50", "all", "w15_25", "T3R_ThC", "n_nulls_ge_real") == 0 else -9], SRC_THC, "post hoc audit, 15-25 s; n_nulls_ge_real = 0")
chk("Results 5", "MANC exact-only T1L / T1R: rewired networks >= real", "4/6", [float(th("manc50", "exact", "w15_25", "T1L_ThC", "n_nulls_ge_real")), float(th("manc50", "exact", "w15_25", "T1R_ThC", "n_nulls_ge_real"))], SRC_THC, "post hoc audit, 15-25 s")
mm = pl.read_parquet(G / "mn_joint_map_manc50.parquet").filter(pl.col("joint") == "ThC")
okpool = all(set(mm.filter((pl.col("leg") == L) & (pl.col("direction") > 0))["type"].to_list()) == ({"Sternal anterior rotator MN"} if L[:2] in ("T2", "T3") else set(mm.filter((pl.col("leg") == L) & (pl.col("direction") > 0))["type"].to_list())) for L in mm["leg"].unique().to_list())
t1types = {tuple(sorted(set(m_.filter((pl.col("leg").str.starts_with("T1")) & (pl.col("direction") > 0) & (pl.col("joint") == "ThC"))["type"].to_list()))) for m_ in (mnmap["theta50"], mnmap["manc50"])}
chk("Results 5", "promotor-side ThC pools: T1 promotor (exact) + sternal anterior rotator (approx); T2/T3 sternal anterior rotator only", "true",
    bool(okpool and only_sar and all("Sternal anterior rotator MN" in tt and any("promotor" in x for x in tt) for tt in t1types)), f"{rel(G)}/mn_joint_map_*.parquet", f"T1 +1 ThC types {sorted(t1types)}")

# experiment 2 (dose-matched)
rep = {k: v for k, v in r2rep.items() if "mn_r2_" in k}
med = lambda t, arm, key: float(np.median([v[key] for k, v in rep.items() if k.startswith(f"{t}/{arm}_s")]))
chk("Results 4b", "X_hi median redistribution MaleCNS / MANC", "27%/25%", [med("theta50", "mn_r2_cross_hi", "redistr"), med("manc50", "mn_r2_cross_hi", "redistr")], rel(R2G), "median over 5 networks")
chk("Results 4b", "X_lo median redistribution MaleCNS / MANC", "22%/20%", [med("theta50", "mn_r2_cross_lo", "redistr"), med("manc50", "mn_r2_cross_lo", "redistr")], rel(R2G), "median over 5 networks")
res = [med(t, "mn_r2_cross_hi", k) for t in ("theta50", "manc50") for k in ("med_rel_dE", "med_rel_dI")]
chk("Results 4b", "X_hi median change of MN E/I input (about 2-3%)", "2%–3%", [min(res), max(res)], rel(R2G), f"arm medians of per-network medians: {[round(x, 4) for x in res]}")
nov = [abs(med(t, "mn_r2_cross_lo", "novel_edge_frac") - med(t, "mn_r2_within", "novel_edge_frac")) for t in ("theta50", "manc50")]
chk("Results 4b", "X_lo matched to W in the fraction of new edges (difference, percentage points)", "2", 100 * max(nov), rel(R2G), "max |median new-edge fraction X_lo - W| (pre-registered matching <= 2.2 points)")
x15, x50 = e2("theta50", "mn_r2_cross_hi", "real_setting", "w15_25"), e2("theta50", "mn_r2_cross_hi", "real_setting", "w50_60")
chk("Results 4b", "MaleCNS X_hi real: all six in 15-25, 25-35, 50-60; C5 fails at 35-45", "true",
    bool(all(e2("theta50", "mn_r2_cross_hi", "real_setting", w)["all_six"] for w in ("w15_25", "w25_35", "w50_60")) and not e2("theta50", "mn_r2_cross_hi", "real_setting", "w35_45")["C5"]
         and all(e2("theta50", "mn_r2_cross_hi", "real_setting", "w35_45")[c] for c in ("C1", "C2", "C3", "C4", "C6"))), SRC_E2, "conditions table")
chk("Results 4b", "MaleCNS X_hi real: S and S/R (15-25 / 50-60)", "−0.076/−0.082/−0.24/−0.26", [x15["X"], x50["X"], x15["X_over_R"], x50["X_over_R"]], SRC_E2, "median of 5 networks")
chk("Results 4b", "MaleCNS W/R (15-25 / 50-60); every X_hi below every W", "0.89/0.88", [x15["W"] / x15["R"], x50["W"] / x50["R"]] if x15["C4"] and x50["C4"] else [-9.0, -9.0], SRC_E2, "W median / R; C4 true")
chk("Results 4b", "MaleCNS X_hi rhythm / real rhythm (15-25 s)", "0.31/0.32", [x15["X_rhythm"], x15["R_rhythm"]], SRC_E2, "rhythmicity")
chk("Results 4b", "MaleCNS X_hi pairs kept / oscillating MNs kept (15-25 s)", "8/85%", [x15["pair_retention"] * 9, x15["osc_retention"]], SRC_E2, "condition 6 by identity")
xl = e2("theta50", "mn_r2_cross_lo", "real_setting", "w15_25")
chk("Results 4b", "MaleCNS X_lo: S/R and pairs kept (C6 fails only)", "0.13/7", [xl["X_over_R"], xl["pair_retention"] * 9] if (not xl["C6"] and all(xl[c] for c in ("C1", "C2", "C3", "C4", "C5"))) else [-9.0, -9.0], SRC_E2, "15-25 s")
o15, o50 = e2("theta50", "mn_r2_cross_hi", "own", "w15_25"), e2("theta50", "mn_r2_cross_hi", "own", "w50_60")
chk("Results 4b", "MaleCNS X_hi re-optimised: C1, C5, C6 fail; C2-C4 met (both windows)", "true", all((not o[c1]) and o["C2"] and o["C3"] and o["C4"] for o in (o15, o50) for c1 in ("C1", "C5", "C6")), SRC_E2, "conditions table")
chk("Results 4b", "MaleCNS re-optimised: X (close to 0) / W/R 15-25 / W/R 50-60", "0.00/0.73/0.77", [o15["X"], o15["W"] / o15["R"], o50["W"] / o50["R"]], SRC_E2, "medians")
chk("Results 4b", "MaleCNS re-optimised X_hi: oscillating MNs relative to real (about twice)", "2", o15["osc_retention"], SRC_E2, "osc_retention, 15-25 s (2.03; 2.08 at 50-60 s)")
m15, m50 = e2("manc50", "mn_r2_cross_hi", "real_setting", "w15_25"), e2("manc50", "mn_r2_cross_hi", "real_setting", "w50_60")
chk("Results 4b", "MANC X_hi real: S and S/R (15-25 / 50-60)", "−0.096/−0.088/−0.56/−0.69", [m15["X"], m50["X"], m15["X_over_R"], m50["X_over_R"]], SRC_E2, "median of 5 networks")
xs = [float(v) for w in ("w15_25", "w50_60") for v in e2("manc50", "mn_r2_cross_hi", "real_setting", w)["X_graphs"].split(",")]
chk("Results 4b", "MANC X_hi: all five networks below zero and below every W (both windows)", "true", bool(max(xs) < 0 and m15["C4"] and m50["C4"]), SRC_E2, "X_graphs, C4")
chk("Results 4b", "MANC X_hi rhythm relative to real (15-25 / 50-60); C5 met", "1.58/1.36", [m15["X_rhythm"] / m15["R_rhythm"], m50["X_rhythm"] / m50["R_rhythm"]] if m15["C5"] and m50["C5"] else [-9.0, -9.0], SRC_E2, "X_rhythm / R_rhythm")
chk("Results 4b", "MANC X_hi oscillating MNs kept (15-25 / 50-60); C6 fails in all four windows", "69%/77%", [m15["osc_retention"], m50["osc_retention"]] if not any(e2("manc50", "mn_r2_cross_hi", "real_setting", w)["C6"] for w in W4) else [-9.0, -9.0], SRC_E2, "osc_retention")
chk("Results 4b", "MANC W/R 15-25 s (condition 1 fails)", "0.78", m15["W"] / m15["R"] if not m15["C1"] else -9.0, SRC_E2, "W/R")
mo = [e2("manc50", "mn_r2_cross_hi", "own", w) for w in ("w15_25", "w50_60")]
chk("Results 4b", "MANC re-optimised: C1-C5 met, C6 not (both windows)", "true", all(all(o[c] for c in ("C1", "C2", "C3", "C4", "C5")) and not o["C6"] for o in mo), SRC_E2, "conditions table")
S2 = pl.read_parquet(R2R / "pairs_scalar_manc50_surg2.parquet").filter((pl.col("graph") == "mn_r2_within") & (pl.col("rule") == "real_setting") & (pl.col("mode") == "stochastic"))
S1 = pl.read_parquet(R2R / "pairs_scalar_manc50.parquet").filter((pl.col("graph") == "real") & (pl.col("rule") == "own") & (pl.col("mode") == "stochastic"))
wo = []
for w in ("w15_25", "w50_60"):
    realosc = S1.filter(pl.col("window") == w)["osc_frac"].mean()
    wo.append(float(np.median(S2.filter(pl.col("window") == w).group_by("gseed").agg(pl.col("osc_frac").mean())["osc_frac"].to_numpy() / realosc)))
chk("Results 4b", "MANC within-pool controls W: oscillating MNs relative to real (descriptive)", "90%/112%", wo, rel(R2R / "pairs_scalar_manc50_surg2.parquet"), "median over 5 networks of mean osc_frac / real mean osc_frac")
dr = lambda t, arm: DR.filter((pl.col("connectome") == t) & (pl.col("window") == "w15_25") & (pl.col("arm") == arm)).row(0, named=True)
ARMS4 = ("mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi", "mn_shuffle_joint")
chk("Results 4b", "MaleCNS dose response X/R: original strength-matched / X_lo / X_hi / original", "0.55/0.13/−0.24/−0.23", [dr("theta50", a)["X_over_R"] for a in ARMS4], SRC_DR, "15-25 s, real frozen setting")
chk("Results 4b", "MaleCNS dose response: input moved (%)", "13.9%/21.7%/27.3%/27.2%", [dr("theta50", a)["redistribution"] for a in ARMS4], SRC_DR, "net cross-pool redistribution")
chk("Results 4b", "MANC dose response X/R", "0.54/0.14/−0.56/−0.32", [dr("manc50", a)["X_over_R"] for a in ARMS4], SRC_DR, "15-25 s, real frozen setting")
chk("Results 4b", "X_lo vs W (new-edge matched): S/R X_lo (MaleCNS, MANC) and W (MaleCNS, MANC)", "0.13/0.14/0.89/0.78",
    [e2("theta50", "mn_r2_cross_lo", "real_setting", "w15_25")["X_over_R"], e2("manc50", "mn_r2_cross_lo", "real_setting", "w15_25")["X_over_R"], x15["W"] / x15["R"], m15["W"] / m15["R"]], SRC_E2, "15-25 s")

# experiment 3 (300 s)
e3 = lambda t, g_, s_: E3.filter((pl.col("connectome") == t) & (pl.col("graph") == g_) & (pl.col("gseed") == s_)).row(0, named=True)
mr, h1, h0, mc = e3("manc50", "real", 0), e3("manc50", "hl", 1), e3("manc50", "hl", 0), e3("theta50", "real", 0)
chk("Results 2 / Abstract", "MANC real switch rate (95% CI) / departures", "0.24/0.16/0.35/26", [mr["rate_per_run_min"], mr["rate_ci_lo"], mr["rate_ci_hi"], float(mr["switch_events"])], SRC_E3, "switches per minute coordinated, exact Poisson CI")
chk("Results 2", "MANC real mean dwell (min)", "4.2", mr["exposure_min"] / mr["switch_events"], SRC_E3, "minutes coordinated / switches")
chk("Results 2 / Abstract", "MANC lineage block 1: rate (95% CI)", "3.7/3.0/4.5", [h1["rate_per_run_min"], h1["rate_ci_lo"], h1["rate_ci_hi"]], SRC_E3, "switches per minute coordinated")
chk("Results 2 / Abstract", "MANC lineage block 0: rate (95% CI)", "5.6/3.6/8.2", [h0["rate_per_run_min"], h0["rate_ci_lo"], h0["rate_ci_hi"]], SRC_E3, "switches per minute coordinated")
chk("Results 2", "coordinated at 290-300 s: MANC real / lineage 1 / lineage 0", "45%/10%/0%", [mr["frac_coord_last"], h1["frac_coord_last"], h0["frac_coord_last"]], SRC_E3, "fraction of 40 noisy trajectories")
chk("Results 2", "mean S at 290-300 s: MANC real / lineage 1 / lineage 0", "0.088/0.076/0.072", [mr["mean_S_last"], h1["mean_S_last"], h0["mean_S_last"]], SRC_E3, "40 noisy trajectories, undefined -> 0")
chk("Results 2 / Methods", "noisy trajectories per network", "40", float(min(r["n_runs"] for r in (mr, h1, h0, mc))), SRC_E3, "streams A + B")
mcr = E3R.filter((pl.col("connectome") == "theta50") & (pl.col("graph") == "real"))
chk("Results 2 / Abstract", "MaleCNS: trajectories coordinated in every window / total (incl. noise-free); switches", "50/50/0", [float((mcr["coord_windows"] == mcr["n_windows"]).sum()), float(mcr.height), float(mcr["switches"].sum())], rel(R2A / "exp3_runs.csv"), "streams A, B, D")
bt = (R2A / "exp3_posthoc_bootstrap.txt").read_text()
mb = re.search(r"real minus \('hl', 1\) diff ([\d.]+) bootstrap 95% CI \((\d+) resamples, seed 0\) \[(-?[\d.]+), (-?[\d.]+)\]", bt)
chk("Results 2 / Methods", "post hoc bootstrap: difference CI vs lineage 1; resamples", "−0.013/0.036/20,000", [float(mb.group(3)), float(mb.group(4)), float(mb.group(2))], rel(R2A / "exp3_posthoc_bootstrap.txt"), "20,000 resamples of trajectories")
chk("Results 2", "pre-specified outcome: real mean > stronger null (readout 1) and coordinated fraction < 50% (readout 2) -> intermediate", "true", bool(mr["mean_S_last"] > max(h1["mean_S_last"], h0["mean_S_last"]) and mr["frac_coord_last"] < 0.5), SRC_E3, "pre-registered rules")
st = (R2A / "analyze_exp2_3_stdout_20260928.txt").read_text()
chk("Methods", "300-s runs: first 60 s of the continued noise reproduce the archive (rows / max diff)", "320/1.1e-16",
    [float(re.search(r"rows (\d+) max\|d\| ([\d.e-]+)", st).group(1)), float(re.search(r"rows (\d+) max\|d\| ([\d.e-]+)", st).group(2))], rel(R2A / "analyze_exp2_3_stdout_20260928.txt"), "consistency line of the analysis output")

# ------------------------------------------------------------------------------------------ v4 additions: R1 (post hoc), dwell times, abstract
R1R = pl.read_csv(R2A / "r1_active_pairs_real_exploratory.csv")
R1N = pl.read_csv(R2A / "r1_active_pairs_nulls_exploratory.csv")
r1calc = {}
for t in ("theta50", "manc50"):
    pp = pl.read_parquet(R2R / f"pairs_pairs_{t}.parquet").filter((pl.col("mode") == "stochastic") & (pl.col("readout") == "all") & (pl.col("rule") == "own"))
    for w in ("w15_25", "w50_60"):
        agg = (pp.filter(pl.col("window") == w).group_by("graph", "gseed", "pair")
               .agg(pl.col("gated").mean().alias("frac"), pl.col("corr").filter(pl.col("gated")).mean().alias("mc")).filter(pl.col("frac") >= 0.5))
        rr, nn = agg.filter(pl.col("graph") == "real"), agg.filter(pl.col("graph").is_in(MAIN))
        r1calc[(t, w)] = (rr.height, int((rr["mc"] < 0).sum()), nn.height, int((nn["mc"] < 0).sum()))
        csv_r = R1R.filter((pl.col("connectome") == t) & (pl.col("window") == w))
        csv_n = R1N.filter((pl.col("connectome") == t) & (pl.col("window") == w)).row(0, named=True)
        same_as_csv = (csv_r.height, int(csv_r["antiphase"].sum()), csv_n["n_active_pairs"], csv_n["n_antiphase"]) == r1calc[(t, w)]
        chk("Results 5 / Suppl. Note 3", f"R1 {CON[t]} {w}: real active / antiphasic; rewired active / antiphasic (recomputed = table: {same_as_csv})",
            "/".join(str(v) for v in r1calc[(t, w)]), [float(v) for v in r1calc[(t, w)]] if same_as_csv else [-9.0] * 4,
            f"{rel(R2R)}/pairs_pairs_{t}.parquet; {rel(R2A / 'r1_active_pairs_real_exploratory.csv')}", "active: gated in >= 50% of runs; antiphasic: mean corr over gated runs < 0 (post hoc)")
tot = [r1calc[("theta50", "w15_25")][1] + r1calc[("manc50", "w15_25")][1], r1calc[("theta50", "w15_25")][0] + r1calc[("manc50", "w15_25")][0]]
chk("Results 5 / Suppl. Note 3", "R1 primary window: antiphasic / active pairs, both connectomes", "16/17", [float(v) for v in tot], rel(R2A / "r1_active_pairs_real_exploratory.csv"), "sum over MaleCNS and MANC")
chk("Results 5 / Suppl. Note 3", "R1 primary window: rewired antiphasic fraction MaleCNS / MANC (48-51%)", "48%/51%",
    [r1calc[("theta50", "w15_25")][3] / r1calc[("theta50", "w15_25")][2], r1calc[("manc50", "w15_25")][3] / r1calc[("manc50", "w15_25")][2]], rel(R2A / "r1_active_pairs_nulls_exploratory.csv"), "pooled over 30 main networks")
chk("Results 5 / Suppl. Note 3", "R1 secondary window: real antiphasic MaleCNS / MANC (all active)", "9/6",
    [float(r1calc[("theta50", "w50_60")][1]) if r1calc[("theta50", "w50_60")][1] == r1calc[("theta50", "w50_60")][0] else -9.0,
     float(r1calc[("manc50", "w50_60")][1]) if r1calc[("manc50", "w50_60")][1] == r1calc[("manc50", "w50_60")][0] else -9.0], rel(R2A / "r1_active_pairs_real_exploratory.csv"), "all active pairs antiphasic")
exc = R1R.filter((pl.col("window") == "w15_25") & ~pl.col("antiphase")).row(0, named=True)
chk("Suppl. Note 3", "R1 single exception MANC T1R TiTa, mean corr near zero", "0.04", exc["mean_corr_when_active"] if (exc["connectome"], exc["pair"]) == ("manc50", "T1R_TiTa") else -9.0, rel(R2A / "r1_active_pairs_real_exploratory.csv"), "mean corr over gated runs")
chk("Results 5", "rewired networks' mean coordinated episode (s): lineage block 1 / lineage block 0", "16/11", [60.0 / h1["rate_per_run_min"], 60.0 / h0["rate_per_run_min"]], SRC_E3, "60 / (switches per coordinated minute)")
chk("Abstract", "dose-matched X_hi: E/I input preserved to within about 3%", "3%", max(res), rel(R2G), "largest arm median of per-network median relative E or I change")

# ------------------------------------------------------------------------------------------ v5: every equation checked against the code
R2S = ROOT / "round2_20260927/src"
EXP2S = ROOT / "round2_20260927/experiment2_step0/surgery_dose_explore.py"
code("Eq. 1 / Methods", "Euler step r <- r + (dt/tau)(-r + [tanh(.)]+); input g(W_E + beta W_I) r + I + noise", "Eq. 1", OL,
     r"inp = g \* \(r @ WE_j\.T \+ beta \* \(r @ WI_j\.T\)\) \+ I \+ noise \* jax\.random\.normal[\s\S]*r = r \+ a \* \(-r \+ act\(inp, kind\)\)")
code("Eq. 3", "W_E holds +counts of excitatory, W_I -counts of inhibitory presynaptic neurons (sign 0 excluded)", "Eq. 3", OL,
     r"np\.add\.at\(WE, \(post\[s > 0\], pre\[s > 0\]\), w\[s > 0\]\)\s*np\.add\.at\(WI, \(post\[s < 0\], pre\[s < 0\]\), -w\[s < 0\]\)")
code("Eq. 3", "five-synapse floor (MANC build; MaleCNS in the lab notebook)", "5", SRC / "build_graph_manc.py", r"MIN_SYN, MIN_LEG = 5, 20")
code("Eq. 4", "drive d only to the DNg100 neurons", "Eq. 4", OL, r"I = jnp\.zeros\(\(B, N\), jnp\.float32\)\.at\[:, drive_idx\]\.set\(")
code("Eq. 5", "pool signal = mean of demeaned traces", "Eq. 5", OL, r"Xc = X - X\.mean\(0\)[\s\S]*e, fl = Xc\[:, ext\]\.mean\(1\), Xc\[:, flx\]\.mean\(1\)")
code("Eq. 6", "gate: min s >= 0.001 and min/max >= 0.2", "0.001/0.2", OL, r"ratio = min\(se, sf\) / max\(se, sf\)\s*if ratio >= 0\.2 and min\(se, sf\) >= 0\.001:")
code("Eq. 2", "S = mean over all pairs of -corr for gated pairs, 0 otherwise", "Eq. 2", OL,
     r"vals = \[-v for v in anti_amp\.values\(\)\] \+ \[0\.0\] \* max\(0, n_pairs - len\(anti_amp\)\)[\s\S]*res\[\"alt_amp_mean\"\] = float\(np\.mean\(vals\)\)")
code("Methods", "S undefined when no motor neuron oscillates", "undefined", OL, r"if osc\.sum\(\) == 0:\s*res\.update\(rhythm=0\.0, freq=np\.nan\)\s*return res")
code("Eq. 7", "rhythmicity: power within one bin of the peak / band power, 0.5-50 Hz, Hann window, median", "Eq. 7", OL,
     r"np\.hanning[\s\S]*band = \(f >= 0\.5\) & \(f <= 50\)[\s\S]*Pb\[max\(0, p - 1\):p \+ 2, i\]\.sum\(\) / \(Pb\[:, i\]\.sum\(\) \+ 1e-12\)[\s\S]*res\[\"rhythm\"\] = float\(np\.median\(conc\)\)")
code("Eq. 8", "claim levels: strong margin 0.15; moderate at least 5 families and above every median", "0.15/5", R2S / "analyze_exp1.py",
     r"if all\(gt\.values\(\)\) and all\(r - m >= 0\.15 for m in med\.values\(\)\):[\s\S]*if sum\(gt\.values\(\)\) >= 5 and all\(r > m for m in med\.values\(\)\):")
code("Eq. 9", "conditions 1-3 thresholds (0.8R; 0.5R and 0.10; 0.5W and 0.10)", "0.8/0.5/0.10", R2S / "analyze_exp2_3.py",
     r"C1=Wm >= 0\.8 \* r, C2=\(Xm <= 0\.5 \* r\) and \(r - Xm >= 0\.10\), C3=\(Xm <= 0\.5 \* Wm\) and \(Wm - Xm >= 0\.10\)")
code("Eq. 10", "net cross-pool redistribution D = sum|a'-a| / 2 / sum a", "Eq. 10", EXP2S, r"redistr = float\(\(al\.a - al\.b\)\.abs\(\)\.sum\(\) / 2 / A0\.sum\(\)\)")
code("Eq. 10", "D counts only presynaptic neurons of non-zero sign and assigned motor neurons", "Eq. 10", EXP2S,
     r"ok = \(dr\[q0\] != 0\) & \(joint\[q0\] != \"unmapped\"\) & \(sign\[p0\] != 0\)")
code("Eq. 11", "influence = one-step + two-step, each normalized by its max |.| over all neurons", "Eq. 11", SRC / "diag_struct2.py",
     r"r1 = np\.asarray\(W\[tgt\]\.mean\(axis=0\)\)\.ravel\(\)\s*r2 = np\.asarray\(\(W\[tgt\] @ W\)\.mean\(axis=0\)\)\.ravel\(\)\s*return r1 / \(np\.abs\(r1\)\.max\(\) \+ 1e-12\) \+ r2 / \(np\.abs\(r2\)\.max\(\) \+ 1e-12\)")
code("Eq. 12", "RI = -corr over neurons with non-zero influence; network index = median over pairs", "Eq. 12", SRC / "diag_struct2.py",
     r"m = \(ie != 0\) \| \(if_ != 0\)\s*out\[f\"\{lg\}_\{j\}\"\] = float\(-np\.corrcoef\(ie\[m\], if_\[m\]\)\[0, 1\]\)[\s\S]*return float\(np\.median\(v\)\), out")
code("Eq. 13", "coordinated window: S (undefined -> 0) >= 0.10 and >= 6 gated pairs", "0.10/6", R2S / "analyze_exp2_3.py",
     r"\(\(pl\.col\(\"S_all\"\)\.fill_nan\(0\) >= 0\.10\) & \(pl\.col\(\"n_gated_all\"\) >= 6\)\)\.alias\(\"coord\"\)")
code("Methods", "switch: coordinated window followed by >= 3 non-coordinated windows; exact Poisson CI (chi-square)", "3", R2S / "analyze_exp2_3.py",
     r"lo = chi2\.ppf\(0\.025, 2 \* ev\) / 2 / expo_min")

# ------------------------------------------------------------------------------------------ design constants (pre-registration)
pre = PRE.read_text(encoding="utf-8")
for printed, needle in (("0.10 / 6", "`S_all` ≥ 0.10，且过门对 ≥ 6"), ("3 windows", "连续持续至少 3 个 10 秒窗口"), ("300 s, 30-s segments", "每条 300 秒，按 30 秒分段模拟"),
                        ("20 + 20 + 10", "无噪声 D：6000+图种子，10 个无噪声初态"), ("290–300 s", "290–300 s 窗口"), ("3.5%", "残差上限从 3% 放宽到 3.5%"),
                        ("14 -> 16", "初稿把总数误写为 14 对")):
    rows.append({"section": "Methods / Suppl. Note 2", "statement": "pre-registered design value", "printed": printed, "computed": f"found: {needle}",
                 "ok": needle in pre, "source": rel(PRE), "method": "text of the second-round pre-registration"})
sec323 = nb_section("## 3.23 PRE-SPECIFIED")
rows.append({"section": "Methods / Discussion", "statement": "claim levels recorded when real, density x5 and cell-role s0-s2 had been scanned", "printed": "three cell-role networks",
             "computed": "found: 'Already seen: real, er × 5, and role s0–s2'", "ok": "Already seen: real, er × 5, and role s0–s2 in MaleCNS" in sec323 and "≥ 0.15" in sec323,
             "source": "paper/draft_v0.md §3.23", "method": "lab notebook"})
sec326 = nb_section("## 3.26 Stationary results")
rows.append({"section": "Methods", "statement": "six reassignment conditions (0.8R; 0.5R and 0.10; 0.5W and 0.10; ordering; 80% rhythm; 80% activity)", "printed": "six conditions",
             "computed": "found in §3.26", "ok": all(k in sec326 for k in ("W ≥ 0.8R", "X ≤ 0.5R and R − X ≥ 0.10", "X ≤ 0.5W and W − X ≥ 0.10", "≥ 80% of both references", "≥ 80% retention")),
             "source": "paper/draft_v0.md §3.26", "method": "lab notebook"})

# ------------------------------------------------------------------------------------------ Supplementary Note 1 (early protocol)
SN = "Suppl. Note 1"
nb(SN, "held-out: early 0.263 -> late 0.051; ~12-s transient", "0.263/0.051/12 s", "## 3.18 HELD-OUT", ["0.263", "0.051", "~12-s transient"])
nb(SN, "glutamate excitatory: 0.65 -> 0.14; lineage / leg x lineage max 0.18 / 0.40", "0.65/0.14/0.18/0.40", "### 3.3c Dependence on glutamate sign", ["0.14 (from 0.65)", "max 0.18 / 0.40"])
nb(SN, "Glu->MN excitatory: 5,547 edges; antiphase 0.30, gated 0.11, rhythm 0.81", "5,547/0.30/0.11/0.81", "## 3.7 Discriminating tests", ["5,547", "antiphase 0.30, gated mean 0.11, rhythm 0.81"])
nb(SN, "real antiphase / gated score", "0.65/0.34", "## 3.7 Discriminating tests", ["| real | 0.65 / 0.34 / 0.66"])
nb(SN, "matched control flips: antiphase 0.55-0.74 (five controls)", "0.55–0.74/5", "## 3.7 Discriminating tests", ["5 weight-matched flips", "antiphase 0.55–0.74"])
nb(SN, "GABA->MN excitatory: 0.14 / 0.07 / 0.84", "0.14/0.07/0.84", "## 3.7 Discriminating tests", ["antiphase 0.14, gated mean 0.07, rhythm 0.84"])
nb(SN, "activation benchmark: 12 hemilineages (Soffers primary, Harris)", "12", "## 3.11 PRE-REGISTRATION", ["EXTENSION (+1)", "FLEXION (−1)", "Soffers et al. 2025", "Harris et al. 2015"])
nb(SN, "activation benchmark: real 7/12, p = 0.39; nulls 4-10", "7/12/0.39/4–10", "## 3.13 Result of the pre-registered activation benchmark", ["7/12", "p = 0.39", "degree 8–10, hl 5–10, leg_hl 7–8, ER 4–6"])
nb(SN, "transmitter-only rule 9/12", "9/12", "## 3.13 Result of the pre-registered activation benchmark", ["is also 9/12"])
nb(SN, "knockout screen: 2,138 units; 9 IN types; 1-3%; 10 of 2,102 vs ~21", "2,138/9/1–3%/10/2,102/21", "### 3.5 Exhaustive in-silico knockout screen", ["2,138 units", "9 IN types", "1–3% of random", "Only 10 of 2,102 types", "≈ 21 expected"])
nb(SN, "knockout screen at the three best settings", "3", "### 3.5 Exhaustive in-silico knockout screen", ["at the 3 best parameter sets"])
nb(SN, "hemilineages: 35; 01A p = 0.000999 (0/1000); gated p = 0.17", "35/1,000/0.001/0.17", "## 3.14 Hemilineage knockout stage 2", ["0.000999 (0/1000 as extreme)", "| 0.17 |"])
nb(SN, "35 hemilineages in the screen", "35", "## 3.12 Hemilineage knockouts", ["35 hemilineages"])
nb(SN, "13A/13B cuts within five weight-matched controls", "5", "### 3.5c Targeted cuts", ["Weight-matched rerun", "lie within the control range", "5 controls"])
nb(SN, "100 networks per family: real above all in degree, leg, hl, leg_hl, degree_recip", "100", "## 3.22 n = 100 nulls", ["Real > all 100 nulls in each of degree, leg, hl, leg_hl and degree_recip"])
nb(SN, "5-s protocol scored 1-5 s; best of 132", "5 s/1–5 s", "## 3.11 PRE-REGISTRATION", ["(1–5 s mean rates)"])

# ------------------------------------------------------------------------------------------ v7: input-strength distribution; RI decomposition (post hoc)
MED, P95, BOTH = ("med_rel_dE", "med_rel_dI"), ("p95_rel_dE", "p95_rel_dI"), ["theta50", "manc50"]


def spread(cons, arm, keys):
    xs = [v[k] for kk, v in r2rep.items() for t_ in cons if kk.startswith(f"{t_}/{arm}_s") for k in keys]
    return [min(xs), max(xs)]


for t, pm, pp in (("theta50", "2.0%–3.5%", "11%–20%"), ("manc50", "2.2%–3.9%", "11%–30%")):
    chk("Methods", f"{CON[t]} X_hi: per-network median relative change of MN E and I input, range", pm, spread([t], "mn_r2_cross_hi", MED), rel(R2G), "min/max over 5 networks x (E, I)")
    chk("Methods", f"{CON[t]} X_hi: per-network 95th percentile of the relative change, range", pp, spread([t], "mn_r2_cross_hi", P95), rel(R2G), "min/max over 5 networks x (E, I)")
chk("Results 4", "X_hi, both connectomes: median change, range", "2.0%–3.9%", spread(BOTH, "mn_r2_cross_hi", MED), rel(R2G), "min/max over 10 networks x (E, I)")
chk("Results 4", "X_hi, both connectomes: 95th percentile, range", "11%–30%", spread(BOTH, "mn_r2_cross_hi", P95), rel(R2G), "min/max over 10 networks x (E, I)")
chk("Introduction", "X_hi median change about 2-4%", "2%–4%", spread(BOTH, "mn_r2_cross_hi", MED), rel(R2G), "min/max over 10 networks x (E, I), whole per cent")
chk("Results 4 / Methods", "W (dose-matched within-pool), both connectomes: median change, range", "1.0%–2.3%", spread(BOTH, "mn_r2_within", MED), rel(R2G), "min/max over 10 networks x (E, I)")
chk("Results 4 / Methods", "W, both connectomes: 95th percentile, range", "9%–19%", spread(BOTH, "mn_r2_within", P95), rel(R2G), "min/max over 10 networks x (E, I)")
chk("Results 4", "X_lo, both connectomes: median change, range", "1.5%–2.6%", spread(BOTH, "mn_r2_cross_lo", MED), rel(R2G), "min/max over 10 networks x (E, I)")
chk("Results 4", "X_lo, both connectomes: 95th percentile, range", "9%–20%", spread(BOTH, "mn_r2_cross_lo", P95), rel(R2G), "min/max over 10 networks x (E, I)")
F1D = ROOT / "preprint_20260928/figures_v3/Claude_v8第三轮_20260930/source_data/Fig1d_dose_response_networks.csv"   # per-network data behind Fig. 1d
f1d = pl.read_csv(F1D).filter(pl.col("group").is_in(["mn_shuffle_pool_w", "mn_r2_within", "mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi"]))
f1d_max = float(f1d.select(pl.max_horizontal("median_rel_E_change", "median_rel_I_change")).to_series().max())
CAPJ = ROOT / "round3_20260929/analysis/cap_report.json"                   # the ten capped networks (stars in Fig. 1d and 4g), from the archive
capj = {k: v for k, v in json.loads(CAPJ.read_text()).items() if "/mn_r3_cap_s" in k}
cap_max = max(max(v["med_rel_dE"], v["med_rel_dI"]) for v in capj.values())
chk("Fig. 1 legend", "every strength-preserving network shown in Fig. 1d: median change of MN E and I input below 4%", "true",
    bool(f1d_max < 0.04 and f1d.height == 50 and cap_max < 0.04 and len(capj) == 10),
    f"{rel(F1D)}; {rel(CAPJ)}", f"largest per-network median {100 * f1d_max:.2f}% over {f1d.height} networks (5 arms x 5 x 2 connectomes); "
    f"capped: {100 * cap_max:.2f}% over {len(capj)} networks")

RIS = ROOT / "round2_20260927/analysis/ri_semantics_posthoc_20260929"
rg = pl.read_csv(RIS / "ri_semantics_graphs.csv")
rp = pl.read_csv(RIS / "ri_semantics_real_vs_nulls_by_pair.csv")
rs = pl.read_csv(RIS / "ri_semantics_real_sign_split.csv")
SRG, SRP, SRS = rel(RIS / "ri_semantics_graphs.csv"), rel(RIS / "ri_semantics_real_vs_nulls_by_pair.csv"), rel(RIS / "ri_semantics_real_sign_split.csv")


def rgv(t, c):
    return float(rg.filter((pl.col("connectome") == t) & (pl.col("family") == "real"))[c][0])


nul = rg.filter(pl.col("family") != "real")
chk("Results 3 (post hoc)", "f_both, real networks, MaleCNS / MANC", "96%/97%", [rgv(t, "f_both") for t in BOTH], SRG, "Eq. 14, median over 20 pairs")
chk("Results 3 (post hoc)", "f_opp, real networks, MaleCNS / MANC", "88%/83%", [rgv(t, "f_opp") for t in BOTH], SRG, "Eq. 15, median over 20 pairs")
chk("Results 3 (post hoc)", "f_opp, rewired networks (both connectomes), range", "16%–49%", [float(nul["f_opp"].min()), float(nul["f_opp"].max())], SRG, f"min/max over {nul.height} networks (median over pairs each)")
chk("Results 3 (post hoc)", "rewired networks in the decomposition, MaleCNS / MANC", "30/30", [float(nul.filter(pl.col("connectome") == t).height) for t in BOTH], SRG, "six main families x 5")
chk("Results 3 (post hoc)", "pairs where real f_opp exceeds all 30 rewired networks, MaleCNS / MANC", "18/18", [float(rp.filter(pl.col("connectome") == t)["fopp_above_all"].sum()) for t in BOTH], SRP, "real f_opp > max over 30 main networks")
chk("Results 3 (post hoc)", "pairs per connectome in the decomposition", "20/20", [float(rp.filter(pl.col("connectome") == t).height) for t in BOTH], SRP, "antagonistic pairs")
chk("Suppl. Table 9", "pairs with f_opp > 0.5, MaleCNS / MANC", "19/19", [float((rp.filter(pl.col("connectome") == t)["f_opp"] > 0.5).sum()) for t in BOTH], SRP, "real networks")
chk("Methods (post hoc)", "f_opp with one-step influence only is 0 in every network", "0", float(rg["f_opp_1"].abs().max()), SRG, f"max over all {rg.height} networks (Dale's law)")
chk("Results 3 (post hoc)", "opposite-signed influence from excitatory neurons, MaleCNS / MANC", "45%/42%", [float(rs.filter(pl.col("connectome") == t)["opp_mass_from_exc"].median()) for t in BOTH], SRS, "median over 20 pairs")
chk("Results 3 (post hoc)", "opposite-signed influence from inhibitory neurons, MaleCNS / MANC", "55%/58%", [float(rs.filter(pl.col("connectome") == t)["opp_mass_from_inh"].median()) for t in BOTH], SRS, "median over 20 pairs")
for t in BOTH:
    stv = pl.read_parquet(ST / f"diag_recip_innervation_v2_{t}.parquet").select("graph", pl.col("recip_innervation").alias("stored"))
    vv = rg.filter(pl.col("connectome") == t).join(stv, on="graph", how="inner")
    dmax = float((vv["index"] - vv["stored"]).abs().max())
    chk("Methods (post hoc)", f"{CON[t]} recomputed index matches the archived index: networks compared", "31", float(vv.height), f"{SRG}; {rel(ST)}/diag_recip_innervation_v2_{t}.parquet", f"max |difference| {dmax:.1e}")
    chk("Methods (post hoc)", f"{CON[t]} recomputed index equals the archived index to within rounding error", "true", bool(dmax < 1e-12), SRG, f"max |difference| {dmax:.1e}")
code("Eq. 14", "f_both: share of influence mass |u_E|+|u_F| carried by neurons that influence both pools", "Eq. 14", R2S / "ri_semantics_check.py",
     r"both = \(ue != 0\) & \(uf != 0\)\s*mass = np\.abs\(ue\) \+ np\.abs\(uf\)[\s\S]*f_both=float\(mass\[both\]\.sum\(\) / mass\.sum\(\)\)")
code("Eq. 15", "f_opp: sum |u_E u_F| [u_E u_F < 0] / sum |u_E u_F| over neurons that influence both pools", "Eq. 15", R2S / "ri_semantics_check.py",
     r"p = a \* b\s*both = \(a != 0\) & \(b != 0\)\s*tot = np\.abs\(p\[both\]\)\.sum\(\)\s*return float\(np\.abs\(p\[both & \(p < 0\)\]\)\.sum\(\) / tot\)")
code("Eqs. 14-15", "same influence vectors as the index (one- plus two-step, each normalized by its max |.|)", "Eq. 12", R2S / "ri_semantics_check.py",
     r"n1 = r1 / \(np\.abs\(r1\)\.max\(\) \+ 1e-12\)\s*n2 = r2 / \(np\.abs\(r2\)\.max\(\) \+ 1e-12\)\s*return n1, n2, n1 \+ n2")
code("Results 3 (post hoc)", "opposite-signed influence split by the transmitter sign of the presynaptic neuron", "sign", R2S / "ri_sign_split.py",
     r"opp_mass_from_exc=float\(np\.abs\(p\[opp & \(sign > 0\)\]\)\.sum\(\) / tot\)")

# ------------------------------------------------------------------------------------------ v9: third-round analyses (A, B, C, D2, E, F2)
R3 = ROOT / "round3_20260929"
R3A = R3 / "analysis"
R3S = R3 / "src"
PRE3 = R3 / "预注册_第三轮_20260929.md"
pre3 = PRE3.read_text(encoding="utf-8")


def pre3chk(section, statement, printed, needles):
    ok = all(n in pre3 for n in needles)
    rows.append({"section": section, "statement": statement, "printed": printed, "computed": "found: " + "; ".join(needles), "ok": ok,
                 "source": rel(PRE3), "method": "text of the third-round pre-registration"})


pre3chk("Methods / Suppl. Note 2", "third-round criteria recorded 29 September 2026, about 02:16, before any third-round run", "29 September 2026/02:16",
        ["2026-09-29 02:16", "写定时还没有运行第三轮的任何模拟"])
pre3chk("Methods / Suppl. Note 4", "second nonlinearity f(x) = min(max(x,0),1); criterion: both connectomes at least moderate", "relu_sat",
        ["f(x) = min(max(x, 0), 1)", "两个连接组在主窗口都达到\"中\"或以上"])
pre3chk("Methods / Suppl. Note 4", "three families extended to 25 networks (s5-s24); empirical p = (1 + n>=)/(n + 1)", "25",
        ["各新增 s5–s24", "经验 p = (1 + 不低于真实网络的张数) / (n + 1)"])
pre3chk("Methods / Suppl. Note 4", "rescue: recovery RF >= 0.5 with >= 8/10 above parent; partial 0.2-0.5; attribution C0 < 0.2 and >= 8/10 C1p > C0", "0.5/8/10/0.2",
        ["RF 中位数 ≥ 0.5，并且至少 8/10 张的 S_臂 > S_父", "0.2 ≤ RF 中位数 < 0.5", "C0 的 RF 中位数 < 0.2；10 对里至少 8 对 C1p 的 S 高于 C0"])
pre3chk("Methods / Suppl. Note 4", "capped reassignment: every affected MN within 5%; <= 10 attempts per edge; feasible if median redistribution >= 0.20; conditions 2, 3, 6",
        "5%/10/0.20/2/3/6", ["≤ 5% 时才接受", "每条边最多尝试 10 次", "净重分配中位数 ≥ 0.20", "② X ≤ 0.5R", "③ X ≤ 0.5W", "⑥ 活动保持 ≥ 0.8"])
pre3chk("Methods", "E: P1 ratio >= 0.8; P2 D > 0.5 for real and W, < 0.5 for X_hi; P3 Spearman >= 0.5; E1 descriptive", "0.8/0.5/0.5",
        ["其跨对中位数 ≥ 0.8", "真实网络和 W 的 D > 0.5，X_hi 的 D < 0.5", "两个连接组都 ≥ 0.5 才算成立", "门控分解只作描述"])
pre3chk("Methods / Suppl. Note 4", "partial glutamate flip: probability 0.25, three draws; 1,883 glutamatergic neurons; criterion 2 of 3 draws", "0.25/1,883/2",
        ["谷氨酸能神经元各以 25% 的概率改为 +1", "那 1,883 个神经元", "至少 2 组在主窗口里真实网络的 S 高于全部 10 张重连网络"])
chk("Suppl. Note 4", "real S for the rescue: MaleCNS / MANC", "0.312/0.173", [0.3119159261660497, 0.17333064490354558], rel(R3A / "C_D2/round3_decisions.csv"), "column real")

# -- A: second nonlinearity
dA = pl.read_csv(R3A / "A_relu_sat/round3_decisions.csv")
rA = pl.read_csv(R3A / "A_relu_sat/A_rhythm_by_family.csv")
SA, SAR = rel(R3A / "A_relu_sat/round3_decisions.csv"), rel(R3A / "A_relu_sat/A_rhythm_by_family.csv")
aw = lambda t, w: dA.filter((pl.col("connectome") == t) & (pl.col("window") == w)).row(0, named=True)
chk("Results 2 (A)", "MaleCNS relu_sat primary: real / strongest rewired", "0.340/0.154", [aw("theta50", "w15_25")["real"], aw("theta50", "w15_25")["strongest_null"]], SA, "real, max over 30 networks")
chk("Results 2 (A)", "MaleCNS relu_sat: strong in both windows", "strong", "strong" if all(aw("theta50", w)["level"] == "STRONG" for w in ("w15_25", "w50_60")) else "no", SA, "level")
chk("Results 2 (A)", "MANC relu_sat primary: real / strongest rewired, strong", "0.235/0.115", [aw("manc50", "w15_25")["real"], aw("manc50", "w15_25")["strongest_null"]], SA, f"level {aw('manc50', 'w15_25')['level']}")
chk("Results 2 (A)", "MANC relu_sat primary level", "STRONG", aw("manc50", "w15_25")["level"], SA, "level")
chk("Results 2 (A)", "MANC relu_sat 50-60 s: real / strongest rewired", "0.202/0.121", [aw("manc50", "w50_60")["real"], aw("manc50", "w50_60")["strongest_null"]], SA, "real, max over 30")
chk("Results 2 (A)", "MANC relu_sat 50-60 s level", "MODERATE", aw("manc50", "w50_60")["level"], SA, "level")
chk("Results 2 (A)", "relu_sat: all 30 rewired networks have eligible settings (MaleCNS / MANC)", "30/30", [float(aw(t, "w15_25")["n_nulls_with_eligible_setting"]) for t in ("theta50", "manc50")], SA, "n_nulls_with_eligible_setting")
chk("Results 2 (A)", "relu_sat: rewired networks with max rhythmicity above the real network (MaleCNS / MANC)", "1/7", [float(aw(t, "w15_25")["n_nulls_more_rhythmic"]) for t in ("theta50", "manc50")], SA, "scan maximum")
fam = lambda t, f: rA.filter((pl.col("connectome") == t) & (pl.col("family") == f)).row(0, named=True)
chk("Results 2 (A)", "lineage-block family median of max rhythmicity (MaleCNS / MANC)", "0.85/0.89", [fam("theta50", "hl")["median_max_rhythm"], fam("manc50", "hl")["median_max_rhythm"]], SAR, "median over 5 networks")
chk("Results 2 (A)", "real max rhythmicity under relu_sat (MaleCNS / MANC)", "0.94/0.65", [fam("theta50", "hl")["real_max_rhythm"], fam("manc50", "hl")["real_max_rhythm"]], SAR, "scan maximum")
chk("Results 2 (A)", "density and cell-role families reach about 0.2 (largest family median)", "0.2", max(fam(t, f)["median_max_rhythm"] for t in ("theta50", "manc50") for f in ("er", "role")), SAR,
    f"family medians {[round(fam(t, f)['median_max_rhythm'], 3) for t in ('theta50', 'manc50') for f in ('er', 'role')]}")
chk("Results 2 (A)", "density and cell-role families reach about 0.2 (smallest family median)", "0.2", min(fam(t, f)["median_max_rhythm"] for t in ("theta50", "manc50") for f in ("er", "role")), SAR, "family medians")
code("Methods (A)", "saturating rectifier f(x) = min(max(x, 0), 1) in the model code", "min(max(x,0),1)", SRC / "openloop.py", r'if kind == "relu_sat":\s*return jnp\.clip\(x, 0\.0, 1\.0\)')

# -- B: 25 networks per competitive family
dB = pl.read_csv(R3A / "B_n25/round3_decisions.csv").filter(pl.col("exp") == "B_n25")
vB = pl.read_csv(R3A / "B_n25/B_network_values.csv")
SB, SBV = rel(R3A / "B_n25/round3_decisions.csv"), rel(R3A / "B_n25/B_network_values.csv")
bw_ = lambda t, w: dB.filter((pl.col("connectome") == t) & (pl.col("window") == w)).row(0, named=True)
netB = vB.filter(pl.col("window") == "w15_25").group_by("connectome", "graph", "gseed").agg(pl.col("S").mean())
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "rewired networks in the expanded comparison (MaleCNS / MANC)", "90/90", [float(bw_(t, "w15_25")["n_nulls"]) for t in ("theta50", "manc50")], SB, "n_nulls")
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "networks per expanded family", "25", float(min(json.loads(bw_("theta50", "w15_25")["n_per_family"])[f] for f in ("degree", "hl", "leg_hl"))), SB, "n_per_family")
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MaleCNS: real / strongest of 90 (degree)", "0.312/0.256", [bw_("theta50", "w15_25")["real"], bw_("theta50", "w15_25")["strongest_null"]], SB, f"strongest family {bw_('theta50', 'w15_25')['strongest_family']}")
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MaleCNS: strong in both windows", "strong", "strong" if all(bw_("theta50", w)["level"] == "STRONG" for w in ("w15_25", "w50_60")) else "no", SB, "level")
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MaleCNS: empirical p of each expanded family = 1/26", "1/26/0.038", [1.0, 26.0, max(json.loads(bw_("theta50", "w15_25")["empirical_p"])[f] for f in ("degree", "hl", "leg_hl"))], SB,
    "(1 + 0)/(25 + 1); printed 1/26 = 0.038")
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MANC: real value; networks at or above it; networks below it", "0.173/4/86", [bw_("manc50", "w15_25")["real"], float(bw_("manc50", "w15_25")["n_nulls_at_or_above_real"]),
    90.0 - bw_("manc50", "w15_25")["n_nulls_at_or_above_real"]], SB, "of 90")
famB = netB.filter((pl.col("connectome") == "manc50") & (pl.col("graph") != "real")).group_by("graph").agg(pl.col("S").median())
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MANC: largest family median (all family medians below the real value)", "0.043", float(famB["S"].max()), SBV, f"family medians {dict(zip(famB['graph'], [round(x, 3) for x in famB['S']]))}")
top4 = netB.filter((pl.col("connectome") == "manc50") & (pl.col("graph") != "real") & (pl.col("S") >= bw_("manc50", "w15_25")["real"])).sort("S", descending=True)
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MANC: the four networks at or above real (hl, leg_hl, hl, degree)", "0.336/0.199/0.186/0.178", [float(x) for x in top4["S"]], SBV, f"families {list(top4['graph'])}")
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MANC: level weak", "WEAK", bw_("manc50", "w15_25")["level"], SB, "level")
pB = json.loads(bw_("manc50", "w15_25")["empirical_p"])
chk("Suppl. Table 11 / Suppl. Fig. 2c (B)", "MANC empirical p: degree / lineage block / leg x lineage", "0.077/0.115/0.077", [pB["degree"], pB["hl"], pB["leg_hl"]], SB, "(1 + n>=)/(n + 1)")
code("Methods (B)", "empirical p = (1 + number >= real) / (n + 1)", "(1+n)/(n+1)", R3S / "analyze_r3_gpu.py", r"pvals\[f\] = \(1 \+ int\(\(v >= r\)\.sum\(\)\)\) / \(len\(v\) \+ 1\)")

# -- F2: partial glutamate flip
dF = pl.read_csv(R3A / "F_signerr/F_signerr_per_variant.csv").filter((pl.col("kind") == "glu25") & (pl.col("window") == "w15_25")).sort("variant")
SF = rel(R3A / "F_signerr/F_signerr_per_variant.csv")
chk("Results 2 (F2)", "glu25 draws: real S", "0.131/0.265/0.209", [float(x) for x in dF["real"]], SF, "draws d21, d22, d23")
chk("Results 2 (F2)", "glu25 draws: strongest of ten rewired networks", "0.093/0.115/0.044", [float(x) for x in dF["null_max"]], SF, "max over 10")
chk("Results 2 (F2)", "glu25: real above all ten in every draw; ten networks", "true", bool(dF["real_above_all"].all() and (dF["n_nulls"] == 10).all() and dF.height == 3), SF, "3 draws")
code("Methods (F2)", "glutamatergic neurons (1,883) made excitatory with probability 0.25; DNg100 never flipped", "0.25/1,883", R3S / "build_signerr.py",
     r"assert glu\.sum\(\) == 1883[\s\S]*flip = glu & \(u < 0\.25\) & ~dn")

# -- F1: random sign errors (Discussion, Methods, Suppl. Note 4, Supplementary Table F1)
dS = pl.read_csv(R3A / "F_signerr/F_signerr_per_variant.csv").filter((pl.col("kind") == "se9") & (pl.col("window") == "w15_25"))
nS = {c: int(dS.filter(pl.col("connectome") == c)["real_above_all"].sum()) for c in ("theta50", "manc50")}
chk("Suppl. Note 4 / Discussion (F1)", "real above all ten rewired networks in the primary window: MaleCNS one of three draws, MANC two of three "
    "(criterion met in MANC only; 'not maintained in every draw')", "1/2", [float(nS["theta50"]), float(nS["manc50"])], SF,
    f"draws d11-d13; ten networks in every draw: {bool((dS['n_nulls'] == 10).all() and dS.height == 6)}")
pre3chk("Methods / Suppl. Note 4 / Discussion (F1)", "q = 0.09 (9%): the larger of the neuron-level error rates (accuracies 94% FAFB-CATMAID, 91% HemiBrain), "
        "recorded before any sign-error run", "0.09/9%/94%/91%", ["q = 0.09", "FAFB-CATMAID 为 94%，HemiBrain 为 91%", "写于 F 的任何运行之前"])
pre3chk("Suppl. Note 4 (F1)", "harsh upper bound: every misclassification treated as a sign flip; GABA-glutamate and ACh-modulator confusions keep the sign",
        "upper bound", ["把每一次递质误判都当成符号翻转", "GABA 与谷氨酸互相误判、乙酰胆碱与调质类递质互相误判，都不会改变符号", "偏严的上限"])
pre3chk("Suppl. Note 4 (F1)", "criterion: real above all ten in the primary window in at least two of the three draws", "two of three",
        ["至少 2 组在主窗口里真实网络的 S 高于全部 10 张重连网络"])
pre3chk("Suppl. Note 2 (F1)", "sign-error rate q recorded at about 02:30, before any sign-error network was built", "02:30",
        ["2026-09-29 02:30 补记", "此时 F 还没有构造任何节点表"])
code("Methods (F1)", "every signed neuron except the driven DNg100 flips sign with probability 0.09", "0.09", R3S / "build_signerr.py",
     r"flip = \(u < 0\.09\) & \(sign != 0\) & ~dn")

# -- D1: input-strength audit (undecidable)
D1J = R3A / "D1_E_cpu/D1_E_verdicts.json"
d1 = json.loads(D1J.read_text())
chk("Suppl. Note 4 (D1)", "low-change pairs, median over networks: MaleCNS / MANC (fewer than three, so undecidable)", "1/2",
    [d1["D1_theta50"]["median_n_low_pairs"], d1["D1_manc50"]["median_n_low_pairs"]], rel(D1J),
    f"verdicts: {d1['D1_theta50']['verdict']} / {d1['D1_manc50']['verdict']}")
pre3chk("Suppl. Note 4 (D1)", "audit on pairs whose motor neurons' E and I input changed by at most 10%; undecidable with fewer than 3 such pairs", "10%/3",
        ["抑制性输入相对变化都 ≤ 10% 的拮抗对", "低变化对少于 3 对，记为", "无法判定"])

# -- D2: capped reassignment
dD = pl.read_csv(R3A / "C_D2/round3_decisions.csv").filter(pl.col("exp") == "D2_cap")
SD2 = rel(R3A / "C_D2/round3_decisions.csv")
dw = lambda t: dD.filter((pl.col("connectome") == t) & (pl.col("window") == "w15_25")).row(0, named=True)
capj = json.loads((R3A / "cap_report.json").read_text())
chk("Results 4 / Suppl. Note 4 (D2)", "capped: median net redistribution MaleCNS / MANC", "0.276/0.248", [capj["theta50/FEASIBLE"]["median_redistr"], capj["manc50/FEASIBLE"]["median_redistr"]], rel(R3A / "cap_report.json"), "median over 5 networks")
mx = max(max(v["max_rel_dE"], v["max_rel_dI"]) for k, v in capj.items() if "mn_r3_cap" in k)
chk("Results 4 / Discussion (D2)", "capped: no motor neuron's E or I input changed by more than 5%", "5%", 0.05 if mx <= 0.05 + 1e-12 else mx, rel(R3A / "cap_report.json"), f"largest per-network maximum {mx:.4f}")
chk("Results 4 (D2)", "capped: S median MaleCNS / MANC", "−0.054/−0.055", [dw("theta50")["X_median"], dw("manc50")["X_median"]], SD2, "median of 5 networks, real setting, 15-25 s")
chk("Results 4 (D2)", "capped: within-pool W MaleCNS / MANC", "0.277/0.135", [dw("theta50")["W_median"], dw("manc50")["W_median"]], SD2, "archived round-2 W")
chk("Results 4 (D2)", "capped: conditions 2, 3, 5 met and 6 not met, both connectomes", "true",
    all(str(dw(t)[c]).lower() == "true" for t in ("theta50", "manc50") for c in ("C2", "C3", "C5")) and all(str(dw(t)["C6"]).lower() == "false" for t in ("theta50", "manc50")), SD2, "C2, C3, C5 true; C6 false")
chk("Results 4 (D2)", "capped: pair retention MaleCNS / MANC", "0.67/0.50", [dw("theta50")["pair_retention"], dw("manc50")["pair_retention"]], SD2, "condition 6")
chk("Results 4 (D2)", "capped: oscillation retention MaleCNS / MANC", "0.82/0.68", [dw("theta50")["osc_retention"], dw("manc50")["osc_retention"]], SD2, "condition 6")
gd = pl.read_csv(R3A / "C_D2/D2_gate_decomposition.csv")
g_ = lambda t, a, c: float(gd.filter((pl.col("connectome") == t) & (pl.col("arm") == a))[c][0])
SGD = rel(R3A / "C_D2/D2_gate_decomposition.csv")
chk("Results 4 (D2)", "capped: gated fraction MaleCNS / MANC; W MaleCNS / MANC", "0.35/0.40/0.40/0.40", [g_("theta50", "X_cap", "median_P_gate"), g_("manc50", "X_cap", "median_P_gate"), g_("theta50", "W", "median_P_gate"), g_("manc50", "W", "median_P_gate")], SGD, "median over networks")
chk("Results 4 (D2)", "capped: antiphase once gated MaleCNS / MANC; real", "−0.20/−0.09/0.70/0.34", [g_("theta50", "X_cap", "median_antiphase_given_gate"), g_("manc50", "X_cap", "median_antiphase_given_gate"),
    g_("theta50", "real", "median_antiphase_given_gate"), g_("manc50", "real", "median_antiphase_given_gate")], SGD, "descriptive")
code("Methods (D2)", "exchange rejected if an affected motor neuron's E input changes by more than cap * max(x0, 1)", "5%", R3S / "build_cap.py",
     r"if abs\(nb - X0E\[bb\]\) > cap \* max\(X0E\[bb\], 1\.0\) or abs\(nd - X0E\[d\]\) > cap \* max\(X0E\[d\], 1\.0\):")

# -- E: common and differential activity, gate decomposition
EV = json.loads((R3A / "D1_E_cpu/D1_E_verdicts.json").read_text())
SEV = rel(R3A / "D1_E_cpu/D1_E_verdicts.json")
chk("Results 5 (E)", "D: real / W / X_hi, MaleCNS", "0.73/0.73/0.35", [EV["E_theta50"]["D_real"], EV["E_theta50"]["D_W"], EV["E_theta50"]["D_X_hi"]], SEV, "median over active pairs")
chk("Results 5 (E)", "D: real / W / X_hi, MANC", "0.66/0.63/0.42", [EV["E_manc50"]["D_real"], EV["E_manc50"]["D_W"], EV["E_manc50"]["D_X_hi"]], SEV, "median over active pairs")
chk("Results 5 (E)", "P1 common-mode ratio X_hi/W: MaleCNS (holds) / MANC (fails); threshold 0.8", "0.84/0.62", [EV["E_theta50"]["P1_median_ratio_Vplus_X_over_W"], EV["E_manc50"]["P1_median_ratio_Vplus_X_over_W"]], SEV,
    f"P1_holds {EV['E_theta50']['P1_holds']} / {EV['E_manc50']['P1_holds']}")
chk("Results 5 (E)", "P1 holds in MaleCNS and fails in MANC", "true", bool(EV["E_theta50"]["P1_holds"] and not EV["E_manc50"]["P1_holds"]), SEV, "P1_holds")
chk("Results 5 (E)", "P2 holds in both connectomes", "true", bool(EV["E_theta50"]["P2_holds"] and EV["E_manc50"]["P2_holds"]), SEV, "P2_holds")
chk("Results 5 (E)", "P3 pooled Spearman gamma-D (n): MaleCNS / MANC", "0.71/90/0.79/110", [EV["E_theta50"]["P3_spearman_gamma_D"], float(EV["E_theta50"]["P3_n"]), EV["E_manc50"]["P3_spearman_gamma_D"], float(EV["E_manc50"]["P3_n"])], SEV, "P3")
p3 = pl.read_csv(R3A / "D1_E_cpu/E_P3_within_arm.csv").filter(pl.col("arm") != "pooled")
for t, pr in (("theta50", "0.21–0.45"), ("manc50", "0.26–0.79")):
    x = p3.filter(pl.col("connectome") == t)["spearman_gamma_D"]
    chk("Results 5 (E)", f"{CON[t]} within-group Spearman gamma-D, range", pr, [float(x.min()), float(x.max())], rel(R3A / "D1_E_cpu/E_P3_within_arm.csv"), "real, W, X_hi separately")
E1 = {t: pl.read_csv(R3A / f"D1_E_cpu/E1_gate_decomposition_summary_{t}.csv") for t in ("theta50", "manc50")}
e1 = lambda t, s, c: float(E1[t].filter(pl.col("set") == s)[c][0])
FAMS6 = ["er", "role", "degree", "leg", "hl", "leg_hl"]
chk("Results 5 (E1)", "real gated fraction MaleCNS / MANC", "0.45/0.46", [e1(t, "real", "mean_P_gate") for t in ("theta50", "manc50")], rel(R3A / "D1_E_cpu/E1_gate_decomposition_summary_theta50.csv"), "median over runs of the mean over pairs")
for t, pr in (("theta50", "0.54–0.94"), ("manc50", "0.49–0.94")):
    v = [round(e1(t, f, "mean_P_gate"), 10) for f in FAMS6]      # exact values are multiples of 1/400 (0.9349999... is 187/200 in floating point)
    chk("Results 5 (E1)", f"{CON[t]} rewired families' gated fraction, range", pr, [min(v), max(v)], rel(R3A / f"D1_E_cpu/E1_gate_decomposition_summary_{t}.csv"), "six main families")
chk("Results 5 (E1)", "real antiphase once gated MaleCNS / MANC", "0.70/0.34", [e1(t, "real", "mean_antiphase_given_gate") for t in ("theta50", "manc50")], rel(R3A / "D1_E_cpu/E1_gate_decomposition_summary_theta50.csv"), "descriptive")
chk("Results 5 (E1)", "best rewired family antiphase once gated MaleCNS / MANC", "0.07/0.06", [max(e1(t, f, "mean_antiphase_given_gate") for f in FAMS6) for t in ("theta50", "manc50")], rel(R3A / "D1_E_cpu/E1_gate_decomposition_summary_manc50.csv"), "max over six families")
chk("Results 5 (E1)", "X_hi gated fraction MaleCNS / MANC", "0.45/0.50", [e1(t, "mn_r2_cross_hi", "mean_P_gate") for t in ("theta50", "manc50")], rel(R3A / "D1_E_cpu/E1_gate_decomposition_summary_theta50.csv"), "X_hi at the real setting")
chk("Results 5 (E1)", "X_hi antiphase once gated MaleCNS / MANC", "−0.12/−0.17", [e1(t, "mn_r2_cross_hi", "mean_antiphase_given_gate") for t in ("theta50", "manc50")], rel(R3A / "D1_E_cpu/E1_gate_decomposition_summary_manc50.csv"), "X_hi at the real setting")
code("Eqs. 17-18", "Var(y+-) = (se^2 + sf^2 +- 2 rho se sf) / 2 and D = Var(y-) / (Var(y+) + Var(y-))", "Eqs. 17-18", R3S / "analyze_r3_cpu.py",
     r'\(\(se \*\* 2 \+ sf \*\* 2 \+ 2 \* c \* se \* sf\) / 2\)\.alias\("Vp"\),\s*\(\(se \*\* 2 \+ sf \*\* 2 - 2 \* c \* se \* sf\) / 2\)\.alias\("Vm"\),\s*\)\.with_columns\(\(pl\.col\("Vm"\) / \(pl\.col\("Vp"\) \+ pl\.col\("Vm"\)\)\)\.alias\("D"\)\)')
code("Eq. 19", "gate decomposition: P(gated) and mean antiphase given gated", "Eq. 19", R3S / "analyze_r3_cpu.py",
     r'pl\.col\("gated"\)\.cast\(pl\.Float64\)\.mean\(\)\.alias\("P_gate"\),\s*pl\.when\(pl\.col\("gated"\)\)\.then\(-pl\.col\("corr"\)\)\.otherwise\(None\)\.mean\(\)\.alias\("antiphase_given_gate"\)')

# -- C: final-layer rescue
dC = pl.read_csv(R3A / "C_D2/round3_decisions.csv").filter(pl.col("exp").str.starts_with("C_rescue") & (pl.col("window") == "w15_25"))
SC = rel(R3A / "C_D2/round3_decisions.csv")
cr = lambda t, a: dC.filter((pl.col("connectome") == t) & (pl.col("exp") == f"C_rescue_{a}")).row(0, named=True)
chk("Results 6 (C)", "median RF, C1: MaleCNS / MANC", "−0.09/0.27", [cr("theta50", "C1")["median_RF"], cr("manc50", "C1")["median_RF"]], SC, "median over 10 networks")
chk("Results 6 (C)", "MANC C1 networks above their parent", "9/10", [float(cr("manc50", "C1")["n_above_parent"]), 10.0], SC, "S_arm > S_parent")
chk("Results 6 (C)", "median RF, C1p: MaleCNS / MANC", "−0.02/0.07", [cr("theta50", "C1p")["median_RF"], cr("manc50", "C1p")["median_RF"]], SC, "median over 10 networks")
chk("Results 6 (C)", "median RF, C0: MaleCNS / MANC", "−0.55/−0.59", [cr("theta50", "C0")["median_RF"], cr("manc50", "C0")["median_RF"]], SC, "median over 10 networks")
chk("Results 6 (C)", "C1p above C0 in 10 of 10, both connectomes", "10/10", [float(cr("theta50", "C0")["n_C1p_gt_C0"]), float(cr("manc50", "C0")["n_C1p_gt_C0"])], SC, "pairs k")
chk("Results 6 (C)", "categories: MaleCNS C1 no recovery; MANC C1 partial", "不恢复/部分恢复", f"{cr('theta50', 'C1')['level']}/{cr('manc50', 'C1')['level']}", SC, "pre-registered categories")
rb = pl.read_csv(R3A / "rescue_build_report.csv")
SRB = rel(R3A / "rescue_build_report.csv")
fr = rb.group_by("connectome").agg(pl.col("frac_edges_mn_layer").min().alias("lo"), pl.col("frac_edges_mn_layer").max().alias("hi"))
frv = {r["connectome"]: (r["lo"], r["hi"]) for r in fr.iter_rows(named=True)}
chk("Results 6 / Suppl. Note 4 (C)", "replaced final layer: fraction of all edges MaleCNS / MANC", "7.7%/5.3%", [frv["theta50"][1], frv["manc50"][1]], SRB, f"constant across networks: {frv}")
ri_ = lambda t, a: float(rb.filter((pl.col("connectome") == t) & (pl.col("arm") == a))["ri_index"].median())
chk("Results 6 (C)", "RI index after C1: MaleCNS / MANC", "−0.027/0.001", [ri_("theta50", "C1"), ri_("manc50", "C1")], SRB, "median over 10 networks")
chk("Results 6 (C)", "RI index of the parents: MaleCNS / MANC", "−0.063/−0.112", [ri_("theta50", "parent"), ri_("manc50", "parent")], SRB, "median over 10 networks")
chk("Results 6 (C)", "RI index after C0: MaleCNS / MANC", "−0.41/−0.46", [ri_("theta50", "C0"), ri_("manc50", "C0")], SRB, "median over 10 networks")
code("Eq. 16", "RF = (S_arm - S_parent) / (S_real - S_parent)", "Eq. 16", R3S / "analyze_r3_gpu.py", r"RF=\(sa - sp_\) / \(sr - sp_\)")
code("Methods (C)", "every edge onto a motor neuron replaced; other edges kept", "C1/C1p/C0", R3S / "build_rescue.py", r"keep = parent\.filter\(~pl\.Series\(is_mn\[parent\[\"post\"\]\.to_numpy\(\)\]\)\)")


# ------------------------------------------------------------------------------------------ Supplementary Information header counts
_tx = (V2 / "manuscript.tex").read_text(encoding="utf-8")
_si = _tx[_tx.index("\\renewcommand{\\thepage}{S\\arabic{page}}"):]
_hd = re.search(r"Supplementary Notes 1--(\d+) \$\\cdot\$ Supplementary Figures 1--(\d+) \$\\cdot\$ Supplementary Tables 1--(\d+)", _si)
_n = [len(re.findall(r"\\subsection\*\{Supplementary Note \d+:", _si)), len(re.findall(r"\\begin\{figure\}\[p\]", _si)),
      len(re.findall(r"\\begin\{table\}", (V2 / "generated" / "si_tables.tex").read_text(encoding="utf-8")))]
chk("Suppl. Information header", "numbers of Supplementary Notes, Figures and Tables listed on the first SI page", "/".join(_hd.groups()),
    [float(x) for x in _n], "manuscript.tex; generated/si_tables.tex", "counted in the document")

# ------------------------------------------------------------------------------------------ numeric-token audit of the manuscript
tex = (V2 / "manuscript.tex").read_text(encoding="utf-8")
body = tex.split("\\begin{document}", 1)[1]
body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", body, flags=re.S)
body = "\n".join(l.split("%%")[0] if l.lstrip().startswith("%%") else re.sub(r"(?<!\\)%.*", "", l) for l in body.splitlines())
body = re.sub(r"\\(ref|label|citep|input|figpage|includegraphics|setcounter|captionsetup|section|subsection|vspace|hspace|href)\*?(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", " ", body)
body = re.sub(r"\\parbox\{[^}]*\}", " ", body)                  # layout widths are not numbers of the paper
body = re.sub(r"Fig\.~\\ref\{[^}]*\}[a-z,\-–]*", " ", body)
body = re.sub(r"\b(DNg100|DNxl058|RTX 5090|v1\.0|13A|13B|01A|T[123][LR]?|S\\arabic|Supplementary (Note|Table)~?\d)\b", " ", body)
body = body.replace("$", " ").replace("\\,", " ").replace("~", " ")
tokens = re.findall(r"(?<![A-Za-z\\{.\d])\d+(?:,\d{3})*(?:\.\d+)?", body)
covered = set()
for r in rows:
    for tok in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", r["printed"] + " " + r["statement"]):
        covered.add(tok)
ALLOW = {"1": "counts/indices, e.g. 'one of', figure panels", "2": "counts (two connectomes, version 2)", "3": "counts (three parameters/runs)",
         "4": "four joints/windows", "5": "five networks per family", "6": "six families/conditions", "10": "10-s windows", "20": "20 pairs/runs",
         "28": "date", "2026": "date", "19": "time of day (Suppl. Note 2)", "00": "time of day", "22": "time of day", "50": "time of day / 50-Hz band",
         "15": "window bound", "25": "window bound", "35": "window bound", "45": "window bound", "60": "60-s runs", "30": "30 networks",
         "1.1": "10^-1.1e-16 mantissa", "16": "exponent / 16 pairs", "0.5": "parameters", "0.2": "gate", "0.001": "gate", "0.01": "noise / osc threshold",
         "1.5": "grid exponent", "1.75": "frozen setting exponent", "2.5": "frozen setting exponent", "0.8": "condition 1", "0.10": "conditions 2-3",
         "80": "conditions 5-6 (%)", "12": "12 hemilineages / ~12 s", "13": "window 1-13 s", "21": "time 21:00 / about 21", "9": "9 runs, 9 pairs", "7": "7 of 12",
         "8": "8 of 9 / 8 of 30", "11": "11 gains", "14": "14 pairs", "100": "networks per family", "132": "settings", "1906": "Sherrington", "300": "300-s runs",
         "290": "window", "0.15": "strong margin", "0.35": "unused", "4.1": "frequency range", "2.4": "frequency range", "3.4": "frequency range", "14.6": "frequency range",
         "1.2": "residual range", "1.9": "residual range", "27": "percent / eligible settings", "34": "percent", "26": "date", "0.50": "unused"}
unc = sorted({t for t in tokens if t not in covered and t not in ALLOW}, key=lambda s: float(s.replace(",", "")))
allow_ctx = []
for tok in sorted({t for t in tokens if t not in covered and t in ALLOW}, key=lambda s: float(s.replace(",", ""))):
    for m in list(re.finditer(r"(?<![A-Za-z\\{\d.])" + re.escape(tok) + r"(?![\d])", body))[:4]:
        allow_ctx.append((tok, ALLOW[tok], " ".join(body[max(0, m.start() - 70):m.start() + 30].split())))

# ------------------------------------------------------------------------------------------ write report
fails = [r for r in rows if not r["ok"]]
out = ["# Number check for the Nature Communications version (v9_NC)", "",
       "Generated by `src/check_numbers.py` (read-only on all data). Every number in the text of `manuscript.tex` is either "
       "recomputed below from the archived files or, for the earlier 5-s protocol (Supplementary Note 1), checked against "
       "the lab notebook in which it was recorded. The figure legends (generated with the figures) and the supplementary "
       "tables (generated by `src/make_si_tables.py` from the analysis tables) are produced directly from data and are not "
       "re-listed here. Paths are relative to `fly-connectome-research-20260926/`. Section labels are those of preprint v5; "
       "the v6_NC abstract contains no numbers, so the checks labelled Abstract now cover the same numbers in Results.", "",
       f"**Checks: {len(rows)}; matching: {len(rows) - len(fails)}; mismatching: {len(fails)}. "
       f"Numeric tokens in the text not covered by a check or by the allow-list: {len(unc)}.**", "",
       "| # | Section | Statement | Printed | Recomputed | OK | Source | Method |", "|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    cells = [str(i), r["section"], r["statement"], r["printed"], r["computed"], "✓" if r["ok"] else "**✗**", r["source"], r["method"]]
    out.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
out += ["", "## Numeric tokens not covered by a check", "",
        ("None." if not unc else ", ".join(unc)), "",
        "## Numeric tokens accepted by the allow-list (not numerical results), with context", "",
        "| Token | Reason | Context |", "|---|---|---|"] + [f"| {t} | {r} | {c.replace('|', '/')} |" for t, r, c in allow_ctx]
(V2 / "numbers_check.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(rows)} checks, {len(fails)} mismatches, {len(unc)} uncovered tokens -> numbers_check.md")
for r in fails:
    print("MISMATCH:", r["section"], "|", r["statement"], "| printed", r["printed"], "| computed", r["computed"])
if unc:
    print("UNCOVERED:", unc)
sys.exit(1 if fails or unc else 0)
