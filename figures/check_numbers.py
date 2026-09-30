"""Recompute every quantitative statement of preprint v1 that the figures depict, and compare it with
the value printed in the manuscript. Writes qa/numbers_check.csv (one row per statement).

A statement "matches" when the computed value, rounded to the precision used in the manuscript,
equals the printed value. Mismatches are reported, never corrected in the data.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import common as C

rows = []


def check(section, statement, printed, computed, nd, source, definition):
    if isinstance(printed, str):
        ok = str(printed) == str(computed)
        comp = computed
    else:
        comp = float(computed)
        ok = abs(round(comp, nd) - printed) < 10 ** (-nd) / 2 + 1e-12
    rows.append(dict(section=section, statement=statement, printed=printed,
                     computed=(round(comp, 6) if isinstance(comp, float) else comp),
                     match=bool(ok), source=source, definition=definition))


# ------------------------------------------------------------------ subnetworks (Fig. 1)
for tag, n_nodes, n_edges, n_mn in [("theta50", 10173, 462650, 381), ("manc50", 10440, 634207, 396)]:
    nodes = pd.read_parquet(C.GRAPHS / f"nodes_{tag}.parquet"); edges = pd.read_parquet(C.GRAPHS / f"edges_{tag}.parquet")
    mn = pd.read_parquet(C.GRAPHS / f"mn_joint_map_{tag}.parquet")
    src = f"graphs/nodes_{tag}, edges_{tag}"
    check("Model", f"{C.TAGS[tag]} neurons", n_nodes, len(nodes), 0, src, "rows of the node table")
    check("Model", f"{C.TAGS[tag]} connections", n_edges, len(edges), 0, src, "rows of the edge table (>=5 synapses, no self-loops)")
    check("Model", f"{C.TAGS[tag]} motor neurons", n_mn, int((nodes.role == "MN").sum()), 0, src, "role == MN")
    pairs = mn.dropna(subset=["joint"]).groupby(["leg", "joint"]).direction.agg(lambda d: (d > 0).any() and (d < 0).any())
    check("Model", f"{C.TAGS[tag]} antagonistic pairs", 20, int(pairs.sum()), 0, f"graphs/mn_joint_map_{tag}", "leg x joint with both directions")
    if tag == "theta50":
        for role, v in [("IN", 8702), ("DN", 424), ("SN", 666)]:
            check("Model", f"MaleCNS {role}", v, int((nodes.role == role).sum()), 0, src, f"role == {role}")
        check("Methods", "MaleCNS mapped MNs", 302, int(mn.joint.notna().sum()), 0, "mn_joint_map_theta50", "joint assigned")
        check("Methods", "MaleCNS approximately mapped MNs", 76, int((mn.joint.notna() & mn.approx.fillna(False)).sum()), 0,
              "mn_joint_map_theta50", "joint assigned and approx flag")

# ------------------------------------------------------------------ scan (Fig. 2)
sc = {t: C.scan(t) for t in C.TAGS}
check("Methods", "parameter settings per network", 132, sc["theta50"][["g", "beta", "drive"]].drop_duplicates().shape[0], 0,
      "nullscan_sets_stat_*", "unique (g, beta, drive)")
best = {}
for t, df in sc.items():
    e = df[df.eligible]
    b = e.groupby(["graph", "gseed"]).agg(bestS=("gmean_stat", "max"), maxrh=("rhythm_stat", "max")).reset_index()
    best[t] = b
    rw = b[b.graph.isin(C.MAIN + C.EXTRA)]
    rv = b[b.graph == "real"].iloc[0]
    n_rw = len(rw)
    check("Rhythm", f"{C.TAGS[t]} scan: real best S", {"theta50": 0.330, "manc50": 0.215}[t], rv.bestS, 3, "nullscan", "max S over eligible settings")
    check("Rhythm", f"{C.TAGS[t]} scan: max best S of rewired", {"theta50": 0.139, "manc50": 0.106}[t], rw.bestS.max(), 3, "nullscan", "max over rewired networks")
    check("Rhythm", f"{C.TAGS[t]} scan: number of rewired networks", {"theta50": 40, "manc50": 30}[t], n_rw, 0, "nullscan", "main + secondary families")
    if t == "theta50":
        check("Rhythm", "MaleCNS scan: real max rhythmicity", 0.70, rv.maxrh, 2, "nullscan", "max rhythmicity over eligible settings")
        check("Rhythm", "MaleCNS scan: rewired max rhythmicity", 0.96, rw.maxrh.max(), 2, "nullscan", "max over rewired networks")

# ------------------------------------------------------------------ confirmation (Fig. 2)
gm = {t: C.graph_window_means(t) for t in C.TAGS}
printed = {("theta50", "w15_25"): (0.312, 0.144), ("theta50", "w50_60"): (0.312, 0.147),
           ("manc50", "w15_25"): (0.173, 0.084), ("manc50", "w50_60"): (0.128, 0.080)}
for (t, w), (pr, pn) in printed.items():
    g = gm[t][gm[t].window == w]
    r = g[g.graph == "real"].S.iloc[0]
    nul = g[g.graph.isin(C.MAIN + C.EXTRA) & (g.rule == "own")]
    check("Coordination", f"{C.TAGS[t]} {w}: real S", pr, r, 3, "heldout_stat", "mean of 20 stochastic runs")
    check("Coordination", f"{C.TAGS[t]} {w}: strongest rewired S", pn, nul.S.max(), 3, "heldout_stat", "max network mean over rewired networks")
    if t == "manc50":
        hl_med = np.median(g[(g.graph == "hl") & (g.rule == "own")].S)
        check("Coordination", f"MANC {w}: real minus lineage-family median", {"w15_25": 0.134, "w50_60": 0.086}[w], r - hl_med, 3,
              "heldout_stat", "real − median of 5 lineage networks")
# claim levels in all four windows
for t in C.TAGS:
    for w in C.WINDOWS:
        g = gm[t][(gm[t].window == w) & (gm[t].rule == "own")]
        r = g[g.graph == "real"].S.iloc[0]
        fams = {f: g[g.graph == f].S.to_numpy() for f in C.MAIN}
        gt = sum(bool((r > v).all()) for v in fams.values())
        strong = gt == 6 and all(r - np.median(v) >= 0.15 for v in fams.values())
        moderate = gt >= 5 and all(r > np.median(v) for v in fams.values())
        lvl = "STRONG" if strong else ("MODERATE" if moderate else "WEAK")
        check("Coordination", f"{C.TAGS[t]} {w}: pre-specified level", {"theta50": "STRONG", "manc50": "MODERATE"}[t], lvl, 0,
              "heldout_stat", "§3.23 rule on the six main families")
# noise-free MaleCNS runs
hd = C.heldout("theta50"); det = hd[(hd.graph == "real") & (hd["mode"] == "deterministic")]
for run, v in zip(range(3), (0.325, 0.233, 0.325)):
    x = det[det.run == run].gmean
    check("Coordination", f"MaleCNS noise-free run {run}: S (all windows)", v, x.iloc[0], 3, "heldout_stat",
          f"windows span {x.min():.3f}–{x.max():.3f}")
# MANC persistence
hm = C.heldout("manc50"); rm = hm[(hm.graph == "real")]
piv_S = rm.pivot_table(index=["mode", "run"], columns="window", values="gmean")
piv_f = rm.pivot_table(index=["mode", "run"], columns="window", values="freq")
piv_n = rm.pivot_table(index=["mode", "run"], columns="window", values="n_gated_pairs")
fast = (piv_f >= 3) & (piv_n <= 3)
switched = (~fast["w15_25"]) & fast[["w25_35", "w35_45", "w50_60"]].any(axis=1)
sw_st = switched.loc["stochastic"]; sw_det = switched.loc["deterministic"]
check("Coordination", "MANC stochastic runs that switched to a fast state after 25 s", 4, int(sw_st.sum()), 0, "heldout_stat",
      "fast state: dominant frequency >= 3 Hz and <= 3 gated pairs; not fast at 15–25 s")
check("Coordination", "MANC noise-free runs that switched", 1, int(sw_det.sum()), 0, "heldout_stat", "same rule")
ns = piv_S.loc["stochastic"][~sw_st.values]
check("Coordination", "MANC non-switching runs: mean S 15–25 s", 0.17, ns["w15_25"].mean(), 2, "heldout_stat", "mean over 16 runs")
check("Coordination", "MANC non-switching runs: mean S 50–60 s", 0.16, ns["w50_60"].mean(), 2, "heldout_stat", "mean over 16 runs")
fsw = [piv_f.loc["stochastic"].loc[r][fast.loc["stochastic"].loc[r]].values for r in sw_st[sw_st].index]
fsw = np.concatenate(fsw)
check("Coordination", "MANC switched-state frequency range (Hz)", "3–15", f"{fsw.min():.0f}–{fsw.max():.0f}", 0, "heldout_stat",
      f"exact {fsw.min():.1f}–{fsw.max():.1f} Hz")
mal = hd[(hd.graph == "real") & (hd["mode"] == "stochastic") & (hd.window == "w15_25")]
mal_all = hd[(hd.graph == "real") & (hd["mode"] == "stochastic")]
check("Coordination", "MaleCNS dominant frequency range (Hz), 15–25 s", "2.4–4.1", f"{mal.freq.min():.1f}–{mal.freq.max():.1f}", 0,
      "heldout_stat", f"stochastic runs, 15–25 s (all four windows: {mal_all.freq.min():.1f}–{mal_all.freq.max():.1f} Hz)")
fz = [piv_f.loc["stochastic"].loc[r, "w15_25"] for r in piv_f.loc["stochastic"].index if not sw_st.loc[r]]
check("Coordination", "MANC coordinated-state frequency (Hz)", 1.0, np.median(fz), 0, "heldout_stat",
      "median over non-switching stochastic runs, 15–25 s (manuscript: 'near 1 Hz')")
# rhythm at frozen settings
for t in C.TAGS:
    g = gm[t][(gm[t].window == "w15_25") & (gm[t].rule == "own") & gm[t].graph.isin(C.MAIN + C.EXTRA)]
    if t == "theta50":
        check("Rhythm", "MaleCNS: max rewired rhythmicity at frozen settings (15–25 s)", 0.90, g.rhythm.max(), 2, "heldout_stat", "network mean over runs")

# ------------------------------------------------------------------ structure (Fig. 3)
rho_in = {}
for t, pr, n_ex in [("theta50", 0.48, 19), ("manc50", 0.38, 18)]:
    rg = C.recip_graph(t); rp = C.recip_pairs(t)
    real = rg[rg.graph == "real"].recip_innervation.iloc[0]
    check("Structure", f"{C.TAGS[t]} real index (median over pairs)", pr, real, 2, "diag_recip_innervation_v2", "median over 20 pairs")
    nul = rg[(rg.kind == "null") & rg.level.isin(C.MAIN)]
    check("Structure", f"{C.TAGS[t]} max index of the 30 main rewired networks <= 0", "True", str(bool((nul.recip_innervation <= 0).all())), 0,
          "diag_recip_innervation_v2", f"max = {nul.recip_innervation.max():.4f}")
    rp = rp.assign(level=rp.graph.str.replace(r"_s\d+$", "", regex=True))
    pr_real = rp[rp.graph == "real"].set_index("pair").recip_innervation
    pr_null = rp[rp.level.isin(C.MAIN)].groupby("pair").recip_innervation.max()
    exceed = (pr_real > pr_null.reindex(pr_real.index))
    check("Structure", f"{C.TAGS[t]} pairs above all 30 rewired networks", n_ex, int(exceed.sum()), 0, "…_perjoint", "real > max over 30")
    check("Structure", f"{C.TAGS[t]} exceptions", "TiTa only", "TiTa only" if all(p.endswith("TiTa") for p in exceed[~exceed].index) else ",".join(exceed[~exceed].index),
          0, "…_perjoint", "pairs not above all rewired networks")
    rho_in[t] = pr_real
j = pd.concat(rho_in, axis=1).dropna()
check("Structure", "Spearman rho of real per-pair index, MaleCNS vs MANC", 0.82, spearmanr(j["theta50"], j["manc50"])[0], 2,
      "…_perjoint", f"n = {len(j)} pairs")

# ------------------------------------------------------------------ surgery (Fig. 4)
retention = []
for t in C.TAGS:
    g = gm[t]
    for w in ["w15_25", "w50_60"]:
        x = g[(g.window == w) & (g.rule == "real_setting")]
        r = g[(g.window == w) & (g.graph == "real")].S.iloc[0]
        med = x.groupby("graph").S.median()
        pX = {("theta50", "w15_25"): -0.071, ("theta50", "w50_60"): -0.070, ("manc50", "w15_25"): -0.056, ("manc50", "w50_60"): -0.051}[(t, w)]
        pW = {("theta50", "w15_25"): 0.298, ("theta50", "w50_60"): 0.297, ("manc50", "w15_25"): 0.140, ("manc50", "w50_60"): 0.108}[(t, w)]
        pXw = {("theta50", "w15_25"): 0.170, ("theta50", "w50_60"): 0.163, ("manc50", "w15_25"): 0.094, ("manc50", "w50_60"): 0.065}[(t, w)]
        pWw = {("theta50", "w15_25"): 0.297, ("theta50", "w50_60"): 0.309, ("manc50", "w15_25"): 0.166, ("manc50", "w50_60"): 0.139}[(t, w)]
        for lab, key, pv in [("cross-pool", "mn_shuffle_joint", pX), ("within-pool", "mn_shuffle_pool", pW),
                             ("strength-matched cross-pool", "mn_shuffle_joint_w", pXw), ("strength-matched within-pool", "mn_shuffle_pool_w", pWw)]:
            check("Surgery", f"{C.TAGS[t]} {w}: {lab} median S", pv, med[key], 3, "heldout_stat (rule real_setting)", "median of 5 network means")
        retention.append(med["mn_shuffle_pool"] / r)
check("Surgery", "within-pool retention W/R (both connectomes, 15–25 and 50–60 s)", "81–96%",
      f"{100*min(retention):.0f}–{100*max(retention):.0f}%", 0, "heldout_stat", "W = median of 5 within-pool networks; R = real")
aud = C.intervention_audit()
for t, pp, pw in [("theta50", 0.272, 0.139), ("manc50", 0.265, 0.131)]:
    a = aud[aud.connectome == t]
    check("Surgery", f"{C.TAGS[t]} net cross-pool redistribution, original", pp,
          a[a.graph == "mn_shuffle_joint"].pool_allocation_redistribution_fraction.median(), 3, "intervention_edge_audit", "median of 5 networks")
    check("Surgery", f"{C.TAGS[t]} net cross-pool redistribution, strength-matched", pw,
          a[a.graph == "mn_shuffle_joint_w"].pool_allocation_redistribution_fraction.median(), 3, "intervention_edge_audit", "median of 5 networks")
rep = C.exp2_report()
xh = [v["redistr"] for k, v in rep.items() if "cross_hi" in k]
check("Surgery", "dose-matched design: redistribution range", "25–28%", f"{100*min(xh):.0f}–{100*max(xh):.0f}%", 0, "surgery_r2_report.json",
      f"X_hi graphs, both connectomes ({min(xh):.3f}–{max(xh):.3f})")
c6 = pd.read_csv(C.R2 / "analysis/exp1_C6_pair_identity.csv")
c6x = c6[(c6.connectome == "theta50") & (c6.surgery == "mn_shuffle_joint") & c6.window.isin(["w15_25", "w50_60"])]
check("Surgery", "MaleCNS pairs still modulated after cross-pool reassignment", "8 of 9",
      f"{round(c6x.pair_retention_median.iloc[0]*c6x.n_contributing_pairs.iloc[0])} of {c6x.n_contributing_pairs.iloc[0]}", 0,
      "exp1_C6_pair_identity", "median over 5 networks, 15–25 s")
check("Surgery", "MaleCNS oscillating-MN retention (15–25 and 50–60 s)", "97–102%",
      f"{100*c6x.osc_retention_median.min():.0f}–{100*c6x.osc_retention_median.max():.0f}%", 0, "exp1_C6_pair_identity", "median over networks")

# ------------------------------------------------------------------ joints (Fig. 5)
cl = pd.read_csv(C.R2 / "analysis/exp1_claim_levels.csv")
w = cl[(cl.connectome == "theta50") & (cl.window == "w15_25")].set_index("readout")
for rd, pr, pn in [("S_ThC", 0.71, 0.22), ("S_CTr", 0.25, 0.24), ("S_FTi", 0.09, None), ("S_TiTa", 0.0, None),
                   ("S_exact", 0.200, 0.126), ("S_nonThC", 0.143, 0.169)]:
    nd = 3 if rd in ("S_exact", "S_nonThC") else 2
    check("Joints", f"MaleCNS {rd} real", pr, w.loc[rd, "real"], nd, "exp1_claim_levels", "mean of 20 runs, 15–25 s")
    if pn is not None:
        check("Joints", f"MaleCNS {rd} strongest rewired", pn, w.loc[rd, "strongest_null"], nd, "exp1_claim_levels", "max over 30 networks")
for win, lvl in [("w15_25", "MODERATE"), ("w50_60", "WEAK")]:
    check("Joints", f"MaleCNS S_nonThC level {win}", lvl, cl[(cl.connectome == "theta50") & (cl.window == win) & (cl.readout == "S_nonThC")].level.iloc[0], 0,
          "exp1_claim_levels", "§3.23 rule")
for win in ["w15_25", "w50_60"]:
    check("Joints", f"MaleCNS S_exact level {win}", "STRONG", cl[(cl.connectome == "theta50") & (cl.window == win) & (cl.readout == "S_exact")].level.iloc[0], 0,
          "exp1_claim_levels", "§3.23 rule")

out = pd.DataFrame(rows)
out.to_csv(C.QA / "numbers_check.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 200)
print(out[["section", "statement", "printed", "computed", "match"]].to_string())
print("\nmismatches:", int((~out.match).sum()))
