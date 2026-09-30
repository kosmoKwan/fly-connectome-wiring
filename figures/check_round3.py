"""Round 3: every result used by the figures, recomputed from the archived parquet files (read-only) and checked
against round3_20260929/analysis/ (and the round-2 post hoc sign analysis). Rules as in the pre-registration
(round3_20260929/预注册_第三轮_20260929.md) and the analysis scripts analyze_r3_gpu.py / analyze_r3_cpu.py.

A   saturating-ReLU neuron: network values, claim level, strongest rewired network; scan maximum rhythmicity.
B   25 networks per family for the three most competitive families: all 90 rewired networks, level, count >= real,
    empirical p per family.
C   rescue: S of every rescued network, S of its parent, recovery fraction RF.
D2  5 % cap per motor neuron: S of the five capped networks at the real setting.
F2  25 % of glutamatergic neurons excitatory (MaleCNS): real network and ten rewired networks per draw.
E   common / differential activity from the round-2 pair-level data (identity Var(y±) = (se² + sf² ± 2ρ se sf) / 2):
    per network-pair values, P1-P3, within-arm correlations.
E1  gate decomposition: fraction of pair-runs past the amplitude gate and antiphase given the gate, per set.
Writes ../derived/r3_*.csv (inputs of the figure scripts) and ../qa/round3_check.csv.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import common as C

RES3 = C.R3 / "results"; AN3 = C.R3A; RES2 = C.R2 / "results"


def rcsv(path, **kw):
    """Exact (round-trip) float parsing: the default parser can change the last bit, which turns near-ties into ties and moves ranks."""
    return pd.read_csv(path, float_precision="round_trip", **kw)
chk = []


def check(section, item, computed, reference, tol=1e-9, note=""):
    if isinstance(reference, str) or isinstance(computed, str):
        ok = str(computed) == str(reference)
    else:
        ok = bool(np.isclose(float(computed), float(reference), atol=tol, rtol=0, equal_nan=True))
        exact = bool(np.isclose(float(computed), float(reference), atol=1e-9, rtol=0, equal_nan=True))
        if exact:
            note = ""
        elif ok and not note:
            note = f"within tolerance {tol}"
    chk.append(dict(section=section, item=item, computed=computed, reference=reference, match=ok, note=note))
    return ok





def fam_values(G, w, fams, extra=None):
    out = {}
    for f in fams:
        v = G[(G.graph == f) & (G.window == w)].sort_values("gseed").S
        out[f] = v.dropna().to_numpy()
    return out


def summarise(real, fams):
    allv = np.concatenate(list(fams.values())); fam_of = np.concatenate([[f] * len(v) for f, v in fams.items()])
    k = int(np.nanargmax(allv))
    return dict(real=real, level=C.level_rule(real, fams), strongest_null=float(allv[k]), strongest_family=str(fam_of[k]),
                families_all_below=int(sum(bool((real > v).all()) for v in fams.values())),
                min_margin_over_family_median=float(min(real - np.median(v) for v in fams.values())), n_nulls=int(len(allv)),
                n_at_or_above_real=int((allv >= real).sum()))


# =============================================================================================== A
dec = rcsv(AN3 / "全部判定/round3_decisions.csv")
A_rows, A_lv, A_rh = [], [], []
for tag in C.TAGS:
    G = C.r3_gmeans(RES3 / f"pairs_scalar_{tag}_relu.parquet"); G = G[G.rule == "own"]
    assert set(G.graph) == {"real"} | set(C.MAIN) and G[G.graph != "real"].groupby("graph").gseed.nunique().eq(5).all()
    A_rows.append(G.assign(connectome=tag))
    for w in C.WINDOWS:
        real = float(G[(G.graph == "real") & (G.window == w)].S.iloc[0])
        s = summarise(real, fam_values(G, w, C.MAIN)); A_lv.append(dict(connectome=tag, window=w, **s))
        ref = dec[(dec.exp == "A_relu_sat") & (dec.connectome == tag) & (dec.window == w)]
        if len(ref):
            r = ref.iloc[0]
            for k in ["real", "strongest_null", "min_margin_over_family_median"]:
                check("A", f"{tag} {w} {k}", s[k], r[k])
            check("A", f"{tag} {w} level", s["level"], r.level); check("A", f"{tag} {w} strongest family", s["strongest_family"], r.strongest_family)
            check("A", f"{tag} {w} families all below real", s["families_all_below"], int(r.families_all_below))
    # scan: maximum rhythmicity over eligible settings (paper convention: eligible only, main families, strictly greater)
    S = pd.read_parquet(RES3 / f"nullscan_sets_relu_{tag}.parquet")
    S["eligible"] = (S.osc_stat > 0.2) & (S.freq_stat >= 1) & (S.freq_stat <= 50)
    mx = S[S.eligible].groupby(["graph", "seed"]).rhythm_stat.max().reset_index().rename(columns={"seed": "gseed", "rhythm_stat": "max_rhythm"})
    real_mx = float(mx[mx.graph == "real"].max_rhythm.iloc[0]); nul = mx[mx.graph != "real"]
    A_rh.append(mx.assign(connectome=tag, real_max_rhythm=real_mx))
    r = dec[(dec.exp == "A_relu_sat") & (dec.connectome == tag)].iloc[0]
    check("A", f"{tag} real scan max rhythmicity", real_mx, r.real_scan_max_rhythm)
    check("A", f"{tag} rewired networks more rhythmic than real", int((nul.max_rhythm > real_mx).sum()), int(r.n_nulls_more_rhythmic))
    check("A", f"{tag} rewired networks with an eligible setting", int(len(nul)), int(r.n_nulls_with_eligible_setting))
    fr = rcsv(AN3 / "A_relu_sat/A_rhythm_by_family.csv"); fr = fr[fr.connectome == tag].set_index("family")
    for f in C.MAIN:
        q = nul[nul.graph == f].max_rhythm
        check("A", f"{tag} {f} median max rhythmicity", float(q.median()), fr.loc[f, "median_max_rhythm"])
        check("A", f"{tag} {f} max max rhythmicity", float(q.max()), fr.loc[f, "max_max_rhythm"])
        check("A", f"{tag} {f} networks above real", int((q > real_mx).sum()), int(fr.loc[f, "n_above_real"]))
pd.concat(A_rows).to_csv(C.DERIVED / "r3_A_network_values.csv", index=False)
pd.DataFrame(A_lv).to_csv(C.DERIVED / "r3_A_levels.csv", index=False)
pd.concat(A_rh).to_csv(C.DERIVED / "r3_A_scan_max_rhythm.csv", index=False)

# =============================================================================================== B
BV = rcsv(AN3 / "B_n25/B_network_values.csv")
B_rows, B_lv = [], []
EXP = ["hl", "degree", "leg_hl"]
for tag in C.TAGS:
    r2 = C.r3_gmeans(RES2 / f"pairs_scalar_{tag}.parquet"); r2 = r2[(r2.rule == "own") & r2.graph.isin(["real"] + C.MAIN)]
    n25 = C.r3_gmeans(RES3 / f"pairs_scalar_{tag}_n25.parquet"); n25 = n25[n25.rule == "own"]
    assert set(n25.graph) == set(EXP) and n25.groupby("graph").gseed.apply(lambda s: sorted(s.unique()) == list(range(5, 25))).all()
    G = pd.concat([r2, n25], ignore_index=True)
    B_rows.append(G.assign(connectome=tag))
    ref = BV[BV.connectome == tag].merge(G, on=["graph", "gseed", "window"], how="left", suffixes=("_ref", ""))
    assert ref.S.notna().all() and len(ref) == len(BV[BV.connectome == tag])
    check("B", f"{tag} network values (all {len(ref)} rows, max |diff|)", float(np.abs(ref.S - ref.S_ref).max()), 0.0, tol=1e-12)
    for w in C.WINDOWS:
        real = float(G[(G.graph == "real") & (G.window == w)].S.iloc[0])
        fams = fam_values(G, w, C.MAIN)
        s = summarise(real, fams)
        emp = {f: (1 + int((fams[f] >= real).sum())) / (len(fams[f]) + 1) for f in C.MAIN}
        B_lv.append(dict(connectome=tag, window=w, **s, n_per_family=json.dumps({f: len(v) for f, v in fams.items()}),
                         empirical_p=json.dumps({f: round(p, 4) for f, p in emp.items()})))
        rr = dec[(dec.exp == "B_n25") & (dec.connectome == tag) & (dec.window == w)]
        if len(rr):
            r = rr.iloc[0]
            for k in ["real", "strongest_null", "min_margin_over_family_median"]:
                check("B", f"{tag} {w} {k}", s[k], r[k])
            check("B", f"{tag} {w} level", s["level"], r.level); check("B", f"{tag} {w} rewired at or above real", s["n_at_or_above_real"], int(r.n_nulls_at_or_above_real))
            check("B", f"{tag} {w} number of rewired networks", s["n_nulls"], int(r.n_nulls))
            ep = json.loads(r.empirical_p)
            for f in C.MAIN:
                check("B", f"{tag} {w} empirical p {f}", round(emp[f], 4), ep[f], tol=1e-4)
pd.concat(B_rows).to_csv(C.DERIVED / "r3_B_network_values.csv", index=False)
pd.DataFrame(B_lv).to_csv(C.DERIVED / "r3_B_levels.csv", index=False)

# =============================================================================================== C
CR = rcsv(AN3 / "C_D2/C_rescue_per_network.csv")
C_rows = []
for tag in C.TAGS:
    Rs = C.r3_gmeans(RES3 / f"pairs_scalar_{tag}_rescue.parquet")
    r2 = C.r3_gmeans(RES2 / f"pairs_scalar_{tag}.parquet"); r2 = r2[r2.rule == "own"]
    for r in CR[CR.connectome == tag].itertuples():
        fam, k = r.parent.rsplit("_s", 1); k = int(k)
        s_arm = float(Rs[(Rs.graph == f"{r.arm}_{fam}") & (Rs.gseed == k) & (Rs.window == r.window)].S.iloc[0])
        s_par = float(r2[(r2.graph == fam) & (r2.gseed == k) & (r2.window == r.window)].S.iloc[0])
        s_real = float(r2[(r2.graph == "real") & (r2.window == r.window)].S.iloc[0])
        rf = (s_arm - s_par) / (s_real - s_par)
        C_rows.append(dict(connectome=tag, window=r.window, arm=r.arm, parent=r.parent, parent_family=fam, gseed=k, S_arm=s_arm, S_parent=s_par,
                           S_real=s_real, RF=rf))
        check("C", f"{tag} {r.window} {r.arm} {r.parent} S_arm", s_arm, r.S_arm)
        check("C", f"{tag} {r.window} {r.arm} {r.parent} RF", rf, r.RF)
CC = pd.DataFrame(C_rows)
for (tag, w), q in CC.groupby(["connectome", "window"]):
    piv = q.pivot(index="parent", columns="arm", values="S_arm")
    for arm in ["C1", "C1p", "C0"]:
        a = q[q.arm == arm]; rr = dec[(dec.exp == f"C_rescue_{arm}") & (dec.connectome == tag) & (dec.window == w)].iloc[0]
        check("C", f"{tag} {w} {arm} median RF", float(a.RF.median()), rr.median_RF)
        check("C", f"{tag} {w} {arm} networks above parent", int((a.S_arm > a.S_parent).sum()), int(rr.n_above_parent))
        check("C", f"{tag} {w} {arm} median S", float(a.S_arm.median()), rr.median_S_arm)
    check("C", f"{tag} {w} pairs with C1p above C0", int((piv.C1p > piv.C0).sum()), int(rr.n_C1p_gt_C0))
CC.to_csv(C.DERIVED / "r3_C_per_network.csv", index=False)
RB = rcsv(AN3 / "rescue_build_report.csv")
RB.to_csv(C.DERIVED / "r3_C_structure.csv", index=False)

# =============================================================================================== D2
cap = json.loads((AN3 / "cap_report.json").read_text())
D_rows = []
for tag in C.TAGS:
    Cp = C.r3_gmeans(RES3 / f"pairs_scalar_{tag}_cap.parquet"); assert set(Cp.rule) == {"real_setting"}
    for w in ["w15_25", "w50_60"]:
        xs = Cp[Cp.window == w].sort_values("gseed")
        rr = dec[(dec.exp == "D2_cap") & (dec.connectome == tag) & (dec.window == w)].iloc[0]
        check("D2", f"{tag} {w} X per network (3 decimals)", ",".join(f"{v:.3f}" for v in xs.S), rr.X_graphs)
        check("D2", f"{tag} {w} X median", float(xs.S.median()), rr.X_median)
        for x in xs.itertuples():
            m = cap[f"{tag}/mn_r3_cap_s{x.gseed}"]
            D_rows.append(dict(connectome=tag, window=w, gseed=x.gseed, S=x.S, rhythm=x.rhythm, R=rr.real, W=rr.W_median, S_over_R=x.S / rr.real,
                               redistribution=m["redistr"], median_rel_E_change=m["med_rel_dE"], median_rel_I_change=m["med_rel_dI"],
                               p95_rel_E_change=m["p95_rel_dE"], p95_rel_I_change=m["p95_rel_dI"], max_rel_E_change=m["max_rel_dE"],
                               max_rel_I_change=m["max_rel_dI"], cap=m["cap"]))
pd.DataFrame(D_rows).to_csv(C.DERIVED / "r3_D2_per_network.csv", index=False)

# =============================================================================================== F2
FV = rcsv(AN3 / "F_signerr/F_signerr_per_variant.csv")
F_rows = []
for d in [21, 22, 23]:
    G = C.r3_gmeans(RES3 / f"pairs_scalar_theta50_theta50_glu25_d{d}.parquet"); G = G[G.rule == "own"]
    for w in ["w15_25", "w50_60"]:
        q = G[G.window == w]; real = float(q[q.graph == "real"].S.iloc[0]); nul = q[q.graph != "real"]
        assert len(nul) == 10
        rr = FV[(FV.variant == f"theta50_glu25_d{d}") & (FV.window == w)].iloc[0]
        check("F2", f"draw {d} {w} real", real, rr.real); check("F2", f"draw {d} {w} rewired maximum", float(nul.S.max()), rr.null_max)
        check("F2", f"draw {d} {w} real above all 10", str(bool((real > nul.S).all())).lower(), str(rr.real_above_all).lower())
        for x in q.itertuples():
            F_rows.append(dict(draw=d, window=w, graph=x.graph, gseed=x.gseed, S=x.S, rhythm=x.rhythm))
pd.DataFrame(F_rows).to_csv(C.DERIVED / "r3_F2_per_network.csv", index=False)

# =============================================================================================== E and E1 (round-2 pair-level data)
verd = json.loads((AN3 / "D1_E_cpu/D1_E_verdicts.json").read_text())
P3W = rcsv(AN3 / "D1_E_cpu/E_P3_within_arm.csv")


def pair_frame(p):
    p = p[(p["mode"] == "stochastic") & (p.window == "w15_25") & (p.readout == "all")].copy()
    c = p["corr"].fillna(0.0)
    p["contrib"] = np.where(p.gated, -p["corr"], 0.0)
    p["Vp"] = (p.se ** 2 + p.sf ** 2 + 2 * c * p.se * p.sf) / 2
    p["Vm"] = (p.se ** 2 + p.sf ** 2 - 2 * c * p.se * p.sf) / 2
    p["D"] = p.Vm / (p.Vp + p.Vm)
    return p


def net_pair(df):
    return (df.groupby(["gseed", "pair"])
              .apply(lambda g: pd.Series(dict(n_gated=int(g.gated.sum()), Vp_mean=g.Vp.mean(), D_gated_mean=g.loc[g.gated, "D"].mean())),
                     include_groups=False)
              .reset_index())


E_sum = []
for tag in C.TAGS:
    raw = pd.read_parquet(RES2 / f"pairs_pairs_{tag}.parquet")
    preal = pair_frame(raw[(raw.graph == "real") & (raw.rule == "own")])
    sraw = pd.read_parquet(RES2 / f"pairs_pairs_{tag}_surg2.parquet")
    psurg = pair_frame(sraw[sraw.rule == "real_setting"])
    NP = pd.concat([net_pair(preal).assign(arm="real"), net_pair(psurg[psurg.graph == "mn_r2_within"]).assign(arm="W"),
                    net_pair(psurg[psurg.graph == "mn_r2_cross_hi"]).assign(arm="X_hi")], ignore_index=True)
    NP["active"] = NP.n_gated >= 10
    ref = rcsv(AN3 / f"D1_E_cpu/E_network_pair_{tag}.csv").merge(NP, on=["arm", "gseed", "pair"], suffixes=("_ref", ""))
    assert len(ref) == len(NP) == 220
    check("E", f"{tag} n_gated per network-pair (220 rows, max |diff|)", float(np.abs(ref.n_gated - ref.n_gated_ref).max()), 0.0)
    check("E", f"{tag} Var(y+) per network-pair (max relative diff)", float((np.abs(ref.Vp_mean - ref.Vp_mean_ref) / ref.Vp_mean_ref.abs()).max()), 0.0, tol=1e-9)
    dd = ref.dropna(subset=["D_gated_mean_ref"])
    check("E", f"{tag} D per network-pair (max |diff|)", float(np.abs(dd.D_gated_mean - dd.D_gated_mean_ref).max()), 0.0, tol=1e-12)
    # P1
    nw = NP[NP.arm == "W"]; nx = NP[NP.arm == "X_hi"]
    w_act = nw[nw.n_gated >= 10].groupby("pair").size(); w_act = w_act[w_act >= 3].index
    rat = (nx.groupby("pair").Vp_mean.median() / nw.groupby("pair").Vp_mean.median()).loc[w_act]
    P1 = float(rat.median())
    # P2
    def arm_D(arm):
        a = NP[(NP.arm == arm) & NP.active].groupby("pair").D_gated_mean.median()
        return float(a.median()), a
    Dr, dr = arm_D("real"); Dw, dw = arm_D("W"); Dx, dx = arm_D("X_hi")
    # P3
    GM = rcsv(AN3 / f"D1_E_cpu/E_gamma_{tag}.csv")
    J = NP[NP.active].merge(GM, on=["arm", "gseed", "pair"]).dropna(subset=["D_gated_mean"])
    rho = float(spearmanr(J.gamma, J.D_gated_mean).statistic)
    v = verd[f"E_{tag}"]
    check("E", f"{tag} P1 median Var(y+) ratio X_hi/W", P1, v["P1_median_ratio_Vplus_X_over_W"]); check("E", f"{tag} P1 pairs", len(w_act), v["P1_n_pairs"])
    check("E", f"{tag} D real", Dr, v["D_real"]); check("E", f"{tag} D W", Dw, v["D_W"]); check("E", f"{tag} D X_hi", Dx, v["D_X_hi"])
    for arm, dd_ in [("real", dr), ("W", dw), ("X_hi", dx)]:
        check("E", f"{tag} active pairs {arm}", len(dd_), v["n_active_pairs"][arm])
    tie = {}
    check("E", f"{tag} P3 Spearman (pooled)", rho, v["P3_spearman_gamma_D"], **tie); check("E", f"{tag} P3 n", len(J), v["P3_n"])
    for arm in ["real", "W", "X_hi"]:
        q = J[J.arm == arm]; ra = float(spearmanr(q.gamma, q.D_gated_mean).statistic)
        rr = P3W[(P3W.connectome == tag) & (P3W.arm == arm)].iloc[0]
        check("E", f"{tag} Spearman within {arm}", ra, rr.spearman_gamma_D, **tie); check("E", f"{tag} n within {arm}", len(q), int(rr.n))
        E_sum.append(dict(connectome=tag, arm=arm, spearman_within=ra, n=len(q)))
    E_sum.append(dict(connectome=tag, arm="pooled", spearman_within=rho, n=len(J), P1=P1, P1_pairs=len(w_act), D_real=Dr, D_W=Dw, D_X_hi=Dx))
    NP.assign(connectome=tag).to_csv(C.DERIVED / f"r3_E_network_pair_{tag}.csv", index=False)
    rat.rename("VpX_over_VpW").reset_index().assign(connectome=tag).to_csv(C.DERIVED / f"r3_E_P1_pairs_{tag}.csv", index=False)
    J.assign(connectome=tag).to_csv(C.DERIVED / f"r3_E_gamma_D_{tag}.csv", index=False)
    pd.concat([dr.rename("D").reset_index().assign(arm="real"), dw.rename("D").reset_index().assign(arm="W"),
               dx.rename("D").reset_index().assign(arm="X_hi")]).assign(connectome=tag).to_csv(C.DERIVED / f"r3_E_D_pairs_{tag}.csv", index=False)
    # E1
    pall = pair_frame(raw[raw.rule == "own"]); pall = pall[pall.graph.isin(["real"] + C.MAIN)].assign(set=lambda d: d.graph)
    ps = psurg[psurg.graph.isin(["mn_r2_within", "mn_r2_cross_hi"])].assign(set=lambda d: d.graph)
    e = pd.concat([pall, ps], ignore_index=True)
    e["anti_g"] = np.where(e.gated, -e["corr"], np.nan)
    e1 = e.groupby(["set", "gseed", "pair"]).agg(P_gate=("gated", "mean"), antiphase_given_gate=("anti_g", "mean"), contrib=("contrib", "mean")).reset_index()
    sm = (e1.groupby(["set", "gseed"]).agg(mean_P_gate=("P_gate", "mean"), mean_antiphase_given_gate=("antiphase_given_gate", "mean"), S=("contrib", "mean"))
            .groupby("set").median())
    ref1 = rcsv(AN3 / f"D1_E_cpu/E1_gate_decomposition_summary_{tag}.csv").set_index("set")
    for st in ref1.index:
        for k in ["mean_P_gate", "mean_antiphase_given_gate", "S"]:
            check("E1", f"{tag} {st} {k}", float(sm.loc[st, k]), float(ref1.loc[st, k]))
    sm.reset_index().assign(connectome=tag).to_csv(C.DERIVED / f"r3_E1_summary_{tag}.csv", index=False)
    e1.assign(connectome=tag).to_csv(C.DERIVED / f"r3_E1_pairs_{tag}.csv", index=False)
pd.DataFrame(E_sum).to_csv(C.DERIVED / "r3_E_summary.csv", index=False)

# =============================================================================================== sign structure of the index (post hoc)
RG = rcsv(C.RISEM / "ri_semantics_graphs.csv"); SS = rcsv(C.RISEM / "ri_semantics_real_sign_split.csv")
RI_rows = []
for tag in C.TAGS:
    q = RG[RG.connectome == tag]; real = q[q.graph == "real"].iloc[0]; nul = q[q.family.isin(C.MAIN)]
    assert len(nul) == 30
    sp = SS[SS.connectome == tag]
    RI_rows.append(dict(connectome=tag, f_opp_real=real.f_opp, f_both_real=real.f_both, f_opp_rewired_min=nul.f_opp.min(), f_opp_rewired_max=nul.f_opp.max(),
                        share_exc_median=sp.opp_mass_from_exc.median(), share_inh_median=sp.opp_mass_from_inh.median(), n_pairs=len(sp)))
RI = pd.DataFrame(RI_rows).set_index("connectome")
RI.reset_index().to_csv(C.DERIVED / "r3_RI_sign_summary.csv", index=False)

# =============================================================================================== numbers stated in the requests (v7 writing prompt, v8 figure prompt)
REQ = [("theta50", "f_opp_real", 0.88, 0.005), ("manc50", "f_opp_real", 0.83, 0.005), ("theta50", "share_exc_median", 0.45, 0.005),
       ("manc50", "share_exc_median", 0.42, 0.005), ("theta50", "share_inh_median", 0.55, 0.005), ("manc50", "share_inh_median", 0.58, 0.005),
       ("theta50", "f_both_real", 0.96, 0.01), ("manc50", "f_both_real", 0.97, 0.01)]
for tag, k, v, tol in REQ:
    check("request", f"{C.TAGS[tag]} {k}", round(float(RI.loc[tag, k]), 2), v, tol=tol)
check("request", "rewired f_opp range, lower end (both)", round(float(RI.f_opp_rewired_min.min()), 2), 0.16, tol=0.005)
check("request", "rewired f_opp range, upper end (both)", round(float(RI.f_opp_rewired_max.max()), 2), 0.49, tol=0.005)
bl = pd.DataFrame(B_lv).set_index(["connectome", "window"])
check("request", "B MaleCNS rewired at or above real (15-25 s)", int(bl.loc[("theta50", "w15_25"), "n_at_or_above_real"]), 0)
check("request", "B MANC rewired at or above real (15-25 s)", int(bl.loc[("manc50", "w15_25"), "n_at_or_above_real"]), 4)
check("request", "B number of rewired networks", int(bl.loc[("theta50", "w15_25"), "n_nulls"]), 90)

out = pd.DataFrame(chk)
out.to_csv(C.QA / "round3_check.csv", index=False)
bad = out[~out.match]
print(f"round-3 check: {int(out.match.sum())} / {len(out)} items match")
if len(bad):
    print(bad.to_string())
    raise SystemExit("round-3 check failed")
