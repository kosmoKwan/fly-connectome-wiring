"""Build the documentation for the figure set from the computed outputs (no hand-typed numbers):

  figure_numbers.csv     every number printed on a figure, with its source and (where the manuscript states it) the check
  captions_draft.md      English figure legends (numbers filled in from the source data)
  README.md              Chinese overview: contents, data sources and filters per panel, verification, open items
  Claude插图_v8第三轮_手机版.pdf  one PDF for reading on a phone: summary, the 13 figures with legends, verification tables
v8 (2026-09-30): round-3 results added; Fig. 1-6 and Extended Data (Supplementary) Fig. 1-7, numbered as agreed with the writing session.
"""
import json, math, re, textwrap, datetime
import numpy as np, pandas as pd
import pymupdf
import common as C

SD = C.SRC_DATA; QA = C.QA; BASE = C.BASE
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
f3 = lambda x: C.fmt(float(x), 3).replace("−", "−")
f2 = lambda x: C.fmt(float(x), 2)
pct = lambda x: f"{100 * float(x):.0f}%"
rd = lambda name: pd.read_csv(SD / f"{name}.csv")

# ------------------------------------------------------------------------------------------------ gather values
V = {}
b1 = rd("ED1b_composition").set_index("connectome")
d1 = rd("ED1d_scan_real_malecns")
best1 = d1[d1.eligible].sort_values("S", ascending=False).iloc[0]
ex = pd.read_csv(C.DERIVED / "example_runs_summary.csv").set_index("example")
sc2 = rd("Fig2bc_scan_best")
st2 = rd("Fig2d-g_confirmation_by_family")
ph2 = rd("Fig2hi_real_runs_by_window")
a3 = rd("Fig3a_influences_T2L_ThC"); b3 = rd("Fig3b_network_index"); c3 = rd("ED3a_pair_index_real_vs_rewired"); d3 = rd("ED3c_index_malecns_vs_manc")
cd4 = rd("Fig4cd_S_relative_to_real"); e4 = rd("Fig4e_activity_kept"); f4 = rd("Fig4f_pair_correlations"); g4 = rd("Fig4g_dose")
ed1d = rd("ED7d_switching_rates"); ed1ab = rd("ED7ab_ensemble_trajectories"); ed1e = rd("ED7e_last_window")
ed2a = rd("ED4a_design_check"); TL = ed2a[ed2a.arm.str.startswith("mn_r2")].copy()
TLR = {}
TLR_C = {(t_, a_): (min(q_.p95_rel_E_change.min(), q_.p95_rel_I_change.min()), max(q_.p95_rel_E_change.max(), q_.p95_rel_I_change.max()))
         for (t_, a_), q_ in TL.groupby(["connectome", "arm"])}
for arm_, lab_ in [("mn_r2_within", "W"), ("mn_r2_cross_lo", "X_lo"), ("mn_r2_cross_hi", "X_hi")]:
    q_ = TL[TL.arm == arm_]
    TLR[lab_] = dict(med=(min(q_.median_rel_E_change.min(), q_.median_rel_I_change.min()), max(q_.median_rel_E_change.max(), q_.median_rel_I_change.max())),
                     p95=(min(q_.p95_rel_E_change.min(), q_.p95_rel_I_change.min()), max(q_.p95_rel_E_change.max(), q_.p95_rel_I_change.max())),
                     p95_plot=(q_.p95_plotted.min(), q_.p95_plotted.max()))
rng = lambda t: f"{100 * t[0]:.1f}–{100 * t[1]:.1f}%"
XLO_SR = rd("Fig4g_dose").query("group == 'mn_r2_cross_lo'").groupby("connectome").S_over_R.median()
assert all(abs(TLR["W"]["p95"][k] - TLR["X_lo"]["p95"][k]) <= 0.01 for k in (0, 1)), "W and X_lo tails no longer similar"
assert (XLO_SR < 0.5).all(), "X_lo no longer lost most of the coordination"
rng0 = lambda t: f"{100 * t[0]:.0f}–{100 * t[1]:.0f}%"
ed2b = rd("ED4b_conditions"); ed2c = rd("ED4c_reoptimised"); ed2d = rd("ED4d_activity_kept")
ed3a = rd("ED6a_levels"); ed3b = rd("ED6b_pair_scores_by_window")
R2CHK = pd.read_csv(QA / "round2_check.csv")
a5 = rd("Fig6a_pair_scores_map"); b5 = rd("Fig6b_joint_scores"); c5 = rd("Fig6c_readouts_levels")
rep = C.exp2_report()
ova = rd("Fig1a_network_sizes").set_index("connectome"); ovc = rd("Fig1c_rhythm_and_coordination")
ovd1 = rd("Fig1d_reciprocal_index"); ovdm = rd("Fig1d_dose_response_medians"); ovdn = rd("Fig1d_dose_response_networks")
ove = rd("Fig1e_joint_scores")
LVL = pd.read_csv(C.DERIVED / "r2_exp1_levels.csv")
nums = []


def add(fig, panel, label, printed, source, definition, manuscript="", note=""):
    nums.append(dict(figure=fig, panel=panel, quantity=label, value_on_figure=printed, source=source, definition=definition,
                     manuscript_value=manuscript, note=note))


# Fig. 1 (overview)
OV = {}
for t in C.TAGS:
    add("1", "a", f"{C.TAGS[t]} neurons / connections", f"{int(ova.loc[t].neurons):,} / {int(ova.loc[t].connections):,}", "Fig1a_network_sizes.csv",
        "rows of the node / edge tables", {"theta50": "10,173 / 462,650", "manc50": "10,440 / 634,207"}[t])
    q = ovc[ovc.connectome == t]; rv = q[q.network == "real"].iloc[0]; nul = q[q.network != "real"]
    lv = LVL[(LVL.connectome == t) & (LVL.window == "w15_25") & (LVL.readout == "S_all")].level.iloc[0].lower()
    ri = ovd1[ovd1.connectome == t]
    OV[t] = dict(N=int(ova.loc[t].neurons), E=int(ova.loc[t].connections), rh=rv.rhythmicity, n_rh=int((nul.rhythmicity >= rv.rhythmicity).sum()),
                 S=rv.S, Smax=nul.S.max(), lv=lv, ri=ri[ri.network == "real"].recip_index.iloc[0], rimax=ri[ri.network != "real"].recip_index.max())
    add("1", "c", f"{C.TAGS[t]} real rhythmicity", f2(rv.rhythmicity), "Fig1c_rhythm_and_coordination.csv",
        "own frozen setting, 15–25 s, mean of 20 noisy runs")
    add("1", "c", f"{C.TAGS[t]} rewired networks at least as rhythmic as real", f"{OV[t]['n_rh']} of {len(nul)}", "Fig1c_rhythm_and_coordination.csv",
        "rhythmicity ≥ real")
    add("1", "c", f"{C.TAGS[t]} real S", f2(rv.S), "Fig1c_rhythm_and_coordination.csv", "mean of 20 noisy runs",
        {"theta50": "0.31", "manc50": "0.17"}[t] + " (request)")
    add("1", "c", f"{C.TAGS[t]} best rewired S", f2(nul.S.max()), "Fig1c_rhythm_and_coordination.csv", "maximum over the 30 rewired networks",
        {"theta50": "0.14", "manc50": "0.08"}[t] + " (request)")
    add("1", "c", f"{C.TAGS[t]} pre-specified level", lv, "derived/r2_exp1_levels.csv", "§3.23 rule, 15–25 s", {"theta50": "strong", "manc50": "moderate"}[t])
    add("1", "d", f"{C.TAGS[t]} reciprocal-innervation index, real (max rewired)", f"{f2(OV[t]['ri'])} ({f3(OV[t]['rimax'])})",
        "Fig1d_reciprocal_index.csv", "median over the 20 pairs", {"theta50": "0.48", "manc50": "0.38"}[t])
    for _, s_ in ovdm[ovdm.connectome == t].iterrows():
        add("1", "d", f"{C.TAGS[t]} {s_.step}: x, median S/R", f"{100 * s_.x:.1f}%, {f2(s_.median_S_over_R)}", "Fig1d_dose_response_medians.csv",
            f"median of {int(s_.n)} networks, 15–25 s")
MAXPCT = math.ceil(100 * ovdn.strength_change.max())
BELOW = "below" if 100 * ovdn.strength_change.max() < MAXPCT else "at most"
add("1", "d", "input-strength change, strength-preserving arms", f"median change {'<' if BELOW == 'below' else '≤'} {MAXPCT}%", "Fig1d_dose_response_networks.csv",
    "larger of the median relative E and I input change, every network", note=f"maximum {100 * ovdn.strength_change.max():.2f}%")
for _, r in ove.iterrows():
    add("1", "e", f"{C.TAGS[r.connectome]} {r.joint}: real S (best rewired)", f"{f2(r.real)} ({f2(r.best_rewired)})", "Fig1e_joint_scores.csv",
        "joint readout, 15–25 s", note=f"level {r.level_15_25}; strong in {int(r.windows_strong)} of {int(r.n_windows)} windows")
thc_all = bool((ove[ove.joint == "ThC"].windows_strong == ove[ove.joint == "ThC"].n_windows).all())
add("1", "e", "ThC 'strong' in both connectomes, all windows", "all 4 windows" if thc_all else "no", "Fig1e_joint_scores.csv", "windows_strong")
# Extended Data Fig. 1 (formerly Fig. 1)
for t in C.TAGS:
    r = b1.loc[t]
    add("ED1", "b", f"{C.TAGS[t]} neurons", f"{int(r.neurons):,}", "ED1b_composition.csv", "rows of the node table",
        {"theta50": "10,173", "manc50": "10,440"}[t])
    add("ED1", "b", f"{C.TAGS[t]} connections", f"{int(r.connections):,}", "ED1b_composition.csv", "rows of the edge table",
        {"theta50": "462,650", "manc50": "634,207"}[t])
    add("ED1", "b", f"{C.TAGS[t]} DN / SN / MN / IN", f"{int(r.n_DN)} / {int(r.n_SN)} / {int(r.n_MN)} / {int(r.n_IN):,}", "ED1b_composition.csv",
        "neurons by role", "424 / 666 / 381 / 8,702" if t == "theta50" else "MN 396")
    add("ED1", "b", f"{C.TAGS[t]} MNs exact / approx / unmapped", f"{int(r.MN_exact)} / {int(r.MN_approx)} / {int(r.MN_unmapped)}",
        "ED1b_composition.csv", "joint assignment in the MN map", "302 mapped, 76 approximate" if t == "theta50" else "")
    add("ED1", "b", f"{C.TAGS[t]} antagonistic pairs", f"{int(r.pairs)}", "ED1b_composition.csv", "leg x joint with both directions", "20")
add("ED1", "d", "eligible scan settings, real MaleCNS", f"{int(d1.eligible.sum())} of {len(d1)}", "ED1d_scan_real_malecns.csv",
    ">20% MNs oscillating and median dominant frequency 1-50 Hz", "132 settings")
add("ED1", "d", "frozen setting, real MaleCNS", f"log10 g = {best1.log10_g:.1f}, β = {best1.beta:g}, d = {best1.drive:g}",
    "ED1d_scan_real_malecns.csv", "eligible setting with the highest S")
# Fig. 2
add("2", "a", "example-run S, real / lineage block 3", f"{f2(ex.loc['malecns_real','cpu_S'])} / {f2(ex.loc['malecns_hl_s3','cpu_S'])}",
    "derived/example_runs_summary.csv", "CPU example simulation, 15-25 s, all 20 pairs",
    note=f"confirmation runs of the same networks: {f3(ex.loc['malecns_real','gpu_S_min'])}–{f3(ex.loc['malecns_real','gpu_S_max'])} and "
         f"{f3(ex.loc['malecns_hl_s3','gpu_S_min'])}–{f3(ex.loc['malecns_hl_s3','gpu_S_max'])}")
a2 = rd("Fig2a_example_ThC_pairs")
for _, r in a2.iterrows():
    add("2", "a", f"r, {r.pair}, {'real' if r.example == 'malecns_real' else 'lineage block 3'}",
        f2(r.corr_15_25) if r.gated else "n.m.", "Fig2a_example_ThC_pairs.csv", "extensor–flexor correlation, 15–25 s")
for t, (ms, mn_) in {"theta50": ("0.330", "0.139"), "manc50": ("0.215", "0.106")}.items():
    q = sc2[sc2.connectome == t]; rv = q[q.graph == "real"].iloc[0]; nul = q[q.graph != "real"]
    add("2", "b" if t == "theta50" else "c", f"{C.TAGS[t]} scan: best S real", f3(rv.bestS), "Fig2bc_scan_best.csv", "max S over eligible settings", ms)
    add("2", "b" if t == "theta50" else "c", f"{C.TAGS[t]} scan: best S of any rewired network", f3(nul.bestS.max()), "Fig2bc_scan_best.csv",
        "max over rewired networks", mn_)
    add("2", "b" if t == "theta50" else "c", f"{C.TAGS[t]} rewired networks more rhythmic than real", f"{int(rv.n_rewired_more_rhythmic_than_real)} of {len(nul)}",
        "Fig2bc_scan_best.csv", "max rhythmicity over eligible settings", "several (up to 0.96 vs 0.70)" if t == "theta50" else "")
for (t, w), p in zip([("theta50", "w15_25"), ("theta50", "w50_60"), ("manc50", "w15_25"), ("manc50", "w50_60")], "defg"):
    q = st2[(st2.connectome == t) & (st2.window == w)]
    lvl = q[q.family == "__level__"].level.iloc[0]
    add("2", p, f"{C.TAGS[t]} {C.WIN_LABEL[w]} real S", f3(q.real.iloc[0]), "Fig2d-g_confirmation_by_family.csv", "mean of 20 runs",
        {"theta50": "0.312", "manc50": {"w15_25": "0.173", "w50_60": "0.128"}[w]}[t])
    add("2", p, f"{C.TAGS[t]} {C.WIN_LABEL[w]} pre-specified level", lvl, "Fig2d-g_confirmation_by_family.csv", "§3.23 rule",
        {"theta50": "strong", "manc50": "moderate"}[t])
    und = q[q.n_undefined_runs > 0]
    if len(und):
        u = und.iloc[0]
        add("2", p, "undefined runs (leg × lin. + recip., network 2)", f"{int(u.n_undefined_runs)} of 20, mean of {int(u.n_defined_runs)}",
            "Fig2d-g_confirmation_by_family.csv", "runs without an oscillating MN", "9 of 20, mean of 11")
for t, p in [("theta50", "h"), ("manc50", "i")]:
    q = ph2[ph2.connectome == t]
    swr = q[(q.switched_run) & (q.window == "w15_25")]
    n_sw = int((swr["mode"] == "stochastic").sum()); n_dsw = int((swr["mode"] == "deterministic").sum())
    add("2", p, f"{C.TAGS[t]} runs that switched to a fast state", f"{n_sw} of 20 noisy, {n_dsw} of 3 noise-free", "Fig2hi_real_runs_by_window.csv",
        "fast state: ≥3 Hz and ≤3 gated pairs, absent at 15–25 s", {"theta50": "", "manc50": "4 of 20; 1 of 3"}[t])
    if t == "theta50":
        f15 = q[(q["mode"] == "stochastic") & (q.window == "w15_25")].freq_hz
        add("2", p, "MaleCNS dominant frequency, 15–25 s", f"{f15.min():.1f}–{f15.max():.1f} Hz", "Fig2hi_real_runs_by_window.csv",
            "stochastic runs", "2.4–4.1 Hz")
        add("2", p, "MaleCNS run-windows in a fast state", f"{int(q.fast_state.sum())} of {len(q)}", "Fig2hi_real_runs_by_window.csv", "all runs")
    else:
        fs = q[(q.switched_run) & q.fast_state & (q["mode"] == "stochastic")].freq_hz
        st15 = q[(q["mode"] == "stochastic") & (q.window == "w15_25")]
        start = st15[st15.fast_state]
        keep = st15[(~st15.switched_run) & (~st15.fast_state)].freq_hz
        F2I = dict(start=", ".join(f"run {int(r_.run)} ({r_.freq_hz:.1f} Hz)" for r_ in start.itertuples()), lo=keep.min(), hi=keep.max())
        add("2", p, "MANC switched-state frequency", f"{fs.min():.0f}–{fs.max():.0f} Hz", "Fig2hi_real_runs_by_window.csv", "fast windows of switched runs",
            "3–15 Hz", note=f"exact {fs.min():.1f}–{fs.max():.1f} Hz")
        add("2", p, "MANC run fast from the start", F2I["start"], "Fig2hi_real_runs_by_window.csv", "fast state already at 15–25 s",
            "run 18 at 7.6 Hz (writing-session note)")
        add("2", p, "MANC frequency of the other runs, 15–25 s", f"{keep.min():.1f}–{keep.max():.1f} Hz", "Fig2hi_real_runs_by_window.csv",
            "neither switched nor fast from the start", "near 1 Hz", note=f"median {keep.median():.2f} Hz; was printed as '≈1 Hz' before this version")
# Fig. 3
for g in ["real", "leg_s0"]:
    q = a3[a3.network == g].iloc[0]
    add("3", "a", f"index of T2L_ThC, {g}", f2(q["index"]), "Fig3a_influences_T2L_ThC.csv", "−Pearson r across neurons", note=f"archived {q.archived_index:.6f}")
for t in C.TAGS:
    q = b3[b3.connectome == t]
    add("3", "b", f"{C.TAGS[t]} real network index", f2(q.real.iloc[0]), "Fig3b_network_index.csv", "median over 20 pairs",
        {"theta50": "0.48", "manc50": "0.38"}[t])
    add("3", "b", f"{C.TAGS[t]} maximum over 30 rewired networks", f3(q[q.family == "__max_main__"]["index"].iloc[0]), "Fig3b_network_index.csv",
        "all ≤ 0 expected", "zero or negative")
    qc = c3[c3.connectome == t]
    add("ED3", "a", f"{C.TAGS[t]} pairs with real above all 30 rewired", f"{int(qc.real_above_all.sum())} of {len(qc)}", "ED3a_pair_index_real_vs_rewired.csv",
        "real > max of 30", {"theta50": "19 of 20", "manc50": "18 of 20"}[t])
add("ED3", "c", "Spearman ρ, MaleCNS vs MANC per-pair index", f2(d3.spearman_rho.iloc[0]), "ED3c_index_malecns_vs_manc.csv", f"n = {len(d3)} pairs", "0.82")
# Fig. 4
TG = {"theta50": "c", "manc50": "d"}
GNAME = {"mn_shuffle_pool": "within-pool, original", "mn_shuffle_joint": "cross-pool, original", "mn_shuffle_pool_w": "within-pool, strength-matched",
         "mn_shuffle_joint_w": "cross-pool, strength-matched", "mn_r2_within": "W (dose-matched within-pool)", "mn_r2_cross_lo": "X_lo",
         "mn_r2_cross_hi": "X_hi (dose-matched cross-pool)"}
MAN4 = {("theta50", "mn_shuffle_joint"): "−0.071 / −0.070", ("manc50", "mn_shuffle_joint"): "−0.056 / −0.051",
        ("theta50", "mn_shuffle_pool"): "0.298 / 0.297", ("manc50", "mn_shuffle_pool"): "0.140 / 0.108",
        ("theta50", "mn_shuffle_joint_w"): "0.170 / 0.163", ("manc50", "mn_shuffle_joint_w"): "0.094 / 0.065",
        ("theta50", "mn_shuffle_pool_w"): "0.297 / 0.309", ("manc50", "mn_shuffle_pool_w"): "0.166 / 0.139",
        ("theta50", "mn_r2_within"): "W 0.277 (request)", ("manc50", "mn_r2_within"): "W 0.135 (request)",
        ("theta50", "mn_r2_cross_hi"): "X −0.076 (request)", ("manc50", "mn_r2_cross_hi"): "X −0.096 (request)"}
for t in C.TAGS:
    q = cd4[cd4.connectome == t]
    R = q.groupby("window").R.first()
    add("4", TG[t], f"{C.TAGS[t]} R (15–25 / 50–60 s)", f"{f3(R['w15_25'])} / {f3(R['w50_60'])}", "Fig4cd_S_relative_to_real.csv", "real network S",
        {"theta50": "0.312 / 0.312", "manc50": "0.173 / 0.128"}[t])
    for g, lab in GNAME.items():
        m = q[q.group == g].groupby("window").median_S.first(); mr = q[q.group == g].groupby("window").median_S_over_R.first()
        add("4", TG[t], f"{C.TAGS[t]} {lab}: median S (15–25 / 50–60 s)", f"{f3(m['w15_25'])} / {f3(m['w50_60'])}", "Fig4cd_S_relative_to_real.csv",
            "median of 5 networks, plotted as S/R", MAN4.get((t, g), ""), note=f"S/R {f2(mr['w15_25'])} / {f2(mr['w50_60'])}")
    for arm in ["cross_lo", "cross_hi"]:
        xs = [v["redistr"] for k, v in rep.items() if k.startswith(f"{t}/mn_r2_{arm}_s")]
        add("4", TG[t], f"{C.TAGS[t]} {arm} tick label (designed redistribution)", pct(np.median(xs)), "surgery_r2_report.json", "median of 5 graphs",
            "25–28% for cross_hi" if arm == "cross_hi" else "")
for _, r in e4.iterrows():
    add("4", "e", f"{C.TAGS[r.connectome]} {C.WIN_LABEL[r.window]} {r.arm} cross-pool / real, {r.metric}", f2(r.ratio), "Fig4e_activity_kept.csv",
        "median of 5 networks / real", "MANC osc 0.69 (request)" if (r.connectome == "manc50" and r.arm == "dose-matched" and r.metric == "osc"
                                                                     and r.window == "w15_25") else "")
for t in C.TAGS:
    q = f4[(f4.connectome == t) & (f4.network == "mn_shuffle_joint")]
    n_pairs = q.pair.nunique()
    ret = q.groupby("gseed").apply(lambda d: (d.gated_frac >= 0.5).mean(), include_groups=False)
    add("4", "f", f"{C.TAGS[t]} contributing pairs still modulated after cross-pool", f"{int(round(np.median(ret) * n_pairs))} of {n_pairs}",
        "Fig4f_pair_correlations.csv", "median over 5 networks", {"theta50": "8 of 9", "manc50": "7 of 8 (progress log)"}[t])
for t in C.TAGS:
    for g in ["mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi", "mn_shuffle_joint"]:
        q = g4[(g4.connectome == t) & (g4.group == g)]
        req = {("theta50", "mn_shuffle_joint_w"): "0.55", ("theta50", "mn_r2_cross_lo"): "0.13", ("theta50", "mn_r2_cross_hi"): "−0.24",
               ("theta50", "mn_shuffle_joint"): "−0.23", ("manc50", "mn_shuffle_joint_w"): "0.54", ("manc50", "mn_r2_cross_lo"): "0.14",
               ("manc50", "mn_r2_cross_hi"): "−0.56", ("manc50", "mn_shuffle_joint"): "−0.32"}[(t, g)]
        add("4", "g", f"{C.TAGS[t]} {GNAME[g]}: redistribution, median S/R (15–25 s)", f"{q.redistribution.median():.3f}, {f2(q.S_over_R.median())}",
            "Fig4g_dose.csv", "per-network points; medians shown", f"X/R {req} (request)",
            note=f"median |ΔE| {pct(q.median_rel_E_change.median())}, |ΔI| {pct(q.median_rel_I_change.median())}")
add("4", "b", "example-run S, real / cross-pool 0", f"{f2(ex.loc['malecns_real','cpu_S'])} / {f2(ex.loc['malecns_cross_s0','cpu_S'])}",
    "derived/example_runs_summary.csv", "CPU example simulation")
# Fig. 6 (formerly 5)
for t in C.TAGS:
    qa_ = a5[a5.connectome == t]
    add("6", "a", f"{C.TAGS[t]} pairs where real exceeds all 30 rewired (rings)", f"{int(qa_.real_above_all_rewired.sum())} of {int(qa_.real_score.notna().sum())}",
        "Fig6a_pair_scores_map.csv", "pair score, 15–25 s")
    for j in C.JOINTS:
        q = b5[(b5.connectome == t) & (b5.joint == j)]
        rv = q[q.network == "real"].score.iloc[0]; nm = q[q.network != "real"].score.max()
        man = {("theta50", "ThC"): "0.71 (≤0.22)", ("theta50", "CTr"): "0.25 (≤0.24)", ("theta50", "FTi"): "0.09", ("theta50", "TiTa"): "0",
               ("manc50", "ThC"): "0.271 (≤0.105) (request)"}.get((t, j), "")
        add("6", "b", f"{C.TAGS[t]} {j} mean pair score, real (max rewired)", f"{f2(rv)} ({f2(nm)})", "Fig6b_joint_scores.csv", "mean of 20 runs, 15–25 s", man)
    for _, r in c5[c5.connectome == t].iterrows():
        man = {("theta50", "S_exact", "w15_25"): "0.200, strong", ("theta50", "S_nonThC", "w15_25"): "0.143, moderate",
               ("manc50", "S_exact", "w15_25"): "0.122, moderate (request)", ("manc50", "S_nonThC", "w15_25"): "0.131, moderate (request)"}.get(
            (t, r.readout, r.window), "")
        add("6", "c", f"{C.TAGS[t]} {r.readout} {C.WIN_LABEL[r.window]}: real, level (max rewired)", f"{f3(r.real)}, {r.level} ({f3(r.max_rewired)})",
            "Fig6c_readouts_levels.csv", "pre-specified readout and level", man)
# Extended Data Fig. 7 (formerly 2)
REQ3 = {("manc50", "real", 0): ("0.088", "45%", "0.24"), ("manc50", "hl", 1): ("0.076", "10%", "3.7"), ("manc50", "hl", 0): ("0.072", "0%", "5.6"),
        ("theta50", "real", 0): ("0.312", "100%", "0")}
LAB3 = {("manc50", "real", 0): "MANC real", ("manc50", "hl", 1): "MANC lineage block 1", ("manc50", "hl", 0): "MANC lineage block 0",
        ("theta50", "real", 0): "MaleCNS real"}
for key, lab in LAB3.items():
    tr = ed1ab[(ed1ab.connectome == key[0]) & (ed1ab.graph == key[1]) & (ed1ab.gseed == key[2])]
    last = tr[tr.t_start == tr.t_start.max()].iloc[0]
    add("ED7", "a", f"{lab}: coordinated at 290–300 s", f"{100 * last.frac_coord:.0f}%", "ED7ab_ensemble_trajectories.csv", "fraction of 40 noisy runs",
        f"{REQ3[key][1]} (request)")
    add("ED7", "b", f"{lab}: mean S at 290–300 s", f3(last.mean_S), "ED7ab_ensemble_trajectories.csv", "undefined S counted as 0", f"{REQ3[key][0]} (request)")
    r = ed1d[(ed1d.connectome == key[0]) & (ed1d.graph == key[1]) & (ed1d.gseed == key[2])].iloc[0]
    txt = f"{r.rate_per_min:.2g} ({int(r.switch_events)} events)" if r.switch_events else f"0 events (95% CI < {r.ci_hi:.2f})"
    add("ED7", "d", f"{lab}: switching rate per minute coordinated", txt, "ED7d_switching_rates.csv", "exact Poisson 95% CI",
        f"{REQ3[key][2]} (request)", note=f"95% CI {r.ci_lo:.2f}–{r.ci_hi:.2f}; mean dwell {r.mean_dwell_min:.2f} min" if r.switch_events else "")
# Extended Data Fig. 4 (formerly 3)
for (t, arm, rule), q in ed2b.groupby(["connectome", "arm", "rule"]):
    for w in ["w15_25", "w50_60"]:
        r = q[q.window == w].iloc[0]
        failed = [c for c in ["C1", "C2", "C3", "C4", "C5", "C6"] if not r[c]]
        add("ED4", "b", f"{C.TAGS[t]} {arm} {rule} {C.WIN_LABEL[w]}", "all six met" if r.all_six else "not met: " + ", ".join(failed),
            "ED4b_conditions.csv", "pre-registered C1–C6",
            {("theta50", "mn_r2_cross_hi", "real_setting", "w15_25"): "all six met (request)",
             ("manc50", "mn_r2_cross_hi", "real_setting", "w15_25"): "C6 fails (request; C1 also fails, see README)"}.get((t, arm, rule, w), ""))
for lab_, v_ in TLR.items():
    add("ED4", "a", f"{lab_}: median / 95th percentile of the MN input change (E and I, both connectomes)", f"{rng(v_['med'])} / {rng(v_['p95'])}",
        "ED4a_design_check.csv", "per-network median and 95th percentile of the relative change of E and of I input",
        {"W": "1.0–2.3% / 9–19%", "X_lo": "1.5–2.6% / 9–20%", "X_hi": "2.0–3.9% / 11–30%"}[lab_] + " (writing session)",
        note=f"plotted 95th percentile (larger of E and I): {rng(v_['p95_plot'])}")
for (t, arm), q in ed2a.groupby(["connectome", "arm"]):
    add("ED4", "a", f"{C.TAGS[t]} {arm}: redistribution, median |ΔE| / |ΔI|", f"{q.redistribution.median():.3f}, {pct(q.median_rel_E_change.median())} / "
        f"{pct(q.median_rel_I_change.median())}", "ED4a_design_check.csv", "median of 5 networks")
for (t, g, w), q in ed2c.groupby(["connectome", "graph", "window"]):
    add("ED4", "c", f"{C.TAGS[t]} {g} own setting {C.WIN_LABEL[w]}: median S", f3(q.S.median()), "ED4c_reoptimised.csv", "median of 5 networks")
for _, r in ed2d.iterrows():
    add("ED4", "d", f"{C.TAGS[r.connectome]} {r.arm} {r.rule} {C.WIN_LABEL[r.window]}: osc coverage / pairs kept", f"{f2(r.osc_coverage)} / {f2(r.pair_retention)}",
        "ED4d_activity_kept.csv", "relative to real; criterion 0.8")
# Extended Data Fig. 6 (formerly 4)
for _, r in ed3a.iterrows():
    add("ED6", "a", f"{C.TAGS[r.connectome]} {r.readout} {C.WIN_LABEL[r.window]}", f"{f2(r.real)} ({r.level})", "ED6a_levels.csv",
        "real value and pre-specified level")
# ---------------- round 3 (v8): new and extended panels
R3 = {k: pd.read_csv(C.DERIVED / f"r3_{k}.csv") for k in ["A_levels", "A_scan_max_rhythm", "B_levels", "C_per_network", "C_structure",
                                                          "D2_per_network", "F2_per_network", "E_summary", "RI_sign_summary"]}
R3CHK = pd.read_csv(QA / "round3_check.csv")
DEC3 = pd.read_csv(C.R3A / "全部判定/round3_decisions.csv")
RIS = R3["RI_sign_summary"].set_index("connectome"); ES3 = R3["E_summary"]; EP = ES3[ES3.arm == "pooled"].set_index("connectome")
E1S = {t: pd.read_csv(C.DERIVED / f"r3_E1_summary_{t}.csv").set_index("set") for t in C.TAGS}
for t in C.TAGS:
    r = RIS.loc[t]
    add("3", "c", f"{C.TAGS[t]} f_opp real (rewired range)", f"{f2(r.f_opp_real)} ({f2(r.f_opp_rewired_min)}–{f2(r.f_opp_rewired_max)})", "Fig3c_f_opp.csv",
        "opposite-signed share of the shared influence, median over 20 pairs", {"theta50": "0.88 (0.16–0.49)", "manc50": "0.83 (0.16–0.49)"}[t] + " (request)")
    add("3", "c", f"{C.TAGS[t]} influence from neurons reaching both pools", f"{100 * r.f_both_real:.0f}%", "Fig3c_f_opp.csv", "f_both, real network",
        {"theta50": "96%", "manc50": "97%"}[t] + " (v7 request)")
    add("3", "d", f"{C.TAGS[t]} route shares (excitatory / inhibitory)", f"{100 * r.share_exc_median:.0f}% / {100 * r.share_inh_median:.0f}%", "Fig3d_route_shares.csv",
        "median over 20 pairs", {"theta50": "45% / 55%", "manc50": "42% / 58%"}[t] + " (request)")
D2 = R3["D2_per_network"]; D2w = D2[D2.window == "w15_25"]
for t in C.TAGS:
    q = D2w[D2w.connectome == t]; dd = DEC3[(DEC3.exp == "D2_cap") & (DEC3.connectome == t) & (DEC3.window == "w15_25")].iloc[0]
    add("4", "g", f"{C.TAGS[t]} cap 5%: redistribution, median S/R (15–25 s)", f"{q.redistribution.median():.3f}, {f2(q.S_over_R.median())}", "Fig4g_dose.csv",
        "group mn_r3_cap; median of 5 networks", note=f"X median {f3(dd.X_median)}; pairs kept {f2(dd.pair_retention)}, osc kept {f2(dd.osc_retention)} (criterion 6 not met)")
    add("4", "g", f"{C.TAGS[t]} cap 5%: largest per-MN input change", f"{100 * max(q.max_rel_E_change.max(), q.max_rel_I_change.max()):.1f}%", "derived/r3_D2_per_network.csv",
        "cap_report.json max_rel_dE/dI", "≤ 5%")
for t in C.TAGS:
    e = E1S[t]; fam = e.loc[C.MAIN]
    for st, lab in [("real", "real"), ("mn_r2_within", "W"), ("mn_r2_cross_hi", "X_hi")]:
        add("5", "b", f"{C.TAGS[t]} {lab}: recruited / antiphase of recruited", f"{f2(e.loc[st, 'mean_P_gate'])} / {f2(e.loc[st, 'mean_antiphase_given_gate'])}",
            "Fig5b_recruitment_vs_antiphase.csv", "median over networks")
    add("5", "b", f"{C.TAGS[t]} rewired families: recruited range / max antiphase",
        f"{f2(fam.mean_P_gate.min())}–{f2(fam.mean_P_gate.max())} / {f2(fam.mean_antiphase_given_gate.max())}", "Fig5b_recruitment_vs_antiphase.csv", "six family medians")
    p = EP.loc[t]
    add("5", "c", f"{C.TAGS[t]} D real / W / X_hi", f"{f2(p.D_real)} / {f2(p.D_W)} / {f2(p.D_X_hi)}", "Fig5c_differential_share.csv", "pre-registered P2",
        {"theta50": "0.73 / 0.73 / 0.35", "manc50": "0.66 / 0.63 / 0.42"}[t] + " (progress log)")
    add("5", "d", f"{C.TAGS[t]} common component X_hi/W (P1)", f"{f2(p.P1)} ({'met' if p.P1 >= 0.8 else 'not met'})", "Fig5d_common_component.csv",
        f"median over {int(p.P1_pairs)} pairs; criterion 0.8", {"theta50": "0.84 met", "manc50": "0.62 not met"}[t] + " (progress log)")
    w_ = ES3[(ES3.connectome == t) & (ES3.arm != "pooled")].set_index("arm")
    add("5", "e", f"{C.TAGS[t]} Spearman γ–D pooled (n)", f"{f2(p.spearman_within)} ({int(p.n)})", "Fig5e_gamma_vs_D.csv", "pre-registered P3, criterion 0.5",
        {"theta50": "0.71 (90)", "manc50": "0.79 (110)"}[t] + " (progress log)")
    add("5", "e", f"{C.TAGS[t]} Spearman within real / W / X_hi", " / ".join(f2(w_.loc[a, "spearman_within"]) for a in ["real", "W", "X_hi"]), "Fig5e_gamma_vs_D.csv",
        "post hoc", {"theta50": "0.43 / 0.21 / 0.45", "manc50": "0.79 / 0.68 / 0.26"}[t] + " (progress log)")
AL3 = R3["A_levels"].set_index(["connectome", "window"]); AR3 = R3["A_scan_max_rhythm"]
for t in C.TAGS:
    for w in ["w15_25", "w50_60"]:
        r = AL3.loc[(t, w)]
        add("ED2", "a", f"{C.TAGS[t]} ReLU {C.WIN_LABEL[w]}: real S, best rewired, level", f"{f2(r.real)}, {f2(r.strongest_null)}, {r.level.lower()}",
            "ED2a_relu_coordination.csv", "own frozen settings")
    q = AR3[AR3.connectome == t]; rm = float(q.real_max_rhythm.iloc[0]); nl = q[q.graph != "real"]
    add("ED2", "b", f"{C.TAGS[t]} ReLU scan: real max rhythmicity; rewired above", f"{f2(rm)}; {int((nl.max_rhythm > rm).sum())} of {len(nl)}", "ED2b_relu_rhythm.csv",
        "eligible settings; strictly greater", {"theta50": "0.94; 1/30", "manc50": "0.65; 7/30"}[t] + " (progress log)")
    add("ED2", "b", f"{C.TAGS[t]} ReLU lineage-block median max rhythmicity", f2(nl[nl.graph == 'hl'].max_rhythm.median()), "ED2b_relu_rhythm.csv", "family median",
        {"theta50": "0.85", "manc50": "0.89"}[t] + " (progress log)")
BL3 = R3["B_levels"].set_index(["connectome", "window"])
for t in C.TAGS:
    for w in ["w15_25", "w50_60"]:
        r = BL3.loc[(t, w)]
        add("ED2", "c", f"{C.TAGS[t]} 25 per family {C.WIN_LABEL[w]}: level; at or above real; best", f"{r.level.lower()}; {int(r.n_at_or_above_real)} of {int(r.n_nulls)}; {f3(r.strongest_null)}",
            "ED2c_n25_distribution.csv", "pre-registered rule over 90 networks", {("theta50", "w15_25"): "strong; 0 of 90", ("manc50", "w15_25"): "weak; 4 of 90"}.get((t, w), ""))
    ep = json.loads(BL3.loc[(t, "w15_25")].empirical_p)
    add("ED2", "c", f"{C.TAGS[t]} empirical p (degree / lineage / leg × lineage)", " / ".join(f"{ep[f]:.3f}" for f in ["degree", "hl", "leg_hl"]), "ED2c_n25_distribution.csv",
        "(1 + n ≥ real)/(n + 1)", {"theta50": "0.038 each", "manc50": "0.077 / 0.115 / 0.077"}[t] + " (progress log)")
F2v = R3["F2_per_network"]; F2w = F2v[F2v.window == "w15_25"]
for d_ in [21, 22, 23]:
    z = F2w[F2w.draw == d_]; rr = float(z[z.graph == "real"].S.iloc[0]); nm = float(z[z.graph != "real"].S.max())
    add("ED2", "d", f"glutamate 25% draw {d_}: real / best rewired", f"{f2(rr)} / {f2(nm)}", "ED2d_glutamate_excitatory.csv", "15–25 s")
CRn = R3["C_per_network"]; CRw = CRn[CRn.window == "w15_25"]
for t in C.TAGS:
    for arm in ["C1", "C1p", "C0"]:
        a = CRw[(CRw.connectome == t) & (CRw.arm == arm)]
        add("ED5", "b", f"{C.TAGS[t]} {arm}: median RF; above parent", f"{f2(a.RF.median())}; {int((a.S_arm > a.S_parent).sum())} of {len(a)}", "ED5b_recovery_fraction.csv",
            "15–25 s; thresholds 0.5 / 0.2")
    pv = CRw[CRw.connectome == t].pivot(index="parent", columns="arm", values="S_arm")
    add("ED5", "b", f"{C.TAGS[t]} parents with C1p above C0", f"{int((pv.C1p > pv.C0).sum())} of {len(pv)}", "ED5b_recovery_fraction.csv", "15–25 s", "10/10 (progress log)")
    cs = R3["C_structure"][R3["C_structure"].connectome == t]
    add("ED5", "c", f"{C.TAGS[t]} structural index: parent / C1 / C1p / C0 / real (medians)",
        " / ".join(f3(cs[cs.arm == a].ri_index.median()) for a in ["parent", "C1", "C1p", "C0", "real"]), "ED5c_structure_index.csv", "rescue_build_report.csv")
RP3 = pd.read_csv(SD / "ED3b_pair_f_opp_real_vs_rewired.csv")
for t in C.TAGS:
    add("ED3", "b", f"{C.TAGS[t]} pairs with f_opp above all 30 rewired", f"{int(RP3[RP3.connectome == t].real_above_all.sum())} of 20", "ED3b_pair_f_opp_real_vs_rewired.csv",
        "per pair", "18/20 (v7 request)")
NUM = pd.DataFrame(nums)
NUM.to_csv(BASE / "figure_numbers.csv", index=False)

# ------------------------------------------------------------------------------------------------ legends (English)
rb = b1.loc["theta50"]; rbm = b1.loc["manc50"]
und = st2[st2.n_undefined_runs > 0].iloc[0]
c3m = c3.groupby("connectome").real_above_all.sum()
x_hi = {t: pct(np.median([v["redistr"] for k, v in rep.items() if k.startswith(f"{t}/mn_r2_cross_hi_s")])) for t in C.TAGS}
x_lo = {t: pct(np.median([v["redistr"] for k, v in rep.items() if k.startswith(f"{t}/mn_r2_cross_lo_s")])) for t in C.TAGS}
cnt4 = {}
for t in C.TAGS:
    q = f4[(f4.connectome == t) & (f4.network == "mn_shuffle_joint")]
    cnt4[t] = (int(round(np.median(q.groupby("gseed").apply(lambda d: (d.gated_frac >= 0.5).mean(), include_groups=False)) * q.pair.nunique())),
               q.pair.nunique())
rings = a5.groupby("connectome").real_above_all_rewired.sum()
r1 = {k: ed1d[(ed1d.connectome == k[0]) & (ed1d.graph == k[1]) & (ed1d.gseed == k[2])].iloc[0] for k in LAB3}
LEG = {}
tm, tn = OV["theta50"], OV["manc50"]
RIMAX = max(tm["rimax"], tn["rimax"])
RI_TXT = "all rewired networks below 0" if RIMAX < 0 else f"rewired at most {f2(RIMAX)}"
RI_ZH = "全部低于 0" if RIMAX < 0 else f"最高 {f2(RIMAX)}"
LEG["1"] = (
    "Fig. 1 | Rhythm is common to real and rewired wiring; antagonist coordination is strongest in the real wiring.",
    f"a, Question and system (schematic): leg-motor subnetworks of two male fly ventral-nerve-cord connectomes (MaleCNS, {tm['N']:,} neurons, "
    f"{tm['E']:,} connections; MANC, {tn['N']:,} and {tn['E']:,}); dots, the four joints of a leg. b, Approach (schematic): a rate "
    "network with the signed synapse counts as fixed weights, driven through the DNg100 descending neurons; 20 extensor–flexor pool pairs are scored "
    "for rhythmicity and for antagonist coordination S (mean −corr(extensor, flexor)). Thirty rewired networks per connectome (six families of "
    "five, preserving progressively more of the real structure) receive the same protocol and pre-specified claim levels (Extended Data Fig. 1). "
    "c, Rhythmicity (left) and S (right) of the real networks (bars; dark teal, MaleCNS; teal, MANC) and the rewired networks (points, coloured "
    f"by family), each at its own frozen setting (60-s confirmation, 15–25 s, mean of 20 noisy runs). {tm['n_rh']} and {tn['n_rh']} of 30 rewired "
    f"networks are at least as rhythmic as the real networks, but none reaches their S ({f2(tm['S'])} against at most {f2(tm['Smax'])}; "
    f"{f2(tn['S'])} against at most {f2(tn['Smax'])}; levels {tm['lv']} and {tn['lv']}). d, Reciprocal innervation (schematic): for example, a premotor neuron "
    "excites one pool and, through an inhibitory neuron, inhibits its antagonist, so the pools alternate. Bottom, reciprocal-innervation index "
    f"(real {f2(tm['ri'])} and {f2(tn['ri'])}; {RI_TXT}). Right, S relative to the real network after "
    f"premotor input is moved across antagonistic pools with motor neurons' input strength nearly unchanged (median change {BELOW} {MAXPCT}%): "
    "within-pool controls (x = 0), "
    "strength-matched and dose-matched cross-pool reassignments (points, networks; lines, medians; stars, reassignment with every motor neuron's "
    "input change capped at 5%; 15–25 s; Fig. 4). e, Joint scores of the real "
    "networks and of the best rewired network (lines), 15–25 s; only the thorax–coxa joint is 'strong' in both connectomes in all four windows "
    "(Extended Data Fig. 6). Source data are provided as a Source Data file.")
LEG["ED1"] = (
    "Extended Data Fig. 1 | Model, rewired networks and protocol.",
    "a, Model (schematic). The leg-motor subnetwork of each connectome is simulated as a rate network with fixed connection weights (equation; W, signed synapse "
    "counts; g, gain; β, inhibition scale) driven only by a sustained input d to the left and right DNg100 descending neurons. Motor neurons "
    "are grouped by annotated target muscle into extensor (coral) and flexor (green) pools at four joints of each leg: thorax–coxa (ThC), "
    "coxa–trochanter (CTr), femur–tibia (FTi) and tibia–tarsus (TiTa). The antagonist score S averages, over the 20 antagonistic pool pairs, "
    "−corr(extensor, flexor) for pairs in which both pools are modulated and 0 otherwise. "
    f"b, Composition of the extracted subnetworks: MaleCNS {int(rb.neurons):,} neurons and {int(rb.connections):,} connections; MANC "
    f"{int(rbm.neurons):,} neurons and {int(rbm.connections):,} connections. Lower bars, motor neurons with an exact or approximate joint "
    "assignment, or unmapped. c, The six families of rewired networks and the structure each preserves (filled circles); five networks per "
    "family. d, Protocol. Every network is simulated at the same 132 settings (11 gains × 3 inhibition scales × 4 drives; 25-s runs scored at "
    f"15–25 s); shown are the scores of the real MaleCNS network at its {int(d1.eligible.sum())} eligible settings (data; sand, not eligible; amber ring, selected "
    "setting). The best eligible setting is frozen and re-simulated for 60 s with 20 new noise realizations and 3 noise-free initial states, "
    "scored in four windows. Source data are provided as a Source Data file.")
LEG["2"] = (
    "Fig. 2 | Rewired networks produce rhythm but weaker antagonist coordination.",
    "a, Example activity of the six thorax–coxa pairs in the real MaleCNS network and in the rewired network with the highest confirmation "
    "score (lineage block), each at its own frozen setting (example simulation with a CPU re-implementation of the model; not a confirmation "
    f"run; S = {f2(ex.loc['malecns_real','cpu_S'])} and {f2(ex.loc['malecns_hl_s3','cpu_S'])}). Each pair is scaled by its larger pool s.d.; "
    "r, extensor–flexor correlation over 15–25 s; n.m., amplitude gate not met. b,c, Parameter scan: best S against maximum rhythmicity over the "
    "eligible settings of the 132-setting scan (one 25-s simulation per setting). Shaded, rhythmicity above the real network's maximum. "
    "d–g, 60-s confirmation at each network's frozen setting: rewired networks (points, mean ± s.e.m. of 20 runs; hollow, secondary MaleCNS "
    f"families that also preserve reciprocal connections; diamond, a network with {int(und.n_undefined_runs)} of 20 undefined runs, mean of "
    f"{int(und.n_defined_runs)}), family medians (dark ticks), real network (line; band ± s.e.m.) and the margin required of every family "
    "median for the pre-specified 'strong' level (dotted). Panel titles give the level reached. h,i, S of every confirmation run of the real "
    "networks; amber, runs that entered a fast state (≥ 3 Hz with ≤ 3 modulated pairs; open circles) after 15–25 s; dashed, noise-free runs; "
    "thick line, mean of 20 noisy runs. Tests with a second neuron model, with 25 networks per competitive family and with part of the "
    "glutamatergic neurons made excitatory: Extended Data Fig. 2; the 300-s continuation: Extended Data Fig. 7. Source data are provided as a "
    "Source Data file.")
LEG["3"] = (
    "Fig. 3 | The real wiring innervates antagonistic pools reciprocally, through an interneuron.",
    "a, The reciprocal-innervation index on data: each point is one neuron's signed one- plus two-step influence on the extensor pool (x) and on "
    "the flexor pool (y) of the MaleCNS T2L thorax–coxa pair (one of the two pairs defining the real network's median index), in the real network "
    "and in leg-block rewired network 0 (symmetric-log axes, linear within ±0.01); index = −Pearson r across neurons "
    f"({f2(a3[a3.network == 'real']['index'].iloc[0])} and {f2(a3[a3.network == 'leg_s0']['index'].iloc[0])}). b, Network-level index (median "
    "over the 20 pairs) of every rewired network (points; dark ticks, family medians) and of the real networks (lines). c, The same neurons act on "
    f"both pools with opposite signs (post hoc): {100 * RIS.loc['theta50', 'f_both_real']:.0f}% (MaleCNS) and {100 * RIS.loc['manc50', 'f_both_real']:.0f}% (MANC) "
    "of the influence on a pair comes from neurons that reach both pools, and the opposite-signed share of this shared influence, f_opp, is "
    f"{f2(RIS.loc['theta50', 'f_opp_real'])} and {f2(RIS.loc['manc50', 'f_opp_real'])} in the real networks (bars) against "
    f"{f2(RIS.f_opp_rewired_min.min())}–{f2(RIS.f_opp_rewired_max.max())} in the 30 rewired networks (points; grey numbers, highest rewired). "
    "d, A neuron's direct outputs share its sign, so opposite signs need an intervening neuron (schematic): an excitatory neuron excites one pool "
    "and inhibits the other through an inhibitory interneuron (route 1), or an inhibitory neuron inhibits one pool and disinhibits the other "
    "(route 2). Right, share of the opposite-signed influence carried by each route in the real networks (median over the 20 pairs): "
    f"{100 * RIS.loc['theta50', 'share_exc_median']:.0f}% and {100 * RIS.loc['theta50', 'share_inh_median']:.0f}% (MaleCNS), "
    f"{100 * RIS.loc['manc50', 'share_exc_median']:.0f}% and {100 * RIS.loc['manc50', 'share_inh_median']:.0f}% (MANC). "
    "Pair-by-pair versions of b and c: Extended Data Fig. 3. Source data are provided as a Source Data file.")
LEG["4"] = (
    "Fig. 4 | Reassigning premotor inputs across antagonistic pools abolishes coordination.",
    "a, Reassignment (schematic): edges onto motor neurons exchange targets within a pool (within-pool control) or between the extensor and "
    "flexor pools of the same leg and joint (cross-pool); strength-matched versions exchange only edges of the same sign and synapse-count "
    "decile. b, Example activity of the six thorax–coxa pairs in the real MaleCNS network and in cross-pool network 0 at the real frozen "
    "setting (example simulation as in Fig. 2a). c,d, S relative to the real network in the same window (R above each panel) for every "
    "reassigned network at the real frozen setting: original, strength-matched, and dose-matched reassignments (labels: designed net fraction "
    f"of input moved across pools; MaleCNS {x_lo['theta50']} and {x_hi['theta50']}, MANC {x_lo['manc50']} and {x_hi['manc50']}). Filled "
    "circles, 15–25 s; open diamonds, 50–60 s; bars, medians of five networks; error bars, s.e.m. of 20 runs divided by R; lines at R, 0.8 R "
    "(within-pool criterion) and 0.5 R (cross-pool criterion). e, Cross-pool arms relative to the real network: S, rhythmicity and the "
    "fraction of oscillating motor neurons, for the original reassignment and the dose-matched one (medians of five networks; circles "
    "MaleCNS, squares MANC; filled 15–25 s, open 50–60 s); shaded, pre-specified criterion met. f, Extensor–flexor correlation of the pairs "
    f"that contribute in the real networks (MaleCNS {cnt4['theta50'][1]}, MANC {cnt4['manc50'][1]}), real (bars) and within-pool (mint) or "
    f"original cross-pool (dark red) networks (hollow, pair modulated in fewer than half the runs); {cnt4['theta50'][0]} of {cnt4['theta50'][1]} "
    f"and {cnt4['manc50'][0]} of {cnt4['manc50'][1]} pairs stay modulated after cross-pool reassignment. g, S relative to the real network "
    "against the net fraction of premotor input moved across antagonistic pools, every network (15–25 s); dashed, medians of the strength-"
    "matched cross-pool series; circles MaleCNS, squares MANC; stars, cross-pool reassignment in which every motor neuron's excitatory and "
    f"inhibitory input changes by at most 5% (round 3; median S/R {f2(D2w[D2w.connectome == 'theta50'].S_over_R.median())} and {f2(D2w[D2w.connectome == 'manc50'].S_over_R.median())}); coordination again falls below zero "
    "while rhythm is kept, but fewer contributing pairs stay modulated than pre-specified "
    f"({f2(DEC3[(DEC3.exp == 'D2_cap') & (DEC3.connectome == 'theta50') & (DEC3.window == 'w15_25')].pair_retention.iloc[0])} and "
    f"{f2(DEC3[(DEC3.exp == 'D2_cap') & (DEC3.connectome == 'manc50') & (DEC3.window == 'w15_25')].pair_retention.iloc[0])}; criterion 0.8). "
    "Dose-matched design: Extended Data Fig. 4. Source data are provided as a Source Data file.")
LEG["5"] = (
    "Fig. 5 | The real wiring wins on coordination, not on recruitment.",
    "a, Common and differential components of an antagonistic pair (schematic): y± = (e ± f)/√2 for the extensor (e) and flexor (f) pool signals; "
    "the differential share D = Var(y−)/[Var(y+) + Var(y−)] exceeds 0.5 exactly when the two pools are antiphasic. b, Recruitment against "
    "coordination (descriptive): for each set of networks, the fraction of pair-runs in which both pools pass the amplitude gate against the "
    "antiphase (−corr) of those pairs (medians over networks; rewired families at their own frozen settings; real network, within-pool control W "
    "and cross-pool arm X_hi at the real frozen setting; 15–25 s). The real networks recruit no more pairs than rewired networks "
    f"({f2(E1S['theta50'].loc['real', 'mean_P_gate'])} and {f2(E1S['manc50'].loc['real', 'mean_P_gate'])} against "
    f"{f2(E1S['theta50'].loc[C.MAIN].mean_P_gate.min())}–{f2(E1S['theta50'].loc[C.MAIN].mean_P_gate.max())} and "
    f"{f2(E1S['manc50'].loc[C.MAIN].mean_P_gate.min())}–{f2(E1S['manc50'].loc[C.MAIN].mean_P_gate.max())}), but their recruited pairs are antiphasic "
    f"({f2(E1S['theta50'].loc['real', 'mean_antiphase_given_gate'])} and {f2(E1S['manc50'].loc['real', 'mean_antiphase_given_gate'])} against at most "
    f"{f2(E1S['theta50'].loc[C.MAIN].mean_antiphase_given_gate.max())} and {f2(E1S['manc50'].loc[C.MAIN].mean_antiphase_given_gate.max())}); "
    f"X_hi recruits about as many pairs ({f2(E1S['theta50'].loc['mn_r2_cross_hi', 'mean_P_gate'])} and {f2(E1S['manc50'].loc['mn_r2_cross_hi', 'mean_P_gate'])}) "
    f"but they turn in phase ({f2(E1S['theta50'].loc['mn_r2_cross_hi', 'mean_antiphase_given_gate'])} and "
    f"{f2(E1S['manc50'].loc['mn_r2_cross_hi', 'mean_antiphase_given_gate'])}; arrows). c, D of the active pairs (points, median over networks; bars, "
    f"medians; pre-registered): {f2(EP.loc['theta50', 'D_real'])}, {f2(EP.loc['theta50', 'D_W'])} and {f2(EP.loc['theta50', 'D_X_hi'])} (MaleCNS) and "
    f"{f2(EP.loc['manc50', 'D_real'])}, {f2(EP.loc['manc50', 'D_W'])} and {f2(EP.loc['manc50', 'D_X_hi'])} (MANC) for real, W and X_hi. d, Common "
    "component: Var(y+) of X_hi relative to W on W's active pairs (points; bars, medians; pre-registered criterion 0.8, dotted): "
    f"{f2(EP.loc['theta50', 'P1'])} (MaleCNS, met) and {f2(EP.loc['manc50', 'P1'])} (MANC, not met). e, The structural opposition γ of a pair's inputs "
    "(uncentred −cos of its influence vectors) predicts D across the real network, 5 W and 5 X_hi networks (pre-registered): Spearman "
    f"ρ = {f2(EP.loc['theta50', 'spearman_within'])} (n = {int(EP.loc['theta50', 'n'])}) and {f2(EP.loc['manc50', 'spearman_within'])} "
    f"(n = {int(EP.loc['manc50', 'n'])}); within groups (coloured numbers; post hoc) the correlations are weaker, so part of the pooled ρ reflects "
    "the difference between groups. Source data are provided as a Source Data file.")
LEG["6"] = (
    "Fig. 6 | Wiring-specific coordination is concentrated at the thorax–coxa joint.",
    "a, Score of each antagonistic pair in the real network (fill; mean of 20 runs, 15–25 s) drawn at its joint on a schematic fly, MaleCNS "
    f"and MANC; ring, the real network exceeds all 30 main rewired networks for that pair ({int(rings['theta50'])} of 20 and "
    f"{int(rings['manc50'])} of 20 pairs); ×, joint without an antagonistic pair. b, Mean pair score per joint for the real network (bar; pale "
    "band ± s.e.m.) and the 30 rewired networks (points sorted within each joint, coloured by family; ± s.e.m.). c, Pre-specified readouts: S "
    "over all 20 pairs, the 16 pairs of exactly assigned motor neurons, and the 14 pairs outside the thorax–coxa joint; squares, "
    "pre-specified level at 15–25 s (left) and 50–60 s (right). The thorax–coxa joint alone is 'strong' in both connectomes and all four windows "
    "(Extended Data Fig. 6). Source data are provided as a Source Data file.")
LEG["ED7"] = (
    "Extended Data Fig. 7 | The coordinated state over five minutes.",
    "300-s simulations at the frozen settings of the real MANC network, its two strongest rewired networks (lineage block 1 and 0) and the real "
    "MaleCNS network; 40 noisy trajectories per network (20 continuing the confirmation runs, 20 new) and 10 noise-free trajectories. A 10-s window "
    "counts as coordinated when S ≥ 0.10 with at least six modulated pairs (pre-registered). a, Fraction of noisy trajectories in the coordinated "
    "state. b, Mean S (undefined S counted as 0; band, ± s.e.m.). Sand band, the last window (290–300 s). c, State of every trajectory in every window "
    "(filled, coordinated; below the line, noise-free trajectories). d, Switching rate: departures from the coordinated state (followed by at "
    "least three non-coordinated windows) per minute spent coordinated, with exact Poisson 95% confidence intervals; MANC real "
    f"{r1[('manc50', 'real', 0)].rate_per_min:.2f} per minute (mean dwell {r1[('manc50', 'real', 0)].mean_dwell_min:.1f} min), lineage block 1 "
    f"{r1[('manc50', 'hl', 1)].rate_per_min:.1f} and lineage block 0 {r1[('manc50', 'hl', 0)].rate_per_min:.1f}; MaleCNS real, no departure. "
    "e, S of every noisy trajectory at 290–300 s; bars, means. Source data are provided as a Source Data file.")
LEG["ED4"] = (
    "Extended Data Fig. 4 | Dose-matched reassignment of premotor inputs.",
    "a, Design check: net fraction of premotor input moved across antagonistic pools against the change in the motor neurons' input strength "
    "(median relative change, larger of excitatory and inhibitory), every reassigned network; hollow, strength-matched constructions (exchanges "
    "restricted to the same sign and synapse-count decile); W, dose-matched within-pool control; X_lo and X_hi, dose-matched cross-pool arms. "
    "Right, each dose-matched network's median change (marker) and 95th percentile (bar top), larger of excitatory and inhibitory; across both "
    f"connectomes and both input signs the medians are {rng(TLR['W']['med'])} (W), {rng(TLR['X_lo']['med'])} (X_lo) and {rng(TLR['X_hi']['med'])} "
    f"(X_hi) and the 95th percentiles {rng0(TLR['W']['p95'])}, {rng0(TLR['X_lo']['p95'])} and {rng0(TLR['X_hi']['p95'])}, so the within-pool "
    "control, which kept coordination, changed its tail by a similar amount to X_lo, which lost most of it. "
    "b, Pre-registered conditions for a selective loss (filled, met): primary test X_hi against W at the real network's frozen setting; "
    "secondary, X_hi at re-optimised settings and X_lo at the real setting. c, S of W and X_hi at their own re-optimised frozen settings "
    "(filled, 15–25 s; open, 50–60 s; mean ± s.e.m. of 20 runs; lines, real network). d, Activity maintenance relative to the real network: "
    "fraction of oscillating motor neurons (circles) and of the real network's contributing pairs still modulated (triangles), medians of five "
    "networks; shaded, criterion (≥ 0.8) met. Source data are provided as a Source Data file.")
LEG["ED6"] = (
    "Extended Data Fig. 6 | Where and when the advantage of the real wiring holds.",
    "a, Pre-specified level (dark teal strong, light teal moderate, white weak) of each readout in each confirmation window, with the real network's value: "
    "S over all 20 pairs, over the 16 pairs of exactly assigned motor neurons, over the 14 pairs outside the thorax–coxa joint, and each joint "
    "alone. b, Score of every antagonistic pair of the real networks in the four windows (dot, modulated in at least half of the runs); in MANC "
    "the contributing pairs shift over time towards the thorax–coxa joints. Source data are provided as a Source Data file.")
AL_ = R3["A_levels"].set_index(["connectome", "window"]); BL_ = R3["B_levels"].set_index(["connectome", "window"])
ARh = R3["A_scan_max_rhythm"]
def _rh(t):
    q = ARh[ARh.connectome == t]; rm = float(q.real_max_rhythm.iloc[0]); nl = q[q.graph != "real"]
    return rm, int((nl.max_rhythm > rm).sum()), float(nl[nl.graph == "hl"].max_rhythm.median())
rhm, rhn = _rh("theta50"), _rh("manc50")
epm, epn = json.loads(BL_.loc[("theta50", "w15_25")].empirical_p), json.loads(BL_.loc[("manc50", "w15_25")].empirical_p)
F2p = sum(bool((float(z[z.graph == "real"].S.iloc[0]) > z[z.graph != "real"].S).all()) for _, z in F2w.groupby("draw"))
LEG["ED2"] = (
    "Extended Data Fig. 2 | Further tests of the advantage: a second neuron model, more rewired networks and a partial glutamate sign flip.",
    "a, Saturating ReLU neurons, f(x) = min(max(x, 0), 1), instead of the rectified tanh (round 3, pre-registered): S of the real networks (lines) "
    "and of the 30 rewired networks (points; ticks, family medians) at their own frozen settings, 15–25 s; "
    f"{AL_.loc[('theta50', 'w15_25')].level.lower()} in MaleCNS ({f2(AL_.loc[('theta50', 'w15_25')].real)} against at most {f2(AL_.loc[('theta50', 'w15_25')].strongest_null)}) "
    f"and {AL_.loc[('manc50', 'w15_25')].level.lower()} in MANC ({f2(AL_.loc[('manc50', 'w15_25')].real)} against at most {f2(AL_.loc[('manc50', 'w15_25')].strongest_null)}; "
    f"{AL_.loc[('manc50', 'w50_60')].level.lower()} at 50–60 s). b, Same model: maximum rhythmicity over the eligible settings of each network's scan; "
    f"{rhm[1]} and {rhn[1]} of 30 rewired networks exceed the real networks ({f2(rhm[0])} and {f2(rhn[0])}), and lineage-block networks are about as rhythmic "
    f"(family medians {f2(rhm[2])} and {f2(rhn[2])}) while their coordination stays low (a). c, Twenty-five networks for each of the three most "
    "competitive families (lineage block, degree, leg × lineage; round 3, pre-registered) and five for the others, 90 in all, rectified tanh "
    f"model, 15–25 s: MaleCNS {BL_.loc[('theta50', 'w15_25')].level.lower()}, {int(BL_.loc[('theta50', 'w15_25')].n_at_or_above_real)} of 90 at or above "
    f"the real network (empirical p = {epm['degree']:.3f} for each expanded family); MANC {BL_.loc[('manc50', 'w15_25')].level.lower()}, "
    f"{int(BL_.loc[('manc50', 'w15_25')].n_at_or_above_real)} of 90 at or above it (rings; p = {epn['hl']:.3f}, {epn['degree']:.3f} and {epn['leg_hl']:.3f}, respectively), "
    "so the real MANC network lies in the upper tail of the rewired distribution. d, A quarter of the glutamatergic neurons made excitatory "
    "(MaleCNS; three random draws; round 3, pre-registered): real network (bar) and 10 rewired networks (lineage block and degree); the real "
    f"network exceeds all 10 in {F2p} of 3 draws. Source data are provided as a Source Data file.")
LEG["ED3"] = (
    "Extended Data Fig. 3 | Reciprocal innervation pair by pair.",
    "a, Real reciprocal-innervation index of each antagonistic pair (circles; filled, above all 30 main rewired networks) against the range "
    f"(bars) and median (ticks) of the rewired networks: {int(c3m['theta50'])} of 20 pairs in MaleCNS and {int(c3m['manc50'])} of 20 in MANC. "
    f"b, The same for f_opp (post hoc; dashed, 0.5): {int(RP3[RP3.connectome == 'theta50'].real_above_all.sum())} and "
    f"{int(RP3[RP3.connectome == 'manc50'].real_above_all.sum())} of 20 pairs. c, Real per-pair index in the two connectomes; Spearman "
    f"ρ = {f2(d3.spearman_rho.iloc[0])} (n = {len(d3)} pairs). Source data are provided as a Source Data file.")
CRw_ = CRw.groupby(["connectome", "arm"]).RF.median()
CSr = R3["C_structure"]; frac_ = CSr[CSr.arm == "C1"].groupby("connectome").frac_edges_mn_layer.median()
c1pc0 = {t: int((CRw[CRw.connectome == t].pivot(index="parent", columns="arm", values="S_arm").pipe(lambda d: d.C1p > d.C0)).sum()) for t in C.TAGS}
LEG["ED5"] = (
    "Extended Data Fig. 5 | Restoring only the last layer restores coordination at most partially.",
    "a, Rescue (round 3, pre-registered; schematic): in parent rewired networks (MaleCNS lineage block and degree, MANC lineage block and leg × "
    "lineage; five each) every edge onto a motor neuron is replaced by the real network's (C1), the within-pool control's (C1p) or the cross-pool "
    f"arm X_hi's (C0) edges onto motor neurons; this last layer is {100 * frac_['theta50']:.1f}% and {100 * frac_['manc50']:.1f}% of all edges and "
    "every neuron keeps its out-degree. b, Recovery fraction RF = (S_arm − S_parent)/(S_real − S_parent) of each rescued network (lines join the "
    "versions of one parent; bars, medians; 15–25 s; dotted, 0.5, recovery; dashed, 0.2, partial): MaleCNS "
    f"{f2(CRw_[('theta50', 'C1')])}, {f2(CRw_[('theta50', 'C1p')])} and {f2(CRw_[('theta50', 'C0')])}; MANC {f2(CRw_[('manc50', 'C1')])} (partial), "
    f"{f2(CRw_[('manc50', 'C1p')])} and {f2(CRw_[('manc50', 'C0')])} for C1, C1p and C0; C1p exceeds C0 for {c1pc0['theta50']} and {c1pc0['manc50']} of 10 "
    "parents. c, Reciprocal-innervation index of parents, rescued versions and real networks: restoring the last layer brings the index only to "
    "about zero, because the opposite-signed routes run through upstream interneurons (Fig. 3d); scrambling it (C0) drives the index negative. "
    "Source data are provided as a Source Data file.")
ORDER = ["1", "2", "3", "4", "5", "6", "ED1", "ED2", "ED3", "ED4", "ED5", "ED6", "ED7"]
md = ["# Figure legends (draft, v8: round-3 results added)", "", "Numbers in these legends are filled in by `src/make_docs.py` from the source data; do not edit them by hand.", ""]
for k in ORDER:
    md += [f"**{LEG[k][0]}** {LEG[k][1]}", ""]
(BASE / "captions_draft.md").write_text("\n".join(md), encoding="utf-8")
print("legend words:", {k: len((LEG[k][0] + ' ' + LEG[k][1]).split()) for k in ORDER})

# ------------------------------------------------------------------------------------------------ README (Chinese)
chk = pd.read_csv(QA / "numbers_check.csv")
FIGFILES = ["Fig1_overview", "Fig2_rhythm_vs_coordination", "Fig3_reciprocal_innervation", "Fig4_motor_input_reassignment",
            "Fig5_differential_activity", "Fig6_joint_localization", "EDFig1_model_and_protocol", "EDFig2_robustness",
            "EDFig3_reciprocal_innervation_by_pair", "EDFig4_dose_matched_reassignment", "EDFig5_rescue", "EDFig6_readouts_across_windows",
            "EDFig7_persistence_300s"]
SHORT = {f: (f[:6] if f.startswith("EDFig") else f[:4]) for f in FIGFILES}
al = {f: json.loads((QA / f"{f}.alignment.json").read_text())["verdict"] for f in FIGFILES}
col = {f: json.loads((QA / f"{SHORT[f]}.collision-audit.json").read_text()) for f in FIGFILES}
txt = {f: (QA / f"{SHORT[f]}.text-audit.txt").read_text() for f in FIGFILES}
minpt = {f: re.search(r"minimum found: ([\d.]+)", v).group(1) for f, v in txt.items()}
man_c = CD2 = pd.read_csv(C.DERIVED / "r2_exp2_conditions.csv")
mc = man_c[(man_c.connectome == "manc50") & (man_c.window == "w15_25") & (man_c.arm == "mn_r2_cross_hi") & (man_c.rule == "real_setting")].iloc[0]
e4m = e4[(e4.connectome == "manc50") & (e4.arm == "dose-matched") & (e4.metric == "rhythm")]
reopt = ed2d[(ed2d.connectome == "theta50") & (ed2d.arm == "mn_r2_cross_hi") & (ed2d.rule == "own")]
src_notes = sorted(SD.glob("*.README.txt"))
sc2m = sc2[sc2.graph == "real"].set_index("connectome")
nr_scan = {t: (int(sc2m.loc[t].n_rewired_more_rhythmic_than_real), int((sc2.connectome == t).sum() - 1)) for t in C.TAGS}
BVd = pd.read_csv(C.DERIVED / "r3_B_network_values.csv")
_bm = BVd[(BVd.connectome == "manc50") & (BVd.window == "w15_25")]; _br = float(_bm[_bm.graph == "real"].S.iloc[0])
B_hi = _bm[(_bm.graph != "real") & (_bm.S >= _br)].sort_values("S", ascending=False)
B_HI_TXT = "、".join(f"{C.FAM_LABEL[r.graph]} s{r.gseed} {r.S:.3f}" for r in B_hi.itertuples())
d2d = {t: DEC3[(DEC3.exp == "D2_cap") & (DEC3.connectome == t) & (DEC3.window == "w15_25")].iloc[0] for t in C.TAGS}
TIE = R3CHK[R3CHK.note.notna() & (R3CHK.note.astype(str) != "")]
readme = f"""# 论文插图 · v8（NC 投稿版）· 加入第三轮结果（Claude，2026-09-30）

在 `Claude_总览图版_20260928/` 的基础上，加入第三轮（2026-09-29 预注册）的结果，并按主线重新组织。`Claude_总览图版_20260928/` 是 v7 用的那一版，保留不动。

**主线**：
- 节律只需要粗略的接线，拮抗协调需要具体到神经元的接线。
- 这种接线是交互支配：同一批神经元让一侧兴奋、另一侧受抑制，而且要经过一个中间神经元。
- 协调丧失，是因为"此消彼长"的差分活动没了，"一起涨落"的共同活动还在。

所有图上的数字都由脚本从 NAS 原始数据算出。第三轮每一项都从原始 parquet 重算，再和 `round3_20260929/analysis/` 的判定表逐项核对。原始数据和归档文件只读，没有改动。

- **手机上看**：`Claude插图_v8第三轮_手机版.pdf`（13 张图 + 图注 + 核对结果 + 说明）。
- **图文件**：
  - `Fig1…Fig6_*.pdf` 是正文图；`EDFig1…EDFig7_*.pdf` 是补充图，NC 版里叫 Supplementary Fig.。
  - 矢量、文字可编辑、183 mm 双栏宽；另有 `.svg` 和 600 dpi `.png`。
- **图注草稿**：`captions_draft.md`，英文，数字由脚本填入。
- **图上每个数字**：`figure_numbers.csv`。
- **源数据**：`source_data/`，每个图版一个 CSV 加一份说明。
- **脚本**：在 `src/`，重新生成全部用 `bash src/run_all.sh`。第三轮核对脚本是 `src/check_round3.py`，新机制图是 `src/fig5.py`。

## 新旧图号对照（写作 session 按这张表改正文）

补充图按正文首次引用的顺序编号，已和写作 session 对过。

| v8 图号 | 内容 | v7（总览图版）图号 | 文件 | 变化 |
|---|---|---|---|---|
| Fig. 1 | 总览 | Fig. 1 | `Fig1_overview` | d 加了 D2 封顶点（星形）；其余不变 |
| Fig. 2 | 节律 vs 协调 | Fig. 2 | `Fig2_rhythm_vs_coordination` | 图不变；图注加了指向 S2（稳健性）和 S7（300 秒）的引用 |
| **Fig. 3** | 两层交互支配 | Fig. 3 | `Fig3_reciprocal_innervation` | a、b 不变；**c 新**（f_opp）；**d 新**（两条通路示意 + 占比）；原 3c、3d 移到 S3 |
| Fig. 4 | 调换输入 | Fig. 4 | `Fig4_motor_input_reassignment` | g 加了 D2 封顶点（星形）；a 下方说明句改为 "input changes little"（原来写 "preserving each motor neuron's input"，按 v7 口径不成立）；其余不变 |
| **Fig. 5** | **差分 / 共同活动（新）** | — | `Fig5_differential_activity` | 新图 |
| Fig. 6 | ThC 定位 | Fig. 5 | `Fig6_joint_localization` | 只改图号 |
| S1 | 模型与流程 | S1 | `EDFig1_model_and_protocol` | 不变 |
| **S2** | **稳健性（新）** | — | `EDFig2_robustness` | 新：A 饱和 ReLU、B 每类 25 张、F2 谷氨酸 |
| **S3** | **交互支配逐对（新）** | Fig. 3c、3d | `EDFig3_reciprocal_innervation_by_pair` | 原 3c → S3a，原 3d → S3c；新 S3b（逐对 f_opp） |
| S4 | 剂量匹配细节 | S3 | `EDFig4_dose_matched_reassignment` | 只改图号 |
| **S5** | **恢复实验（新）** | — | `EDFig5_rescue` | 新 |
| S6 | 等级矩阵 | S4 | `EDFig6_readouts_across_windows` | 只改图号 |
| S7 | 300 秒 | S2 | `EDFig7_persistence_300s` | 只改图号 |

改正文时：
- 原来引用 "Fig. 3c"、"Fig. 3d" 的地方，改成 "Supplementary Fig. 3a"、"3c"。
- 原 "Fig. 5" 改成 "Fig. 6"。
- 补充图：原 S2（300 秒）改为 S7，原 S3（剂量匹配）改为 S4，原 S4（等级矩阵）改为 S6。
- 图注之间的交叉引用已经按新图号改好。
- 图注源文件里写的是 "Extended Data Fig."，写作 session 的转换脚本会换成 "Supplementary Fig."。

## 第三轮的 8 个图块放在哪里

| 图块 | 位置 |
|---|---|
| A 饱和 ReLU（协调、节律） | S2a（协调）、S2b（各家族最大节律性） |
| B 每类 25 张 | S2c（90 张重连网络的分布，达到或超过真实网络的用琥珀色圈标出） |
| 两层交互支配 | Fig. 3c（f_opp）、Fig. 3d（两条通路示意 + 占比）；逐对见 S3b |
| E1 招募与协调的拆解 | Fig. 5b |
| E 共同 / 差分 | Fig. 5a（示意）、5c（D，0.5 参考线）、5d（共同活动 X_hi/W）、5e（γ 对 D，合并与组内相关） |
| D2 封顶 5% | Fig. 4g、Fig. 1d（星形） |
| C 恢复实验 | S5 |
| F2 谷氨酸 25% 改兴奋性 | S2d |
| F1 9% 随机翻转 | 按用户决定不画 |

## 第三轮需要照实写的地方

1. **B（每类 25 张）MANC 判为"弱"**：
   - 90 张里有 {len(B_hi)} 张达到或超过真实网络：{B_HI_TXT}。
   - S2c 用琥珀色圈照实标出。
   - MaleCNS 仍为"强"（0/90）。
2. **D2（每个运动神经元都 ≤ 5%）按预注册判"不成立"**：
   - S 在两个连接组都掉到负值，节律也保留了。
   - 但第 ⑥ 条活动保持没过：参与协调的肌池对保留 {f2(d2d['theta50'].pair_retention)} / {f2(d2d['manc50'].pair_retention)}，振荡运动神经元保留 {f2(d2d['theta50'].osc_retention)} / {f2(d2d['manc50'].osc_retention)}，要求 ≥ 0.8。
   - 图 4g、图 1d 只画了 S，图 4 的图注写明了这一点。
3. **E 的 P1（共同活动保留）只在 MaleCNS 成立**：X_hi/W 为 {f2(EP.loc['theta50', 'P1'])}（过）和 {f2(EP.loc['manc50', 'P1'])}（不过）。所以"共同活动还在"在 MANC 只部分成立，图 5d 照实画出。
4. **E 的 P3（γ 预测 D）**：合并相关达标，组内相关更弱，说明合并 ρ 有一部分来自组间差异。图 5e 两种都标了。
5. **P3 的数值复算（已解决）**：第一次复算时，MANC 合并 ρ 和 W 组内 ρ 与判定表差在第三位小数。
   - 原因在我的核对脚本：pandas 默认的浮点解析不保证逐位还原，把两个几乎相等的 γ 读成了完全相等，秩随之改变。复查 session 指出了这一点。
   - 改用逐位还原精度读 CSV 后，与判定表逐位一致：MANC 合并 {EP.loc['manc50', 'spearman_within']:.4f}，W 组内 {ES3[(ES3.connectome == 'manc50') & (ES3.arm == 'W')].spearman_within.iloc[0]:.4f}。
   - 数据和判定都没有问题。
6. **C（恢复实验）不恢复**：只有 MANC 的 C1（精确放回）部分恢复。C0 让协调比父网络还差。结论是"最后一层的拮抗分配必要但不充分"，驱动在上游中间神经元，这与图 3d 一致。
7. **A（饱和 ReLU）**：MANC 在 50–60 s 为"中"，主窗口是"强"。
8. **D1** 无法判定（低变化对不到 3 对），没有画。

## 09-30 修订（按写作 session 的意见，文件夹原地更新）

图注源文件 `captions_draft.md` 采纳了写作 session 的 6 处措辞，数字都没改：

1. Fig. 2 末句：由"稳健"改为"检验"。F1 中 MaleCNS 没达标，不能声称对递质符号误差稳健。
2. S2 标题：改为 "Further tests of the advantage: …"，理由同上。
3. S2c：p 值改按"谱系块、度、腿×谱系"的顺序列出。原来的顺序与前文列出的家族对错了，这是我的错；图面上的文字本来是对的。
4. S2b："reach them" 改为 "are about as rhythmic"。MaleCNS 谱系块的中位数是 0.85，低于真实网络的 0.94。
5. Fig. 5 标题：改为 "The real wiring wins on coordination, not on recruitment."
6. S5 标题：改为 "… restores coordination at most partially"。MANC 的 C1 部分恢复。

图面文字改了两处：

- Fig. 5c 小标题：改为 "Reassignment lowers the push–pull share below 0.5"。D 只降到 0.35 / 0.42，没有消失。
- S2 各面板标题：去掉内部代号 "(round 3, A)"、"(round 3, B)"、"(F2)"。

## 之前版本的修订记录

- 09-29 v7 修订（Fig. 1 两处文字、原 S3a 第 95 百分位、原 S1a 标题、图注措辞）和 09-28 总览图版的改动，都保留在 `Claude_总览图版_20260928/README.md` 里。
- 图号以上表为准。

## 核对

- **第三轮**：`src/check_round3.py` 从原始 parquet 重算 A、B、C、D2、F2，从第二轮逐对数据重算 E、E1，与判定表逐项比对：**{int(R3CHK.match.sum())} / {len(R3CHK)} 项一致**（`qa/round3_check.csv`）。{"全部逐位一致（CSV 按逐位还原精度读取）。" if len(TIE) == 0 else f"其中 {len(TIE)} 项在容差内，见上面第 5 条。"}
- **第二轮**：从原始 parquet 独立重算实验 1、2、3（规则照搬 `analyze_exp1.py`、`analyze_exp2_3.py`），与判定表和本次任务给的核对数逐项比对：**{int(R2CHK.match.sum())} / {len(R2CHK)} 项一致**（`qa/round2_check.csv`）。其中任务说明里的 {int((R2CHK.section == 'request').sum())} 个核对数全部一致。
- **第一轮稿件数字**：{int(chk.match.sum())} / {len(chk)} 一致（`qa/numbers_check.csv`）。
  - 过程中这张表曾报 1 处不一致："MaleCNS 振荡覆盖 97–102%" 算成了 96–102%。
  - 原因在我的核对脚本：判定表后来加入了 MANC 的行，脚本没有限定只看 MaleCNS。修正后一致，数据本身没有问题。
- 两批手术的"挪动量"用的是同一套计算：实验 2 的造图脚本复现出原强度受控手术的值（MaleCNS 0.1388，MANC 0.1308），和审计表逐位相同，所以可以画在同一条剂量轴上。

## 需要写论文时注意的地方

1. **MANC 实验 2 的结果比任务说明表里写的多一条没过**：任务说明写"第 ⑥ 条没过"。按判定表，15–25 s 窗口的第 ① 条也没过（W/R = {mc.W / mc.R:.2f}，要求 ≥ 0.8）；50–60 s 的第 ① 条满足。补充图 S4b 照实画出。
2. **MANC 剂量匹配手术后节律反而更强**：X_hi 的节律是真实图的 {e4m[e4m.window == 'w15_25'].ratio.iloc[0]:.2f} / {e4m[e4m.window == 'w50_60'].ratio.iloc[0]:.2f} 倍（15–25 / 50–60 s）。这满足第 ⑤ 条，也符合"节律不稀奇"的主线。
3. **MaleCNS 的 X_hi 重新选参后，振荡运动神经元是真实图的约 {reopt.osc_coverage.iloc[0]:.1f} 倍**：它选到的参数区间与真实图不同，所以"重新选参"这一行的活动指标不宜和真实参数直接比较（补充图 S4d）。
4. **实验 3 在 290–300 s 的平均 S，MANC 真实图只比零模型名义上高**：补充图 S7e 画的是每条轨迹的分布，能看出重叠。事后自助法区间见 `analysis/exp3_posthoc_bootstrap.txt`，不在图里。
5. **总览图 c 的"节律"用的是每个网络在自己冻结参数下的节律性**。冻结参数是按 S 最大挑的，所以这里比较的是"协调最好时的节律"。如果正文要说"重连网络能产生和真实图一样强、甚至更强的节律"，图 2b、c 的扫描结果更直接：扫描里最大节律性高于真实图的重连网络，MaleCNS 有 {nr_scan['theta50'][0]}/{nr_scan['theta50'][1]} 个，MANC 有 {nr_scan['manc50'][0]}/{nr_scan['manc50'][1]} 个。
6. **总览图 d 的剂量反应只放了"输入强度几乎不变"的几档**：池内对照（x = 0，两种设计合起来共 10 个网络）、强度匹配的跨池手术、剂量匹配的 X_lo 和 X_hi。改变输入强度的原始跨池手术只在图 4 里。
7. **逐个运动神经元的分析和逆推理 R2（腿间配合）**：结果文件已在 `round2_20260927/analysis/`，v8 的任务没要求，没有画。

## 质检

| 图 | 版面对齐（1.5 pt） | 文字碰撞 | 最小字号 |
|---|---|---|---|
""" + "\n".join(f"| {f.split('_')[0]} | {al[f]} | {col[f]['verdict']}（FAIL {col[f]['summary']['fail']}，WARN {col[f]['summary']['warn']}） | {minpt[f]} pt |"
                 for f in FIGFILES) + """

- 剩余 WARN 都逐个看过：文字压在浅色底纹或栅格散点边缘，属有意设计，成品尺寸下可读。
- 自检"数据点不得落在坐标范围外"这一版又拦下三处，都已扩大坐标范围、全部画出：
  - 图 4e，MANC 节律比 1.58；
  - 图 5b，关节得分 −0.41；
  - 扩展数据图 3d（上一版编号 2d），振荡覆盖 2.03。


## 配色（全套统一）

同一种颜色在所有图里表示同一件事：

| 颜色 | 含义 |
|---|---|
| 深青 `#005F73` | 真实网络（线、柱、点、"强"等级） |
| 浅沙黄 → 琥珀 → 橙 → 锈红 → 砖红 → 酒红 | 6 类零模型：密度、细胞类型、度、腿、谱系块、腿+谱系块，越接近真实结构颜色越深 |
| 珊瑚橙 `#E76F51` / 青绿 `#2A9D8F` | 伸肌池 / 屈肌池 |
| 薄荷绿 `#5FB49C` / 深红 `#9B2226` | 池内对照手术 / 跨池手术 |
| 白 → 青 → 深青 | 协调强度（热图、肌池对得分） |
| 琥珀 `#E08A1E` | 强调：选中的参数、"高于全部零模型"的圈、快态切换 |
| 沙色底纹 / 浅青底纹 | 不合格或被丢弃的区段 / 预设标准达标的区域 |

- 灰色只用于次要文字、坐标轴和"未保留"的空心点，不再有紫色。
- 色盲模拟（Machado 矩阵）：零模型六色两两色差在绿色盲下最小 ΔE = 11.5，红色盲 13.5，蓝色盲 13.4。

## 各图版的数据来源与筛选（摘自 `source_data/*.README.txt`）

""" + "\n".join(f"### {p.name.replace('.README.txt', '')}\n\n```\n{p.read_text(encoding='utf-8').strip()}\n```\n" for p in src_notes) + f"""
生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}（本机时间）。
"""
(BASE / "README.md").write_text(readme, encoding="utf-8")

# ------------------------------------------------------------------------------------------------ phone PDF
e1m, e1n = E1S["theta50"], E1S["manc50"]
KEY = {"1": "总览（与 v7 相同）；d 加了星形：每个运动神经元的输入变化都 ≤ 5% 时，协调同样掉到负值。",
       "2": "节律 vs 协调（不变）。稳健性见 S2，300 秒见 S7。",
       "3": f"新 c、d：同一批神经元一正一负。{100 * RIS.loc['theta50', 'f_both_real']:.0f}% / {100 * RIS.loc['manc50', 'f_both_real']:.0f}% 的影响来自同时作用两侧的神经元，"
            f"其中一正一负的比例 {f2(RIS.loc['theta50', 'f_opp_real'])} / {f2(RIS.loc['manc50', 'f_opp_real'])}（重连 {f2(RIS.f_opp_rewired_min.min())}–{f2(RIS.f_opp_rewired_max.max())}）；"
            f"一正一负要经过中间神经元：兴奋性通路 {100 * RIS.loc['theta50', 'share_exc_median']:.0f}% / {100 * RIS.loc['manc50', 'share_exc_median']:.0f}%，"
            f"抑制-去抑制通路 {100 * RIS.loc['theta50', 'share_inh_median']:.0f}% / {100 * RIS.loc['manc50', 'share_inh_median']:.0f}%。",
       "4": f"调换输入（不变），g 加星形：每个运动神经元 ≤ 5% 时 S/R 仍为 {f2(D2w[D2w.connectome == 'theta50'].S_over_R.median())} / "
            f"{f2(D2w[D2w.connectome == 'manc50'].S_over_R.median())}；但预设的活动保持没过。",
       "5": f"新：赢在协调，不在招募。真实网络的过门比例 {f2(e1m.loc['real', 'mean_P_gate'])} / {f2(e1n.loc['real', 'mean_P_gate'])} 并不比重连网络高，"
            f"但被招募的对都在反相（{f2(e1m.loc['real', 'mean_antiphase_given_gate'])} / {f2(e1n.loc['real', 'mean_antiphase_given_gate'])}，重连 ≤ "
            f"{f2(e1m.loc[C.MAIN].mean_antiphase_given_gate.max())} / {f2(e1n.loc[C.MAIN].mean_antiphase_given_gate.max())}）；X_hi 招募不变却变成同相。"
            f"差分比例 D：真实 {f2(EP.loc['theta50', 'D_real'])} / {f2(EP.loc['manc50', 'D_real'])}，X_hi {f2(EP.loc['theta50', 'D_X_hi'])} / {f2(EP.loc['manc50', 'D_X_hi'])}；"
            f"共同活动 MaleCNS 保留（{f2(EP.loc['theta50', 'P1'])}），MANC 只部分保留（{f2(EP.loc['manc50', 'P1'])}）。",
       "6": "ThC 定位（原图 5，只改图号）。",
       "ED1": "模型与流程（不变）。",
       "ED2": f"新：进一步检验。饱和 ReLU 下两个连接组都是强；谱系块网络的节律与真实网络相当（家族中位数 {f2(rhm[2])} / {f2(rhn[2])}，真实 {f2(rhm[0])} / {f2(rhn[0])}），但协调不行。每类 25 张：MaleCNS 仍强（0/90），"
              f"MANC 弱（{len(B_hi)}/90 达到或超过，圈出）。谷氨酸 25% 改兴奋性：真实网络 {F2p}/3 次高于全部重连网络。",
       "ED3": "新（原图 3c、3d 移来）：交互支配逐对，加逐对 f_opp。",
       "ED4": "剂量匹配细节（原 S3，只改图号）。",
       "ED5": f"新：恢复实验。只放回最后一层不能恢复协调（MaleCNS C1 的 RF {f2(CRw_[('theta50', 'C1')])}；MANC C1 部分恢复 {f2(CRw_[('manc50', 'C1')])}）；"
              "打乱最后一层（C0）协调更差。结构指数只回到 0 附近：一正一负在上游中间神经元。",
       "ED6": "等级矩阵（原 S4，只改图号）。",
       "ED7": "300 秒（原 S2，只改图号）。"}
W_, H_ = 595.0, 842.0; M = 34.0
doc = pymupdf.open()


def page_text(page, rect, text, size=9.5, color=(0, 0, 0), align=0):
    return page.insert_textbox(rect, text, fontname="AU", fontfile=FONT, fontsize=size, color=color, align=align, lineheight=1.35)


def new_page():
    p = doc.new_page(width=W_, height=H_)
    p.insert_font(fontname="AU", fontfile=FONT)
    return p


p = new_page(); y = M
page_text(p, pymupdf.Rect(M, y, W_ - M, y + 40), "论文插图 · v8 · 加入第三轮结果 · 手机阅读合集", size=16); y += 32
page_text(p, pymupdf.Rect(M, y, W_ - M, y + 20), f"Claude · 生成于 {datetime.date.today()} · 文件夹 preprint_20260928/figures_v3/Claude_v8第三轮_20260930/",
          size=8.3, color=(0.35, 0.35, 0.35)); y += 24
summary = (
    "• v8：加入第三轮结果，按主线重排。节律只需粗接线；协调需要精细接线，即交互支配，要经过中间神经元；协调丧失是差分活动没了，共同活动还在。\n"
    "• 正文 6 张：新图 5 讲差分 / 共同活动。补充图 7 张：新 S2 稳健性、S3 交互支配逐对、S5 恢复实验。图号对照见 README，已和写作 session 对过。\n"
    f"• 数字：第三轮 {int(R3CHK.match.sum())}/{len(R3CHK)} 项与判定表{'逐位' if len(TIE) == 0 else ''}一致；第二轮 {int(R2CHK.match.sum())}/{len(R2CHK)}；第一轮 {int(chk.match.sum())}/{len(chk)}。\n"
    "• 照实画出的不利结果：MANC 每类 25 张下为弱；封顶 5% 的活动保持没过；MANC 的共同活动只部分保留；恢复实验不恢复。F1 按你的决定不画。\n"
    "• 所有图都过了对齐、文字碰撞、最小字号和“数据点不得被裁掉”的检查。配色沿用统一色板（见下方色块）。")
rest = page_text(p, pymupdf.Rect(M, y, W_ - M, y + 260), summary, size=10); y += 260 - rest + 12
def _rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
SW_ROWS = [("真实网络", [C.REAL]),
           ("6 类零模型（密度→腿+谱系块）", [C.FAM_COLOR[f] for f in C.MAIN]),
           ("伸肌 / 屈肌", [C.EXT, C.FLX]),
           ("池内 / 跨池手术", [C.WITHIN, C.CROSS]),
           ("协调强度（低→高）", C.COORD_COLORS),
           ("强调", [C.RING])]
page_text(p, pymupdf.Rect(M, y, W_ - M, y + 16), "配色（每种颜色在全部图里含义相同）", size=10); y += 18
for lab, cols in SW_ROWS:
    x = M
    for c_ in cols:
        p.draw_rect(pymupdf.Rect(x, y + 1, x + 16, y + 11), color=(0.8, 0.8, 0.8), fill=_rgb(c_), width=0.3); x += 18
    page_text(p, pymupdf.Rect(M + 6 * 18 + 10, y, W_ - M, y + 14), lab, size=8.6, color=(0.2, 0.2, 0.2))
    y += 15
y += 8
p.draw_line((M, y), (W_ - M, y), color=(0.7, 0.7, 0.7), width=0.5); y += 12
page_text(p, pymupdf.Rect(M, y, W_ - M, y + 40), "每张图一页：上面是图，下面是中文要点和英文图注草稿。顺序：图 1–6，补充图 S1–S7。", size=9.5,
          color=(0.35, 0.35, 0.35))
for name, k in zip(FIGFILES, ORDER):
    p = new_page()
    src = pymupdf.open(BASE / f"{name}.pdf"); r0 = src[0].rect
    wbox = W_ - 2 * M; hfig = wbox * r0.height / r0.width
    if hfig > 520:
        wbox = 520 * r0.width / r0.height; hfig = 520
    x0 = (W_ - wbox) / 2
    p.show_pdf_page(pymupdf.Rect(x0, M, x0 + wbox, M + hfig), src, 0)
    y = M + hfig + 10
    rest = page_text(p, pymupdf.Rect(M, y, W_ - M, y + 70), "要点：" + KEY[k], size=9.5, color=(0.1, 0.1, 0.1))
    y += 70 - max(rest, 0) + 6
    legend = LEG[k][0] + " " + LEG[k][1]
    rest = page_text(p, pymupdf.Rect(M, y, W_ - M, H_ - M), legend, size=7.6, color=(0.25, 0.25, 0.25))
    if rest < 0:
        p2 = new_page()
        page_text(p2, pymupdf.Rect(M, M, W_ - M, H_ - M), f"（{LEG[k][0].split('|')[0].strip()} 图注）\n" + legend, size=8.5, color=(0.25, 0.25, 0.25))
# checks page
p = new_page(); y = M
page_text(p, pymupdf.Rect(M, y, W_ - M, y + 24), "核对结果", size=13); y += 26
req = pd.concat([R3CHK[R3CHK.section == "request"], R2CHK[R2CHK.section == "request"]], ignore_index=True)
lines = [f"{'✓' if r.match else '✗'}  {r.item}：数据 {r.computed} · 任务说明 {r.reference}" for r in req.itertuples()]
page_text(p, pymupdf.Rect(M, y, W_ - M, H_ - M), "任务说明里给的核对数（全部从原始数据重算）：\n" + "\n".join(lines) +
          f"\n\n此外，实验 1、2、3 的 {len(R2CHK) - len(req)} 项与 analysis/ 下预注册判定表逐项一致；第一轮稿件数字 {int(chk.match.sum())}/{len(chk)} 一致。",
          size=7.4, color=(0.15, 0.15, 0.15))
# notes page
p = new_page(); y = M
page_text(p, pymupdf.Rect(M, y, W_ - M, y + 24), "写论文时需要注意的地方", size=13); y += 28
notes = ("第三轮（照实写）\n" + re.search(r"## 第三轮需要照实写的地方\n\n(.*?)\n\n## 之前版本的修订记录", readme, re.S).group(1)
         + "\n\n第二轮起的注意事项\n" + re.search(r"## 需要写论文时注意的地方\n\n(.*?)\n\n## 质检", readme, re.S).group(1))
page_text(p, pymupdf.Rect(M, y, W_ - M, H_ - M), re.sub(r"\*\*|`", "", notes), size=9.0)
doc.subset_fonts()
out = BASE / "Claude插图_v8第三轮_手机版.pdf"
doc.save(out, garbage=4, deflate=True)
print("written", out, round(out.stat().st_size / 1e6, 2), "MB,", doc.page_count, "pages")
