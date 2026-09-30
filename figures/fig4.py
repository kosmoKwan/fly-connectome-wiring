"""Fig. 4 | Reassigning premotor inputs across antagonistic pools abolishes coordination but not activity.

a  Schematic of the motor-input reassignment (within-pool control, cross-pool exchange; strength-matched variant).
b  Example pool activity (CPU re-simulation, labelled): the six ThC pairs, real MaleCNS vs cross-pool network 0,
   both at the real network's frozen setting.
c,d  S relative to the real network (same window) for every reassigned network at the real frozen setting: original,
   strength-matched and dose-matched (round 2, experiment 2) reassignments.
e  Coordination lost, activity kept? S, rhythmicity and oscillating-MN fraction of the cross-pool arms relative to real.
f  Pair by pair (round-2 re-run): pool correlation of the pairs that contribute in the real network, both connectomes.
g  Dose: S/R against the net fraction of premotor input moved across antagonistic pools, all arms.
All values from the round-2 tables re-computed and checked by check_round2.py.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
import common as C
import schem as SC

C.setup_style()
W, H = 183.0, 150.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)

R1_B, R1_H = 104.0, 38.0
R2_B, R2_H = 60.0, 32.0
R3_B, R3_H = 14.0, 30.0
ax_a = A(4.0, R1_B, 82.0, R1_H)
ax_b = A(100.0, R1_B, 71.0, R1_H)
ax_c = A(16.0, R2_B, 45.0, R2_H)
ax_d = A(66.0, R2_B, 45.0, R2_H)
ax_e = A(140.0, R2_B, 40.0, R2_H)
ax_f = A(16.0, R3_B, 100.0, R3_H)
ax_g = A(130.0, R3_B, 50.0, R3_H)
PURPLE = C.CROSS; GREYW = C.WITHIN   # cross-pool = deep red, within-pool = mint (palette v2)

# ================================================================================ a: schematic
ax = ax_a
ax.set_xlim(0, 82); ax.set_ylim(0, 38); ax.set_aspect("equal"); ax.axis("off")
P_X = [5.0, 13.0, 21.0]; P_Y = 28.5
M_X = [3.0, 9.0, 17.0, 23.0]; M_Y = 13.0
M_COL = [C.EXT, C.EXT, C.FLX, C.FLX]
real_edges = [(0, 0, 1.4), (1, 1, 0.7), (2, 2, 1.4), (2, 3, 0.7), (1, 3, 0.7)]   # (premotor, MN, width)
designs = [("Real wiring", None), ("Within-pool exchange", ((0, 0), (1, 1))), ("Cross-pool exchange", ((0, 0), (2, 2)))]
for k, (title, swap) in enumerate(designs):
    ox = 2.0 + k * 27.0
    edges = [list(e) for e in real_edges]
    hl = set()
    if swap is not None:
        (pa_, ma_), (pb_, mb_) = swap
        for e in edges:
            if (e[0], e[1]) == (pa_, ma_):
                e[1] = mb_; hl.add((pa_, mb_))
            elif (e[0], e[1]) == (pb_, mb_):
                e[1] = ma_; hl.add((pb_, ma_))
    for (p, m, w) in edges:
        col = (PURPLE if k == 2 else GREYW) if (p, m) in hl else C.LIGHT
        lw = w * (1.0 if (p, m) not in hl else 1.15)
        SC.arrow(ax, (ox + P_X[p], P_Y - 1.2), (ox + M_X[m], M_Y + 1.25), color=col, lw=lw, head=3.2, z=3)
    for i, x in enumerate(P_X):
        SC.node(ax, ox + x, P_Y, r=1.25, fc=C.NODE, ec="white", lw=0.5)
    for i, x in enumerate(M_X):
        SC.node(ax, ox + x, M_Y, r=1.25, fc=M_COL[i], ec="white", lw=0.5)
    ax.plot([ox + M_X[0] - 1.2, ox + M_X[1] + 1.2], [M_Y - 2.6] * 2, color=C.EXT, lw=0.8, solid_capstyle="butt")
    ax.plot([ox + M_X[2] - 1.2, ox + M_X[3] + 1.2], [M_Y - 2.6] * 2, color=C.FLX, lw=0.8, solid_capstyle="butt")
    ax.text(ox + (M_X[0] + M_X[1]) / 2, M_Y - 3.3, "Ext.", fontsize=5.4, ha="center", va="top", color=C.EXT)
    ax.text(ox + (M_X[2] + M_X[3]) / 2, M_Y - 3.3, "Flx.", fontsize=5.4, ha="center", va="top", color=C.FLX)
    ax.text(ox + 13.0, 37.5, title, fontsize=6, ha="center", va="top",
            color=PURPLE if k == 2 else (C.REAL if k == 0 else GREYW), fontweight="bold" if k == 0 else "normal")
    if k == 0:
        ax.text(ox - 0.6, P_Y, "Premotor", fontsize=5.3, ha="center", va="bottom", color=C.MUTED, rotation=90, rotation_mode="anchor")
        ax.text(ox - 0.6, M_Y, "MNs", fontsize=5.3, ha="center", va="bottom", color=C.MUTED, rotation=90, rotation_mode="anchor")
ax.text(0.5, 0.2, "Schematic. Strength-matched versions exchange only edges of the same sign and synapse-count decile,\n"
        "so motor neurons' excitatory and inhibitory input changes little.", fontsize=5.3, color=C.MUTED, ha="left", va="bottom",
        linespacing=1.25)
C.panel_label(ax_a, "a", x_pt=0, y_pt=4)

# ================================================================================ b: example traces (real vs cross-pool)
ex = {k: np.load(C.DERIVED / f"example_traces_{k}.npz") for k in ("malecns_real", "malecns_cross_s0")}
exs = pd.read_csv(C.DERIVED / "example_runs_summary.csv").set_index("example")
exp = {k: pd.read_csv(C.DERIVED / f"example_pairs_{k}.csv").set_index("pair") for k in ex}
SEG = (15.0, 17.0); LEG_ROWS = ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]; SP = 4.6
COL_X = {"malecns_real": 0.0, "malecns_cross_s0": 2.78}
rows_b = []
for k, x0 in COL_X.items():
    z = ex[k]; X = z["X"]; fs = float(z["fs"]); leg, joint, dd = z["leg"], z["joint"], z["direction"]
    Xc = X - X.mean(0); i0, i1 = int((SEG[0] - 15) * fs), int((SEG[1] - 15) * fs); t = np.arange(i1 - i0) / fs
    for r, L in enumerate(LEG_ROWS):
        e = (leg == L) & (joint == "ThC") & (dd > 0); f = (leg == L) & (joint == "ThC") & (dd < 0)
        a, b = Xc[:, e].mean(1), Xc[:, f].mean(1); sc = max(a.std(), b.std()); y0 = -r * SP
        info = exp[k].loc[f"{L}_ThC"]; gated = bool(info["gated"])
        ca, cb = (C.EXT, C.FLX) if gated else (C.EXT_FADED, C.FLX_FADED)
        ax_b.plot(x0 + t, y0 + b[i0:i1] / sc, color=cb, lw=0.6, zorder=2)
        ax_b.plot(x0 + t, y0 + a[i0:i1] / sc, color=ca, lw=0.6, zorder=3)
        ax_b.text(x0 + 2.07, y0, f"r = {C.fmt(info['corr'], 2)}" if gated else "n.m.", fontsize=5.6, va="center", ha="left",
                  color=C.INK if gated else C.MUTED)
        rows_b.append(dict(example=k, pair=f"{L}_ThC", corr_15_25=info["corr"], gated=gated, scale=sc))
for r, L in enumerate(LEG_ROWS):
    ax_b.text(-0.07, -r * SP, L, fontsize=6, ha="right", va="center")
for k, x0 in COL_X.items():
    s = exs.loc[k]
    ax_b.text(x0, 1.55 * SP, "Real network" if k == "malecns_real" else "Cross-pool network 0", fontsize=6.5, ha="left", va="center",
              color=C.REAL if k == "malecns_real" else PURPLE, fontweight="bold" if k == "malecns_real" else "normal")
    ax_b.text(x0, 0.95 * SP, f"S = {C.fmt(s.cpu_S, 2)}", fontsize=5.6, ha="left", va="center", color=C.MUTED)
yb = -5 * SP - 0.8 * SP
ax_b.plot([0.0, 0.5], [yb, yb], color=C.INK, lw=0.9, solid_capstyle="butt")
ax_b.text(0.25, yb - 0.35 * SP, "0.5 s", fontsize=5.6, ha="center", va="top")
ax_b.plot([0.9, 1.12], [yb, yb], color=C.EXT, lw=1.2); ax_b.text(1.16, yb, "Extensor pool", fontsize=5.6, va="center")
ax_b.plot([2.1, 2.32], [yb, yb], color=C.FLX, lw=1.2); ax_b.text(2.36, yb, "Flexor pool", fontsize=5.6, va="center")
ax_b.text(4.95, yb - 0.35 * SP, "Example simulation · MaleCNS · ThC pairs · real frozen setting · 15–17 s", fontsize=5.3,
          va="top", ha="right", color=C.MUTED, style="italic")
ax_b.set_xlim(-0.02, 4.95); ax_b.set_ylim(yb - 0.95 * SP, 1.8 * SP); ax_b.axis("off")
C.panel_label(ax_b, "b", x_pt=-22, y_pt=4)
C.write_source(pd.DataFrame(rows_b), "Fig4b_example_ThC_pairs",
               "Example simulation (CPU re-implementation, numpy noise; not a confirmation run). Real MaleCNS and cross-pool\n"
               "reassigned network 0, both at the real network's frozen setting. corr/gating over 15-25 s; traces 15-17 s.\n"
               f"Example-run S: real {exs.loc['malecns_real','cpu_S']:.4f}; cross-pool s0 {exs.loc['malecns_cross_s0','cpu_S']:.4f} "
               f"(confirmation runs of this network {exs.loc['malecns_cross_s0','gpu_S_min']:.3f}-{exs.loc['malecns_cross_s0','gpu_S_max']:.3f}).")

# ================================================================================ c,d: S/R by reassignment type
GM = {t: pd.read_csv(C.DERIVED / f"r2_exp2_graph_means_{t}.csv") for t in C.TAGS}      # exp-1 + exp-2 graph means (checked)
_rep = C.exp2_report()
_pct = lambda tag, arm: 100 * np.median([v["redistr"] for k, v in _rep.items() if k.startswith(f"{tag}/mn_r2_{arm}_s")])
ARMS_T = {t: [("mn_shuffle_pool", 0.0, "Within"), ("mn_shuffle_joint", 1.0, "Cross"),
              ("mn_shuffle_pool_w", 2.5, "Within"), ("mn_shuffle_joint_w", 3.5, "Cross"),
              ("mn_r2_within", 5.0, "Within"), ("mn_r2_cross_lo", 6.0, f"Cross\n{_pct(t, 'cross_lo'):.0f}%"),
              ("mn_r2_cross_hi", 7.0, f"Cross\n{_pct(t, 'cross_hi'):.0f}%")] for t in C.TAGS}
OFF = {"w15_25": -0.17, "w50_60": 0.17}
cd_rows = []
for ax, tag, lab in [(ax_c, "theta50", "c"), (ax_d, "manc50", "d")]:
    G = GM[tag]
    R = {w: float(G[(G.graph == "real") & (G.window == w)].S_all.iloc[0]) for w in C.WINDOWS}
    ax.axhline(1.0, color=C.REAL, lw=0.9, zorder=1)
    ax.axhline(0.8, color=C.REAL, lw=0.5, ls=(0, (1, 1.6)), zorder=1)
    ax.axhline(0.5, color=C.REAL, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.0, color=C.GRID, lw=0.6, zorder=0)
    for g, xc, _ in ARMS_T[tag]:
        col = GREYW if ("pool" in g or "within" in g) else PURPLE
        for w in ("w15_25", "w50_60"):
            q = G[(G.graph == g) & (G.rule == "real_setting") & (G.window == w)].sort_values("gseed")
            v = q.S_all.to_numpy() / R[w]; e_ = q.S_all_sem.to_numpy() / R[w]
            xx = xc + OFF[w] + np.linspace(-0.05, 0.05, len(v))
            ax.errorbar(xx, v, yerr=e_, fmt="none", ecolor=col, elinewidth=0.5, capsize=0, zorder=2)
            if w == "w15_25":
                ax.scatter(xx, v, s=11, facecolors=col, edgecolors="white", linewidths=0.3, zorder=3)
            else:
                ax.scatter(xx, v, s=11, marker="D", facecolors="white", edgecolors=col, linewidths=0.6, zorder=3)
            med = float(np.median(v))
            ax.plot([xc + OFF[w] - 0.13, xc + OFF[w] + 0.13], [med, med], color=C.INK, lw=1.0, zorder=4, solid_capstyle="butt")
            for _, rr in q.iterrows():
                cd_rows.append(dict(connectome=tag, window=w, group=g, gseed=int(rr.gseed), S=rr.S_all, S_sem=rr.S_all_sem, R=R[w],
                                    S_over_R=rr.S_all / R[w], sem_over_R=rr.S_all_sem / R[w], median_S_over_R=med, median_S=float(np.median(q.S_all))))
    ax.set_xlim(-0.55, 7.55); ax.set_ylim(-1.0, 1.28)
    ax.set_xticks([x for _, x, _ in ARMS_T[tag]]); ax.set_xticklabels([l for _, _, l in ARMS_T[tag]], fontsize=5.3, linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 0.8, 1.0]); ax.set_yticklabels(["−1", "−0.5", "0", "0.5", "0.8", "1"])
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for (x0_, x1_, txt) in [(-0.35, 1.35, "Original"), (2.15, 3.85, "Strength-matched"), (4.65, 7.35, "Dose-matched")]:
        ax.plot([x0_, x1_], [-0.27, -0.27], transform=tr, clip_on=False, color=C.INK, lw=0.5)
        ax.text((x0_ + x1_) / 2, -0.30, txt, transform=tr, fontsize=5.4, ha="center", va="top")
    C.panel_title(ax, f"{C.TAGS[tag]} · R = {C.fmt(R['w15_25'])} / {C.fmt(R['w50_60'])}", y_pt=4)
    C.panel_label(ax, lab, x_pt=-30 if lab == "c" else -12, y_pt=4)
ax_c.set_ylabel("S relative to real")
ax_d.set_yticklabels([])
for yv, tx in [(1.0, "R"), (0.8, "0.8 R"), (0.5, "0.5 R")]:
    ax_d.text(7.5, yv + 0.025, tx, fontsize=5.3, ha="right", va="bottom", color=C.REAL if yv == 1 else C.MUTED)
C.write_source(pd.DataFrame(cd_rows), "Fig4cd_S_relative_to_real",
               "Reassigned networks at the real network's frozen setting (rule real_setting); S = mean over 20 stochastic runs (round-2 re-run\n"
               "of the original arms, bit-identical to the archive; experiment-2 arms for the dose-matched groups); R = real network, same window.\n"
               "Filled circles 15-25 s, open diamonds 50-60 s; bars = medians over 5 networks; error bars = s.e.m. over runs / R.\n"
               "Dose-matched labels: median designed net cross-pool redistribution of the 5 graphs (surgery_r2_report.json).")

# ================================================================================ e: coordination lost, activity kept?
CD2 = pd.read_csv(C.DERIVED / "r2_exp2_conditions.csv")
e_rows = []
for tag in C.TAGS:
    G = GM[tag]
    for w in ("w15_25", "w50_60"):
        Rr = G[(G.graph == "real") & (G.window == w)].iloc[0]
        for arm, g in [("original", "mn_shuffle_joint"), ("dose-matched", "mn_r2_cross_hi")]:
            q = G[(G.graph == g) & (G.rule == "real_setting") & (G.window == w)]
            for metric, col_, ref in [("S", "S_all", Rr.S_all), ("rhythm", "rhythm", Rr.rhythm), ("osc", "osc_frac", Rr.osc_frac)]:
                e_rows.append(dict(connectome=tag, window=w, arm=arm, metric=metric, ratio=float(np.median(q[col_])) / float(ref)))
E = pd.DataFrame(e_rows)
# cross-check with the pre-registered condition table (dose-matched arm)
for _, r_ in E[E.arm == "dose-matched"].iterrows():
    c_ = CD2[(CD2.connectome == r_.connectome) & (CD2.window == r_.window) & (CD2.arm == "mn_r2_cross_hi") & (CD2.rule == "real_setting")].iloc[0]
    ref = {"S": c_.X_over_R, "rhythm": c_.X_rhythm / c_.R_rhythm, "osc": c_.osc_retention}[r_.metric]
    assert abs(ref - r_.ratio) < 1e-9, (r_, ref)
ROW = {("S", "original"): 0.0, ("S", "dose-matched"): 0.8, ("rhythm", "original"): 2.2, ("rhythm", "dose-matched"): 3.0,
       ("osc", "original"): 4.4, ("osc", "dose-matched"): 5.2}
ax = ax_e
PASS = C.PASS
ax.axvline(1.0, color=C.REAL, lw=0.9, zorder=1)
ax.fill_between([-1.0, 0.5], -0.45, 1.25, color=PASS, lw=0, zorder=0)
ax.plot([0.5, 0.5], [-0.45, 1.25], color=C.REAL, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.plot([0, 0], [-0.45, 1.25], color=C.GRID, lw=0.6, zorder=0)
XMAX = max(1.5, float(np.ceil((E.ratio.max() + 0.08) * 10) / 10))
for y0 in (1.75, 3.95):
    ax.fill_between([0.8, XMAX], y0, y0 + 1.7, color=PASS, lw=0, zorder=0)
    ax.plot([0.8, 0.8], [y0, y0 + 1.7], color=C.REAL, lw=0.5, ls=(0, (1, 1.6)), zorder=1)
DY = {("theta50", "w15_25"): -0.2, ("theta50", "w50_60"): -0.07, ("manc50", "w15_25"): 0.07, ("manc50", "w50_60"): 0.2}
for _, rr in E.iterrows():
    y = ROW[(rr.metric, rr.arm)] + DY[(rr.connectome, rr.window)]
    ax.scatter([rr.ratio], [y], s=11, marker="o" if rr.connectome == "theta50" else "s",
               facecolors=PURPLE if rr.window == "w15_25" else "white", edgecolors=PURPLE, linewidths=0.6, zorder=3)
ax.set_ylim(5.75, -0.55); ax.set_xlim(-1.0, XMAX)
ax.set_yticks(list(ROW.values()))
ax.set_yticklabels(["S · original", "S · dose-matched", "Rhythm · original", "Rhythm · dose-matched", "Osc. MNs · original",
                    "Osc. MNs · dose-matched"], fontsize=5.4)
ax.tick_params(axis="y", length=0, pad=2)
ax.spines["left"].set_visible(False)
ax.set_xticks([-1, -0.5, 0, 0.5, 1.0, 1.5]); ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1", "1.5"])
ax.set_xlabel("Cross-pool arm / real")
C.panel_title(ax, "Coordination lost, activity kept?", y_pt=4)
C.panel_label(ax, "e", x_pt=-66, y_pt=4)
eh = [Line2D([], [], ls="none", marker="o", ms=3.0, mfc=PURPLE, mec=PURPLE, label="MaleCNS"),
      Line2D([], [], ls="none", marker="s", ms=2.8, mfc=PURPLE, mec=PURPLE, label="MANC"),
      Line2D([], [], ls="none", marker="o", ms=3.0, mfc=C.MUTED, mec=C.MUTED, label="15–25 s"),
      Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=C.MUTED, mew=0.6, label="50–60 s"),
      Rectangle((0, 0), 1, 1, facecolor=PASS, edgecolor="none", label="criterion met")]
ax.legend(handles=eh, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=5.1, handletextpad=0.25, columnspacing=0.6,
          labelspacing=0.3, frameon=False, borderaxespad=0.25, handlelength=1.0)
C.write_source(E, "Fig4e_activity_kept",
               "Median over the 5 networks of each cross-pool arm at the real frozen setting, divided by the real network (same window).\n"
               "original = cross-pool reassignment of the first round (strength not matched); dose-matched = experiment-2 arm X_hi.\n"
               "Criteria (pre-registered): S <= 0.5 R; rhythmicity >= 0.8 of real (and of the within-pool control); oscillating MNs >= 0.8 of real.\n"
               "Dose-matched ratios equal the X/R, X_rhythm/R_rhythm and osc_retention columns of exp2_conditions.csv (asserted).")

# ================================================================================ f: pair by pair, both connectomes
f_rows = []
ax = ax_f
ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
x0 = 0.0; xt, xl = [], []
counts = {}
for tag in C.TAGS:
    fr = pd.read_csv(C.DERIVED / f"r2_exp1_pair_scores_{tag}.csv")
    fr = fr[fr.window == "w15_25"]
    real = fr[fr.graph == "real"].set_index("pair")
    pairs = sorted(real[real.fg >= 0.5].index, key=C.pair_sort_key)
    ret = []
    for fam, off, col in [("mn_shuffle_pool", -0.2, GREYW), ("mn_shuffle_joint", 0.2, PURPLE)]:
        q = fr[(fr.graph == fam) & (fr.rule == "real_setting") & fr.pair.isin(pairs)]
        for s_ in sorted(q.gseed.unique()):
            qs = q[q.gseed == s_].set_index("pair").loc[pairs]
            if fam == "mn_shuffle_joint":
                ret.append(float((qs.fg >= 0.5).mean()))
            for i, p_ in enumerate(pairs):
                gv = qs.loc[p_, "fg"] >= 0.5
                ax.scatter([x0 + i + off + (s_ - 2) * 0.035], [qs.loc[p_, "corr"]], s=8, facecolors=col if gv else "white",
                           edgecolors="white" if gv else col, linewidths=0.3 if gv else 0.6, zorder=3)
                f_rows.append(dict(connectome=tag, network=fam, gseed=int(s_), pair=p_, corr=qs.loc[p_, "corr"], gated_frac=qs.loc[p_, "fg"]))
    for i, p_ in enumerate(pairs):
        ax.plot([x0 + i - 0.3, x0 + i + 0.3], [real.loc[p_, "corr"]] * 2, color=C.REAL, lw=1.2, zorder=4, solid_capstyle="butt")
        f_rows.append(dict(connectome=tag, network="real", gseed=0, pair=p_, corr=real.loc[p_, "corr"], gated_frac=real.loc[p_, "fg"]))
        xt.append(x0 + i); xl.append(p_.replace("_", "\n"))
    counts[tag] = (int(round(np.median(ret) * len(pairs))), len(pairs))
    ax.text(x0 + (len(pairs) - 1) / 2, 1.5, f"{C.TAGS[tag]}: {counts[tag][0]} of {counts[tag][1]} pairs still modulated after cross-pool",
            fontsize=5.4, ha="center", va="center", color=PURPLE)
    x0 += len(pairs) + 0.8
    if tag == "theta50":
        ax.axvline(x0 - 0.9, color=C.LIGHT, lw=0.5, zorder=0)
ax.set_xlim(-0.6, x0 - 1.2); ax.set_ylim(-1.08, 1.72)
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=5.0, linespacing=1.2)
ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["−1", "−0.5", "0", "0.5", "1"])
ax.set_ylabel("Extensor–flexor\ncorrelation")
ax.tick_params(axis="x", length=0, pad=2)
C.panel_title(ax, "Pairs that contribute in the real network, 15–25 s (median of 5 networks for the counts)", y_pt=4)
C.panel_label(ax, "f", x_pt=-30, y_pt=4)
fh = [Line2D([], [], color=C.REAL, lw=1.2, label="Real"),
      Line2D([], [], ls="none", marker="o", ms=3, mfc=GREYW, mec=GREYW, label="Within-pool"),
      Line2D([], [], ls="none", marker="o", ms=3, mfc=PURPLE, mec=PURPLE, label="Cross-pool (original)"),
      Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=C.MUTED, mew=0.6, label="not modulated")]
ax.legend(handles=fh, loc="upper left", bbox_to_anchor=(0.0, -0.30), ncol=4, fontsize=5.3, handletextpad=0.3, columnspacing=1.2,
          frameon=False, borderaxespad=0)
C.write_source(pd.DataFrame(f_rows), "Fig4f_pair_correlations",
               "Round-2 pair-level re-run (bit-identical to the archived confirmation runs), 15-25 s, stochastic runs, both connectomes.\n"
               "corr = mean over 20 runs of the extensor-flexor pool correlation; gated_frac = fraction of runs passing the amplitude gate.\n"
               "Pairs = those gated in >= 50% of the real network's runs (MaleCNS 9, MANC 8). Reassigned networks at the real frozen setting.")

# ================================================================================ g: dose
ax = ax_g
aud = C.intervention_audit()
g_rows = []
ax.axhline(1.0, color=C.REAL, lw=0.9, zorder=1); ax.axhline(0.5, color=C.REAL, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.axhline(0.0, color=C.GRID, lw=0.6, zorder=0)
SERIES = {"within": ["mn_shuffle_pool", "mn_shuffle_pool_w", "mn_r2_within"], "matched": ["mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi"],
          "original": ["mn_shuffle_joint"]}
for tag, mk in [("theta50", "o"), ("manc50", "s")]:
    G = GM[tag]; Rv = float(G[(G.graph == "real") & (G.window == "w15_25")].S_all.iloc[0])
    med_line = []
    for kind, graphs in SERIES.items():
        for g in graphs:
            q = G[(G.graph == g) & (G.rule == "real_setting") & (G.window == "w15_25")].set_index("gseed")
            xs_, ys_ = [], []
            for s_ in q.index:
                if g.startswith("mn_r2"):
                    x_ = 0.0 if g == "mn_r2_within" else float(_rep[f"{tag}/{g}_s{s_}"]["redistr"])
                    dE = float(_rep[f"{tag}/{g}_s{s_}"]["med_rel_dE"]); dI = float(_rep[f"{tag}/{g}_s{s_}"]["med_rel_dI"])
                else:
                    a = aud[(aud.connectome == tag) & (aud.graph == g)].set_index("seed")
                    x_ = float(a.loc[s_, "pool_allocation_redistribution_fraction"])
                    dE = float(a.loc[s_, "median_relative_E_change"]); dI = float(a.loc[s_, "median_relative_I_change"])
                y_ = float(q.loc[s_, "S_all"]) / Rv; e_ = float(q.loc[s_, "S_all_sem"]) / Rv
                col = GREYW if kind == "within" else PURPLE
                filled = kind != "matched"
                ax.errorbar([x_], [y_], yerr=[e_], fmt="none", ecolor=col, elinewidth=0.5, capsize=0, zorder=2)
                ax.scatter([x_], [y_], s=10, marker=mk, facecolors=col if filled else "white", edgecolors="white" if filled else col,
                           linewidths=0.3 if filled else 0.6, zorder=3)
                xs_.append(x_); ys_.append(y_)
                g_rows.append(dict(connectome=tag, group=g, kind=kind, gseed=int(s_), redistribution=x_, S_over_R=y_, S=float(q.loc[s_, "S_all"]),
                                   R=Rv, median_rel_E_change=dE, median_rel_I_change=dI))
            if kind == "matched":
                med_line.append((float(np.median(xs_)), float(np.median(ys_))))
    med_line = sorted(med_line)
    ax.plot([m[0] for m in med_line], [m[1] for m in med_line], color=PURPLE, lw=0.8, ls=(0, (2.5, 1.5)), zorder=1,
            marker=mk, ms=3.2, mfc="white", mec=PURPLE, mew=0.8)
    # round 3 (D2): cross-pool reassignment with every motor neuron's E and I input change capped at 5%
    cp = pd.read_parquet(C.R3 / "results" / f"pairs_scalar_{tag}_cap.parquet")
    cp = cp[(cp["mode"] == "stochastic") & (cp.window == "w15_25")].groupby("gseed").S_all.agg(
        ["mean", lambda v: v.std(ddof=1) / np.sqrt(v.notna().sum())])
    cp.columns = ["S", "sem"]
    d2 = pd.read_csv(C.DERIVED / "r3_D2_per_network.csv").query("connectome == @tag and window == 'w15_25'").set_index("gseed")
    assert np.allclose(d2.loc[cp.index, "S"], cp.S, atol=1e-12)                           # same values as check_round3.py
    for s_ in cp.index:
        x_ = float(d2.loc[s_, "redistribution"]); y_ = float(cp.loc[s_, "S"]) / Rv; e_ = float(cp.loc[s_, "sem"]) / Rv
        ax.errorbar([x_], [y_], yerr=[e_], fmt="none", ecolor=PURPLE, elinewidth=0.5, capsize=0, zorder=2)
        ax.scatter([x_], [y_], s=26, marker="*", facecolors=PURPLE, edgecolors="white", linewidths=0.3, zorder=4)
        g_rows.append(dict(connectome=tag, group="mn_r3_cap", kind="capped", gseed=int(s_), redistribution=x_, S_over_R=y_, S=float(cp.loc[s_, "S"]),
                           R=Rv, median_rel_E_change=float(d2.loc[s_, "median_rel_E_change"]), median_rel_I_change=float(d2.loc[s_, "median_rel_I_change"])))
ax.set_xlim(-0.015, 0.3); ax.set_ylim(-0.95, 1.3)
ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_yticks([-0.5, 0, 0.5, 1.0]); ax.set_yticklabels(["−0.5", "0", "0.5", "1"])
ax.set_xlabel("Premotor input moved across pools")
ax.set_ylabel("S relative to real")
C.panel_title(ax, "Dose (15–25 s)", y_pt=4)
C.panel_label(ax, "g", x_pt=-26, y_pt=4)
gh = [Line2D([], [], ls="none", marker="o", ms=3, mfc=PURPLE, mec=PURPLE, label="Cross, strength not matched"),
      Line2D([], [], ls=(0, (2.5, 1.5)), lw=0.8, color=PURPLE, marker="o", ms=3, mfc="white", mec=PURPLE, label="Cross, strength-matched"),
      Line2D([], [], ls="none", marker="*", ms=5, mfc=PURPLE, mec="white", mew=0.3, label="Cross, every MN ≤ 5%"),
      Line2D([], [], ls="none", marker="o", ms=3, mfc=GREYW, mec=GREYW, label="Within-pool"),
      Line2D([], [], ls="none", marker="s", ms=2.8, mfc=C.MUTED, mec=C.MUTED, label="MANC (squares)")]
ax.legend(handles=gh, loc="lower left", ncol=1, fontsize=5.2, handletextpad=0.3, labelspacing=0.25, frameon=False, borderaxespad=0.2)
C.write_source(pd.DataFrame(g_rows), "Fig4g_dose",
               "x = net cross-pool redistribution of premotor input of each network (original arms: structural audit table; experiment-2 arms:\n"
               "surgery_r2_report.json; same definition, the report reproduces the audit value of the strength-matched network s0 exactly);\n"
               "within-pool arms move no input across pools (x = 0). y = S / R at the real frozen setting, 15-25 s; error bars s.e.m. / R.\n"
               "Dashed line: medians of the strength-matched cross-pool series (original _w, X_lo, X_hi). median_rel_E/I_change = median relative\n"
               "change of each MN's excitatory/inhibitory input. group mn_r3_cap (round 3, D2): cross-pool exchanges accepted only if all four\n"
               "affected MNs keep |relative change| <= 5% of both E and I input (cap_report.json); x from cap_report.json, S from pairs_scalar_*_cap.")

C.save(fig, "Fig4_motor_input_reassignment", dict(
    axes=[ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g], panel_ids=list("abcdefg"),
    row_groups=[["a", "b"], ["c", "d", "e"], ["f", "g"]],
    column_groups=[["c", "f"]],
    exemptions=[{"panels": ["e"], "checks": ["panel-width", "horizontal-gutter"],
                 "reason": "ratio summary panel e is narrower and needs extra room for its row labels"},
                {"panels": ["c", "f"], "checks": ["column"], "reason": "f is a wider pair-level panel that only shares the left edge with c"}],
))
print("Fig. 4 written")
