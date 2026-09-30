"""Fig. 1 | Overview: rhythm is everywhere, antagonist coordination is written in the wiring.

a  The question and the system (schematic; network sizes are data from the node and edge tables).
b  The approach (schematic): fixed-weight rate model driven through DNg100, extensor and flexor pools at every
   joint, and the same pre-registered test on six families of rewired networks that keep progressively more structure.
c  The answer (data): rhythmicity and antagonist score S of the real network and of the 30 rewired networks, each at its
   own frozen setting (60-s confirmation, 15-25 s window, mean of 20 noisy runs).
d  The mechanism: reciprocal innervation (schematic); reciprocal-innervation index of real and rewired networks (data);
   moving premotor input across antagonistic pools at preserved input strength removes coordination (data).
e  Where: joint-level score of the real networks against the best of the 30 rewired networks (data) on a leg schematic.
Every number is computed here from the archived results (read-only) and cross-checked against the round-2 analysis
tables; this script runs after fig4.py, whose dose table (source_data/Fig4g_dose.csv) it reuses.
"""
import math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import common as C
import schem as SC

C.setup_style()
W, H = 183.0, 172.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
COL = {"theta50": C.REAL, "manc50": C.REAL_MID}          # the two real networks (as in Extended Data Fig. 2)
MK = {"theta50": "o", "manc50": "s"}
ROW_Y = {"theta50": 1.0, "manc50": 0.0}
FAM_DY = {f: (k - 2.5) * 0.085 for k, f in enumerate(C.MAIN)}   # fixed vertical offset per family inside a strip row

# =============================================================================================== data
LV = pd.read_csv(C.DERIVED / "r2_exp1_levels.csv")          # recomputed from the parquet files; == analysis/exp1_claim_levels.csv
LV_AN = pd.read_csv(C.R2 / "analysis/exp1_claim_levels.csv")
conf, rows_c = {}, []
for tag in C.TAGS:
    g = C.graph_window_means(tag)
    w = g[(g.window == "w15_25") & (g.rule == "own")]
    real = w[w.graph == "real"].iloc[0]
    rw = w[w.graph.isin(C.MAIN)].sort_values(["graph", "gseed"])
    assert len(rw) == 30
    lv = LV[(LV.connectome == tag) & (LV.window == "w15_25") & (LV.readout == "S_all")].iloc[0]
    assert abs(lv.real - real.S) < 1e-9 and abs(lv.strongest_rewired - rw.S.max()) < 1e-9     # same numbers as the analysis
    conf[tag] = dict(real=real, rw=rw, level=lv.level.lower(), n_rhythm_ge=int((rw.rhythm >= real.rhythm).sum()),
                     n_S_ge=int((rw.S >= real.S).sum()))
    rows_c.append(dict(connectome=tag, network="real", family="real", gseed=0, rhythmicity=real.rhythm, S=real.S, S_sem=real.S_sem))
    for r in rw.itertuples():
        rows_c.append(dict(connectome=tag, network=f"{r.graph}_s{r.gseed}", family=r.graph, gseed=r.gseed, rhythmicity=r.rhythm,
                           S=r.S, S_sem=r.S_sem))

rec, rows_d1 = {}, []
for tag in C.TAGS:
    rg = C.recip_graph(tag)
    real_i = float(rg.loc[rg.kind == "real", "recip_innervation"].iloc[0])
    nul = rg[(rg.kind == "null") & rg.level.isin(C.MAIN)].copy()
    nul["gseed"] = nul.graph.str.extract(r"_s(\d+)$").astype(int)
    assert len(nul) == 30
    rec[tag] = dict(real=real_i, rw=nul, max=float(nul.recip_innervation.max()))
    rows_d1.append(dict(connectome=tag, network="real", family="real", recip_index=real_i))
    rows_d1 += [dict(connectome=tag, network=r.graph, family=r.level, recip_index=r.recip_innervation) for r in nul.itertuples()]

dose = pd.read_csv(C.SRC_DATA / "Fig4g_dose.csv")            # written by fig4.py in this run (per-network S / R and x)
dose["strength_change"] = dose[["median_rel_E_change", "median_rel_I_change"]].max(axis=1)
KEEP = ["mn_shuffle_pool_w", "mn_r2_within", "mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi"]   # strength-preserving arms
dk = dose[dose.group.isin(KEEP)].copy()
dr = pd.read_csv(C.R2 / "analysis/exp2_dose_response.csv").query("window == 'w15_25'")
for r in dr[dr.arm.isin(KEEP)].itertuples():                  # medians agree with the pre-registered analysis table
    q = dk[(dk.connectome == r.connectome) & (dk.group == r.arm)]
    assert len(q) == 5 and abs(q.S_over_R.median() - r.X_over_R) < 1e-6 and abs(q.redistribution.median() - r.redistribution) < 5e-4
max_change = float(dk.strength_change.max())
MAXPCT = math.ceil(100 * max_change)                          # printed bound, rounded up so it never understates
orig_change = float(dose[dose.group == "mn_shuffle_joint"].strength_change.median())
steps = []
for tag in C.TAGS:
    q = dk[dk.connectome == tag]
    wq = q[q.kind == "within"]
    steps.append(dict(connectome=tag, step="within-pool (both designs)", n=len(wq), x=0.0, median_S_over_R=wq.S_over_R.median()))
    for arm, lab in [("mn_shuffle_joint_w", "cross, strength-matched"), ("mn_r2_cross_lo", "cross, dose-matched X_lo"),
                     ("mn_r2_cross_hi", "cross, dose-matched X_hi")]:
        a = q[q.group == arm]
        steps.append(dict(connectome=tag, step=lab, n=len(a), x=a.redistribution.median(), median_S_over_R=a.S_over_R.median()))
steps = pd.DataFrame(steps)

JR = {"ThC": "S_ThC", "CTr": "S_CTr", "FTi": "S_FTi", "TiTa": "S_TiTa"}
rows_e = []
for tag in C.TAGS:
    for j, rd in JR.items():
        q = LV[(LV.connectome == tag) & (LV.readout == rd)]
        r15 = q[q.window == "w15_25"].iloc[0]
        an = LV_AN[(LV_AN.connectome == tag) & (LV_AN.window == "w15_25") & (LV_AN.readout == rd)]
        if len(an):
            assert an.iloc[0].level == r15.level
        rows_e.append(dict(connectome=tag, joint=j, real=r15.real, real_sem=r15.real_sem, best_rewired=r15.strongest_rewired,
                           best_family=r15.strongest_family, level_15_25=r15.level.lower(),
                           windows_strong=int((q.level == "STRONG").sum()), n_windows=len(q)))
E_ = pd.DataFrame(rows_e)

sizes = {}
for tag in C.TAGS:
    sizes[tag] = (len(pd.read_parquet(C.GRAPHS / f"nodes_{tag}.parquet")), len(pd.read_parquet(C.GRAPHS / f"edges_{tag}.parquet")))


def f2(x):
    return C.fmt(x, 2)


# =============================================================================================== a: the question
ax_a = A(0.5, 116.0, 57.0, 55.0)
ax = ax_a
ax.set_xlim(0, 57); ax.set_ylim(0, 55); ax.set_aspect("equal"); ax.axis("off")
ax.text(4.2, 54.0, "Which features of walking\nare written in the wiring?", fontsize=7.4, fontweight="bold", ha="left", va="top",
        linespacing=1.15)
CX, CY, s = 12.8, 21.0, 0.56                                   # fly schematic, top view (same outline as Fig. 5a)
P = lambda x, y: (CX + s * (x - 31.0), CY + s * (y - 46.5))
ax.add_patch(Ellipse(P(31, 63.5), 8.0 * s, 6.0 * s, fc=C.BODY, ec=C.BODY_EDGE, lw=0.6, zorder=1))
ax.add_patch(Ellipse(P(31, 46.5), 12.5 * s, 23.0 * s, fc=C.BODY, ec=C.BODY_EDGE, lw=0.6, zorder=1))
ax.add_patch(Ellipse(P(31, 24.0), 11.0 * s, 19.0 * s, fc=C.BODY, ec=C.BODY_EDGE, lw=0.6, zorder=1))
ax.add_patch(Ellipse(P(31, 46.8), 5.2 * s, 18.5 * s, fc=C.REAL_PALE, ec=C.REAL_MID, lw=0.6, zorder=3))   # ventral nerve cord
rng = np.random.default_rng(7)                                  # decorative texture inside the VNC outline (schematic)
vn = np.array([P(31 + 1.4 * np.sin(k * 2.4), 39.5 + k * 1.05) for k in range(15)])
for i in range(len(vn) - 2):
    ax.plot(*zip(vn[i], vn[i + 2]), color=C.REAL_LIGHT, lw=0.35, zorder=4)
ax.scatter(vn[:, 0], vn[:, 1], s=1.6, color=C.REAL_MID, zorder=5, linewidths=0)
LEGS = {"T1": ((36.2, 54.5), [(2.6, 1.8), (4.6, 5.0), (4.6, 5.6), (2.2, 3.2)]),
        "T2": ((37.2, 46.5), [(3.0, 0.2), (6.0, 2.6), (6.0, -2.8), (3.0, -2.0)]),
        "T3": ((36.2, 38.5), [(2.6, -1.6), (5.2, -4.4), (4.6, -6.2), (2.2, -3.8)])}
leg_pts = {}
for seg, (p0, vecs) in LEGS.items():
    for side in ("L", "R"):
        sg = 1 if side == "R" else -1
        pts = [np.array([31 + sg * (p0[0] - 31), p0[1]])]
        for v in vecs:
            pts.append(pts[-1] + np.array([sg * v[0], v[1]]))
        pts = [np.array(P(*p)) for p in pts]
        leg_pts[seg + side] = pts
        for i in range(4):
            SC.capsule(ax, pts[i], pts[i + 1], width=[1.9, 1.6, 1.2, 0.8][i] * s, fc=C.SEG, ec=C.BODY_EDGE, lw=0.4, z=1)
for j, joint in enumerate(C.JOINTS):                           # the four joints of one leg (T2R)
    xy = leg_pts["T2R"][j]
    ax.add_patch(Circle(xy, 0.62, fc=C.JOINT_COLOR[joint], ec=C.REAL_MID if joint == "TiTa" else "white", lw=0.35, zorder=6))
ax.annotate("VNC", P(31.8, 51.0), xytext=(24.6, 30.5), fontsize=5.8, ha="left", va="center",
            color=C.REAL, fontweight="bold", arrowprops=dict(arrowstyle="-", lw=0.45, color=C.REAL_MID, shrinkA=0.5, shrinkB=0.3))
tip = leg_pts["T2R"][3]
ax.annotate("4 leg\njoints", tip, xytext=(25.0, 13.0), fontsize=5.4, ha="left", va="center", color=C.MUTED, linespacing=1.1,
            arrowprops=dict(arrowstyle="-", lw=0.45, color=C.MUTED, shrinkA=0.3, shrinkB=1.0))
ax.text(CX, 1.2, "schematic", fontsize=5.2, ha="center", va="bottom", color=C.MUTED, style="italic")
# the two connectomes (sizes are data)
for k, tag in enumerate(C.TAGS):
    y0 = 25.0 - k * 15.0
    SC.box(ax, 32.5, y0, 24.2, 12.5, fc=C.CLOUD, ec=C.REAL_LIGHT, lw=0.6, r=1.6, z=1)
    ax.text(34.0, y0 + 10.4, C.TAGS[tag], fontsize=6.6, fontweight="bold", ha="left", va="top", color=COL[tag])
    ax.text(34.0, y0 + 6.2, f"{sizes[tag][0]:,} neurons\n{sizes[tag][1]:,} connections", fontsize=5.6, ha="left", va="top",
            linespacing=1.25)
ax.text(32.5, 41.0, "Two connectomes of the\nmale fly nerve cord", fontsize=5.5, ha="left", va="bottom",
        color=C.MUTED, linespacing=1.2)
C.panel_label(ax_a, "a", x_pt=0, y_pt=-7.2)

# =============================================================================================== b: the approach
ax_b = A(61.0, 116.0, 121.5, 55.0)
ax = ax_b
ax.set_xlim(0, 121.5); ax.set_ylim(0, 55); ax.set_aspect("equal"); ax.axis("off")
ax.text(0.5, 54.0, "Simulate the wiring as it is, then test it against rewired wiring", fontsize=7.4, fontweight="bold", ha="left",
        va="top")
BY, BH = 26.0, 19.0
boxes = [(0.5, 25.0), (31.5, 27.0), (64.5, 25.0), (95.5, 25.5)]
for (x0, w_) in boxes:
    SC.box(ax, x0, BY, w_, BH, fc="white", ec=C.REAL_LIGHT, lw=0.6, r=1.8, z=1)
for (xa, xb) in [(25.5, 31.5), (58.5, 64.5), (89.5, 95.5)]:
    SC.arrow(ax, (xa + 0.4, BY + BH / 2), (xb - 0.4, BY + BH / 2), color=C.REAL_MID, lw=0.8, head=3.5)
# 1 connectome
cxn = np.array([[4.5, 40.5], [9.0, 42.5], [13.5, 40.0], [7.0, 37.0], [12.0, 36.0], [17.0, 38.0], [20.5, 41.5], [16.0, 43.0]])
for i, j in [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (2, 5), (5, 6), (6, 7), (7, 1), (4, 5), (3, 1)]:
    ax.plot(*zip(cxn[i], cxn[j]), color=C.REAL_LIGHT, lw=0.5, zorder=2)
ax.scatter(cxn[:, 0], cxn[:, 1], s=7, color=C.REAL_MID, edgecolors="white", linewidths=0.3, zorder=3)
ax.text(13.0, 32.2, "Connectome\nsigned synapse counts", fontsize=5.6, ha="center", va="center", linespacing=1.2)
# 2 model
ax.add_patch(Circle((38.0, 39.5), 3.2, fc=C.REAL_PALE, ec=C.REAL, lw=0.6, zorder=2))
xs_ = np.linspace(-1, 1, 40)
ax.plot(38.0 + 2.2 * xs_, 39.5 + 1.6 * np.tanh(2.2 * xs_), color=C.REAL, lw=0.7, zorder=3)
ax.text(49.0, 39.5, "rate\nmodel", fontsize=5.6, ha="center", va="center", linespacing=1.1)
ax.text(45.0, 30.6, "weights fixed to the\nconnectome; 3 global\nparameters scanned", fontsize=5.4, ha="center", va="center", linespacing=1.2)
SC.arrow(ax, (38.0, 51.0), (38.0, 43.2), color=C.REAL, lw=0.9, head=3.5)
ax.text(39.3, 49.2, "DNg100 drive", fontsize=5.5, ha="left", va="center", color=C.REAL)
# 3 motor pools
for (col, dx) in [(C.EXT, -3.4), (C.FLX, 3.4)]:
    for m in range(3):
        ax.add_patch(Circle((77.0 + dx + (m - 1) * 1.45, 39.8 + (1.05 if m == 1 else -0.35)), 0.9, fc=col, ec="white", lw=0.3, zorder=3))
ax.text(73.6, 42.3, "ext.", fontsize=5.4, ha="center", va="bottom", color=C.EXT)
ax.text(80.4, 42.3, "flx.", fontsize=5.4, ha="center", va="bottom", color=C.FLX)
ax.text(77.0, 32.0, "extensor and flexor\npools, 20 pairs", fontsize=5.6, ha="center", va="center", linespacing=1.2)
# 4 readouts (idealised traces)
tt = np.linspace(0, 1, 120)
ax.plot(98.0 + 9.0 * tt, 40.8 + 1.4 * np.sin(2 * np.pi * 2 * tt), color=C.INK, lw=0.7, zorder=3)
ax.text(108.5, 40.8, "rhythm", fontsize=5.6, ha="left", va="center")
ax.plot(98.0 + 9.0 * tt, 34.6 + 1.4 * np.sin(2 * np.pi * 2 * tt), color=C.EXT, lw=0.7, zorder=3)
ax.plot(98.0 + 9.0 * tt, 34.6 - 1.4 * np.sin(2 * np.pi * 2 * tt), color=C.FLX, lw=0.7, zorder=3)
ax.text(108.5, 34.6, "antagonist\nscore S", fontsize=5.6, ha="left", va="center", linespacing=1.1)
ax.text(108.2, 28.9, "S = −corr(ext., flx.),\nmean of 20 pairs", fontsize=5.1, ha="center", va="center", color=C.MUTED, linespacing=1.15)
# rewired families: the same test on networks that keep more and more of the real structure
ax.text(0.5, 21.5, "The same test on 30 rewired networks: six families, five networks each", fontsize=6.0, ha="left", va="center")
chips = [(C.FAM_LABEL[f], C.FAM_COLOR[f], "white" if f in ("leg", "hl", "leg_hl") else C.INK) for f in C.MAIN] + \
        [("Real wiring", C.REAL, "white")]
cw_, gap_ = 15.6, 1.05
for k, (lab, fc, tc) in enumerate(chips):
    x0 = 0.5 + k * (cw_ + gap_)
    SC.box(ax, x0, 11.0, cw_, 6.2, fc=fc, ec="none", lw=0, r=1.4, z=2)
    ax.text(x0 + cw_ / 2, 14.1, lab, fontsize=5.4, ha="center", va="center", color=tc, fontweight="bold" if k == 6 else "normal", zorder=3)
SC.arrow(ax, (1.0, 7.8), (6 * (cw_ + gap_) - 1.0, 7.8), color=C.MUTED, lw=0.6, head=3.2)
ax.text(0.5, 4.8, "more of the real structure preserved (edge count → cell roles → degrees → leg and lineage blocks)", fontsize=5.2,
        ha="left", va="center", color=C.MUTED)
ax.text(0.5, 1.0, "Every network: 132-setting scan, best setting frozen, 60-s confirmation with 20 noisy runs; claim levels fixed in advance",
        fontsize=5.2, ha="left", va="center", color=C.MUTED)
C.panel_label(ax_b, "b", x_pt=-8, y_pt=-7.2)

# =============================================================================================== c: the answer (data)
CB, CHt = 80.0, 23.0
ax_c1 = A(24.0, CB, 62.0, CHt)
ax_c2 = A(116.0, CB, 62.0, CHt)


def strip(ax, key, xlim, xticks, xlabel):
    for tag in C.TAGS:
        d = conf[tag] if key in ("rhythm", "S") else rec[tag]
        rw = d["rw"]
        vals = rw[key].to_numpy() if key in ("rhythm", "S") else rw.recip_innervation.to_numpy()
        fams = rw.graph.to_numpy() if key in ("rhythm", "S") else rw.level.to_numpy()
        yy = ROW_Y[tag] + np.array([FAM_DY[f] for f in fams])
        ax.scatter(vals, yy, s=13, c=[C.FAM_COLOR[f] for f in fams], edgecolors="white", linewidths=0.3, zorder=3)
        rv = float(d["real"][key]) if key in ("rhythm", "S") else d["real"]
        ax.plot([rv, rv], [ROW_Y[tag] - 0.34, ROW_Y[tag] + 0.34], color=COL[tag], lw=2.2, solid_capstyle="butt", zorder=4)
    ax.set_xlim(*xlim); ax.set_ylim(-0.55, 1.72)
    ax.set_xticks(xticks); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel)


strip(ax_c1, "rhythm", (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], "Rhythmicity")
strip(ax_c2, "S", (-0.2, 0.45), [-0.2, 0, 0.2, 0.4], "Antagonist score S")
ax_c2.axvline(0, ymax=0.86, color=C.GRID, lw=0.5, zorder=0)
for ax in (ax_c1, ax_c2):
    for tag in C.TAGS:
        tr_ = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.02, ROW_Y[tag] + (0.12 if ax is ax_c2 else 0), C.TAGS[tag], transform=tr_, fontsize=6.0, ha="right", va="center",
                color=COL[tag], fontweight="bold")
        if ax is ax_c2:                                          # pre-specified level of the real network (analysis table)
            ax.text(-0.02, ROW_Y[tag] - 0.2, conf[tag]["level"], transform=tr_, fontsize=5.5, ha="right", va="center", color=COL[tag])
for tag in C.TAGS:
    d = conf[tag]; y = ROW_Y[tag]
    ax_c1.text(d["real"].rhythm, y + 0.42, f2(d["real"].rhythm), fontsize=5.6, ha="center", va="bottom", color=COL[tag])
    ax_c1.text(0.99, y + 0.42, f"{d['n_rhythm_ge']} of 30 rewired ≥ real", fontsize=5.3, ha="right", va="bottom", color=C.MUTED,
               transform=blended_transform_factory(ax_c1.transAxes, ax_c1.transData))
    ax_c2.text(d["real"].S, y + 0.42, f2(d["real"].S), fontsize=5.8, ha="center", va="bottom", color=COL[tag], fontweight="bold")
    mx = float(d["rw"].S.max())
    ax_c2.text(mx, y + 0.27, f2(mx), fontsize=5.4, ha="center", va="bottom", color=C.MUTED)
ax_c2.text(0.01, 0.975, "grey numbers: best of 30 rewired", transform=ax_c2.transAxes, fontsize=5.2, ha="left", va="top",
           color=C.MUTED)
C.panel_title(ax_c1, "Rhythm: real and rewired wiring alike", x_pt=-58, y_pt=9, fontsize=6.8, fontweight="bold")
C.panel_title(ax_c2, "Antagonist coordination: strongest in the real wiring", x_pt=-58, y_pt=9, fontsize=6.8, fontweight="bold")
C.panel_label(ax_c1, "c", x_pt=-66, y_pt=9)
lh = [Line2D([], [], ls="none", marker="o", ms=3.2, mfc=C.FAM_COLOR[f], mec="white", mew=0.3, label=C.FAM_LABEL[f]) for f in C.MAIN]
lh += [Line2D([], [], color=C.REAL, lw=2.0, label="real network")]
fig.legend(handles=lh, loc="center left", bbox_to_anchor=(24.0 / W, 69.2 / H), ncol=7, fontsize=5.5, handlelength=1.2,
           columnspacing=1.4, handletextpad=0.4, frameon=False, borderaxespad=0)
fig.text(24.0 / W, 66.2 / H, "Each network at its own frozen setting; 60-s confirmation, 15–25 s, mean of 20 noisy runs; "
         "30 rewired networks per connectome", fontsize=5.2, color=C.MUTED, ha="left", va="center")
C.write_source(pd.DataFrame(rows_c), "Fig1c_rhythm_and_coordination",
               "Network values at each network's own frozen setting, 60-s confirmation, 15-25 s window, mean over 20 noisy runs\n"
               "(undefined runs excluded, as in the archived analysis). rhythmicity = mean MN rhythmicity; S = antagonist score.\n"
               "Same values as Fig. 2d,f; real S and the best rewired S equal analysis/exp1_claim_levels.csv (asserted).")

# =============================================================================================== d: the mechanism
ax_d = A(0.5, 27.5, 62.0, 33.0)
ax = ax_d
ax.set_xlim(0, 62); ax.set_ylim(0, 33); ax.set_aspect("equal"); ax.axis("off")
ax.text(4.2, 32.5, "Why: reciprocal innervation", fontsize=6.6, fontweight="bold", ha="left", va="top")
PX, PY = 29.0, 23.5
SC.node(ax, PX, PY, r=2.2, fc=C.NODE, ec="white", lw=0.6, z=5)
ax.text(PX + 3.2, PY + 1.0, "premotor neuron", fontsize=5.5, ha="left", va="center")
t2 = np.linspace(0, 1, 100)
ax.plot(PX + 3.4 + 9.0 * t2, PY - 2.0 + 1.0 * np.sin(2 * np.pi * 2 * t2), color=C.NODE, lw=0.6)
EXX, FLX_, POY = 13.0, 45.0, 11.0
for (cx_, col) in [(EXX, C.EXT), (FLX_, C.FLX)]:
    for m in range(3):
        ax.add_patch(Circle((cx_ + (m - 1) * 1.9, POY + (1.3 if m == 1 else -0.2)), 1.15, fc=col, ec="white", lw=0.3, zorder=4))
SC.arrow(ax, (PX - 1.8, PY - 1.4), (EXX + 2.2, POY + 3.2), color=C.INK, lw=0.8, head=3.4)
ax.text((PX + EXX) / 2 - 2.2, (PY + POY) / 2 + 2.4, "+", fontsize=7.5, ha="center", va="center", color=C.EXT, fontweight="bold")
IX, IY = 38.5, 17.5
SC.arrow(ax, (PX + 1.6, PY - 1.6), (IX - 0.9, IY + 0.9), color=C.INK, lw=0.8, head=3.0)
SC.node(ax, IX, IY, r=1.1, fc="white", ec=C.NODE, lw=0.7, z=5)
SC.inhib(ax, (IX + 0.7, IY - 0.9), (FLX_ - 0.6, POY + 3.0), color=C.INK, lw=0.8, bar=0.9)
ax.text(IX + 5.4, IY - 0.6, "−", fontsize=8.0, ha="center", va="center", color=C.FLX, fontweight="bold")
ax.text(IX - 2.0, IY - 0.2, "inh.", fontsize=5.2, ha="right", va="center", color=C.MUTED)
for (cx_, col, sg, lab) in [(EXX, C.EXT, 1, "extensor pool"), (FLX_, C.FLX, -1, "flexor pool")]:
    ax.plot(cx_ - 5.0 + 10.0 * t2, 2.6 + sg * 1.0 * np.sin(2 * np.pi * 2 * t2), color=col, lw=0.8)
    ax.text(cx_, POY - 2.9, lab, fontsize=5.4, ha="center", va="center", color=col)
ax.text(PX, 2.6, "antiphase", fontsize=5.2, ha="center", va="center", color=C.MUTED)
ax.text(61.5, 0.3, "schematic", fontsize=5.2, ha="right", va="bottom", color=C.MUTED, style="italic")
C.panel_label(ax_d, "d", x_pt=0, y_pt=-7.2)
# d, data 1: the real wiring is reciprocally organised
ax_d1 = A(20.0, 9.0, 40.0, 11.0)
ax = ax_d1
for tag in C.TAGS:
    d = rec[tag]
    yy = ROW_Y[tag] + np.array([FAM_DY[f] for f in d["rw"].level])
    ax.scatter(d["rw"].recip_innervation, yy, s=9, c=[C.FAM_COLOR[f] for f in d["rw"].level], edgecolors="white", linewidths=0.25, zorder=3)
    ax.plot([d["real"]] * 2, [ROW_Y[tag] - 0.36, ROW_Y[tag] + 0.36], color=COL[tag], lw=2.0, solid_capstyle="butt", zorder=4)
    ax.text(d["real"] + 0.03, ROW_Y[tag], f2(d["real"]), fontsize=5.4, ha="left", va="center", color=COL[tag])
    ax.text(-0.02, ROW_Y[tag], C.TAGS[tag], transform=blended_transform_factory(ax.transAxes, ax.transData), fontsize=5.4,
            ha="right", va="center", color=COL[tag], fontweight="bold")
ax.set_xlim(-0.5, 0.7); ax.set_ylim(-0.55, 1.55); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([-0.4, 0, 0.4]); ax.axvline(0, color=C.GRID, lw=0.5, zorder=0)
ax.set_xlabel("Reciprocal-innervation index", labelpad=1.5)
C.panel_title(ax_d1, "In the wiring (data): real vs 30 rewired", x_pt=-54, y_pt=4, fontsize=5.8)
C.write_source(pd.DataFrame(rows_d1), "Fig1d_reciprocal_index",
               "Network-level reciprocal-innervation index (median over the 20 antagonistic pairs of -Pearson r between each neuron's signed\n"
               "one- plus two-step influence on the extensor and on the flexor pool); archived structural diagnostic, as in Fig. 3b.")
# d, data 2: moving input across the pools at preserved strength removes coordination
ax_d2 = A(80.0, 13.0, 40.0, 32.0)
ax = ax_d2
ax.axhline(1.0, color=C.REAL, lw=0.7, zorder=1)
ax.axhline(0.0, color=C.GRID, lw=0.6, zorder=0)
for tag in C.TAGS:
    q = dk[dk.connectome == tag]
    jit = {"theta50": -0.0035, "manc50": 0.0035}[tag]
    ax.scatter(100 * (q.redistribution + np.where(q.kind == "within", jit, 0)), q.S_over_R, s=7, marker=MK[tag],
               facecolors="white", edgecolors=COL[tag], linewidths=0.5, zorder=3, alpha=0.9)
    st = steps[steps.connectome == tag]
    ax.plot(100 * st.x, st.median_S_over_R, color=COL[tag], lw=1.3, marker=MK[tag], ms=3.4, mfc=COL[tag], mec="white", mew=0.4, zorder=4)
    cq = dose[(dose.group == "mn_r3_cap") & (dose.connectome == tag)]           # round 3: every motor neuron's change capped at 5%
    assert len(cq) == 5
    ax.scatter(100 * cq.redistribution, cq.S_over_R, s=24, marker="*", facecolors=COL[tag], edgecolors="white", linewidths=0.3, zorder=5)
ax.set_xlim(-2.5, 30.0); ax.set_ylim(-0.85, 1.25)
ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([-0.5, 0, 0.5, 1.0])
ax.set_xlabel("Premotor input moved across\nantagonist pools (%)", linespacing=1.15)
ax.set_ylabel("S relative to real")
ax.text(29.5, 1.05, "real", fontsize=5.4, ha="right", va="bottom", color=C.REAL)
C.panel_title(ax_d2, "Cross-pool input removes coordination", x_pt=-26, y_pt=12, fontsize=6.6, fontweight="bold")
C.panel_title(ax_d2, f"input strength nearly unchanged (median change {'<' if 100 * max_change < MAXPCT else '≤'} {MAXPCT}%)", x_pt=-26, y_pt=4.5,
              fontsize=5.2, color=C.MUTED)
dh = [Line2D([], [], color=COL[t], lw=1.2, marker=MK[t], ms=3.0, mfc=COL[t], mec=COL[t], mew=0.4, label=C.TAGS[t]) for t in C.TAGS]
dh += [Line2D([], [], ls="none", marker="*", ms=5, mfc=C.MUTED, mec="white", mew=0.3, label="every MN ≤ 5%")]
ax.legend(handles=dh, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=5.4, handlelength=1.8, numpoints=1, frameon=False,
          borderaxespad=0.3)
C.write_source(dk.drop(columns=["S", "R"]).assign(S_over_R=dk.S_over_R), "Fig1d_dose_response_networks",
               "Strength-preserving reassignments only (median relative change of each MN's excitatory and inhibitory input <= "
               f"{100 * max_change:.1f}%): within-pool controls (x = 0; strength-matched and dose-matched designs), strength-matched\n"
               "cross-pool, and the dose-matched cross-pool arms X_lo and X_hi. y = S / R at the real frozen setting, 15-25 s.\n"
               "Values from Fig4g_dose.csv; arm medians equal analysis/exp2_dose_response.csv (asserted). The original, unmatched\n"
               f"cross-pool reassignment (median input-strength change {100 * orig_change:.0f}%) is shown in Fig. 4, not here.")
C.write_source(steps, "Fig1d_dose_response_medians", "Line in Fig. 1d: median S / R per design step (5 networks each; 10 at x = 0).")
C.write_source(dose[dose.group == "mn_r3_cap"], "Fig1d_cap_networks",
               "Stars in Fig. 1d (round 3, D2): cross-pool reassignment in which every motor neuron's excitatory and inhibitory input changes by\n"
               "at most 5% (cap_report.json); x = net redistribution, y = S / R at the real frozen setting, 15-25 s (as in Fig. 4g).")

# =============================================================================================== e: where
ax_e = A(139.0, 13.0, 42.0, 32.0)
ax_el = A(139.0, 46.0, 42.0, 7.5)
ax = ax_e
BW = 0.34
for i, j in enumerate(C.JOINTS):
    for k, tag in enumerate(C.TAGS):
        r = E_[(E_.connectome == tag) & (E_.joint == j)].iloc[0]
        x = i + (k - 0.5) * (BW + 0.04)
        ax.bar(x, max(r.real, 0), width=BW, color=COL[tag], edgecolor="none", zorder=2)
        ax.plot([x - BW / 2 - 0.03, x + BW / 2 + 0.03], [r.best_rewired] * 2, color=C.FAM_COLOR["leg"], lw=1.1, zorder=3,
                solid_capstyle="butt")
ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(C.JOINTS)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_ylabel("Joint score S")
thc = E_[E_.joint == "ThC"]
n_all = "all 4" if (thc.windows_strong == thc.n_windows).all() else f"{int(thc.windows_strong.min())} of {int(thc.n_windows.max())}"
ax.text(0.42, 0.78, f"ThC alone is\n'strong' in both\nconnectomes,\n{n_all} windows", fontsize=5.4, ha="left", va="top", color=C.INK,
        linespacing=1.2)
eh = [Line2D([], [], color=C.REAL, lw=4, label="MaleCNS"), Line2D([], [], color=C.REAL_MID, lw=4, label="MANC"),
      Line2D([], [], color=C.FAM_COLOR["leg"], lw=1.1, label="best rewired")]
ax.legend(handles=eh, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=5.3, handlelength=1.2, frameon=False, borderaxespad=0)
C.panel_title(ax_el, "Where: the thorax–coxa joint", x_pt=-26, y_pt=4, fontsize=6.6, fontweight="bold")
C.panel_label(ax_el, "e", x_pt=-34, y_pt=4)
# leg schematic whose joints sit above the bars
ax = ax_el
ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 1); ax.axis("off")
jy = {"ThC": 0.55, "CTr": 0.40, "FTi": 0.62, "TiTa": 0.36}
SC.capsule(ax, (-0.62, 0.62), (-0.08, 0.57), width=0.34, fc=C.BODY, ec=C.BODY_EDGE, lw=0.5, z=1)
pts_ = [np.array([i, jy[j]]) for i, j in enumerate(C.JOINTS)] + [np.array([3.55, 0.2])]
for i in range(4):
    SC.capsule(ax, pts_[i], pts_[i + 1], width=[0.2, 0.17, 0.13, 0.09][i], fc=C.SEG, ec=C.BODY_EDGE, lw=0.45, z=2)
for i, j in enumerate(C.JOINTS):
    ax.add_patch(Ellipse(pts_[i], 0.2, 0.2 * (42.0 / 4.2) / (7.5 / 1.0), fc=C.JOINT_COLOR[j],
                         ec=C.REAL_MID if j == "TiTa" else "white", lw=0.4, zorder=4))
ax.add_patch(Ellipse(pts_[0], 0.36, 0.36 * (42.0 / 4.2) / (7.5 / 1.0), fc="none", ec=C.RING, lw=1.2, zorder=5))
ax.text(3.6, 0.98, "leg, schematic", fontsize=5.2, ha="right", va="top", color=C.MUTED, style="italic")
C.write_source(E_, "Fig1e_joint_scores",
               "Joint readouts of the round-2 pair-level re-run (S over the pairs of one joint), 15-25 s: real network (mean of 20 runs)\n"
               "and the best of the 30 main rewired networks; level = pre-specified claim level; windows_strong = number of the four\n"
               "confirmation windows in which the joint alone reaches 'strong'. Values from r2_exp1_levels.csv (= analysis table).")
C.write_source(pd.DataFrame([dict(connectome=t, neurons=sizes[t][0], connections=sizes[t][1]) for t in C.TAGS]), "Fig1a_network_sizes",
               "Rows of the node and edge tables of the extracted leg-motor subnetworks used by every simulation.")

C.save(fig, "Fig1_overview", dict(
    axes=[ax_c1, ax_c2, ax_d2, ax_e], panel_ids=["c1", "c2", "d2", "e"],
    row_groups=[["c1", "c2"], ["d2", "e"]],
))
print("Fig. 1 (overview) written")
