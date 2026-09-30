"""Fig. 6 | Wiring-specific coordination is concentrated at the proximal leg joints.

a  Where the real MaleCNS network coordinates: each antagonistic pair drawn at its joint on a schematic fly
   (body and legs schematic; colours are data). Fill = the pair's score in the real network; amber ring = the
   real network exceeds all 30 main rewired networks for that pair.
b  Mean pair score per joint: real network vs the 30 main rewired networks (MaleCNS; MANC when available).
c  Pre-specified readouts: all 20 pairs, exactly assigned motor neurons only (16 pairs), and all joints except
   thorax-coxa (14 pairs); pre-specified levels in both analysis windows.
All values from the round-2 pair-level re-run of the confirmation trajectories (bit-identical to the archive).
"""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import common as C
import schem as SC

C.setup_style()
W, H = 183.0, 128.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
ax_a = A(1.0, 54.0, 62.0, 70.0)
ax_a2 = A(63.0, 54.0, 62.0, 70.0)
ax_b1 = A(14.0, 14.0, 36.0, 28.0)
ax_b2 = A(56.0, 14.0, 36.0, 28.0)
ax_c1 = A(103.0, 14.0, 36.0, 28.0)
ax_c2 = A(145.0, 14.0, 36.0, 28.0)
READS = ["S_all", "S_exact", "S_nonThC"]
READ_LAB = {"S_all": "All\n(20)", "S_exact": "Exact\n(16)", "S_nonThC": "No ThC\n(14)"}
JOINT_READ = {"ThC": "S_ThC", "CTr": "S_CTr", "FTi": "S_FTi", "TiTa": "S_TiTa"}


def level(r, fams):
    gt = {f: bool((r > v).all()) for f, v in fams.items()}; med = {f: float(np.median(v)) for f, v in fams.items()}
    if all(gt.values()) and all(r - m >= 0.15 for m in med.values()):
        return "strong"
    if sum(gt.values()) >= 5 and all(r > m for m in med.values()):
        return "moderate"
    return "weak"


def load(tag):
    S = C.round2_scalar(tag)
    if S is None:
        return None
    st = S[S["mode"] == "stochastic"]
    reads = ["S_all", "S_exact", "S_nonThC", "S_ThC", "S_CTr", "S_FTi", "S_TiTa"]
    gm = st.groupby(["graph", "gseed", "rule", "window"])[reads].mean().reset_index()   # NaN (undefined S_all) skipped
    sem = st.groupby(["graph", "gseed", "rule", "window"])[reads].agg(lambda v: v.std(ddof=1) / np.sqrt(v.notna().sum()))
    sem.columns = [c + "_sem" for c in sem.columns]
    return gm.merge(sem.reset_index(), on=["graph", "gseed", "rule", "window"])


def pair_scores(tag, window):
    P = C.round2_pairs(tag)
    pa = P[(P["mode"] == "stochastic") & (P.readout == "all") & (P.window == window)].copy()
    pa["contrib"] = np.where(pa.gated, -pa["corr"], 0.0)
    return pa.groupby(["graph", "gseed", "rule", "pair"]).agg(contrib=("contrib", "mean"), gated_frac=("gated", "mean")).reset_index()


# ================================================================================ a: anatomical maps (both connectomes)
cmap = LinearSegmentedColormap.from_list("coord", C.COORD_COLORS)
norm = Normalize(0, 0.9)
from matplotlib.patches import Ellipse, Circle
BODY = C.BODY; EDGE = C.BODY_EDGE; RJ = 1.25
a_rows = []


def draw_fly(ax, tag, label_joints):
    ps = pair_scores(tag, "w15_25")
    real_p = ps[ps.graph == "real"].set_index("pair")
    null_p = ps[(ps.rule == "own") & ps.graph.isin(C.MAIN)].groupby("pair").contrib.max()
    chk = pd.read_csv(C.R2 / f"analysis/exp1_pair_real_vs_nulls_{tag}.csv").query("window == 'w15_25'").set_index("pair")
    assert np.allclose(chk.real.reindex(real_p.index), real_p.contrib, atol=1e-9)          # cross-check with the analysis table
    assert np.allclose(chk.null_max.reindex(null_p.index), null_p, atol=1e-9)
    ax.set_xlim(0, 62); ax.set_ylim(8, 78); ax.set_aspect("equal"); ax.axis("off")
    CX = 31.0; oy = -4.0
    ax.add_patch(Ellipse((CX, 67.5 + oy), 8.0, 6.0, fc=BODY, ec=EDGE, lw=0.6, zorder=1))
    ax.add_patch(Ellipse((CX, 50.5 + oy), 12.5, 23.0, fc=BODY, ec=EDGE, lw=0.6, zorder=1))
    ax.add_patch(Ellipse((CX, 28.0 + oy), 11.0, 19.0, fc=BODY, ec=EDGE, lw=0.6, zorder=1))
    for yy in (54.3, 46.5):
        ax.plot([CX - 5.6, CX + 5.6], [yy + oy, yy + oy], color=EDGE, lw=0.4, zorder=1)
    for (yc, lab) in [(58.5, "T1"), (50.5, "T2"), (42.5, "T3")]:
        ax.text(CX, yc + oy, lab, fontsize=5.4, ha="center", va="center", color=C.MUTED, zorder=2)
    LEG_R = {"T1": ((CX + 5.2, 58.5 + oy), [(2.6, 1.8), (4.6, 5.0), (4.6, 5.6), (2.2, 3.2)]),
             "T2": ((CX + 6.2, 50.5 + oy), [(3.0, 0.2), (6.0, 2.6), (6.0, -2.8), (3.0, -2.0)]),
             "T3": ((CX + 5.2, 42.5 + oy), [(2.6, -1.6), (5.2, -4.4), (4.6, -6.2), (2.2, -3.8)])}
    n_above = 0
    for seg, (p0, vecs) in LEG_R.items():
        for side in ("L", "R"):
            sgn = 1 if side == "R" else -1
            pts = [np.array([CX + sgn * (p0[0] - CX), p0[1]])]
            for v in vecs:
                pts.append(pts[-1] + np.array([sgn * v[0], v[1]]))
            for i in range(4):
                SC.capsule(ax, pts[i], pts[i + 1], width=[1.9, 1.6, 1.2, 0.8][i], fc=C.SEG, ec=EDGE, lw=0.45, z=1)
            leg = f"{seg}{side}"
            for j, joint in enumerate(C.JOINTS):
                pair = f"{leg}_{joint}"; xy = pts[j]
                if pair in real_p.index:
                    v = float(real_p.loc[pair, "contrib"]); nm = float(null_p.loc[pair]); above = v > nm; n_above += above
                    SC.node(ax, xy[0], xy[1], r=RJ, fc=cmap(norm(max(v, 0))), ec=C.MUTED, lw=0.45, z=6)
                    if above:
                        ax.add_patch(Circle(xy, RJ + 0.62, fc="none", ec=C.RING, lw=1.1, zorder=7))
                    a_rows.append(dict(connectome=tag, pair=pair, leg=leg, joint=joint, real_score=v, max_rewired_score=nm,
                                       real_above_all_rewired=above, gated_frac_real=float(real_p.loc[pair, "gated_frac"])))
                else:
                    ax.plot([xy[0]], [xy[1]], marker="x", ms=2.4, mew=0.6, color=C.LIGHT, zorder=6)
                    a_rows.append(dict(connectome=tag, pair=pair, leg=leg, joint=joint, real_score=np.nan, max_rewired_score=np.nan,
                                       real_above_all_rewired=False, gated_frac_real=np.nan))
            end = pts[-1]
            ax.text(end[0] + sgn * 1.0, end[1] + (0.8 if seg == "T1" else (-0.8 if seg == "T3" else 0)), leg, fontsize=5.6,
                    ha="left" if side == "R" else "right", va="center")
    if label_joints:
        p0, vecs = LEG_R["T3"]; pts = [np.array(p0)]
        for v in vecs:
            pts.append(pts[-1] + np.array(v))
        for j, joint in enumerate(C.JOINTS):
            ax.annotate(joint, pts[j], xytext=(pts[j][0] + [4.5, 5.5, 5.5, 4.5][j], pts[j][1] + [1.5, 1.5, 1.2, 0.8][j]), fontsize=5.3,
                        ha="left", va="center", color=C.MUTED, arrowprops=dict(arrowstyle="-", lw=0.4, color=C.MUTED, shrinkA=0.3, shrinkB=1.9))
        ax.annotate("", (CX, 76.0 + oy), xytext=(CX, 71.5 + oy), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C.MUTED, mutation_scale=6))
        ax.text(CX + 1.0, 75.2 + oy, "anterior", fontsize=5.2, color=C.MUTED, va="center")
    C.panel_title(ax, f"{C.TAGS[tag]} · real above all rewired at {n_above} of {len(real_p)} pairs", x_pt=10, y_pt=2)
    return n_above


draw_fly(ax_a, "theta50", True)
draw_fly(ax_a2, "manc50", False)
# keys column (right of the two maps)
kx = 130.0
cax = fig.add_axes([kx / W, 100.0 / H, 36.0 / W, 2.0 / H])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_ticks([0, 0.3, 0.6, 0.9]); cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1.2); cb.outline.set_linewidth(0.4)
cax.set_title("Pair score in the real network (15–25 s)", fontsize=5.6, pad=2.5, loc="left")
kax = fig.add_axes([kx / W, 68.0 / H, 50.0 / W, 26.0 / H]); kax.set_xlim(0, 50); kax.set_ylim(0, 26); kax.set_aspect("equal"); kax.axis("off")
kax.add_patch(Circle((2.0, 21.0), RJ, fc=C.REAL_MID, ec=C.MUTED, lw=0.45, zorder=6))
kax.add_patch(Circle((2.0, 21.0), RJ + 0.62, fc="none", ec=C.RING, lw=1.1, zorder=7))
kax.text(4.6, 21.0, "real network above all 30\nrewired networks for this pair", fontsize=5.2, va="center", linespacing=1.2)
kax.plot([2.0], [13.5], marker="x", ms=2.4, mew=0.6, color=C.LIGHT)
kax.text(4.6, 13.5, "no antagonistic pair at this joint", fontsize=5.2, va="center")
kax.text(0.0, 5.0, "Body and legs schematic;\nfills and rings are data", fontsize=5.2, ha="left", va="center", color=C.MUTED, style="italic",
         linespacing=1.2)
C.panel_label(ax_a, "a", x_pt=0, y_pt=2)
C.write_source(pd.DataFrame(a_rows), "Fig6a_pair_scores_map",
               "Both connectomes, 15-25 s. real_score = mean over 20 runs of the pair contribution (-corr if both pools pass the amplitude\n"
               "gate, else 0); max_rewired_score = maximum of that pair's score over the 30 main rewired networks (own frozen settings).\n"
               "NaN = no antagonistic pair at that joint. Source: round-2 pair-level re-run; checked against exp1_pair_real_vs_nulls_*.csv.")

# ================================================================================ b, c
JIT = np.linspace(-0.26, 0.26, 30)
b_rows, c_rows = [], []
for (axb, axc, tag) in [(ax_b1, ax_c1, "theta50"), (ax_b2, ax_c2, "manc50")]:
    gm = load(tag)
    for ax in (axb, axc):
        ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
    if gm is None:
        for ax, lab in ((axb, "per-joint scores"), (axc, "readouts")):
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ("left", "bottom"):
                ax.spines[sp_].set_color(C.LIGHT)
            ax.text(0.5, 0.5, f"{C.TAGS[tag]} {lab}:\npair-level re-run in progress", transform=ax.transAxes, ha="center", va="center",
                    fontsize=5.8, color=C.PENDING, style="italic", linespacing=1.3)
            C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold")
        continue
    w15 = gm[gm.window == "w15_25"]
    real = w15[w15.graph == "real"].iloc[0]
    nul = w15[(w15.rule == "own") & w15.graph.isin(C.MAIN)].sort_values(["graph", "gseed"])
    # --- b: per joint
    for k, joint in enumerate(C.JOINTS):
        col = JOINT_READ[joint]
        vals = nul[col].to_numpy(); fams = nul.graph.to_numpy(); sems = nul[col + "_sem"].to_numpy()
        order = np.argsort(vals)
        for xx_, yy_, ee_, ff_ in zip(k + JIT[:len(vals)], vals[order], sems[order], fams[order]):
            axb.plot([xx_, xx_], [yy_ - ee_, yy_ + ee_], color=C.FAM_COLOR[ff_], lw=0.45, zorder=2)
        axb.scatter(k + JIT[:len(vals)], vals[order], s=8, c=[C.FAM_COLOR[f] for f in fams[order]], edgecolors="white", linewidths=0.25, zorder=3)
        rv = float(real[col]); rsem = float(real[col + "_sem"])
        axb.add_patch(Rectangle((k - 0.34, rv - rsem), 0.68, 2 * rsem, facecolor=C.REAL_PALE, edgecolor="none", zorder=1))
        axb.plot([k - 0.34, k + 0.34], [rv, rv], color=C.REAL, lw=1.6, zorder=4, solid_capstyle="butt")
        nm = float(vals.max())
        axb.text(k, max(rv, nm) + 0.045, f"{C.fmt(rv, 2)}", fontsize=5.6, ha="center", va="bottom", color=C.REAL)
        b_rows += [dict(connectome=tag, joint=joint, network=g_, gseed=int(s_), score=v_, sem=e_)
                   for g_, s_, v_, e_ in zip(nul.graph, nul.gseed, nul[col], nul[col + "_sem"])]
        b_rows.append(dict(connectome=tag, joint=joint, network="real", gseed=0, score=rv, sem=rsem))
    axb.set_xticks(range(4)); axb.set_xticklabels(C.JOINTS)
    axb.set_xlim(-0.55, 3.55); axb.set_ylim(-0.5, 0.88)
    axb.set_yticks([-0.4, 0, 0.4, 0.8]); axb.set_yticklabels(["−0.4", "0", "0.4", "0.8"])
    axb.tick_params(axis="x", length=0, pad=2.5)
    C.panel_title(axb, f"{C.TAGS[tag]} · per joint", y_pt=4)
    # --- c: readouts with levels in both windows
    levels_txt = []
    for k, rd in enumerate(READS):
        vals = nul[rd].to_numpy(); fams = nul.graph.to_numpy(); order = np.argsort(vals); sems = nul[rd + "_sem"].to_numpy()
        for xx_, yy_, ee_, ff_ in zip(k + JIT[:len(vals)], vals[order], sems[order], fams[order]):
            axc.plot([xx_, xx_], [yy_ - ee_, yy_ + ee_], color=C.FAM_COLOR[ff_], lw=0.45, zorder=2)
        axc.scatter(k + JIT[:len(vals)], vals[order], s=8, c=[C.FAM_COLOR[f] for f in fams[order]], edgecolors="white", linewidths=0.25, zorder=3)
        rv = float(real[rd]); rsem = float(real[rd + "_sem"])
        axc.add_patch(Rectangle((k - 0.34, rv - rsem), 0.68, 2 * rsem, facecolor=C.REAL_PALE, edgecolor="none", zorder=1))
        axc.plot([k - 0.34, k + 0.34], [rv, rv], color=C.REAL, lw=1.6, zorder=4, solid_capstyle="butt")
        lv = {}
        for w in ("w15_25", "w50_60"):
            ww = gm[gm.window == w]; r_ = float(ww[ww.graph == "real"][rd].iloc[0])
            fams_w = {f: ww[(ww.graph == f) & (ww.rule == "own")][rd].to_numpy() for f in C.MAIN}
            lv[w] = level(r_, fams_w)
            c_rows.append(dict(connectome=tag, readout=rd, window=w, real=r_, max_rewired=float(max(v.max() for v in fams_w.values())),
                               level=lv[w]))
        axc.text(k, max(rv, vals.max()) + 0.03, f"{C.fmt(rv)}", fontsize=5.6, ha="center", va="bottom", color=C.REAL)
        levels_txt.append(f"{lv['w15_25']} / {lv['w50_60']}")
        for dx_, w_ in [(-0.13, "w15_25"), (0.13, "w50_60")]:
            fc_ = C.LEVEL_FC[lv[w_].upper()]
            axc.add_patch(Rectangle((k + dx_ - 0.1, -0.232), 0.2, 0.034, facecolor=fc_, edgecolor=C.MUTED, lw=0.4, zorder=5))
    cl = pd.read_csv(C.R2 / "analysis/exp1_claim_levels.csv")
    for r_ in c_rows:
        if r_["connectome"] == tag:
            ref = cl[(cl.connectome == tag) & (cl.window == r_["window"]) & (cl.readout == r_["readout"])].iloc[0]
            assert ref.level.lower() == r_["level"] and abs(ref.real - r_["real"]) < 1e-9, (r_, ref.level, ref.real)
    axc.set_xticks(range(3)); axc.set_xticklabels([READ_LAB[r] for r in READS])
    axc.set_xlim(-0.55, 2.55); axc.set_ylim(-0.25, 0.43)
    axc.set_yticks([-0.2, 0, 0.2, 0.4]); axc.set_yticklabels(["−0.2", "0", "0.2", "0.4"])
    axc.tick_params(axis="x", length=0, pad=2.5)
    C.panel_title(axc, f"{C.TAGS[tag]} · pre-specified readouts", y_pt=4)
ax_b1.set_ylabel("Mean pair score")
ax_c1.set_ylabel("S (15–25 s)")

fig.text(14.0 / W, 2.2 / H, "Squares in c: pre-specified level at 15–25 s (left) and 50–60 s (right); dark teal strong, light teal moderate, "
         "white weak", fontsize=5.2, color=C.MUTED, style="italic", ha="left", va="bottom")
C.panel_label(ax_b1, "b", x_pt=-30, y_pt=4)
C.panel_label(ax_c1, "c", x_pt=-26, y_pt=4)
fh = [Line2D([], [], color=C.REAL, lw=1.6, label="Real ± s.e.m.")] + \
     [Line2D([], [], ls="none", marker="o", ms=2.8, mfc=C.FAM_COLOR[f], mec="white", mew=0.25, label=C.FAM_LABEL[f]) for f in C.MAIN]
fig.legend(handles=fh, loc="upper left", bbox_to_anchor=(130.0 / W, 64.0 / H), ncol=2, fontsize=5.4, handlelength=1.3,
           handletextpad=0.35, columnspacing=1.0, labelspacing=0.35, frameon=False, borderaxespad=0,
           title="b, c: rewired networks (mean ± s.e.m.)", title_fontsize=5.4, alignment="left")
C.write_source(pd.DataFrame(b_rows), "Fig6b_joint_scores",
               "Mean pair score per joint (pairs of that joint; non-gated pairs count 0), mean over 20 runs, 15-25 s; 30 main rewired\n"
               "networks at their own frozen settings and the real network. Round-2 pair-level re-run.")
C.write_source(pd.DataFrame(c_rows), "Fig6c_readouts_levels",
               "Pre-specified readouts (round-2 pre-registration): S over all 20 pairs, over the 16 pairs of exactly mapped MNs, and over\n"
               "the 14 non-ThC pairs; level = §3.23 rule on the six main families, per window. Checked against the round-2 analysis table.")

C.save(fig, "Fig6_joint_localization", dict(
    axes=[ax_a, ax_a2, ax_b1, ax_b2, ax_c1, ax_c2], panel_ids=["a1", "a2", "b1", "b2", "c1", "c2"],
    row_groups=[["a1", "a2"], ["b1", "b2"], ["c1", "c2"], ["b1", "c1"]],
))
print("Fig. 6 written")
