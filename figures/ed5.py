"""Extended Data Fig. 5 | Restoring the last layer is not enough (round 3, C: rescue).

a  Construction (schematic): in each parent rewired network all edges onto motor neurons are replaced by the real network's
   (C1), by the within-pool control's (C1p) or by the cross-pool arm's (C0) edges onto motor neurons; everything upstream is kept.
b  Recovery fraction RF = (S_arm - S_parent) / (S_real - S_parent) of every rescued network (lines join the three versions of
   the same parent), 15-25 s; pre-registered thresholds 0.5 (recovery) and 0.2 (partial).
c  Operational check: reciprocal-innervation index of the parents, the three rescued versions and the real network.
Values from derived/r3_C_per_network.csv and r3_C_structure.csv (check_round3.py: recomputed from the round-3 parquet files,
checked against analysis/C_D2 and rescue_build_report.csv).
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common as C
import schem as SC

C.setup_style()
W, H = 183.0, 96.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
ax_a = A(1.0, 10.0, 50.0, 76.0)
ax_b1 = A(66.0, 16.0, 34.0, 60.0); ax_b2 = A(106.0, 16.0, 34.0, 60.0)
ax_c = A(152.0, 16.0, 29.0, 60.0)
CR = pd.read_csv(C.DERIVED / "r3_C_per_network.csv"); CS = pd.read_csv(C.DERIVED / "r3_C_structure.csv")
DEC = pd.read_csv(C.R3A / "全部判定/round3_decisions.csv")
ARMS = ["C1", "C1p", "C0"]
ARM_LAB = {"C1": "C1\nreal", "C1p": "C1p\nwithin", "C0": "C0\ncross"}
ARM_COL = {"C1": C.REAL, "C1p": C.WITHIN, "C0": C.CROSS}
VERDICT = {"恢复": "recovery", "部分恢复": "partial recovery", "不恢复": "no recovery"}

# ================================================================================ a: construction (schematic)
ax = ax_a
ax.set_xlim(0, 50); ax.set_ylim(0, 76); ax.axis("off")
ax.text(4.2, 75.5, "Rescue: put back only the last layer", fontsize=6.6, fontweight="bold", ha="left", va="top")
SC.box(ax, 3.0, 50.0, 44.0, 14.0, fc=C.SAND, ec=C.FAM_COLOR["hl"], lw=0.7, r=1.8, z=1)
ax.text(25.0, 58.8, "parent: a rewired network", fontsize=5.8, ha="center", va="center")
ax.text(25.0, 54.2, "(upstream wiring kept as it is)", fontsize=5.2, ha="center", va="center", color=C.MUTED)
for k, (arm, lab) in enumerate([("C1", "real network's edges"), ("C1p", "within-pool control's"), ("C0", "cross-pool arm's")]):
    x0 = 3.0 + k * 15.0
    SC.arrow(ax, (x0 + 6.5, 45.2), (x0 + 6.5, 40.8), color=C.INK, lw=0.7, head=3.0)
    SC.box(ax, x0, 25.0, 13.0, 15.0, fc="white", ec=ARM_COL[arm], lw=0.9, r=1.4, z=1)
    ax.text(x0 + 6.5, 35.8, arm, fontsize=6.4, ha="center", va="center", color=ARM_COL[arm], fontweight="bold")
    ax.text(x0 + 6.5, 30.0, lab.replace(" ", "\n", 1), fontsize=5.0, ha="center", va="center", linespacing=1.15)
    for m, col in enumerate([C.EXT, C.EXT, C.FLX, C.FLX]):
        ax.add_patch(plt.Circle((x0 + 2.8 + m * 2.5, 20.5), 0.9, fc=col, ec="white", lw=0.3, zorder=3))
ax.text(25.0, 47.6, "edges onto motor neurons replaced by", fontsize=5.2, ha="center", va="center", color=C.MUTED)
ax.text(25.0, 15.5, "motor-neuron pools (ext., flx.)", fontsize=5.2, ha="center", va="center", color=C.MUTED)
n_edge = CS[CS.arm == "C1"].groupby("connectome").frac_edges_mn_layer.median()
ax.text(1.0, 8.5, f"last layer: {100 * n_edge['theta50']:.1f}% (MaleCNS) and {100 * n_edge['manc50']:.1f}% (MANC)\nof all edges; every neuron keeps "
        "its out-degree", fontsize=5.2, ha="left", va="center", color=C.INK, linespacing=1.25)
ax.text(49.5, 0.3, "schematic", fontsize=5.2, ha="right", va="bottom", color=C.MUTED, style="italic")
C.panel_label(ax_a, "a", x_pt=0, y_pt=-8)

# ================================================================================ b: recovery fraction
b_rows = []
for ax, tag in [(ax_b1, "theta50"), (ax_b2, "manc50")]:
    q = CR[(CR.connectome == tag) & (CR.window == "w15_25")]
    ax.axhline(0, color=C.GRID, lw=0.7, zorder=0); ax.axhline(1, color=C.REAL, lw=0.8, zorder=0)
    ax.axhline(0.5, color=C.INK, lw=0.5, ls=(0, (1, 1.6)), zorder=0); ax.axhline(0.2, color=C.INK, lw=0.5, ls=(0, (3, 2)), zorder=0)
    for par, g in q.groupby("parent"):
        g = g.set_index("arm").loc[ARMS]
        fam = g.parent_family.iloc[0]
        ax.plot(range(3), g.RF, color=C.FAM_COLOR[fam], lw=0.5, alpha=0.8, zorder=1)
        ax.scatter(range(3), g.RF, s=10, color=C.FAM_COLOR[fam], edgecolors="white", linewidths=0.3, zorder=3)
    lo = float(q.RF.min())
    for k, arm in enumerate(ARMS):
        a = q[q.arm == arm]; m = float(a.RF.median()); nab = int((a.S_arm > a.S_parent).sum())
        d = DEC[(DEC.exp == f"C_rescue_{arm}") & (DEC.connectome == tag) & (DEC.window == "w15_25")].iloc[0]
        assert abs(m - d.median_RF) < 1e-9 and nab == int(d.n_above_parent)
        cat = "recovery" if (m >= 0.5 and nab >= 8) else ("partial recovery" if m >= 0.2 else "no recovery")
        assert cat == VERDICT[d.level]
        ax.plot([k - 0.28, k + 0.28], [m, m], color=C.INK, lw=1.3, zorder=4, solid_capstyle="butt")
        ax.text(k, 1.32, f"{C.fmt(m, 2)}\n{cat.replace(' recovery', '')}", fontsize=5.2, ha="center", va="top",
                color=ARM_COL[arm], linespacing=1.15)
        for r in a.itertuples():
            b_rows.append(dict(connectome=tag, arm=arm, parent=r.parent, S_parent=r.S_parent, S_arm=r.S_arm, S_real=r.S_real, RF=r.RF,
                               arm_median_RF=m, n_above_parent=nab, verdict=cat))
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(min(-1.1, lo - 0.15), 1.4)
    ax.set_xticks(range(3)); ax.set_xticklabels([ARM_LAB[a] for a in ARMS], fontsize=5.4, linespacing=1.35)
    for tl, a_ in zip(ax.get_xticklabels(), ARMS):
        tl.set_color(ARM_COL[a_])
    C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold", loc="center")
ax_b1.set_ylabel("Recovery fraction RF")
ax_b1.text(-0.45, 0.96, "real", fontsize=5.2, ha="left", va="top", color=C.REAL)
C.panel_label(ax_b1, "b", x_pt=-30, y_pt=4)
fh = [Line2D([], [], ls="none", marker="o", ms=3, mfc=C.FAM_COLOR[f], mec="white", mew=0.3, label=C.FAM_LABEL[f]) for f in ["hl", "degree", "leg_hl"]]
fh += [Line2D([], [], color=C.INK, lw=0.5, ls=(0, (1, 1.6)), label="0.5 recovery"), Line2D([], [], color=C.INK, lw=0.5, ls=(0, (3, 2)), label="0.2 partial")]
fig.legend(handles=fh, loc="lower left", bbox_to_anchor=(66.0 / W, 0.5 / H), ncol=5, fontsize=5.2, frameon=False, handlelength=1.6,
           columnspacing=1.2, handletextpad=0.3, borderaxespad=0)
C.write_source(pd.DataFrame(b_rows), "ED5b_recovery_fraction",
               "Round 3, C: parents = MaleCNS lineage block and degree s0-s4, MANC lineage block and leg x lineage s0-s4 (fixed before the run from\n"
               "the round-2 family maxima). S_arm and S_parent at each network's own frozen setting, 15-25 s (S_parent archived round-2 value);\n"
               "RF = (S_arm - S_parent) / (S_real - S_parent). Verdict (pre-registered): recovery if median RF >= 0.5 and >= 8/10 above the parent;\n"
               "partial if 0.2 <= median < 0.5; otherwise none (asserted against analysis/全部判定/round3_decisions.csv).")

# ================================================================================ c: structural check
ax = ax_c
c_rows = []
ORDER = ["parent", "C1", "C1p", "C0", "real"]
for tag, dx, mk in [("theta50", -0.17, "o"), ("manc50", 0.17, "s")]:
    q = CS[CS.connectome == tag]
    for k, arm in enumerate(ORDER):
        v = q[q.arm == arm].ri_index.to_numpy()
        col = C.REAL if arm == "real" else ARM_COL.get(arm, C.NULL_MED)
        ax.scatter(np.full(len(v), k + dx), v, s=9 if arm != "real" else 16, marker=mk, color=col, edgecolors="white", linewidths=0.3, zorder=3)
        for x_ in v:
            c_rows.append(dict(connectome=tag, arm=arm, ri_index=x_))
ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.6, 0.6); ax.set_yticks([-0.5, 0, 0.5])
ax.set_xticks(range(5)); ax.set_xticklabels(["parent", "C1", "C1p", "C0", "real"], fontsize=5.2, rotation=90)
ax.set_ylabel("Reciprocal-innervation index")
C.panel_title(ax_c, "Structure", y_pt=4, fontweight="bold", loc="center")
ax.text(0.03, 0.97, "circles MaleCNS\nsquares MANC", transform=ax.transAxes, fontsize=5.0, ha="left", va="top", color=C.MUTED, linespacing=1.2)
C.panel_label(ax_c, "c", x_pt=-28, y_pt=4)
C.write_source(pd.DataFrame(c_rows), "ED5c_structure_index",
               "Operational check of the rescue construction (rescue_build_report.csv): reciprocal-innervation index of each parent, of its three\n"
               "rescued versions and of the real network. C1 and C1p have identical indices (within-pool exchanges leave the index unchanged).")

C.save(fig, "EDFig5_rescue", dict(
    axes=[ax_b1, ax_b2, ax_c], panel_ids=["b1", "b2", "c"], row_groups=[["b1", "b2", "c"]],
    exemptions=[{"panels": ["c"], "checks": ["panel-width", "horizontal-gutter"], "reason": "c is a narrower structural check beside the two RF panels"}],
))
print("ED Fig. 5 written")
