"""Extended Data Fig. 6 | Where and when the advantage holds (round 2, experiment 1, both connectomes).

a  Pre-specified level of every readout in every confirmation window: all 20 pairs, exact-map pairs (16), pairs outside
   the thorax-coxa joint (14), and each joint alone; cell text = real-network value.
b  Pair scores of the real networks in the four windows (which antagonistic pairs carry the score, and when).
Values from derived/r2_exp1_levels.csv and r2_exp1_pair_scores_*.csv (recomputed from pairs_*.parquet, checked against
analysis/exp1_claim_levels.csv).
"""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import common as C

C.setup_style()
W, H = 183.0, 96.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
ax_a = A(30.0, 16.0, 74.0, 66.0)
ax_b1 = A(126.0, 16.0, 22.0, 66.0)
ax_b2 = A(158.0, 16.0, 22.0, 66.0)
LV = pd.read_csv(C.DERIVED / "r2_exp1_levels.csv")
READS = [("S_all", "All pairs (20)"), ("S_exact", "Exact map (16)"), ("S_nonThC", "Without ThC (14)"), ("S_ThC", "ThC (6)"),
         ("S_CTr", "CTr (6)"), ("S_FTi", "FTi (6)"), ("S_TiTa", "TiTa (2)")]
LEVEL_FC = C.LEVEL_FC
ax = ax_a
ax.set_xlim(-0.55, 8.95); ax.set_ylim(len(READS) + 0.1, -1.7); ax.axis("off")
a_rows = []
for ci, tag in enumerate(C.TAGS):
    for wi, w in enumerate(C.WINDOWS):
        x = ci * 4.5 + wi
        for ri, (rd, lab) in enumerate(READS):
            r = LV[(LV.connectome == tag) & (LV.window == w) & (LV.readout == rd)].iloc[0]
            ri_ = ri + (0.35 if ri >= 3 else 0)
            ax.add_patch(Rectangle((x - 0.46, ri_ - 0.44), 0.92, 0.88, facecolor=LEVEL_FC[r.level], edgecolor=C.MUTED, lw=0.5))
            ax.text(x, ri_, C.fmt(r.real, 2), fontsize=5.2, ha="center", va="center",
                    color="white" if r.level == "STRONG" else C.INK)
            a_rows.append(dict(connectome=tag, window=w, readout=rd, real=r.real, level=r.level.lower(), strongest_rewired=r.strongest_rewired,
                               strongest_family=r.strongest_family, families_all_below=int(r.families_all_below)))
        ax.text(x, -0.75, C.WIN_LABEL[w].replace(" s", ""), fontsize=5.2, ha="center", va="center", rotation=0)
    ax.text(ci * 4.5 + 1.5, -1.45, C.TAGS[tag], fontsize=6.3, ha="center", va="center", fontweight="bold")
for ri, (rd, lab) in enumerate(READS):
    ax.text(-0.62, ri + (0.35 if ri >= 3 else 0), lab, fontsize=5.8, ha="right", va="center")
ax.text(-0.62, 2.95, "joint alone:", fontsize=5.2, ha="right", va="center", color=C.MUTED, style="italic")
ky = len(READS) + 0.75
for k, (lv, lab) in enumerate([("STRONG", "strong"), ("MODERATE", "moderate"), ("WEAK", "weak")]):
    ax.add_patch(Rectangle((k * 1.9 - 0.3, ky - 0.25), 0.5, 0.5, facecolor=LEVEL_FC[lv], edgecolor=C.MUTED, lw=0.5, clip_on=False))
    ax.text(k * 1.9 + 0.32, ky, lab, fontsize=5.4, va="center", ha="left")
ax.text(5.8, ky, "cell text: real network's value · window (s)", fontsize=5.2, va="center", ha="left", color=C.MUTED)
C.panel_label(ax_a, "a", x_pt=-80, y_pt=4)
C.panel_title(ax_a, "Pre-specified level of each readout, every window", x_pt=-70, y_pt=4)
C.write_source(pd.DataFrame(a_rows), "ED6a_levels",
               "Pre-specified level (§3.23 rule over the six main families) of each readout in each confirmation window; real = mean over 20 runs.\n"
               "Recomputed from pairs_scalar_*.parquet; identical to analysis/exp1_claim_levels.csv (checked).")

# ---------------- b: real-network pair scores across windows
cmap = LinearSegmentedColormap.from_list("coord", C.COORD_COLORS)
norm = Normalize(0, 0.9)
b_rows = []
for ax, tag in [(ax_b1, "theta50"), (ax_b2, "manc50")]:
    fr = pd.read_csv(C.DERIVED / f"r2_exp1_pair_scores_{tag}.csv")
    fr = fr[(fr.graph == "real")]
    pairs = sorted(fr.pair.unique(), key=C.pair_sort_key)
    M = np.zeros((len(pairs), 4)); G_ = np.zeros_like(M)
    for j, w in enumerate(C.WINDOWS):
        q = fr[fr.window == w].set_index("pair")
        M[:, j] = q.loc[pairs, "contrib"].to_numpy(); G_[:, j] = q.loc[pairs, "fg"].to_numpy()
    ax.imshow(np.clip(M, 0, None), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(len(pairs)):
        for j in range(4):
            if G_[i, j] >= 0.5:
                dark = norm(max(M[i, j], 0)) > 0.45
                ax.plot([j], [i], marker="o", ms=1.8, color="white" if dark else C.INK, zorder=4)
            b_rows.append(dict(connectome=tag, pair=pairs[i], window=C.WINDOWS[j], pair_score=M[i, j], gated_fraction=G_[i, j],
                               contributing=bool(G_[i, j] >= 0.5)))
    ax.set_xticks(range(4)); ax.set_xticklabels(["15", "25", "35", "50"], fontsize=5.2)
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p.replace("_", " ") for p in pairs] if tag == "theta50" else [], fontsize=5.0)
    ax.tick_params(length=0, pad=1.5)
    for k, jn in enumerate(C.JOINTS[:-1]):
        n = sum(1 for p in pairs if p.endswith(jn))
        idx = max(i for i, p in enumerate(pairs) if p.endswith(jn))
        ax.axhline(idx + 0.5, color=C.MUTED, lw=0.6)
    ax.set_xlabel("Window start (s)")
    C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold", loc="center")
cax = fig.add_axes([126.0 / W, 5.0 / H, 54.0 / W, 1.6 / H])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_ticks([0, 0.3, 0.6, 0.9]); cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1.0); cb.outline.set_linewidth(0.4)
fig.text(126.0 / W, 8.4 / H, "pair score, real network  ·  dot: pair modulated in ≥ half of the runs", fontsize=5.1, ha="left", va="bottom",
         color=C.MUTED)
C.panel_label(ax_b1, "b", x_pt=-30, y_pt=4)
C.write_source(pd.DataFrame(b_rows), "ED6b_pair_scores_by_window",
               "Real networks at their frozen settings: mean over 20 runs of each pair's score (−corr if both pools modulated, else 0; negative values\n"
               "shown as 0 in the heat map, exact values here); gated_fraction = fraction of runs in which both pools pass the amplitude gate;\n"
               "contributing (dot) = gated in >= 50% of runs, the pair-identity rule of condition C6.")

C.save(fig, "EDFig6_readouts_across_windows", dict(
    axes=[ax_b1, ax_b2], panel_ids=["b1", "b2"], row_groups=[["b1", "b2"]],
))
print("ED Fig. 6 written")
