"""Extended Data Fig. 4 | Dose-matched reassignment of premotor inputs (round 2, experiment 2).

a  Manipulation check: each reassigned network's net cross-pool redistribution against the change in its motor
   neurons' input strength (median relative change; larger of excitatory and inhibitory). Right: for every dose-matched
   network (W, X_lo, X_hi) the median and the 95th percentile of that change (surgery_r2_report.json), so that the
   tails of the within-pool control and of the cross-pool arms can be compared.
b  Pre-registered conditions C1-C6 for the dose-matched arms (X_hi at the real network's frozen setting = primary test;
   X_hi re-optimised; X_lo at the real setting), 15-25 s and 50-60 s.
c  Re-optimised settings: S of the within-pool control W and of X_hi, each network at its own frozen setting.
d  Activity maintenance: oscillating-motor-neuron coverage and retention of the contributing pairs, relative to real.
All values from derived/r2_exp2_*.csv (recomputed from the parquet files and checked against analysis/exp2_*.csv).
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
import common as C

C.setup_style()
W, H = 183.0, 134.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
PURPLE = C.CROSS; GREYW = C.WITHIN   # palette v2
ax_a = A(16.0, 80.0, 43.0, 42.0)
ax_a2 = A(62.0, 80.0, 16.0, 42.0)
ax_b = A(120.0, 80.0, 60.0, 42.0)
ax_c1 = A(16.0, 12.0, 34.0, 44.0)
ax_c2 = A(56.0, 12.0, 34.0, 44.0)
ax_d1 = A(118.0, 12.0, 28.0, 44.0)
ax_d2 = A(152.0, 12.0, 28.0, 44.0)
rep = C.exp2_report(); aud = C.intervention_audit()
CD = pd.read_csv(C.DERIVED / "r2_exp2_conditions.csv")
GM = {t: pd.read_csv(C.DERIVED / f"r2_exp2_graph_means_{t}.csv") for t in C.TAGS}
PS = {t: pd.concat([pd.read_csv(C.DERIVED / f"r2_exp1_pair_scores_{t}.csv"), pd.read_csv(C.DERIVED / f"r2_exp2_pair_scores_{t}.csv")])
      for t in C.TAGS}

# ================================================================================ a: manipulation check
ARM_A = [("mn_shuffle_pool", "within", False, "within, original"), ("mn_shuffle_joint", "cross", False, "cross, original"),
         ("mn_shuffle_pool_w", "within", True, "within, strength-matched"), ("mn_shuffle_joint_w", "cross", True, "cross, strength-matched"),
         ("mn_r2_within", "within", True, "W"), ("mn_r2_cross_lo", "cross", True, "X_lo"),
         ("mn_r2_cross_hi", "cross", True, "X_hi")]
a_rows = []
for tag, mk in [("theta50", "o"), ("manc50", "s")]:
    for g, kind, matched, lab in ARM_A:
        for s_ in range(5):
            if g.startswith("mn_r2"):
                m = rep[f"{tag}/{g}_s{s_}"]; x_ = float(m["redistr"]); dE, dI = float(m["med_rel_dE"]), float(m["med_rel_dI"])
                pE, pI = float(m["p95_rel_dE"]), float(m["p95_rel_dI"])
            else:
                a = aud[(aud.connectome == tag) & (aud.graph == g)].set_index("seed").loc[s_]
                x_ = float(a.pool_allocation_redistribution_fraction); dE, dI = float(a.median_relative_E_change), float(a.median_relative_I_change)
                pE, pI = float(a.p95_relative_E_change), float(a.p95_relative_I_change)
            y_ = max(dE, dI)
            col = PURPLE if kind == "cross" else GREYW
            ax_a.scatter([x_], [y_], s=10, marker=mk, facecolors="white" if matched else col, edgecolors=col if matched else "white",
                         linewidths=0.6 if matched else 0.3, zorder=3)
            a_rows.append(dict(connectome=tag, arm=g, gseed=s_, redistribution=x_, median_rel_E_change=dE, median_rel_I_change=dI, plotted=y_,
                               p95_rel_E_change=pE, p95_rel_I_change=pI, p95_plotted=max(pE, pI) if g.startswith("mn_r2") else np.nan))
A_ = pd.DataFrame(a_rows)
for g, kind, matched, lab in ARM_A:
    q = A_[A_.arm == g]
    xm, ymed, ymax, ymin = float(q.redistribution.median()), float(q.plotted.median()), float(q.plotted.max()), float(q.plotted.min())
    if kind == "within":        # label to the right of the x = 0 column, at the cluster's median height
        ax_a.text(0.014, ymed, lab, fontsize=5.1, ha="left", va="center", color=C.MUTED)
    else:                       # label above the cluster
        ax_a.text(xm, ymax * 1.28, lab.replace("cross, ", "cross,\n") if "original" in lab or "strength" in lab else lab, fontsize=5.1,
                  ha="center", va="bottom", color=PURPLE, linespacing=1.15)
ax_a.set_yscale("log"); ax_a.set_ylim(0.002, 1.6); ax_a.set_xlim(-0.015, 0.31)
ax_a.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1]); ax_a.set_yticklabels(["0.3%", "1%", "3%", "10%", "30%", "100%"]); ax_a.minorticks_off()
ax_a.set_xticks([0, 0.1, 0.2, 0.3])
ax_a.set_xlabel("Premotor input moved across pools")
ax_a.set_ylabel("Change in MN input strength")
C.panel_title(ax_a, "Design check", y_pt=4)
ax_a.text(0.02, 0.98, "hollow: strength-matched\ncircles MaleCNS, squares MANC", transform=ax_a.transAxes, fontsize=5.1, ha="left", va="top",
          color=C.MUTED, linespacing=1.3)
C.panel_label(ax_a, "a", x_pt=-32, y_pt=4)
# a, right: median and 95th percentile of the input change, every dose-matched network (same log axis as the left part)
ax = ax_a2
TAIL = [("mn_r2_within", "W", GREYW), ("mn_r2_cross_lo", "X_lo", PURPLE), ("mn_r2_cross_hi", "X_hi", PURPLE)]
for i, (g, lab, col) in enumerate(TAIL):
    for tag, mk, dx in [("theta50", "o", -0.2), ("manc50", "s", 0.2)]:
        q = A_[(A_.connectome == tag) & (A_.arm == g)].sort_values("gseed")
        for r in q.itertuples():
            x = i + dx + (r.gseed - 2) * 0.045
            ax.plot([x, x], [r.plotted, r.p95_plotted], color=col, lw=0.5, zorder=2, solid_capstyle="butt")
            ax.plot([x - 0.06, x + 0.06], [r.p95_plotted] * 2, color=col, lw=0.8, zorder=2, solid_capstyle="butt")
            ax.scatter([x], [r.plotted], s=7, marker=mk, facecolors="white", edgecolors=col, linewidths=0.5, zorder=3)
ax.set_yscale("log"); ax.set_ylim(0.002, 1.6); ax.set_xlim(-0.6, 2.6)
ax.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1]); ax.set_yticklabels([]); ax.minorticks_off()
ax.set_xticks(range(3)); ax.set_xticklabels([t[1] for t in TAIL], fontsize=5.4)
ax.text(0.5, 0.985, "marker: median\nbar top: 95th pct.", transform=ax.transAxes, fontsize=5.0, ha="center", va="top", color=C.MUTED,
        linespacing=1.25)
C.panel_title(ax_a2, "95th percentile", y_pt=4, fontsize=5.6)
C.write_source(A_, "ED4a_design_check",
               "Per reassigned network: net cross-pool redistribution and the median relative change of each motor neuron's excitatory (E) and\n"
               "inhibitory (I) input; plotted = larger of the two medians. p95_* = 95th percentile of the same relative changes; p95_plotted\n"
               "(right part of panel a, dose-matched arms only) = larger of the two. Original arms: structural audit table; experiment-2 arms:\n"
               "surgery_r2_report.json (same metric code: the report reproduces the audit values of the original strength-matched network exactly).")

# ================================================================================ b: pre-registered conditions
ROWS_B = [("theta50", "mn_r2_cross_hi", "real_setting", "MaleCNS · X_hi, real setting (primary)"), ("theta50", "mn_r2_cross_hi", "own", "MaleCNS · X_hi, re-optimised"),
          ("theta50", "mn_r2_cross_lo", "real_setting", "MaleCNS · X_lo, real setting"), ("manc50", "mn_r2_cross_hi", "real_setting", "MANC · X_hi, real setting (primary)"),
          ("manc50", "mn_r2_cross_hi", "own", "MANC · X_hi, re-optimised"), ("manc50", "mn_r2_cross_lo", "real_setting", "MANC · X_lo, real setting")]
CONDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
ax = ax_b
ax.set_xlim(-0.6, 14.8); ax.set_ylim(len(ROWS_B) + 0.2, -2.3); ax.axis("off")
b_rows = []
for i, (tag, arm, rule, lab) in enumerate(ROWS_B):
    y = i + (0.35 if tag == "manc50" else 0)
    for k, w in enumerate(["w15_25", "w50_60"]):
        r = CD[(CD.connectome == tag) & (CD.window == w) & (CD.arm == arm) & (CD.rule == rule)].iloc[0]
        x0 = k * 8.0
        for j, c_ in enumerate(CONDS):
            ok = bool(r[c_])
            ax.add_patch(Rectangle((x0 + j - 0.36, y - 0.36), 0.72, 0.72, facecolor=C.REAL if ok else "white", edgecolor=C.MUTED, lw=0.5))
        ok = bool(r.all_six)
        ax.add_patch(Rectangle((x0 + 6.3 - 0.4, y - 0.4), 0.8, 0.8, facecolor=C.SWITCH if ok else "white", edgecolor=C.SWITCH, lw=0.8))
        b_rows.append(dict(connectome=tag, arm=arm, rule=rule, window=w, **{c_: bool(r[c_]) for c_ in CONDS}, all_six=bool(r.all_six),
                           R=r.R, W=r.W, X=r.X, X_over_R=r.X_over_R, osc_retention=r.osc_retention, pair_retention=r.pair_retention))
    ax.text(-0.9, y, lab, fontsize=5.4, ha="right", va="center")
for k, w in enumerate(["15–25 s", "50–60 s"]):
    x0 = k * 8.0
    ax.text(x0 + 2.8, -1.9, w, fontsize=6, ha="center", va="center")
    for j, c_ in enumerate(CONDS):
        ax.text(x0 + j, -1.0, c_, fontsize=5.4, ha="center", va="center")
    ax.text(x0 + 6.3, -1.0, "all", fontsize=5.4, ha="center", va="center", color=C.SWITCH)
ax.text(-7.4, 6.45, "Filled: met. C1 W ≥ 0.8 R · C2 X ≤ 0.5 R and R − X ≥ 0.1 · C3 X ≤ 0.5 W and W − X ≥ 0.1 ·\n"
        "C4 every X below R and below every W · C5 rhythmicity of X ≥ 0.8 of R and of W ·\n"
        "C6 ≥ 80% of contributing pairs and of oscillating MNs kept (R real, W within-pool, X cross-pool)",
        fontsize=5.0, ha="left", va="top", color=C.MUTED, linespacing=1.3)
C.panel_label(ax_b, "b", x_pt=-106, y_pt=4)
C.panel_title(ax_b, "Pre-registered conditions for a selective loss", x_pt=-96, y_pt=4)
C.write_source(pd.DataFrame(b_rows), "ED4b_conditions",
               "Pre-registered conditions (§3.26 + amendment 1) for the experiment-2 arms; R = real, W = median of 5 within-pool controls, X = median of\n"
               "5 cross-pool networks, same parameter rule. Identical to analysis/exp2_conditions.csv (checked by check_round2.py).")

# ================================================================================ c: re-optimised settings
c_rows = []
for ax, tag, lab in [(ax_c1, "theta50", "c"), (ax_c2, "manc50", None)]:
    G = GM[tag]
    ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
    for w, dx in [("w15_25", -0.17), ("w50_60", 0.17)]:
        R = float(G[(G.graph == "real") & (G.window == w)].S_all.iloc[0])
        ax.plot([-0.45 + (0.5 if w == "w50_60" else 0), 0.05 + (0.5 if w == "w50_60" else 0)], [R, R], color=C.REAL, lw=1.4)
        for xc, g, rule, col in [(1.4, "mn_r2_within", "own", GREYW), (2.6, "mn_r2_cross_hi", "own", PURPLE)]:
            q = G[(G.graph == g) & (G.rule == rule) & (G.window == w)].sort_values("gseed")
            xx = xc + dx + np.linspace(-0.05, 0.05, len(q))
            ax.errorbar(xx, q.S_all, yerr=q.S_all_sem, fmt="none", ecolor=col, elinewidth=0.5, capsize=0, zorder=2)
            ax.scatter(xx, q.S_all, s=11, marker="o" if w == "w15_25" else "D", facecolors=col if w == "w15_25" else "white",
                       edgecolors="white" if w == "w15_25" else col, linewidths=0.3 if w == "w15_25" else 0.6, zorder=3)
            ax.plot([xc + dx - 0.12, xc + dx + 0.12], [q.S_all.median()] * 2, color=C.INK, lw=1.0, zorder=4)
            for _, rr in q.iterrows():
                c_rows.append(dict(connectome=tag, window=w, graph=g, rule=rule, gseed=int(rr.gseed), S=rr.S_all, S_sem=rr.S_all_sem, R=R))
    ax.set_xticks([0.0, 1.4, 2.6]); ax.set_xticklabels(["Real", "W", "X_hi"])
    ax.set_xlim(-0.6, 3.1); ax.set_ylim(-0.12, 0.36); ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.tick_params(axis="x", length=0)
    C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold")
ax_c1.set_ylabel("S at each network's own setting")
ax_c2.set_yticklabels([])
C.panel_label(ax_c1, "c", x_pt=-30, y_pt=4)
fig.text(16.0 / W, (12.0 + 44.0 + 9.0) / H, "Re-optimised: every network at its own frozen setting (filled 15–25 s, open 50–60 s)", fontsize=6.3,
         ha="left", va="bottom")
C.write_source(pd.DataFrame(c_rows), "ED4c_reoptimised",
               "Experiment-2 arms W (mn_r2_within) and X_hi (mn_r2_cross_hi), each at its own frozen setting from the round-2 scan (rule own);\n"
               "S = mean ± s.e.m. over 20 runs. Real = real network at its frozen setting.")

# ================================================================================ d: activity maintenance
d_rows = []
ARM_D = [("mn_r2_within", "real_setting", "W"), ("mn_r2_cross_lo", "real_setting", "X_lo"), ("mn_r2_cross_hi", "real_setting", "X_hi"),
         ("mn_r2_cross_hi", "own", "X_hi (re-opt.)")]
for ax, tag, lab in [(ax_d1, "theta50", "d"), (ax_d2, "manc50", None)]:
    G = GM[tag]; F = PS[tag]
    ax.axvline(1.0, color=C.REAL, lw=0.8, zorder=1); ax.axvline(0.8, color=C.REAL, lw=0.5, ls=(0, (1, 1.6)), zorder=1)
    for i, (g, rule, lab_) in enumerate(ARM_D):
        for w, dy, fill in [("w15_25", -0.12, True), ("w50_60", 0.12, False)]:
            Rr = G[(G.graph == "real") & (G.window == w)].iloc[0]
            q = G[(G.graph == g) & (G.rule == rule) & (G.window == w)]
            osc = float(np.median(q.osc_frac)) / float(Rr.osc_frac)
            contrib = set(F[(F.graph == "real") & (F.window == w) & (F.fg >= 0.5)].pair)
            ret = [len(contrib & set(F[(F.graph == g) & (F.gseed == s_) & (F.rule == rule) & (F.window == w) & (F.fg >= 0.5)].pair)) / len(contrib)
                   for s_ in range(5)]
            pr = float(np.median(ret))
            if g != "mn_r2_within":   # cross-check with the pre-registered table
                c_ = CD[(CD.connectome == tag) & (CD.window == w) & (CD.arm == g) & (CD.rule == rule)].iloc[0]
                assert abs(c_.osc_retention - osc) < 1e-9 and abs(c_.pair_retention - pr) < 1e-9
            y = i * 1.0
            ax.scatter([osc], [y + dy], s=12, marker="o", facecolors=C.REAL if fill else "white", edgecolors=C.REAL, linewidths=0.6, zorder=3)
            ax.scatter([pr], [y + dy], s=12, marker="^", facecolors=C.SWITCH if fill else "white", edgecolors=C.SWITCH, linewidths=0.6, zorder=3)
            d_rows.append(dict(connectome=tag, arm=g, rule=rule, window=w, osc_coverage=osc, pair_retention=pr, n_contributing_pairs=len(contrib)))
    ax.set_yticks(range(len(ARM_D))); ax.set_ylim(len(ARM_D) - 0.4, -0.6)
    ax.set_yticklabels([a[2] for a in ARM_D] if tag == "theta50" else [], fontsize=5.4)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    xmax_ = max(1.6, float(np.ceil(max(max(r["osc_coverage"], r["pair_retention"]) for r in d_rows) * 10 + 1) / 10))
    ax.set_xlim(0.2, xmax_); ax.set_xticks([0.4, 0.8, 1.2, 1.6, 2.0][: 4 + (xmax_ > 2.0)])
    ax.axvspan(0.8, xmax_, color=C.PASS, lw=0, zorder=0)
    ax.set_xlabel("Relative to real")
    C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold")
C.panel_label(ax_d1, "d", x_pt=-40, y_pt=4)
fig.text(118.0 / W, (12.0 + 44.0 + 9.0) / H, "Activity kept? (real frozen setting unless noted)", fontsize=6.3, ha="left", va="bottom")
dh = [Line2D([], [], ls="none", marker="o", ms=3, mfc=C.REAL, mec=C.REAL, label="oscillating MNs"),
      Line2D([], [], ls="none", marker="^", ms=3, mfc=C.SWITCH, mec=C.SWITCH, label="contributing pairs kept"),
      Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=C.MUTED, mew=0.6, label="open: 50–60 s")]
fig.legend(handles=dh, loc="lower left", bbox_to_anchor=(100.0 / W, 0.3 / H), ncol=3, fontsize=5.3, frameon=False, handletextpad=0.3,
           columnspacing=1.0, borderaxespad=0)
C.write_source(pd.DataFrame(d_rows), "ED4d_activity_kept",
               "osc_coverage = median over 5 networks of the fraction of oscillating MNs / real (same window); pair_retention = median over networks of the\n"
               "fraction of the real network's contributing pairs (gated in >= 50% of runs) still gated in >= 50% of runs. Criterion (C6): both >= 0.8.\n"
               "Cross-pool arms equal analysis/exp2_conditions.csv (asserted); W computed with the same rule (descriptive).")

C.save(fig, "EDFig4_dose_matched_reassignment", dict(
    axes=[ax_a, ax_a2, ax_b, ax_c1, ax_c2, ax_d1, ax_d2], panel_ids=["a", "a2", "b", "c1", "c2", "d1", "d2"],
    row_groups=[["a", "a2", "b"], ["c1", "c2"], ["d1", "d2"], ["c1", "d1"]],
    exemptions=[{"panels": ["b"], "checks": ["row", "panel-width", "horizontal-gutter"], "reason": "b is a table-like condition grid, not a plot area"},
                {"panels": ["a2"], "checks": ["panel-width", "horizontal-gutter"],
                 "reason": "a2 is the narrow right-hand part of panel a and shares its y axis"}],
))
print("ED Fig. 4 written")
