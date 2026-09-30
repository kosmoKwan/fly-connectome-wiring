"""Fig. 5 | The real wiring wins on coordination, not on recruitment.

a  Common and differential activity of an antagonistic pair (schematic): y± = (e ± f)/√2; the differential share
   D = Var(y-) / [Var(y+) + Var(y-)] exceeds 0.5 exactly when the pools are antiphasic.
b  Recruitment versus coordination (round 3, E1, descriptive): for every set of networks, the fraction of pair-runs past the
   amplitude gate against the antiphase of the gated pairs; one point per rewired family, the real network, the within-pool
   control W and the cross-pool arm X_hi (real frozen setting).
c  Differential share D of the active pairs (pre-registered P2): real network, W and X_hi.
d  Common component kept? Var(y+) of X_hi relative to W on W's active pairs (pre-registered P1, criterion 0.8).
e  Structure predicts the differential share (P3): structural opposition γ of each active pair against its D.
All values from derived/r3_E*.csv, recomputed by check_round3.py from the round-2 pair-level data and checked against
round3_20260929/analysis/D1_E_cpu/.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import common as C

C.setup_style()
W, H = 183.0, 134.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
ax_a = A(1.0, 70.0, 74.0, 56.0)
ax_b1 = A(90.0, 78.0, 40.0, 40.0)
ax_b2 = A(141.0, 78.0, 40.0, 40.0)
ax_c = A(16.0, 14.0, 50.0, 40.0)
ax_d = A(80.0, 14.0, 22.0, 40.0)
ax_e1 = A(118.0, 14.0, 27.0, 40.0)
ax_e2 = A(154.0, 14.0, 27.0, 40.0)
ARM_COL = {"real": C.REAL, "W": C.WITHIN, "X_hi": C.CROSS}
ARM_LAB = {"real": "Real", "W": "W", "X_hi": "X_hi"}
COMMON, DIFF = C.SWITCH, C.REAL                 # schematic colours: common (together) and differential (push-pull) component
ES = pd.read_csv(C.DERIVED / "r3_E_summary.csv")
POOL = ES[ES.arm == "pooled"].set_index("connectome")

# ================================================================================ a: schematic
ax = ax_a
ax.set_xlim(0, 74); ax.set_ylim(0, 56); ax.axis("off")
ax.text(4.2, 55.5, "Two components of a pair's activity", fontsize=6.6, fontweight="bold", ha="left", va="top")
t = np.linspace(0, 1, 200)
for k, (x0, title, phase, dlab) in enumerate([(3.0, "Push–pull (antiphase)", np.pi, "D > 0.5"), (40.0, "Together (in phase)", 0.0, "D < 0.5")]):
    e = np.sin(2 * np.pi * 2 * t); f = np.sin(2 * np.pi * 2 * t + phase)
    yp, ym = (e + f) / np.sqrt(2), (e - f) / np.sqrt(2)
    ax.text(x0 + 15.0, 47.5, title, fontsize=5.8, ha="center", va="center", color=C.INK)
    for (yc, sig, col, lw) in [(40.0, e, C.EXT, 0.9), (40.0, f, C.FLX, 0.9), (27.5, yp, COMMON, 1.0), (15.0, ym, DIFF, 1.0)]:
        ax.plot(x0 + 30.0 * t, yc + 2.6 * sig, color=col, lw=lw)
    ax.text(x0 + 15.0, 6.5, dlab, fontsize=6.0, ha="center", va="center", color=DIFF if phase else COMMON, fontweight="bold")
for (yc, lab, col) in [(40.0, "ext. e, flx. f", C.INK), (27.5, "common y+", COMMON), (15.0, "differential y−", DIFF)]:
    ax.text(36.6, yc + 3.9, lab, fontsize=5.2, ha="center", va="center", color=col)
ax.text(1.0, 1.2, "y± = (e ± f)/√2 ;  D = Var(y−) / [Var(y+) + Var(y−)]", fontsize=5.4, ha="left", va="bottom", color=C.INK)
ax.text(73.5, 1.2, "schematic", fontsize=5.2, ha="right", va="bottom", color=C.MUTED, style="italic")
C.panel_label(ax_a, "a", x_pt=0, y_pt=-8)

# ================================================================================ b: recruitment vs antiphase (E1)
b_rows = []
for ax, tag in [(ax_b1, "theta50"), (ax_b2, "manc50")]:
    S1 = pd.read_csv(C.DERIVED / f"r3_E1_summary_{tag}.csv").set_index("set")
    ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
    for f in C.MAIN:
        r = S1.loc[f]
        ax.scatter([r.mean_P_gate], [r.mean_antiphase_given_gate], s=18, color=C.FAM_COLOR[f], edgecolors="white", linewidths=0.3, zorder=3)
        b_rows.append(dict(connectome=tag, set=f, P_gate=r.mean_P_gate, antiphase_given_gate=r.mean_antiphase_given_gate, S=r.S))
    for st, key, s_, dx, dy, ha in [("real", "real", 46, 0.03, 0.06, "left"), ("mn_r2_within", "W", 30, -0.03, 0.06, "right"),
                                   ("mn_r2_cross_hi", "X_hi", 30, -0.035, -0.02, "right")]:
        r = S1.loc[st]
        ax.scatter([r.mean_P_gate], [r.mean_antiphase_given_gate], s=s_, color=ARM_COL[key], edgecolors="white", linewidths=0.5, zorder=5)
        ax.text(r.mean_P_gate + dx, r.mean_antiphase_given_gate + dy, ARM_LAB[key], fontsize=5.8, ha=ha, va="center", color=ARM_COL[key],
                fontweight="bold")
        b_rows.append(dict(connectome=tag, set=key, P_gate=r.mean_P_gate, antiphase_given_gate=r.mean_antiphase_given_gate, S=r.S))
    rr, xx = S1.loc["real"], S1.loc["mn_r2_cross_hi"]
    ax.annotate("", (xx.mean_P_gate, xx.mean_antiphase_given_gate + 0.035), xytext=(rr.mean_P_gate, rr.mean_antiphase_given_gate - 0.045),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C.CROSS, mutation_scale=7, shrinkA=1, shrinkB=1), zorder=4)
    fam = S1.loc[C.MAIN]
    ax.text(0.98, 0.97, f"rewired families:\nrecruited {fam.mean_P_gate.min():.2f}–{fam.mean_P_gate.max():.2f}\nantiphase ≤ "
            f"{fam.mean_antiphase_given_gate.max():.2f}", transform=ax.transAxes, fontsize=5.0, ha="right", va="top", color=C.MUTED,
            linespacing=1.2)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.3, 0.85)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Pair-runs recruited\n(past the amplitude gate)", linespacing=1.15)
    C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold", loc="center")
ax_b1.set_ylabel("Antiphase of recruited pairs")
C.panel_label(ax_b1, "b", x_pt=-30, y_pt=16)
fig.text(90.0 / W, 126.8 / H, "Not more recruitment, but recruited pairs in antiphase", fontsize=6.6, fontweight="bold", ha="left", va="baseline")
fig.text(90.0 / W, 123.4 / H, "One point per set (median of its networks); arrow: cross-pool reassignment X_hi. Families coloured as in Fig. 2.",
         fontsize=5.2, color=C.MUTED, ha="left", va="baseline")
C.write_source(pd.DataFrame(b_rows), "Fig5b_recruitment_vs_antiphase",
               "Round 3, E1 (descriptive): per network and pair, P_gate = fraction of the 20 runs in which both pools pass the amplitude gate,\n"
               "antiphase_given_gate = mean of -corr over the gated runs; per network the means over the 20 pairs; per set the median over its\n"
               "networks (real: one network). Sets: six rewired families (own frozen settings), real, W and X_hi (real frozen setting), 15-25 s.")

# ================================================================================ c: differential share D (P2)
ax = ax_c
c_rows = []
ax.axhline(0.5, color=C.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
for ci, tag in enumerate(C.TAGS):
    Dp = pd.read_csv(C.DERIVED / f"r3_E_D_pairs_{tag}.csv")
    for k, arm in enumerate(["real", "W", "X_hi"]):
        q = Dp[Dp.arm == arm].sort_values("pair").reset_index(drop=True)
        x = ci * 3.8 + k
        jit = np.linspace(-0.22, 0.22, len(q)) if len(q) > 1 else np.zeros(1)
        ax.scatter(x + jit, q.D, s=9, color=ARM_COL[arm], edgecolors="white", linewidths=0.3, zorder=3, alpha=0.9)
        m = float(POOL.loc[tag, f"D_{arm}"])
        assert abs(m - q.D.median()) < 1e-12
        ax.plot([x - 0.32, x + 0.32], [m, m], color=C.INK, lw=1.2, zorder=4, solid_capstyle="butt")
        ax.text(x, 1.0, C.fmt(m, 2), fontsize=5.6, ha="center", va="bottom", color=ARM_COL[arm], fontweight="bold")
        for r in q.itertuples():
            c_rows.append(dict(connectome=tag, arm=arm, pair=r.pair, D_network_median=r.D, arm_median=m, n_active_pairs=len(q)))
    ax.text(ci * 3.8 + 1, -0.27, C.TAGS[tag], fontsize=6.0, ha="center", va="top", fontweight="bold",
            transform=blended_transform_factory(ax.transData, ax.transAxes))
ax.set_xlim(-0.6, 7.2); ax.set_ylim(0, 1.08)
ax.set_xticks([0, 1, 2, 3.8, 4.8, 5.8]); ax.set_xticklabels(["Real", "W", "X_hi"] * 2)
for tl, arm in zip(ax.get_xticklabels(), ["real", "W", "X_hi"] * 2):
    tl.set_color(ARM_COL[arm]); tl.set_fontweight("bold")
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Differential share D")
ax.text(-0.52, 0.52, "antiphase ↑", fontsize=5.2, ha="left", va="bottom", color=C.MUTED)
ax.text(-0.52, 0.48, "in phase ↓", fontsize=5.2, ha="left", va="top", color=C.MUTED)
C.panel_title(ax_c, "Reassignment lowers the push–pull share below 0.5", x_pt=-26, y_pt=12, fontsize=6.6, fontweight="bold")
C.panel_title(ax_c, "points: active pairs (median over networks); bars: medians", x_pt=-26, y_pt=4.5, fontsize=5.2, color=C.MUTED)
C.panel_label(ax_c, "c", x_pt=-34, y_pt=12)
C.write_source(pd.DataFrame(c_rows), "Fig5c_differential_share",
               "Pre-registered P2 (round 3): D = Var(y-) / [Var(y+) + Var(y-)] per run and pair, Var(y±) = (se^2 + sf^2 ± 2 rho se sf)/2 from the\n"
               "round-2 pair-level data (15-25 s, stochastic runs, real frozen setting for W and X_hi); per network and pair the mean over gated\n"
               "runs; active pair = gated in >= 10 of 20 runs; per pair the median over the arm's networks; arm value = median over pairs.")

# ================================================================================ d: common component kept? (P1)
ax = ax_d
d_rows = []
ax.axhline(1.0, color=C.GRID, lw=0.7, zorder=0)
ax.axhline(0.8, color=C.INK, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
for ci, tag in enumerate(C.TAGS):
    q = pd.read_csv(C.DERIVED / f"r3_E_P1_pairs_{tag}.csv").sort_values("pair").reset_index(drop=True)
    jit = np.linspace(-0.2, 0.2, len(q))
    ax.scatter(ci + jit, q.VpX_over_VpW, s=9, color=C.SWITCH, edgecolors="white", linewidths=0.3, zorder=3)
    m = float(POOL.loc[tag, "P1"]); assert abs(m - q.VpX_over_VpW.median()) < 1e-12
    ax.plot([ci - 0.3, ci + 0.3], [m, m], color=C.INK, lw=1.2, zorder=4, solid_capstyle="butt")
    ax.text(ci, 3.3, f"{C.fmt(m, 2)}\n{'met' if m >= 0.8 else 'not met'}", fontsize=5.4, ha="center", va="top", color=C.INK, linespacing=1.15)
    for r in q.itertuples():
        d_rows.append(dict(connectome=tag, pair=r.pair, Vplus_X_over_W=r.VpX_over_VpW, median=m, criterion=0.8, met=bool(m >= 0.8)))
ax.set_yscale("log"); ax.set_ylim(0.09, 3.6); ax.minorticks_off()
ax.set_yticks([0.1, 0.25, 0.5, 1, 2]); ax.set_yticklabels(["0.1", "0.25", "0.5", "1", "2"])
ax.set_xlim(-0.6, 1.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["MaleCNS", "MANC"], fontsize=5.4)
ax.set_ylabel("Common component, X_hi / W")
C.panel_title(ax_d, "Common kept?", x_pt=-20, y_pt=12, fontsize=6.6, fontweight="bold")
C.panel_title(ax_d, "dotted: criterion 0.8", x_pt=-20, y_pt=4.5, fontsize=5.2, color=C.MUTED)
C.panel_label(ax_d, "d", x_pt=-28, y_pt=12)
C.write_source(pd.DataFrame(d_rows), "Fig5d_common_component",
               "Pre-registered P1 (round 3): on W's active pairs (active in >= 3 of 5 W networks), the ratio of the median over networks of Var(y+)\n"
               "(mean over the 20 runs) for X_hi to that for W; criterion: median over pairs >= 0.8. Log scale.")

# ================================================================================ e: structure predicts D (P3)
e_rows = []
for ax, tag in [(ax_e1, "theta50"), (ax_e2, "manc50")]:
    J = pd.read_csv(C.DERIVED / f"r3_E_gamma_D_{tag}.csv")
    ax.axhline(0.5, color=C.INK, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for arm in ["W", "X_hi", "real"]:
        q = J[J.arm == arm]
        ax.scatter(q.gamma, q.D_gated_mean, s=9 if arm != "real" else 16, color=ARM_COL[arm], edgecolors="white", linewidths=0.3,
                   zorder=4 if arm == "real" else 3, alpha=0.9)
        for r in q.itertuples():
            e_rows.append(dict(connectome=tag, arm=arm, gseed=r.gseed, pair=r.pair, gamma=r.gamma, D=r.D_gated_mean))
    sw = ES[(ES.connectome == tag) & (ES.arm != "pooled")].set_index("arm")
    ax.text(-0.76, 1.235, f"pooled ρ = {C.fmt(POOL.loc[tag, 'spearman_within'], 2)} (n = {int(POOL.loc[tag, 'n'])})", fontsize=5.6, ha="left",
            va="top", color=C.INK)
    ax.text(-0.76, 1.11, "within", fontsize=5.2, ha="left", va="top", color=C.MUTED)
    for xx_, a_ in zip([-0.36, -0.06, 0.24], ["real", "W", "X_hi"]):            # within-group rho, coloured by group (real, W, X_hi)
        ax.text(xx_, 1.11, C.fmt(sw.loc[a_, "spearman_within"], 2), fontsize=5.4, ha="left", va="top", color=ARM_COL[a_], fontweight="bold")
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(0, 1.26)
    ax.set_xticks([-0.5, 0, 0.5]); ax.set_xticklabels(["−0.5", "0", "0.5"]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Structural opposition γ")
    C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold", loc="center")
ax_e1.set_ylabel("Differential share D")
C.panel_label(ax_e1, "e", x_pt=-26, y_pt=12)
fig.text(118.0 / W, (54.0 + 12 * 25.4 / 72) / H, "Structure predicts the push–pull share", fontsize=6.6, fontweight="bold", ha="left", va="baseline")
C.write_source(pd.DataFrame(e_rows), "Fig5e_gamma_vs_D",
               "Pre-registered P3 (round 3): gamma = uncentred -cos(u_E, u_F) of the pair's influence vectors (as ri_semantics_check.py) in each of\n"
               "the 11 graphs (real, 5 W, 5 X_hi); D of the pair in that graph (active pairs only). Spearman rho pooled over the 11 graphs\n"
               "(pre-registered, criterion >= 0.5) and within each arm (post hoc, descriptive; weaker, part of the pooled rho comes from the\n"
               "difference between arms). Recomputed values equal D1_E_verdicts.json and E_P3_within_arm.csv exactly (CSV read at full precision).")
C.save(fig, "Fig5_differential_activity", dict(
    axes=[ax_b1, ax_b2, ax_c, ax_d, ax_e1, ax_e2], panel_ids=["b1", "b2", "c", "d", "e1", "e2"],
    row_groups=[["b1", "b2"], ["c", "d", "e1", "e2"]],
    exemptions=[{"panels": ["c", "d", "e1", "e2"], "checks": ["panel-width", "horizontal-gutter"],
                 "reason": "c groups six arms, d two ratios and e1/e2 are square-ish scatter plots: widths follow content"}],
))
print("Fig. 5 written")
