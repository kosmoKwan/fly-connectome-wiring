"""Extended Data Fig. 2 | Further tests of the advantage: a second neuron model, more rewired networks and a partial glutamate sign flip.

a  Saturating ReLU f(x) = min(max(x, 0), 1) instead of the rectified tanh (round 3, A): S of the real network and of the 30 main
   rewired networks, each at its own frozen setting, 15-25 s.
b  Same model, rhythm: maximum rhythmicity over the eligible settings of the 132-setting scan.
c  25 networks for each of the three most competitive families (round 3, B; lineage block, degree, leg x lineage; s0-s4 from
   round 2, s5-s24 new) plus 5 for each other family: 90 rewired networks per connectome, 15-25 s.
d  25% of glutamatergic neurons made excitatory (round 3, F2; MaleCNS; 3 random draws): real network and 10 rewired networks.
All values from derived/r3_*.csv (check_round3.py: recomputed from the round-3 parquet files, checked against analysis/).
"""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.transforms import blended_transform_factory
import common as C

C.setup_style()
W, H = 183.0, 150.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
ax_a1 = A(42.0, 92.0, 30.0, 40.0); ax_a2 = A(77.0, 92.0, 30.0, 40.0)
ax_b1 = A(117.0, 92.0, 30.0, 40.0); ax_b2 = A(152.0, 92.0, 30.0, 40.0)
ax_c1 = A(42.0, 20.0, 44.0, 50.0); ax_c2 = A(94.0, 20.0, 44.0, 50.0)
ax_d = A(150.0, 20.0, 31.0, 50.0)
FAM_Y = {f: k for k, f in enumerate(C.MAIN)}
COLT = {"theta50": C.REAL, "manc50": C.REAL_MID}
AV = pd.read_csv(C.DERIVED / "r3_A_network_values.csv"); AL = pd.read_csv(C.DERIVED / "r3_A_levels.csv")
AR = pd.read_csv(C.DERIVED / "r3_A_scan_max_rhythm.csv")
BV = pd.read_csv(C.DERIVED / "r3_B_network_values.csv"); BL = pd.read_csv(C.DERIVED / "r3_B_levels.csv")
FV = pd.read_csv(C.DERIVED / "r3_F2_per_network.csv")


def family_axis(ax, labels):
    ax.set_ylim(len(C.MAIN) - 0.4, -0.75)
    ax.set_yticks(list(FAM_Y.values()))
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0, pad=7.5)
    if labels:
        ax.set_yticklabels([C.FAM_LABEL[f] for f in C.MAIN])
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        xdot = -1.4 / (ax.get_position().width * W)          # 1.4 mm left of the axis, whatever the panel width
        for f, y in FAM_Y.items():
            ax.scatter([xdot], [y], s=10, transform=tr, clip_on=False, color=C.FAM_COLOR[f], edgecolors="white", linewidths=0.3, zorder=5)
    else:
        ax.set_yticklabels([])


def strip(ax, vals, fams, real, col, jit_scale=0.3, ring=None):
    for f in C.MAIN:
        v = vals[fams == f]
        jit = np.linspace(-jit_scale, jit_scale, len(v)) if len(v) > 1 else np.zeros(1)
        ax.scatter(v, FAM_Y[f] + jit * (0.5 if len(v) <= 5 else 1.0), s=10 if len(v) <= 5 else 7, color=C.FAM_COLOR[f], edgecolors="white",
                   linewidths=0.25, zorder=3)
        ax.plot([np.median(v)] * 2, [FAM_Y[f] - 0.36, FAM_Y[f] + 0.36], color=C.INK, lw=0.9, zorder=4, solid_capstyle="butt")
    ax.axvline(real, color=col, lw=1.4, zorder=2)


# ================================================================================ a: saturating ReLU, coordination
a_rows = []
for ax, tag, lab in [(ax_a1, "theta50", True), (ax_a2, "manc50", False)]:
    q = AV[(AV.connectome == tag) & (AV.window == "w15_25")]; real = float(q[q.graph == "real"].S.iloc[0]); nul = q[q.graph != "real"]
    lv = AL[(AL.connectome == tag) & (AL.window == "w15_25")].iloc[0]
    assert abs(lv.real - real) < 1e-12
    strip(ax, nul.S.to_numpy(), nul.graph.to_numpy(), real, COLT[tag])
    ax.text(real - 0.012, -0.55, C.fmt(real, 2), fontsize=5.6, ha="right", va="center", color=COLT[tag], fontweight="bold")
    family_axis(ax, lab); ax.axvline(0, color=C.GRID, lw=0.6, zorder=0)
    ax.set_xlim(-0.15, 0.4); ax.set_xticks([0, 0.2, 0.4]); ax.set_xlabel("S")
    C.panel_title(ax, f"{C.TAGS[tag]} · {lv.level.lower()}", y_pt=4, loc="center", fontweight="bold")
    for r in nul.itertuples():
        a_rows.append(dict(connectome=tag, family=r.graph, gseed=r.gseed, S=r.S, real=real, level=lv.level, strongest_rewired=lv.strongest_null))
C.panel_label(ax_a1, "a", x_pt=-100, y_pt=4)
fig.text(12.0 / W, 141.0 / H, "Saturating ReLU neuron: coordination, 15–25 s", fontsize=6.6, fontweight="bold", ha="left", va="baseline")
C.write_source(pd.DataFrame(a_rows), "ED2a_relu_coordination",
               "Round 3, A: f(x) = min(max(x, 0), 1). Network value = mean S over the 20 stochastic runs at the network's own frozen setting,\n"
               "15-25 s; level by the pre-registered rule (Sec. 3.23) over the six main families (checked against analysis/A_relu_sat).")

# ================================================================================ b: saturating ReLU, rhythm (scan)
b_rows = []
for ax, tag in [(ax_b1, "theta50"), (ax_b2, "manc50")]:
    q = AR[AR.connectome == tag]; real = float(q.real_max_rhythm.iloc[0]); nul = q[q.graph != "real"]
    strip(ax, nul.max_rhythm.to_numpy(), nul.graph.to_numpy(), real, COLT[tag])
    n_more = int((nul.max_rhythm > real).sum())
    ax.text(real - 0.02, -0.55, C.fmt(real, 2), fontsize=5.6, ha="right", va="center", color=COLT[tag], fontweight="bold")
    family_axis(ax, False)
    ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("Max. rhythmicity")
    C.panel_title(ax, f"{C.TAGS[tag]} · {n_more}/{len(nul)} above real", y_pt=4, loc="center", fontweight="bold")
    for r in nul.itertuples():
        b_rows.append(dict(connectome=tag, family=r.graph, gseed=r.gseed, max_rhythmicity=r.max_rhythm, real_max_rhythmicity=real))
C.panel_label(ax_b1, "b", x_pt=-14, y_pt=4)
fig.text(117.0 / W, 141.0 / H, "Same model: rhythm (scan maximum)", fontsize=6.6, fontweight="bold", ha="left", va="baseline")
C.write_source(pd.DataFrame(b_rows), "ED2b_relu_rhythm",
               "Round 3, A: maximum rhythmicity over the eligible settings of each network's 132-setting scan (eligible: > 20% of MNs oscillating,\n"
               "dominant frequency 1-50 Hz); 'above real' = strictly greater (paper convention). Checked against A_rhythm_by_family.csv.")

# ================================================================================ c: 25 networks per competitive family
c_rows = []
for ax, tag, lab in [(ax_c1, "theta50", True), (ax_c2, "manc50", False)]:
    q = BV[(BV.connectome == tag) & (BV.window == "w15_25")]; real = float(q[q.graph == "real"].S.iloc[0]); nul = q[q.graph != "real"].sort_values(["graph", "gseed"])
    lv = BL[(BL.connectome == tag) & (BL.window == "w15_25")].iloc[0]; emp = json.loads(lv.empirical_p); npf = json.loads(lv.n_per_family)
    assert len(nul) == 90
    strip(ax, nul.S.to_numpy(), nul.graph.to_numpy(), real, COLT[tag], jit_scale=0.34)
    hi = nul[nul.S >= real]
    for r in hi.itertuples():                          # networks that reach the real network: ring (same jitter as in strip())
        v = nul[nul.graph == r.graph].S.to_numpy(); k = int(np.where(np.isclose(v, r.S))[0][0])
        jit = np.linspace(-0.34, 0.34, len(v))[k] * (0.5 if len(v) <= 5 else 1.0)
        ax.scatter([r.S], [FAM_Y[r.graph] + jit], s=30, facecolors="none", edgecolors=C.RING, linewidths=0.9, zorder=5)
    big = [f for f in C.MAIN if npf[f] > 5]
    SHORT = {"degree": "degree", "hl": "lineage", "leg_hl": "leg × lin."}
    ax.text(0.5, -0.27, "empirical p, 25 networks:\n" + " · ".join(f"{SHORT[f]} {emp[f]:.3f}" for f in big), transform=ax.transAxes,
            fontsize=5.0, ha="center", va="top", color=C.MUTED, linespacing=1.25)
    ax.text(real - 0.012, -0.55, C.fmt(real, 2), fontsize=5.6, ha="right", va="center", color=COLT[tag], fontweight="bold")
    family_axis(ax, lab); ax.axvline(0, color=C.GRID, lw=0.6, zorder=0)
    ax.set_xlim(-0.2, 0.4); ax.set_xticks([-0.2, 0, 0.2, 0.4]); ax.set_xlabel("S")
    C.panel_title(ax, f"{C.TAGS[tag]} · {lv.level.lower()} · {int(lv.n_at_or_above_real)} of {len(nul)} at or above real", y_pt=4, loc="center",
                  fontweight="bold")
    for r in nul.itertuples():
        c_rows.append(dict(connectome=tag, family=r.graph, gseed=r.gseed, S=r.S, real=real, at_or_above_real=bool(r.S >= real),
                           empirical_p_family=emp[r.graph], level=lv.level))
C.panel_label(ax_c1, "c", x_pt=-100, y_pt=24)
fig.text(12.0 / W, 79.0 / H, "25 networks per competitive family, 15–25 s", fontsize=6.6, fontweight="bold",
         ha="left", va="baseline")
fig.text(12.0 / W, 75.6 / H, "ring: rewired network at or above the real one; p = (1 + networks ≥ real) / (n + 1)", fontsize=5.2, color=C.MUTED,
         ha="left", va="baseline")
C.write_source(pd.DataFrame(c_rows), "ED2c_n25_distribution",
               "Round 3, B: lineage block, degree and leg x lineage with 25 networks each (s0-s4 round 2, s5-s24 new), other families 5 each;\n"
               "network value as in a (rectified tanh model). Level by the pre-registered rule over all 90; empirical p = (1 + n_at_or_above)/(n + 1).\n"
               "Checked against analysis/B_n25 (network values bit-identical).")

# ================================================================================ d: 25% of glutamatergic neurons excitatory (MaleCNS)
ax = ax_d
d_rows = []
q = FV[FV.window == "w15_25"]
n_pass = 0
for k, d_ in enumerate([21, 22, 23]):
    z = q[q.draw == d_]; real = float(z[z.graph == "real"].S.iloc[0]); nul = z[z.graph != "real"].sort_values(["graph", "gseed"])
    jit = np.linspace(-0.22, 0.22, len(nul))
    ax.scatter(k + jit, nul.S, s=10, c=[C.FAM_COLOR[g] for g in nul.graph], edgecolors="white", linewidths=0.25, zorder=3)
    ax.plot([k - 0.3, k + 0.3], [real, real], color=C.REAL, lw=1.6, zorder=4, solid_capstyle="butt")
    ok = bool((real > nul.S).all()); n_pass += ok
    ax.text(k, real + 0.015, C.fmt(real, 2), fontsize=5.4, ha="center", va="bottom", color=C.REAL, fontweight="bold")
    for r in z.itertuples():
        d_rows.append(dict(draw=d_, graph=r.graph, gseed=r.gseed, S=r.S, real_above_all_10=ok))
ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
ax.set_xlim(-0.6, 2.6); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["draw 1", "draw 2", "draw 3"])
ax.set_ylim(-0.1, 0.40); ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3]); ax.set_ylabel("S (15–25 s)")
fh = [Line2D([], [], color=C.REAL, lw=1.6, label="real"),
      Line2D([], [], ls="none", marker="o", ms=3, mfc=C.FAM_COLOR["hl"], mec="white", mew=0.3, label="lineage block"),
      Line2D([], [], ls="none", marker="o", ms=3, mfc=C.FAM_COLOR["degree"], mec="white", mew=0.3, label="degree")]
ax.legend(handles=fh, loc="upper left", fontsize=5.2, handlelength=1.2, frameon=False, borderaxespad=0.2, labelspacing=0.3)
C.panel_title(ax_d, f"real above all 10 in {n_pass} of 3 draws", y_pt=4, loc="center", fontweight="bold")
C.panel_label(ax_d, "d", x_pt=-30, y_pt=24)
fig.text(144.0 / W, 79.0 / H, "Glutamate partly excitatory", fontsize=6.6, fontweight="bold", ha="left", va="baseline")
fig.text(144.0 / W, 75.6 / H, "MaleCNS, 25% of glutamatergic → excitatory", fontsize=5.2, color=C.MUTED, ha="left", va="baseline")
C.write_source(pd.DataFrame(d_rows), "ED2d_glutamate_excitatory",
               "Round 3, F2 (MaleCNS): each glutamatergic neuron (1,883 neurons whose sign differs between the glutamate-excitatory and the default\n"
               "node tables) made excitatory with probability 0.25; draws with seeds 21, 22, 23. Real network and 10 rewired networks (lineage block\n"
               "and degree, s0-s4), each at its own frozen setting, 15-25 s. Checked against F_signerr_per_variant.csv.")

C.save(fig, "EDFig2_robustness", dict(
    axes=[ax_a1, ax_a2, ax_b1, ax_b2, ax_c1, ax_c2, ax_d], panel_ids=["a1", "a2", "b1", "b2", "c1", "c2", "d"],
    row_groups=[["a1", "a2", "b1", "b2"], ["c1", "c2", "d"]],
    exemptions=[{"panels": ["a1", "a2", "b1", "b2"], "checks": ["horizontal-gutter"], "reason": "a and b are two panels; the gap between them is wider"},
                {"panels": ["d"], "checks": ["panel-width", "horizontal-gutter"], "reason": "d is a separate, narrower panel with its own axis labels"}],
))
print("ED Fig. 2 written")
