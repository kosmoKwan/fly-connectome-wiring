"""Fig. 3 | Reciprocal innervation of antagonistic pools, through an interneuron, is specific to the real wiring.

a  How the index is computed, on real data: every neuron's signed one- plus two-step influence on the extensor pool
   and on the flexor pool of one antagonistic pair (MaleCNS T2L thorax-coxa, one of the two pairs that define the
   median), real network and leg-block rewired network 0. Index = minus the correlation across neurons.
b  Network-level index (median over the 20 pairs) for every rewired network and the real networks.
c  The index reflects opposite-signed influence from the same neurons (post hoc): share of the shared influence that is
   opposite in sign (f_opp), real networks and the 30 main rewired networks.
d  Opposite signs need an intervening neuron (Dale's law): the two routes (schematic) and the share of the
   opposite-signed influence carried by each in the real networks (median over the 20 pairs).
Pair-by-pair versions of b and c are in Extended Data Fig. 3.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import schem as SC
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import common as C

C.setup_style()
W, H = 183.0, 120.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)

R1_B, R1_H = 70.0, 36.0
ax_a1 = A(15.0, R1_B, 36.0, R1_H)
ax_a2 = A(60.0, R1_B, 36.0, R1_H)
ax_b1 = A(123.0, R1_B, 27.0, R1_H)
ax_b2 = A(153.0, R1_B, 27.0, R1_H)
ax_c = A(22.0, 12.0, 56.0, 28.0)
ax_d = A(88.0, 2.0, 62.0, 47.0)
ax_d2 = A(153.0, 12.0, 25.0, 28.0)

# ================================================================================ a: influences for one pair
PAIR = "T2L_ThC"
rp_m = C.recip_pairs("theta50")
real_pairs = rp_m[rp_m.graph == "real"].set_index("pair").recip_innervation
med = float(np.median(real_pairs))
a_rows = []
for ax, g, col, title in [(ax_a1, "real", C.REAL, "Real network"), (ax_a2, "leg_s0", C.FAM_COLOR["leg"], "Rewired, leg block")]:
    d = pd.read_csv(C.DERIVED / f"struct_example_theta50_{g}_{PAIR}.csv")
    r = float(np.corrcoef(d.infl_ext, d.infl_flx)[0, 1])
    arch = float(rp_m[(rp_m.graph == g) & (rp_m.pair == PAIR)].recip_innervation.iloc[0])
    assert abs(-r - arch) < 1e-9, (g, r, arch)
    LT = 0.01
    for (xs_, ys_) in [((0, 2), (-2, 0)), ((-2, 0), (0, 2))]:
        ax.fill_between([xs_[0], xs_[1]], ys_[0], ys_[1], color=C.PASS, lw=0, zorder=0)
    ax.scatter(d.infl_ext, d.infl_flx, s=1.4, c=col, alpha=0.32 if g == "real" else 0.4, lw=0, rasterized=True, zorder=2)
    ax.axhline(0, color=C.LIGHT, lw=0.4, zorder=1); ax.axvline(0, color=C.LIGHT, lw=0.4, zorder=1)
    ax.set_xscale("symlog", linthresh=LT, linscale=0.6); ax.set_yscale("symlog", linthresh=LT, linscale=0.6)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ticks = [-1, -0.1, 0, 0.1, 1]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(["−1", "−0.1", "0", "0.1", "1"]); ax.set_yticklabels(["−1", "−0.1", "0", "0.1", "1"])
    ax.minorticks_off()
    ax.set_xlabel("Influence on extensor pool")
    for (tx, ty, ha, va, txt, tc) in [(0.03, 0.97, "left", "top", "opposite\nsign", C.REAL), (0.97, 0.03, "right", "bottom", "opposite\nsign", C.REAL),
                                      (0.97, 0.97, "right", "top", "same\nsign", C.MUTED), (0.03, 0.03, "left", "bottom", "same\nsign", C.MUTED)]:
        ax.text(tx, ty, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=5.3, color=tc, linespacing=1.3, zorder=6)
    C.panel_title(ax, f"{title} · index {C.fmt(-r, 2)}", y_pt=4)
    for _, rr in d.iterrows():
        pass
    a_rows.append(d.assign(network=g, pearson_r=r, index=-r, archived_index=arch))
ax_a1.set_ylabel("Influence on flexor pool")
C.panel_label(ax_a1, "a", x_pt=-30, y_pt=4)
fig.text(15.0 / W, (R1_B - 13.2) / H,
         f"MaleCNS pair {PAIR.replace('_', ' ')}, one of the two pairs defining the median index ({C.fmt(med, 2)}) · "
         f"one point per neuron · symmetric-log axes, linear within ±{LT}",
         fontsize=5.3, color=C.MUTED, ha="left", va="bottom", style="italic")
C.write_source(pd.concat(a_rows), "Fig3a_influences_T2L_ThC",
               "Per-neuron signed one-step + two-step influence on the extensor and flexor pools of MaleCNS T2L_ThC, each step normalized\n"
               "by its maximum absolute value over neurons (archived definition). index = -Pearson r over neurons with non-zero influence;\n"
               "recomputed values equal the archived per-pair values (archived_index). Networks: real; leg-block rewired network, seed 0.")

# ================================================================================ b: network-level index
FAM_Y = {"er": 0, "role": 1, "degree": 2, "leg": 3, "hl": 4, "leg_hl": 5, "degree_recip": 6.6, "leg_hl_recip": 7.6}
JIT = np.array([-0.22, -0.11, 0.0, 0.11, 0.22])
b_rows = []
for ax, tag, lab in [(ax_b1, "theta50", "b"), (ax_b2, "manc50", None)]:
    rg = C.recip_graph(tag)
    real = float(rg[rg.graph == "real"].recip_innervation.iloc[0])
    nul = rg[(rg.kind == "null") & rg.level.isin(C.MAIN + C.EXTRA)].copy()
    nul["gseed"] = nul.graph.str.extract(r"_s(\d+)$").astype(int)[0]
    ax.axvline(0, color=C.GRID, lw=0.6, zorder=0)
    ax.axvline(real, color=C.REAL, lw=0.9, zorder=1)
    ax.text(real - 0.02, -1.2, C.fmt(real, 2), fontsize=5.6, ha="right", va="center", color=C.REAL)
    for fam in C.MAIN + C.EXTRA:
        q = nul[nul.level == fam].sort_values("gseed")
        if not len(q):
            continue
        hollow = fam in C.EXTRA
        ax.scatter(q.recip_innervation, FAM_Y[fam] + JIT[q.gseed.to_numpy()], s=11,
                   facecolors="white" if hollow else C.FAM_COLOR[fam], edgecolors=C.FAM_COLOR[fam] if hollow else "white",
                   linewidths=0.6 if hollow else 0.3, zorder=3)
        m = float(np.median(q.recip_innervation))
        ax.plot([m, m], [FAM_Y[fam] - 0.36, FAM_Y[fam] + 0.36], color=C.INK, lw=1.0, zorder=4, solid_capstyle="butt")
        for _, rr in q.iterrows():
            b_rows.append(dict(connectome=tag, family=fam, gseed=int(rr.gseed), index=rr.recip_innervation, family_median=m, real=real))
    mx = float(nul[nul.level.isin(C.MAIN)].recip_innervation.max())
    b_rows.append(dict(connectome=tag, family="__max_main__", gseed=-1, index=mx, family_median=np.nan, real=real))
    if tag == "manc50":
        ax.text(0.19, 7.1, "none for\nMANC", fontsize=5.3, color=C.PENDING, ha="center", va="center", style="italic", linespacing=1.3)
    ax.set_ylim(8.2, -1.75); ax.set_xlim(-0.42, 0.58)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(["−0.4", "", "0", "", "0.4"])
    ax.set_yticks(list(FAM_Y.values())); ax.tick_params(axis="y", length=0, pad=8.5)
    ax.spines["left"].set_visible(False)
    if tag == "theta50":
        ax.set_yticklabels([C.FAM_LABEL[f] for f in FAM_Y])
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        for fam, yv in FAM_Y.items():
            hollow = fam in C.EXTRA
            ax.scatter([-0.055], [yv], s=10, transform=tr, clip_on=False, facecolors="white" if hollow else C.FAM_COLOR[fam],
                       edgecolors=C.FAM_COLOR[fam], linewidths=0.6 if hollow else 0.3, zorder=5)
        ax.text(-0.66, 5.95, "Secondary", transform=blended_transform_factory(ax.transAxes, ax.transData), fontsize=5.2,
                color=C.MUTED, ha="left", va="center")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Reciprocal-innervation\nindex")
    C.panel_title(ax, C.TAGS[tag], loc="center", y_pt=4, fontweight="bold")
C.panel_label(ax_b1, "b", x_pt=-72, y_pt=4)
C.write_source(pd.DataFrame(b_rows), "Fig3b_network_index",
               "Network-level reciprocal-innervation index = median over the 20 antagonistic pairs (archived structural diagnostic).\n"
               "Rewired families as in Fig. 2; secondary families (MaleCNS only) hollow. __max_main__ = maximum over the 30 main networks.")

# ================================================================================ c: opposite-signed shared influence (post hoc)
RG = pd.read_csv(C.RISEM / "ri_semantics_graphs.csv")
SUM = pd.read_csv(C.DERIVED / "r3_RI_sign_summary.csv").set_index("connectome")       # checked in check_round3.py
ROW = {"theta50": 1.0, "manc50": 0.0}; COLT = {"theta50": C.REAL, "manc50": C.REAL_MID}
FDY = {f: (k - 2.5) * 0.085 for k, f in enumerate(C.MAIN)}
c_rows = []
ax = ax_c
ax.axvline(0.5, color=C.LIGHT, lw=0.6, ls=(0, (2.5, 2)), zorder=0)
for tag in C.TAGS:
    q = RG[RG.connectome == tag]; real = q[q.graph == "real"].iloc[0]; nul = q[q.family.isin(C.MAIN)].copy()
    assert len(nul) == 30 and abs(real.f_opp - SUM.loc[tag, "f_opp_real"]) < 1e-12
    y = ROW[tag]
    ax.scatter(nul.f_opp, y + nul.family.map(FDY), s=12, c=[C.FAM_COLOR[f] for f in nul.family], edgecolors="white", linewidths=0.3, zorder=3)
    ax.plot([real.f_opp] * 2, [y - 0.34, y + 0.34], color=COLT[tag], lw=2.2, solid_capstyle="butt", zorder=4)
    ax.text(real.f_opp, y + 0.41, C.fmt(real.f_opp, 2), fontsize=5.8, ha="center", va="bottom", color=COLT[tag], fontweight="bold")
    ax.text(float(nul.f_opp.max()) - 0.012, y + 0.27, C.fmt(float(nul.f_opp.max()), 2), fontsize=5.3, ha="right", va="bottom", color=C.MUTED)
    ax.text(-0.02, y, C.TAGS[tag], transform=blended_transform_factory(ax.transAxes, ax.transData), fontsize=6.0, ha="right", va="center",
            color=COLT[tag], fontweight="bold")
    c_rows.append(dict(connectome=tag, network="real", family="real", f_opp=real.f_opp, f_both=real.f_both))
    c_rows += [dict(connectome=tag, network=r.graph, family=r.family, f_opp=r.f_opp, f_both=r.f_both) for r in nul.itertuples()]
ax.set_xlim(0, 1); ax.set_ylim(-0.55, 1.75); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel("Opposite-signed share of shared influence (f_opp)")
ax.text(0.48, 1.74, "← mostly same sign", fontsize=5.2, ha="right", va="top", color=C.MUTED)
ax.text(0.52, 1.74, "mostly opposite sign →", fontsize=5.2, ha="left", va="top", color=C.MUTED)
fb = SUM.f_both_real
C.panel_title(ax_c, "The same neurons push one pool and pull the other", x_pt=-54, y_pt=19.5, fontsize=6.6, fontweight="bold")
C.panel_title(ax_c, f"{100 * fb['theta50']:.0f}% and {100 * fb['manc50']:.0f}% of the influence comes from neurons reaching both pools; "
              "grey: best rewired", x_pt=-54, y_pt=11.5, fontsize=5.2, color=C.MUTED)
C.panel_label(ax_c, "c", x_pt=-62, y_pt=19.5)
C.write_source(pd.DataFrame(c_rows), "Fig3c_f_opp",
               "Post hoc sign analysis of the reciprocal-innervation index (round2_20260927/analysis/ri_semantics_posthoc_20260929/):\n"
               "u = one- plus two-step influence vectors of the index. f_both = share of the influence (|u_E| + |u_F|) from neurons with\n"
               "non-zero influence on both pools; f_opp = sum |u_E u_F| over those neurons with u_E u_F < 0, divided by sum |u_E u_F| (network\n"
               "value: median over the 20 pairs). Real networks and the 30 main rewired networks (6 families x 5).")

# ================================================================================ d: two routes to opposite signs (schematic) + shares
ax = ax_d
ax.set_xlim(0, 62); ax.set_ylim(0, 40); ax.set_aspect("equal"); ax.axis("off")
EXC_COL, INH_COL = C.REAL_LIGHT, C.REAL           # the two routes (same colours in the share bars)
PXE, PXF, PXI = 31.0, 50.0, 40.5                   # extensor pool, flexor pool, interneuron (x, mm)


def pools(y0):
    for (cx_, col) in [(PXE, C.EXT), (PXF, C.FLX)]:
        for m in range(3):
            ax.add_patch(Circle((cx_ + (m - 1) * 1.6, y0 + (1.1 if m == 1 else -0.2)), 1.0, fc=col, ec="white", lw=0.3, zorder=4))


def route(y0, src_col, src_lab, first_sign, text):
    ax.text(0.5, y0 + 7.6, text, fontsize=5.4, ha="left", va="center", color=C.INK)
    SC.node(ax, 12.0, y0, r=2.0, fc=src_col, ec=C.INK, lw=0.6, z=5)
    ax.text(12.0, y0 - 3.4, src_lab, fontsize=5.3, ha="center", va="top", color=C.INK)
    pools(y0)
    if first_sign > 0:
        SC.arrow(ax, (14.2, y0 + 0.6), (PXE - 2.4, y0 + 0.6), color=C.INK, lw=0.8, head=3.2)
    else:
        SC.inhib(ax, (14.2, y0 + 0.6), (PXE - 2.2, y0 + 0.6), color=C.INK, lw=0.8, bar=0.9)
    SC.node(ax, PXI, y0 - 5.2, r=1.1, fc="white", ec=C.NODE, lw=0.7, z=5)                  # inhibitory interneuron
    ax.text(PXI, y0 - 7.0, "inh.", fontsize=5.0, ha="center", va="top", color=C.MUTED)
    if first_sign > 0:
        SC.arrow(ax, (13.9, y0 - 1.2), (PXI - 1.2, y0 - 4.9), color=C.INK, lw=0.8, head=3.0, rad=0.12)
    else:
        SC.inhib(ax, (13.9, y0 - 1.2), (PXI - 1.2, y0 - 4.95), color=C.INK, lw=0.8, bar=0.8, rad=0.12)
    SC.inhib(ax, (PXI + 1.1, y0 - 4.6), (PXF - 1.6, y0 - 1.2), color=C.INK, lw=0.8, bar=0.8)
    s1, s2 = ("+", "−") if first_sign > 0 else ("−", "+")
    ax.text(PXE, y0 + 3.4, s1, fontsize=7.5, ha="center", va="center", color=C.EXT, fontweight="bold")
    ax.text(PXF, y0 + 3.4, s2, fontsize=7.5, ha="center", va="center", color=C.FLX, fontweight="bold")


ax.set_ylim(0, 47)
route(29.0, EXC_COL, "excitatory", +1, "Route 1: excite one pool, inhibit the other")
route(9.5, INH_COL, "inhibitory", -1, "Route 2: inhibit one pool, disinhibit the other")
ax.text(PXE, 41.2, "extensor", fontsize=5.3, ha="center", va="center", color=C.EXT)
ax.text(PXF, 41.2, "flexor", fontsize=5.3, ha="center", va="center", color=C.FLX)
ax.text(12.0, 41.2, "source neuron", fontsize=5.3, ha="center", va="center", color=C.MUTED)
ax.text(PXF + 2.0, 0.2, "schematic", fontsize=5.2, ha="right", va="bottom", color=C.MUTED, style="italic")
C.panel_label(ax_d, "d", x_pt=-2, y_pt=-7)
ax.text(4.2, 47.0, "Opposite signs need an interneuron: two routes", fontsize=6.6, fontweight="bold", ha="left", va="top")
# shares (data): median over the 20 pairs of the real network
SS = pd.read_csv(C.RISEM / "ri_semantics_real_sign_split.csv")
ax = ax_d2
d_rows = []
for tag in C.TAGS:
    q = SS[SS.connectome == tag]; e_ = float(q.opp_mass_from_exc.median()); i_ = float(q.opp_mass_from_inh.median())
    assert abs(e_ - SUM.loc[tag, "share_exc_median"]) < 1e-12
    y = ROW[tag]
    ax.barh(y, e_, height=0.5, color=EXC_COL, edgecolor="white", lw=0.4)
    ax.barh(y, i_, left=e_, height=0.5, color=INH_COL, edgecolor="white", lw=0.4)
    ax.text(e_ / 2, y, f"{100 * e_:.0f}%", fontsize=5.6, ha="center", va="center", color=C.INK)
    ax.text(e_ + i_ / 2, y, f"{100 * i_:.0f}%", fontsize=5.6, ha="center", va="center", color="white")
    ax.text(-0.04, y, C.TAGS[tag], transform=blended_transform_factory(ax.transAxes, ax.transData), fontsize=5.6, ha="right", va="center",
            color=COLT[tag], fontweight="bold")
    d_rows.append(dict(connectome=tag, share_excitatory_route=e_, share_inhibitory_route=i_, n_pairs=len(q)))
ax.set_xlim(0, 1); ax.set_ylim(-0.6, 1.9); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
ax.set_xlabel("Share of opposite-\nsigned influence", linespacing=1.15)
ax.text(0.0, 1.66, "route 1", fontsize=5.2, ha="left", va="center", color=C.MUTED)
ax.text(1.0, 1.66, "route 2", fontsize=5.2, ha="right", va="center", color=C.MUTED)
C.write_source(pd.DataFrame(d_rows), "Fig3d_route_shares",
               "Real networks: share of the opposite-signed overlap mass sum|u_E u_F|[u_E u_F < 0] contributed by excitatory (sign +1) and by\n"
               "inhibitory (sign -1) source neurons, median over the 20 antagonistic pairs (ri_semantics_real_sign_split.csv). A neuron's direct\n"
               "outputs share its sign (Dale), so opposite-signed influence runs through an intervening neuron: excitatory sources excite one pool\n"
               "and inhibit the other through an inhibitory interneuron; inhibitory sources inhibit one pool and disinhibit the other.")

C.save(fig, "Fig3_reciprocal_innervation", dict(
    axes=[ax_a1, ax_a2, ax_b1, ax_b2, ax_c, ax_d2],
    panel_ids=["a1", "a2", "b1", "b2", "c", "d2"],
    row_groups=[["a1", "a2", "b1", "b2"], ["c", "d2"]],
    exemptions=[{"panels": ["b1", "b2"], "checks": ["panel-width", "horizontal-gutter"],
                 "reason": "strip plots of the network index are narrower than the two square influence scatter plots"},
                {"panels": ["d2"], "checks": ["panel-width", "horizontal-gutter"],
                 "reason": "d2 is the narrow data part of panel d, to the right of its schematic"}],
))
print("Fig. 3 written")
