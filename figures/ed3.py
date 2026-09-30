"""Extended Data Fig. 3 | Reciprocal innervation pair by pair (formerly Fig. 3c,d, plus the post hoc sign analysis).

a  Real reciprocal-innervation index of every antagonistic pair against the range of the 30 main rewired networks.
b  The same for f_opp, the opposite-signed share of the shared influence (post hoc; ri_semantics_pairs.csv).
c  Real per-pair index, MaleCNS against MANC.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr
import common as C

C.setup_style()
W, H = 183.0, 128.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
ax_a1 = A(15.0, 100.0, 112.0, 17.0)
ax_a2 = A(15.0, 79.0, 112.0, 17.0)
ax_b1 = A(15.0, 44.0, 112.0, 17.0)
ax_b2 = A(15.0, 23.0, 112.0, 17.0)
ax_c = A(143.0, 79.0, 37.0, 38.0)
rp_m = C.recip_pairs("theta50")
real_pairs = rp_m[rp_m.graph == "real"].set_index("pair").recip_innervation

# ================================================================================ a: index pair by pair (formerly Fig. 3c)
c_rows = []
pairs_order = sorted(real_pairs.index, key=C.pair_sort_key)
xpos, x = {}, 0.0
for i, p in enumerate(pairs_order):
    if i and C.split_pair(p)[1] != C.split_pair(pairs_order[i - 1])[1]:
        x += 0.8
    xpos[p] = x; x += 1.0
for ax, tag in [(ax_a1, "theta50"), (ax_a2, "manc50")]:
    rp = C.recip_pairs(tag).copy()
    rp["level"] = rp.graph.str.replace(r"_s\d+$", "", regex=True)
    rv = rp[rp.graph == "real"].set_index("pair").recip_innervation
    nv = rp[rp.level.isin(C.MAIN)]
    stats = nv.groupby("pair").recip_innervation.agg(["min", "max", "median", "size"])
    n_above = 0
    ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
    for p in pairs_order:
        xx = xpos[p]; s = stats.loc[p]; above = bool(rv[p] > s["max"])
        n_above += above
        ax.plot([xx, xx], [s["min"], s["max"]], color=C.NULL_RANGE, lw=2.4, solid_capstyle="butt", zorder=1)
        ax.plot([xx - 0.22, xx + 0.22], [s["median"]] * 2, color=C.NULL_MED, lw=0.7, zorder=2)
        ax.scatter([xx], [rv[p]], s=14, facecolors=C.REAL if above else "white", edgecolors=C.REAL, linewidths=0.7, zorder=3)
        c_rows.append(dict(connectome=tag, pair=p, joint=C.split_pair(p)[1], real=rv[p], rewired_min=s["min"], rewired_median=s["median"],
                           rewired_max=s["max"], n_rewired=int(s["size"]), real_above_all=above))
    ax.text(1.0, 1.02, f"{C.TAGS[tag]}: real above all 30 rewired networks in {n_above} of {len(pairs_order)} pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=C.INK)
    ax.set_xlim(-0.7, x - 0.3); ax.set_ylim(-0.55, 0.8)
    ax.set_yticks([-0.4, 0, 0.4, 0.8]); ax.set_yticklabels(["−0.4", "0", "0.4", "0.8"])
    ax.set_xticks([xpos[p] for p in pairs_order])
    if tag == "theta50":
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels([C.split_pair(p)[0] for p in pairs_order], fontsize=5.2)
    ax.tick_params(axis="x", length=1.5, pad=1.5)
    ax.set_ylabel("Index")
tr = blended_transform_factory(ax_a2.transData, ax_a2.transAxes)
for j in C.JOINTS:
    ps = [p for p in pairs_order if p.endswith("_" + j)]
    x0_, x1_ = xpos[ps[0]] - 0.3, xpos[ps[-1]] + 0.3
    ax_a2.plot([x0_, x1_], [-0.30, -0.30], transform=tr, clip_on=False, color=C.INK, lw=0.5)
    ax_a2.text((x0_ + x1_) / 2, -0.37, j, transform=tr, fontsize=6, ha="center", va="top", color=C.INK)
C.panel_label(ax_a1, "a", x_pt=-30, y_pt=4)
handles = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=C.REAL, mec=C.REAL, label="Real, above all rewired"),
           Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=C.REAL, mew=0.7, label="Real, not above all"),
           Line2D([], [], color=C.NULL_RANGE, lw=2.4, label="Range of 30 rewired networks"),
           Line2D([], [], color=C.NULL_MED, lw=0.7, label="Median")]

C.write_source(pd.DataFrame(c_rows), "ED3a_pair_index_real_vs_rewired",
               "Per antagonistic pair: real index and min/median/max over the 30 main rewired networks (6 families x 5).")

# ================================================================================ c: MaleCNS vs MANC (formerly Fig. 3d)
cr_all = pd.DataFrame(c_rows).pivot(index="pair", columns="connectome", values="real")
n_before = len(cr_all); cr = cr_all.dropna(); n_after = len(cr)
assert n_before == n_after == 20, (n_before, n_after)   # count tracking: no pair is dropped (both connectomes have the same 20 pairs)
rho, pval = spearmanr(cr["theta50"], cr["manc50"])
lim = (-0.25, 0.72)
ax_c.plot(lim, lim, color=C.LIGHT, lw=0.6, ls=(0, (3, 2)), zorder=0)
for p, row in cr.iterrows():
    j = C.split_pair(p)[1]
    ax_c.scatter(row["theta50"], row["manc50"], s=15, facecolors=C.JOINT_COLOR[j], edgecolors=C.REAL_MID if j == "TiTa" else "white",
                 linewidths=0.4, zorder=3)
for p in [q for q in cr.index if q.endswith("TiTa")]:
    up = cr.loc[p, "manc50"] > -0.05
    ax_c.annotate(p.replace("_", " "), (cr.loc[p, "theta50"], cr.loc[p, "manc50"]), xytext=(4, -3.5) if up else (4, 0),
                  textcoords="offset points", fontsize=5.2, va="top" if up else "center", ha="left", color=C.MUTED)
ax_c.text(0.31, 0.97, f"Spearman ρ = {C.fmt(rho, 2)}\nn = {len(cr)} pairs", transform=ax_c.transAxes, ha="left", va="top", fontsize=5.8,
          linespacing=1.3)
jh = [Line2D([], [], ls="none", marker="o", ms=3.4, mfc=C.JOINT_COLOR[j], mec="white" if j != "TiTa" else C.REAL_MID, mew=0.4, label=j)
      for j in C.JOINTS]
ax_c.legend(handles=jh, loc="lower right", fontsize=5.5, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25, frameon=False)
ax_c.set_xlim(lim); ax_c.set_ylim(lim)
ax_c.set_xticks([0, 0.3, 0.6]); ax_c.set_yticks([0, 0.3, 0.6])
ax_c.set_xlabel("MaleCNS real index"); ax_c.set_ylabel("MANC real index")
C.panel_title(ax_c, "Same pairs, two connectomes", y_pt=4)
C.panel_label(ax_c, "c", x_pt=-24, y_pt=4)
C.write_source(cr.reset_index().assign(spearman_rho=rho, spearman_p=pval), "ED3c_index_malecns_vs_manc",
               "Real per-pair reciprocal-innervation index in the two connectomes (same 20 pair identities); Spearman rank correlation.")


# ================================================================================ b: f_opp pair by pair (post hoc)
RP = pd.read_csv(C.RISEM / "ri_semantics_pairs.csv"); RVN = pd.read_csv(C.RISEM / "ri_semantics_real_vs_nulls_by_pair.csv")
b_rows = []
for ax, tag in [(ax_b1, "theta50"), (ax_b2, "manc50")]:
    q = RP[RP.connectome == tag]
    rv = q[q.graph == "real"].set_index("pair").f_opp
    nv = q[q.family.isin(C.MAIN)]
    assert nv.groupby("pair").size().eq(30).all()
    st = nv.groupby("pair").f_opp.agg(["min", "max", "median"])
    chk = RVN[RVN.connectome == tag].set_index("pair")
    assert np.allclose(chk.loc[st.index, "null_max_fopp"], st["max"], atol=1e-12)          # same as the archived per-pair table
    n_above = 0
    ax.axhline(0.5, color=C.LIGHT, lw=0.6, ls=(0, (2.5, 2)), zorder=0)
    for p in pairs_order:
        xx = xpos[p]; s_ = st.loc[p]; above = bool(rv[p] > s_["max"]); n_above += above
        assert above == bool(chk.loc[p, "fopp_above_all"])
        ax.plot([xx, xx], [s_["min"], s_["max"]], color=C.NULL_RANGE, lw=2.4, solid_capstyle="butt", zorder=1)
        ax.plot([xx - 0.22, xx + 0.22], [s_["median"]] * 2, color=C.NULL_MED, lw=0.7, zorder=2)
        ax.scatter([xx], [rv[p]], s=14, facecolors=C.REAL if above else "white", edgecolors=C.REAL, linewidths=0.7, zorder=3)
        b_rows.append(dict(connectome=tag, pair=p, joint=C.split_pair(p)[1], real_f_opp=rv[p], rewired_min=s_["min"], rewired_median=s_["median"],
                           rewired_max=s_["max"], real_above_all=above))
    ax.text(1.0, 1.02, f"{C.TAGS[tag]}: real above all 30 rewired networks in {n_above} of {len(pairs_order)} pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=C.INK)
    ax.set_xlim(-0.7, xpos[pairs_order[-1]] + 0.7); ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1]); ax.set_xticks([xpos[p] for p in pairs_order])
    ax.set_xticklabels([] if tag == "theta50" else [C.split_pair(p)[0] for p in pairs_order], fontsize=5.2)
    ax.tick_params(axis="x", length=1.5, pad=1.5)
    ax.set_ylabel("f_opp")
tr = blended_transform_factory(ax_b2.transData, ax_b2.transAxes)
for j in C.JOINTS:
    ps = [p for p in pairs_order if p.endswith("_" + j)]
    x0_, x1_ = xpos[ps[0]] - 0.3, xpos[ps[-1]] + 0.3
    ax_b2.plot([x0_, x1_], [-0.30, -0.30], transform=tr, clip_on=False, color=C.INK, lw=0.5)
    ax_b2.text((x0_ + x1_) / 2, -0.37, j, transform=tr, fontsize=6, ha="center", va="top", color=C.INK)
C.panel_label(ax_b1, "b", x_pt=-30, y_pt=4)
ax_b2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.62), ncol=4, fontsize=5.5, handlelength=1.5,
             columnspacing=1.4, frameon=False, borderaxespad=0)
C.write_source(pd.DataFrame(b_rows), "ED3b_pair_f_opp_real_vs_rewired",
               "Post hoc (ri_semantics_pairs.csv): per antagonistic pair, f_opp of the real network and min/median/max over the 30 main rewired\n"
               "networks; real_above_all equals fopp_above_all in ri_semantics_real_vs_nulls_by_pair.csv (asserted).")

C.save(fig, "EDFig3_reciprocal_innervation_by_pair", dict(
    axes=[ax_a1, ax_a2, ax_b1, ax_b2, ax_c], panel_ids=["a1", "a2", "b1", "b2", "c"],
    column_groups=[["a1", "a2"], ["b1", "b2"]],
))
print("ED Fig. 3 written")
