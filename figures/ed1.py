"""Extended Data Fig. 1 | Model, rewired networks and protocol (formerly Fig. 1).

a  Schematic of the model: sustained drive to DNg100, fixed-weight recurrent leg-motor connectome, readout from
   anatomically assigned extensor and flexor motor-neuron pools at four leg joints, antagonist score S.
b  Composition of the two extracted subnetworks (data: node, edge and motor-neuron tables).
c  Six nested families of rewired networks: which structure each family preserves (from the generator code).
d  Protocol: every network gets the same 132-setting scan (real MaleCNS network shown, data); the best eligible
   setting is frozen and re-run with fresh noise for 60 s, scored in four windows.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import blended_transform_factory
import common as C
import schem as SC

C.setup_style()
W, H = 183.0, 150.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
ax_a = A(2.0, 97.0, 179.0, 50.0)
ax_b = A(15.0, 52.0, 70.0, 32.0)
ax_c = A(122.0, 52.0, 58.0, 32.0)
ax_d = A(2.0, 4.0, 179.0, 33.0)
ROLE_COL = {"IN": C.SAND_DEEP, "SN": C.AMBER, "DN": C.REAL, "MN": C.EXT}   # MN drawn as an extensor/flexor split

# ================================================================================ a: model schematic
ax = ax_a
ax.set_xlim(0, 179); ax.set_ylim(0, 50); ax.set_aspect("equal"); ax.axis("off")
# --- drive
for yy in (29.0, 20.0):
    SC.node(ax, 12.5, yy, r=1.9, fc=C.REAL, ec="white", lw=0.6)
    SC.arrow(ax, (2.5, yy), (10.2, yy), color=C.REAL, lw=0.9, head=4)
ax.text(12.5, 35.0, "DNg100", fontsize=6.5, ha="center", va="bottom", fontweight="bold")
ax.text(12.5, 32.4, "left, right", fontsize=5.4, ha="center", va="bottom", color=C.MUTED)
ax.plot([2.5, 4.5, 4.5, 8.5], [11.0, 11.0, 13.4, 13.4], color=C.REAL, lw=0.8)
ax.text(2.5, 8.6, "sustained\ndrive d", fontsize=5.6, ha="left", va="top", linespacing=1.1)
# --- network cloud (schematic)
cx, cy, rx, ry = 46.5, 24.5, 22.0, 15.5
ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc=C.CLOUD, ec=C.REAL_LIGHT, lw=0.6, zorder=0))
# schematic node layout: deterministic golden-angle (sunflower) pattern inside the ellipse; decorative, not data
_k = np.arange(1, 59); _ga = np.pi * (3 - np.sqrt(5))
pts = np.stack([cx + (rx - 2.4) * np.sqrt(_k / 58) * np.cos(_k * _ga), cy + (ry - 2.4) * np.sqrt(_k / 58) * np.sin(_k * _ga)], 1)
for i, p in enumerate(pts):
    d = np.linalg.norm(pts - p, axis=1)
    for j in np.argsort(d)[1:3]:
        ax.plot([p[0], pts[j][0]], [p[1], pts[j][1]], color=C.REAL_LIGHT, lw=0.35, zorder=1)
roles = np.where(pts[:, 0] < cx - 11, "DN", np.where(_k % 8 == 3, "SN", "IN"))
roles = np.where(pts[:, 0] > cx + 13, "MN", roles)
mn_flip = 0
for p, r in zip(pts, roles):
    fc = ROLE_COL[r]
    if r == "MN":
        fc = C.EXT if mn_flip % 2 == 0 else C.FLX; mn_flip += 1
    ax.add_patch(Circle(p, 0.85, fc=fc, ec="white", lw=0.3, zorder=2))
for yy in (29.0, 20.0):
    SC.arrow(ax, (14.6, yy), (cx - rx + 2.6, yy + (1.2 if yy > 24 else -1.2)), color=C.REAL, lw=0.8, head=3.5)
ax.text(cx, 49.5, "Fixed-weight leg-motor connectome", fontsize=6.5, ha="center", va="top", fontweight="bold")
ax.text(cx, 46.3, "weights = signed synapse counts, never trained", fontsize=5.5, ha="center", va="top", color=C.MUTED)
ax.text(cx, 1.5, r"$\tau\dot{r} = -r + [\tanh(g\,(W_E + \beta W_I)\,r + I + \xi)]_+$", fontsize=7.6, ha="center", va="bottom")
# --- motor pools
PX = 90.0
POOL_Y = {"ThC": 38.0, "CTr": 29.0, "FTi": 20.0, "TiTa": 11.0}
for j, yy in POOL_Y.items():
    for (col, dx) in [(C.EXT, -2.7), (C.FLX, 2.7)]:
        for m in range(3):
            ax.add_patch(Circle((PX + dx + (m - 1) * 1.15, yy + (0.85 if m == 1 else -0.35)), 0.72, fc=col, ec="white", lw=0.3, zorder=4))
    ax.text(PX - 7.2, yy, j, fontsize=6, ha="right", va="center", color=C.INK, fontweight="bold")
    SC.arrow(ax, (cx + rx - 0.8, cy + (yy - cy) * 0.6), (PX - 12.5, yy), color=C.REAL_MID, lw=0.6, head=3, z=3)
ax.text(PX, 49.5, "Motor-neuron pools", fontsize=6.5, ha="center", va="top", fontweight="bold")
ax.text(PX - 2.7, 43.0, "ext.", fontsize=5.4, ha="center", va="bottom", color=C.EXT)
ax.text(PX + 2.7, 43.0, "flx.", fontsize=5.4, ha="center", va="bottom", color=C.FLX)
ax.text(PX, 5.6, "by target muscle,\nfor each leg", fontsize=5.4, ha="center", va="top", color=C.MUTED, linespacing=1.1)
# --- leg (side view, schematic), upper right
LX, LY = 110.0, 42.0
ax.add_patch(FancyBboxPatch((LX, LY - 4.0), 10.0, 8.0, boxstyle="round,pad=0,rounding_size=2.5", fc=C.BODY,
                            ec=C.BODY_EDGE, lw=0.6, zorder=1))
ax.text(LX + 5.0, LY, "thorax", fontsize=5.2, ha="center", va="center", color=C.MUTED)
J = {"ThC": np.array([LX + 10.0, LY - 2.2])}
J["CTr"] = J["ThC"] + np.array([5.5, -6.0])
J["FTi"] = J["CTr"] + np.array([19.0, 3.0])
J["TiTa"] = J["FTi"] + np.array([17.0, -10.5])
tip = J["TiTa"] + np.array([12.0, -4.0])
for a_, b_, w_ in [(J["ThC"], J["CTr"], 3.2), (J["CTr"], J["FTi"], 2.4), (J["FTi"], J["TiTa"], 1.6), (J["TiTa"], tip, 1.0)]:
    SC.capsule(ax, a_, b_, width=w_, fc=C.SEG, ec=C.BODY_EDGE, lw=0.5, z=2)
for j, p in J.items():
    ax.add_patch(Circle(p, 1.3, fc=C.JOINT_COLOR[j], ec=C.REAL_MID if j == "TiTa" else "white", lw=0.5, zorder=5))
LBL = {"ThC": (2.2, 2.4, "left", "thorax–coxa"), "CTr": (0.0, -3.6, "center", "coxa–trochanter"),
       "FTi": (0.0, 3.2, "center", "femur–tibia"), "TiTa": (0.0, -3.6, "center", "tibia–tarsus")}
for j, (dx, dy, ha, long) in LBL.items():
    p = J[j]
    va = "bottom" if dy > 0 else "top"
    ax.text(p[0] + dx, p[1] + dy, f"{j}" if dy > 0 else long, fontsize=6 if dy > 0 else 5.0, ha=ha, va=va,
            fontweight="bold" if dy > 0 else "normal", color=C.INK if dy > 0 else C.MUTED)
    ax.text(p[0] + dx, p[1] + dy + (2.6 if dy > 0 else -2.4), long if dy > 0 else f"{j}", fontsize=5.0 if dy > 0 else 6, ha=ha, va=va,
            fontweight="normal" if dy > 0 else "bold", color=C.MUTED if dy > 0 else C.INK)
ax.text(179.0, 49.5, "Four joints of each leg (schematic)", fontsize=6.5, ha="right", va="top", fontweight="bold")
SC.arrow(ax, (PX + 6.3, 38.0), (LX - 0.8, 40.0), color=C.REAL_MID, lw=0.6, head=3, z=3)
# --- readout, lower right
RX, RY, RW, RH = 110.0, 1.0, 69.0, 16.5
ax.add_patch(FancyBboxPatch((RX, RY), RW, RH, boxstyle="round,pad=0,rounding_size=1.6", fc=C.CLOUD, ec=C.REAL_LIGHT, lw=0.6))
ax.text(RX + 2.0, RY + RH - 1.6, "Antagonist score S", fontsize=6.5, ha="left", va="top", fontweight="bold")
ax.text(RX + 2.0, RY + RH - 5.6,
        "For each of the 20 antagonistic pairs: if the extensor-pool and flexor-pool\n"
        "signals are both modulated, score = −corr(extensor, flexor); otherwise 0.\n"
        "S = mean over the 20 pairs: +1 alternating, 0 none, −1 co-active.",
        fontsize=5.5, ha="left", va="top", linespacing=1.3)
SC.arrow(ax, (PX + 6.3, 11.0), (RX - 0.8, 9.5), color=C.REAL_MID, lw=0.6, head=3, z=3)
C.panel_label(ax_a, "a", x_pt=2, y_pt=-2)

# ================================================================================ b: composition (data)
ax = ax_b
rows_b = []
ybar = {"theta50": 2.35, "manc50": 0.0}
for tag, y0 in ybar.items():
    n = pd.read_parquet(C.GRAPHS / f"nodes_{tag}.parquet"); e = pd.read_parquet(C.GRAPHS / f"edges_{tag}.parquet")
    m = pd.read_parquet(C.GRAPHS / f"mn_joint_map_{tag}.parquet")
    N = len(n); E = len(e)
    counts = {r: int((n.role == r).sum()) for r in ["IN", "DN", "SN", "MN"]}
    left = 0.0
    for r in ["IN", "DN", "SN", "MN"]:
        w = counts[r] / N
        if r == "MN":
            ax.barh(y0 + 0.505, w, left=left, height=0.21, color=C.EXT, edgecolor="white", lw=0.4)
            ax.barh(y0 + 0.295, w, left=left, height=0.21, color=C.FLX, edgecolor="white", lw=0.4)
        else:
            ax.barh(y0 + 0.40, w, left=left, height=0.42, color=ROLE_COL[r], edgecolor="white", lw=0.4)
        left += w
    ax.text(counts["IN"] / N / 2, y0 + 0.40, f"interneurons {counts['IN']:,}", fontsize=5.3, ha="center", va="center")
    mapped = m.joint.notna(); approx = mapped & m.approx.fillna(False)
    mc = {"exact": int((mapped & ~approx).sum()), "approx": int(approx.sum()), "unmapped": int((~mapped).sum())}
    left = 0.0
    for k, col in [("exact", C.REAL), ("approx", C.REAL_LIGHT), ("unmapped", C.REAL_PALE)]:
        w = mc[k] / len(m)
        ax.barh(y0 - 0.18, w, left=left, height=0.42, color=col, edgecolor="white", lw=0.4)
        ax.text(left + w / 2, y0 - 0.18, f"{k} {mc[k]}", fontsize=5.2, ha="center", va="center",
                color="white" if k == "exact" else C.INK)
        left += w
    pairs = m[mapped].groupby(["leg", "joint"]).direction.agg(lambda d: (d > 0).any() and (d < 0).any())
    pj = pairs[pairs].reset_index().groupby("joint").size()
    ax.text(-0.02, y0 + 0.40, "Neurons", fontsize=5.6, ha="right", va="center")
    ax.text(-0.02, y0 - 0.18, "Motor\nneurons", fontsize=5.6, ha="right", va="center", linespacing=1.25)
    ax.text(0.0, y0 + 0.74, f"{C.TAGS[tag]}", fontsize=6.3, ha="left", va="bottom", fontweight="bold")
    ax.text(0.2 if tag == "theta50" else 0.13, y0 + 0.74,
            f"{N:,} neurons (DN {counts['DN']}, SN {counts['SN']}, MN {counts['MN']}) · {E:,} connections · {int(pairs.sum())} pairs",
            fontsize=5.5, ha="left", va="bottom")
    rows_b.append(dict(connectome=tag, neurons=N, connections=E, **{f"n_{k}": v for k, v in counts.items()},
                       MN_exact=mc["exact"], MN_approx=mc["approx"], MN_unmapped=mc["unmapped"], pairs=int(pairs.sum()),
                       **{f"pairs_{j}": int(pj.get(j, 0)) for j in C.JOINTS}))
ax.set_xlim(0, 1.0); ax.set_ylim(-0.55, 3.45)
ax.axis("off")
kx, ky = 0.0, -0.62
for lab, col in [("interneuron", ROLE_COL["IN"]), ("descending", ROLE_COL["DN"]), ("sensory", ROLE_COL["SN"]), ("motor", ROLE_COL["MN"])]:
    if lab == "motor":
        ax.add_patch(Rectangle((kx, ky), 0.025, 0.09, fc=C.EXT, ec="none", clip_on=False))
        ax.add_patch(Rectangle((kx, ky - 0.09), 0.025, 0.09, fc=C.FLX, ec="none", clip_on=False))
    else:
        ax.add_patch(Rectangle((kx, ky - 0.09), 0.025, 0.18, fc=col, ec=col, lw=0.3, clip_on=False))
    ax.text(kx + 0.035, ky, lab, fontsize=5.3, va="center")
    kx += 0.035 + 0.0125 * len(lab) + 0.045
ax.text(0.0, -0.98, "Motor neurons: exact or approximate assignment to a joint pool, or unmapped", fontsize=5.2, va="center",
        color=C.MUTED)
C.panel_label(ax_b, "b", x_pt=-36, y_pt=3)
C.panel_title(ax_b, "Extracted subnetworks (data)", x_pt=-26, y_pt=3, fontweight="bold")
C.write_source(pd.DataFrame(rows_b), "ED1b_composition",
               "Counts from the extracted node, edge and motor-neuron tables used by every simulation (connections >= 5 synapses).\n"
               "MN assignment: exact = mapped to a joint without the approximate flag; approx = flagged approximate; unmapped = no joint.\n"
               "Antagonistic pair = leg x joint with both an extensor and a flexor pool.")

# ================================================================================ c: rewired families
ax = ax_c
props = ["Edge\ncount", "Edges\nbetween\ncell roles", "In/out\ndegree of\neach neuron", "Edges\nbetween\nleg blocks",
         "Edges\nbetween\nlineage\nblocks"]
keep = {"er": [1, 0, 0, 0, 0], "role": [1, 1, 0, 0, 0], "degree": [1, 1, 1, 0, 0], "leg": [1, 1, 1, 1, 0],
        "hl": [1, 1, 1, 0, 1], "leg_hl": [1, 1, 1, 1, 1]}
tr = blended_transform_factory(ax.transAxes, ax.transData)
for i, fam in enumerate(C.MAIN):
    y = 5 - i
    ax.scatter([-0.035], [y], s=16, color=C.FAM_COLOR[fam], edgecolors="white", linewidths=0.3, clip_on=False, zorder=3, transform=tr)
    ax.text(-0.075, y, C.FAM_LABEL[fam], fontsize=6, ha="right", va="center", transform=tr)
    for j, v in enumerate(keep[fam]):
        ax.scatter([j], [y], s=15 if v else 6, facecolors=C.REAL if v else "white", edgecolors=C.REAL if v else C.LIGHT,
                   linewidths=0.3 if v else 0.6, zorder=3)
    if i:
        ax.plot([-0.3, 4.3], [y + 0.5, y + 0.5], color=C.GRID, lw=0.5, zorder=0)
for j, p in enumerate(props):
    ax.text(j, 5.75, p, fontsize=5.2, ha="center", va="bottom", linespacing=1.25)
ax.set_xlim(-0.4, 4.4); ax.set_ylim(-2.6, 9.4)
ax.axis("off")
ax.text(-0.37, -0.85, "Filled: preserved. 5 networks per family; every neuron keeps its transmitter sign.\n"
        "Density and cell-role networks place edges at random; the other four exchange edge\n"
        "targets within blocks, which also keeps every neuron's degree and output strength.",
        fontsize=5.1, ha="left", va="top", color=C.MUTED, linespacing=1.2, transform=tr)
C.panel_label(ax_c, "c", x_pt=-66, y_pt=3)
C.panel_title(ax_c, "Six nested families of rewired networks", x_pt=-56, y_pt=3, fontweight="bold")

# ================================================================================ d: protocol
ax = ax_d
ax.set_xlim(0, 179); ax.set_ylim(0, 33); ax.set_aspect("equal"); ax.axis("off")
sc = C.scan("theta50"); r = sc[sc.graph == "real"].copy()
G_ = sorted(r.g.unique()); B_ = sorted(r.beta.unique()); D_ = sorted(r.drive.unique())
cmap = LinearSegmentedColormap.from_list("coord", C.COORD_COLORS)
norm = Normalize(0, 0.35)
HX, HY, cw, ch = 16.0, 5.0, 2.1, 1.75            # heatmap origin (mm) and cell size
rows_d = []
for bi, b in enumerate(B_):
    for di, d in enumerate(D_):
        row = bi * 4 + di
        for gi, g in enumerate(G_):
            q = r[(r.g == g) & (r.beta == b) & (r.drive == d)].iloc[0]
            x0, y0 = HX + gi * cw, HY + (11 - row) * ch + (2 - bi) * 0.6
            if q.eligible:
                fc = cmap(norm(max(q.gmean_stat, 0)))
                ax.add_patch(Rectangle((x0, y0), cw, ch, fc=fc, ec="white", lw=0.3))
            else:
                ax.add_patch(Rectangle((x0, y0), cw, ch, fc=C.SAND, ec="white", lw=0.3))
            rows_d.append(dict(connectome="theta50", graph="real", log10_g=float(np.log10(g)), beta=b, drive=d, eligible=bool(q.eligible),
                               S=q.gmean_stat, rhythmicity=q.rhythm_stat))
        ax.text(HX - 0.6, HY + (11 - row) * ch + (2 - bi) * 0.6 + ch / 2, f"{d:g}", fontsize=5.0, ha="right", va="center")
    ymid = HY + (11 - (bi * 4 + 1.5)) * ch + (2 - bi) * 0.6 + ch / 2
    ax.text(HX - 4.2, ymid, f"β = {b:g}", fontsize=5.2, ha="right", va="center")
best = r[r.eligible].sort_values("gmean_stat", ascending=False).iloc[0]
gi = G_.index(best.g); bi = B_.index(best.beta); di = D_.index(best.drive); row = bi * 4 + di
ax.add_patch(Rectangle((HX + gi * cw - 0.45, HY + (11 - row) * ch + (2 - bi) * 0.6 - 0.45), cw + 0.9, ch + 0.9, fc="none",
                       ec=C.RING, lw=1.1, zorder=5))
for gi_, g in enumerate(G_):
    if gi_ % 2 == 0:
        ax.text(HX + gi_ * cw + cw / 2, HY - 0.6, f"{np.log10(g):g}".replace("-", "−"), fontsize=5.0, ha="center", va="top")
ax.text(HX + 11 * cw / 2, HY - 3.0, "gain g (log10)", fontsize=5.4, ha="center", va="top")
ax.text(HX - 0.6, HY + 12 * ch + 1.2 + 0.6, "d", fontsize=5.2, ha="right", va="bottom")
ax.text(HX, 32.6, "1  Scan: 132 settings, 25-s runs", fontsize=6.3, ha="left", va="top", fontweight="bold")
ax.text(HX + 11 * cw + 1.2, HY + 12 * ch + 1.2, f"real MaleCNS\nnetwork (data):\nS at {int(r.eligible.sum())} eligible\nsettings",
        fontsize=5.2, ha="left", va="top", color=C.MUTED, linespacing=1.25)
# colour bar
cax = fig.add_axes([(2.0 + HX + 11 * cw + 1.4) / W, (4.0 + 7.5) / H, 1.5 / W, 8.0 / H])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
cb.set_ticks([0, 0.1, 0.2, 0.3]); cb.ax.tick_params(labelsize=5.0, length=1.3, pad=1.0); cb.outline.set_linewidth(0.4)
cb.set_label("S", fontsize=5.4, labelpad=1.5)
ax.add_patch(Rectangle((HX + 11 * cw + 1.3, HY - 3.4), 1.6, 1.3, fc=C.SAND, ec=C.BODY_EDGE, lw=0.3))
ax.text(HX + 11 * cw + 3.4, HY - 2.75, "not eligible", fontsize=5.0, ha="left", va="center", color=C.MUTED)
# freeze arrow
FX = 64.0
SC.arrow(ax, (FX - 2.0, 17.0), (FX + 17.0, 17.0), color=C.REAL, lw=0.9, head=4.5)
ax.text(FX + 7.5, 32.6, "2  Freeze", fontsize=6.3, ha="center", va="top", fontweight="bold")
ax.text(FX + 7.5, 25.6, "best eligible\nsetting", fontsize=5.6, ha="center", va="top", linespacing=1.1)
ax.text(FX + 7.5, 14.6, f"MaleCNS real:\nlog10 g = {np.log10(best.g):.1f}\nβ = {best.beta:g}, d = {best.drive:g}".replace("-", "−"),
        fontsize=5.6, ha="center", va="top", linespacing=1.2)
# 60-s confirmation timeline
TX0, TX1, TY = 92.0, 176.0, 15.0
sx = (TX1 - TX0) / 60.0
ax.text(TX0, 32.6, "3  Confirm: 60 s at the frozen setting, fresh noise", fontsize=6.3, ha="left", va="top", fontweight="bold")
ax.add_patch(Rectangle((TX0, TY - 1.2), 60 * sx, 2.4, fc=C.REAL_PALE, ec="none"))
ax.add_patch(Rectangle((TX0, TY - 1.2), 15 * sx, 2.4, fc=C.SAND, ec="none"))
for (a0, a1) in [(15, 25), (25, 35), (35, 45), (50, 60)]:
    ax.add_patch(Rectangle((TX0 + a0 * sx + 0.25, TY - 1.2), (a1 - a0) * sx - 0.5, 2.4, fc=C.REAL, ec="none"))
    ax.text(TX0 + (a0 + a1) / 2 * sx, TY + 2.2, f"{a0}–{a1} s", fontsize=5.4, ha="center", va="bottom")
for t in [0, 15, 30, 45, 60]:
    ax.plot([TX0 + t * sx] * 2, [TY - 1.6, TY - 2.3], color=C.INK, lw=0.5)
    ax.text(TX0 + t * sx, TY - 2.8, f"{t}", fontsize=5.0, ha="center", va="top")
ax.text(TX0 + 7.5 * sx, TY + 2.2, "transient", fontsize=5.2, ha="center", va="bottom", color=C.MUTED)
ax.text(TX0 + 30 * sx, TY - 6.0, "time (s) · scoring windows in dark teal", fontsize=5.2, ha="center", va="top", color=C.MUTED)
ax.text(TX0, 25.3, "20 runs with new noise + 3 noise-free runs from different initial states, for every network\n"
        "network value = mean of the 20 noisy runs · family value = median of its 5 networks", fontsize=5.4, ha="left", va="top",
        linespacing=1.25)
ax.text(TX0, 4.2, "Claim levels, fixed before these results: strong = real above all 30 rewired networks and ≥ 0.15 above every\n"
        "family median; moderate = above every network in ≥ 5 families and above every family median",
        fontsize=5.2, ha="left", va="top", color=C.MUTED, linespacing=1.2)
C.panel_label(ax_d, "d", x_pt=2, y_pt=-4)
C.write_source(pd.DataFrame(rows_d), "ED1d_scan_real_malecns",
               "Real MaleCNS network, 132-setting scan (25-s runs scored at 15-25 s). eligible: >20% of MNs oscillating and median dominant\n"
               f"frequency 1-50 Hz. Frozen setting = eligible setting with the highest S: log10 g = {np.log10(best.g):.2f}, beta = {best.beta:g}, "
               f"d = {best.drive:g}, S = {best.gmean_stat:.4f}.")

C.save(fig, "EDFig1_model_and_protocol", dict(
    axes=[ax_b, ax_c], panel_ids=["b", "c"], row_groups=[["b", "c"]],
))
print("ED Fig. 1 written")
