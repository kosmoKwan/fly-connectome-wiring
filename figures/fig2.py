"""Fig. 2 | Rewired networks produce rhythm but not antagonist coordination.

a  Example pool activity (CPU re-simulation, labelled as such): the six thorax-coxa pairs of the real
   MaleCNS network and of its strongest rewired network, same anatomical pairs, same time segment.
b,c  Parameter scan (132 settings, 25-s runs): each network's best S against its maximum rhythmicity,
   both over eligible settings.
d-g  60-s confirmation at each network's frozen setting (fresh noise): network means of 20 runs by family,
   15-25 s and 50-60 s, MaleCNS and MANC; real network and the pre-specified 'strong' margin.
h,i  Real networks, run by run across the four confirmation windows (persistence of the coordinated state).
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import common as C

C.setup_style()
W, H = 183.0, 175.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)

# ------------------------------------------------------------------------------------------ layout (mm)
R1_B, R1_H = 123.0, 41.0      # row 1: a, b, c
R2_B, R2_H = 62.0, 41.0       # row 2: d-g
R3_B, R3_H = 16.0, 24.0       # row 3: h, i
ax_a = A(14.0, R1_B, 72.0, R1_H)
ax_b = A(104.0, R1_B, 34.0, R1_H)
ax_c = A(146.0, R1_B, 34.0, R1_H)
X0, PW, GAP = 31.0, 34.2, 4.8
ax_d, ax_e, ax_f, ax_g = (A(X0 + i * (PW + GAP), R2_B, PW, R2_H) for i in range(4))
ax_h = A(14.0, R3_B, 72.0, R3_H)
ax_i = A(108.0, R3_B, 72.0, R3_H)


def header(x0_mm, x1_mm, y_mm, text):
    fig.text((x0_mm + x1_mm) / 2 / W, (y_mm + 1.0) / H, text, ha="center", va="bottom", fontsize=6.5)
    fig.lines.append(Line2D([x0_mm / W, x1_mm / W], [y_mm / H] * 2, transform=fig.transFigure, color=C.INK, lw=0.5))


# ================================================================================ a: example traces
ex = {k: np.load(C.DERIVED / f"example_traces_{k}.npz") for k in ("malecns_real", "malecns_hl_s3")}
exs = pd.read_csv(C.DERIVED / "example_runs_summary.csv").set_index("example")
exp = {k: pd.read_csv(C.DERIVED / f"example_pairs_{k}.csv").set_index("pair") for k in ex}
SEG = (15.0, 17.0)                                   # first 2 s of the 15-25-s scoring window
LEG_ROWS = ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]
SP = 4.6                                             # vertical spacing between pairs (normalized units)
COL_X = {"malecns_real": 0.0, "malecns_hl_s3": 2.78}
rows_a = []
for k, x0 in COL_X.items():
    z = ex[k]; X = z["X"]; fs = float(z["fs"]); leg, joint, dd = z["leg"], z["joint"], z["direction"]
    Xc = X - X.mean(0)
    i0, i1 = int((SEG[0] - 15.0) * fs), int((SEG[1] - 15.0) * fs)
    t = np.arange(i1 - i0) / fs
    for r, L in enumerate(LEG_ROWS):
        e = (leg == L) & (joint == "ThC") & (dd > 0); f = (leg == L) & (joint == "ThC") & (dd < 0)
        a, b = Xc[:, e].mean(1), Xc[:, f].mean(1)
        sc = max(a.std(), b.std())
        y0 = -r * SP
        info = exp[k].loc[f"{L}_ThC"]
        gated = bool(info["gated"])
        ca, cb = (C.EXT, C.FLX) if gated else (C.EXT_FADED, C.FLX_FADED)
        ax_a.plot(x0 + t, y0 + b[i0:i1] / sc, color=cb, lw=0.6, zorder=2)
        ax_a.plot(x0 + t, y0 + a[i0:i1] / sc, color=ca, lw=0.6, zorder=3)
        if gated:
            lab, lc = f"r = {C.fmt(info['corr'], 2)}", C.INK
        else:
            lab, lc = "n.m.", C.MUTED
        ax_a.text(x0 + 2.07, y0, lab, fontsize=5.6, va="center", ha="left", color=lc)
        rows_a.append(dict(example=k, pair=f"{L}_ThC", segment_s=f"{SEG[0]}-{SEG[1]}", scale=sc, corr_15_25=info["corr"],
                           gated=gated, contribution=info["contrib"], n_ext=int(e.sum()), n_flx=int(f.sum()),
                           sd_ext=info["sd_ext"], sd_flx=info["sd_flx"]))
for r, L in enumerate(LEG_ROWS):
    ax_a.text(-0.07, -r * SP, L, fontsize=6, ha="right", va="center")
for k, x0 in COL_X.items():
    s = exs.loc[k]
    name = "Real network" if k == "malecns_real" else "Strongest rewired network"
    ax_a.text(x0, 1.55 * SP, name, fontsize=6.5, ha="left", va="center", color=C.REAL if k == "malecns_real" else C.FAM_COLOR["hl"],
              fontweight="bold" if k == "malecns_real" else "normal")
    sub = f"S = {C.fmt(s.cpu_S, 2)}" if k == "malecns_real" else f"lineage block · S = {C.fmt(s.cpu_S, 2)}"
    ax_a.text(x0, 0.95 * SP, sub, fontsize=5.6, ha="left", va="center", color=C.MUTED)
yb = -5 * SP - 0.8 * SP
ax_a.plot([0.0, 0.5], [yb, yb], color=C.INK, lw=0.9, solid_capstyle="butt")
ax_a.text(0.25, yb - 0.35 * SP, "0.5 s", fontsize=5.6, ha="center", va="top")
ax_a.plot([0.9, 1.12], [yb, yb], color=C.EXT, lw=1.2); ax_a.text(1.16, yb, "Extensor pool", fontsize=5.6, va="center")
ax_a.plot([2.1, 2.32], [yb, yb], color=C.FLX, lw=1.2); ax_a.text(2.36, yb, "Flexor pool", fontsize=5.6, va="center")
ax_a.text(3.3, yb, "n.m., not modulated", fontsize=5.6, va="center", color=C.MUTED)
ax_a.text(4.95, yb - 0.35 * SP, "Example simulation · MaleCNS · thorax–coxa (ThC) pairs · 15–17 s", fontsize=5.3, va="top", ha="right",
          color=C.MUTED, style="italic")
ax_a.set_xlim(-0.02, 4.95); ax_a.set_ylim(yb - 0.95 * SP, 1.8 * SP)
ax_a.axis("off")
C.panel_label(ax_a, "a", x_pt=-30, y_pt=5)
C.write_source(pd.DataFrame(rows_a), "Fig2a_example_ThC_pairs",
               "Example simulation (CPU re-implementation of the rate model, numpy noise; not a confirmation run).\n"
               "Networks: real MaleCNS at its frozen setting; lineage-block network 3 (the rewired network with the highest\n"
               "confirmation S) at its own frozen setting. Pairs shown: all six thorax-coxa (ThC) pairs, chosen before viewing\n"
               "the traces because ThC carries most of the real network's score (Fig. 5).\n"
               "Traces: extensor/flexor pool means (demeaned over 15-25 s) for 15-17 s, each pair scaled by its larger pool s.d.\n"
               "corr/gating computed over the full 15-25-s window with the archived readout rules.\n"
               f"Example-run S (all 20 pairs): real {exs.loc['malecns_real','cpu_S']:.4f} "
               f"(confirmation runs {exs.loc['malecns_real','gpu_S_min']:.3f}-{exs.loc['malecns_real','gpu_S_max']:.3f}); "
               f"lineage block s3 {exs.loc['malecns_hl_s3','cpu_S']:.4f} "
               f"(confirmation {exs.loc['malecns_hl_s3','gpu_S_min']:.3f}-{exs.loc['malecns_hl_s3','gpu_S_max']:.3f}).")

# ================================================================================ b,c: scan scatter
scan_rows = []
for ax, tag, lab in [(ax_b, "theta50", "b"), (ax_c, "manc50", "c")]:
    sc = C.scan(tag)
    e = sc[sc.eligible]
    best = e.groupby(["graph", "gseed"]).agg(bestS=("gmean_stat", "max"), max_rhythm=("rhythm_stat", "max"),
                                             n_eligible=("eligible", "size")).reset_index()
    best = best[best.graph.isin(["real"] + C.MAIN + C.EXTRA)].copy()
    best.insert(0, "connectome", tag)
    rv = best[best.graph == "real"].iloc[0]
    nul = best[best.graph != "real"]
    n_more = int((nul.max_rhythm > rv.max_rhythm).sum())
    best["n_rewired_more_rhythmic_than_real"] = n_more
    scan_rows.append(best)
    ax.axhline(0, color=C.GRID, lw=0.6, zorder=0)
    ax.axvspan(rv.max_rhythm, 1.0, color=C.SAND, lw=0, zorder=0)
    for fam in C.MAIN + C.EXTRA:
        q = best[best.graph == fam]
        if not len(q):
            continue
        hollow = fam in C.EXTRA
        ax.scatter(q.max_rhythm, q.bestS, s=12, marker="o", linewidths=0.6 if hollow else 0.3,
                   facecolors="white" if hollow else C.FAM_COLOR[fam], edgecolors=C.FAM_COLOR[fam] if hollow else "white", zorder=3)
    ax.scatter([rv.max_rhythm], [rv.bestS], s=30, marker="o", facecolors=C.REAL, edgecolors="white", linewidths=0.6, zorder=5)
    right = tag == "theta50"
    ax.annotate(f"Real {C.fmt(rv.bestS)}", (rv.max_rhythm, rv.bestS), xytext=(5.5 if right else -5.5, 0), textcoords="offset points",
                ha="left" if right else "right", va="center", fontsize=5.8, color=C.REAL)
    top = nul.loc[nul.bestS.idxmax()]
    ax.annotate(f"Best rewired {C.fmt(top.bestS)}", (top.max_rhythm, top.bestS), xytext=(-4, 9), textcoords="offset points",
                ha="right", va="bottom", fontsize=5.5, color=C.MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C.MUTED, shrinkA=0, shrinkB=2.4, relpos=(1, 0)))
    ax.text(0.03, 0.80 if tag == "theta50" else 0.985, f"Shaded: more rhythmic than real\n({n_more} of {len(nul)} rewired networks)", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.4, color=C.MUTED, linespacing=1.25)
    ax.set_xlim(0.1, 1.0); ax.set_ylim(-0.13, 0.38)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Max. rhythmicity")
    C.panel_title(ax, C.TAGS[tag], y_pt=4, fontweight="bold")
    C.panel_label(ax, lab, x_pt=-24, y_pt=4)
ax_b.set_ylabel("Best S")
header(104.0 - 8, 180.0, R1_B + R1_H + 5.2, "Parameter scan · 132 settings per network, 25-s runs")
C.write_source(pd.concat(scan_rows), "Fig2bc_scan_best",
               "Per network: best S and maximum rhythmicity over eligible settings of the 132-setting scan (15-25 s of 25-s runs).\n"
               "Eligible: >20% of MNs oscillating and median dominant frequency 1-50 Hz. MaleCNS: 40 rewired networks (6 main + 2 secondary\n"
               "families, secondary = hollow); MANC: 30. The two maxima can come from different settings.")

# ================================================================================ d-g: confirmation strips
FAM_Y = {"er": 0, "role": 1, "degree": 2, "leg": 3, "hl": 4, "leg_hl": 5, "degree_recip": 6.6, "leg_hl_recip": 7.6}
JIT = np.array([-0.22, -0.11, 0.0, 0.11, 0.22])
strip_rows = []
panels = [(ax_d, "theta50", "w15_25", "d"), (ax_e, "theta50", "w50_60", "e"), (ax_f, "manc50", "w15_25", "f"), (ax_g, "manc50", "w50_60", "g")]
undef_note = None
for ax, tag, win, lab in panels:
    gmw = C.graph_window_means(tag)
    x = gmw[(gmw.window == win) & (gmw.rule == "own")]
    real = x[x.graph == "real"].S.iloc[0]
    ax.axvline(0, color=C.GRID, lw=0.6, zorder=0)
    real_sem = x[x.graph == "real"].S_sem.iloc[0]
    ax.axvspan(real - real_sem, real + real_sem, color=C.REAL_PALE, lw=0, zorder=0)
    ax.axvline(real, color=C.REAL, lw=0.9, zorder=1)
    ax.axvline(real - 0.15, color=C.REAL, lw=0.6, ls=(0, (1, 1.6)), zorder=1)
    lft = real > 0.25
    ax.text(real + (-0.008 if lft else 0.008), -1.25, f"{C.fmt(real)}", fontsize=5.6, ha="right" if lft else "left",
            va="center", color=C.REAL)
    meds = {}
    for fam in C.MAIN + C.EXTRA:
        q = x[x.graph == fam].sort_values("gseed")
        if not len(q):
            continue
        yy = FAM_Y[fam] + JIT[q.gseed.to_numpy()]
        hollow = fam in C.EXTRA
        undef = q.n_undef.to_numpy() > 0
        ax.errorbar(q.S, yy, xerr=q.S_sem, fmt="none", ecolor=C.FAM_COLOR[fam], elinewidth=0.5, capsize=0, zorder=2)
        for (xv, yv, u) in zip(q.S, yy, undef):
            ax.scatter([xv], [yv], s=13 if u else 11, marker="D" if u else "o",
                       facecolors="white" if (hollow or u) else C.FAM_COLOR[fam],
                       edgecolors=C.FAM_COLOR[fam] if (hollow or u) else "white", linewidths=0.6 if (hollow or u) else 0.3, zorder=3)
        if undef.any():
            uu = q[q.n_undef > 0].iloc[0]
            undef_note = (int(uu.n_undef), int(uu.n_runs), int(uu.n_def))
        med = float(np.median(q.S)); meds[fam] = med
        ax.plot([med, med], [FAM_Y[fam] - 0.36, FAM_Y[fam] + 0.36], color=C.INK, lw=1.0, zorder=4, solid_capstyle="butt")
        for _, rr in q.iterrows():
            strip_rows.append(dict(connectome=tag, window=win, family=fam, gseed=int(rr.gseed), S=rr.S, S_sem=rr.S_sem,
                                   n_defined_runs=int(rr.n_def), n_undefined_runs=int(rr.n_undef), family_median=med, real=real))
    gt = sum(bool((real > x[x.graph == f].S).all()) for f in C.MAIN)
    strong = gt == 6 and all(real - meds[f] >= 0.15 for f in C.MAIN)
    moderate = gt >= 5 and all(real > meds[f] for f in C.MAIN)
    lvl = "strong" if strong else ("moderate" if moderate else "weak")
    if tag == "manc50":
        ax.text(0.27, 7.1, "not generated\nfor MANC", fontsize=5.3, color=C.PENDING, ha="center", va="center", style="italic",
                linespacing=1.0)
    ax.set_ylim(8.2, -1.75)
    ax.set_xlim(-0.15, 0.36)
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.set_yticks(list(FAM_Y.values()))
    ax.tick_params(axis="y", length=0, pad=8.5)
    ax.spines["left"].set_visible(False)
    if lab == "d":
        ax.set_yticklabels([C.FAM_LABEL[f] for f in FAM_Y])
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        for fam, yv in FAM_Y.items():
            hollow = fam in C.EXTRA
            ax.scatter([-0.045], [yv], s=10, transform=tr, clip_on=False, facecolors="white" if hollow else C.FAM_COLOR[fam],
                       edgecolors=C.FAM_COLOR[fam], linewidths=0.6 if hollow else 0.3, zorder=5)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Antagonist score S")
    C.panel_title(ax, f"{C.WIN_LABEL[win]} · {lvl}", loc="center", y_pt=4)
    C.panel_label(ax, lab, x_pt=-62 if lab == "d" else -8, y_pt=4)
    strip_rows.append(dict(connectome=tag, window=win, family="__level__", gseed=-1, S=np.nan, S_sem=np.nan, n_defined_runs=-1,
                           n_undefined_runs=-1, family_median=np.nan, real=real, level=lvl))
for (a1, a2, tag) in [(ax_d, ax_e, "theta50"), (ax_f, ax_g, "manc50")]:
    p1, p2 = a1.get_position(), a2.get_position()
    header(p1.x0 * W + 1, p2.x1 * W - 1, R2_B + R2_H + 5.0, f"{C.TAGS[tag]} · 60-s confirmation, frozen settings, fresh noise")
ax_d.text(-0.52, 5.95, "Secondary", transform=blended_transform_factory(ax_d.transAxes, ax_d.transData),
          fontsize=5.2, color=C.MUTED, ha="left", va="center")
# key for d-g
kx = 31.0 / W; ky = (R2_B - 12.2) / H
key = [Line2D([], [], ls="-", lw=0.5, marker="o", ms=3.2, color=C.FAM_COLOR["degree"], mfc=C.FAM_COLOR["degree"], mec="white", mew=0.3,
              label="Rewired network, mean ± s.e.m. of 20 runs"),
       Line2D([], [], ls="none", marker="|", ms=5.5, mew=1.0, color=C.INK, label="Family median"),
       Line2D([], [], color=C.REAL, lw=0.9, label="Real (band ± s.e.m.)"),
       Line2D([], [], color=C.REAL, lw=0.6, ls=(0, (1, 1.6)), label="Real − 0.15 ('strong' margin)"),
       Line2D([], [], ls="none", marker="D", ms=3.2, mfc="white", mec=C.FAM_COLOR["leg_hl_recip"], mew=0.6,
              label=f"{undef_note[0]}/{undef_note[1]} runs undefined, mean of {undef_note[2]}")]
fig.legend(handles=key, loc="lower left", bbox_to_anchor=(kx - 0.075, ky), ncol=5, fontsize=5.6, handlelength=1.6,
           columnspacing=1.0, handletextpad=0.45, frameon=False, borderaxespad=0)
C.write_source(pd.DataFrame(strip_rows), "Fig2d-g_confirmation_by_family",
               "Network value = mean over 20 stochastic runs of the 60-s confirmation at the network's own frozen setting.\n"
               "Undefined runs (no oscillating MN) are excluded from the mean, as in the archived analysis; n_undefined_runs is listed.\n"
               "family_median = median over the 5 networks. level = pre-specified claim level (§3.23) over the 6 main families:\n"
               "strong = real above all 30 networks and >= 0.15 above every family median; moderate = above every network in >= 5\n"
               "families and above every family median.")

# ================================================================================ h,i: persistence of the real networks
pers_rows = []
xs = np.arange(4)
SW = C.SWITCH
for ax, tag, lab in [(ax_h, "theta50", "h"), (ax_i, "manc50", "i")]:
    hd = C.heldout(tag); r = hd[hd.graph == "real"]
    S = r.pivot_table(index=["mode", "run"], columns="window", values="gmean")[C.WINDOWS]
    Fq = r.pivot_table(index=["mode", "run"], columns="window", values="freq")[C.WINDOWS]
    Ng = r.pivot_table(index=["mode", "run"], columns="window", values="n_gated_pairs")[C.WINDOWS]
    fast = (Fq >= 3) & (Ng <= 3)
    switched = (~fast["w15_25"]) & fast[["w25_35", "w35_45", "w50_60"]].any(axis=1)
    for (mode, run), row in S.iterrows():
        sw = bool(switched.loc[(mode, run)])
        fm = fast.loc[(mode, run)].values
        if mode == "stochastic":
            ax.plot(xs, row.values, color=SW if sw else C.RUN, lw=0.8 if sw else 0.5, zorder=3 if sw else 2)
        else:
            ax.plot(xs, row.values, color=SW if sw else C.REAL, lw=0.75, ls=(0, (3, 1.6)), zorder=3)
        if fm.any():
            ax.scatter(xs[fm], row.values[fm], s=8, facecolors="white", edgecolors=SW, linewidths=0.6, zorder=4)
        for w in C.WINDOWS:
            pers_rows.append(dict(connectome=tag, mode=mode, run=run, window=w, S=row[w], freq_hz=Fq.loc[(mode, run), w],
                                  n_gated_pairs=Ng.loc[(mode, run), w], fast_state=bool(fast.loc[(mode, run), w]), switched_run=sw))
    mean = S.loc["stochastic"].mean(axis=0).values
    ax.plot(xs, mean, color=C.REAL, lw=1.5, zorder=5, marker="o", ms=2.6, mfc=C.REAL, mec="white", mew=0.4)
    n_sw = int(switched.loc["stochastic"].sum()); n_det_sw = int(switched.loc["deterministic"].sum())
    st = r[(r["mode"] == "stochastic")]
    if tag == "theta50":
        f15 = st[st.window == "w15_25"].freq
        n_fast = int(fast.values.sum())
        note = f"No run entered a fast state ({n_fast} of {fast.size} run-windows)\n" \
               f"dominant frequency {f15.min():.1f}–{f15.max():.1f} Hz at 15–25 s"
    else:
        swmask = switched.loc["stochastic"].values
        start = fast.loc["stochastic"]["w15_25"].values              # fast already at 15-25 s: never coordinated
        keep = ~swmask & ~start
        f_oth = Fq.loc["stochastic"][keep]["w15_25"]
        fsw = Fq.loc["stochastic"][swmask].values[fast.loc["stochastic"][swmask].values]
        f_start = Fq.loc["stochastic"][start]["w15_25"]
        start_txt = (f"run {int(f_start.index[0])} fast from the start ({f_start.iloc[0]:.1f} Hz)" if len(f_start) == 1
                     else f"{len(f_start)} runs fast from the start")
        note = f"{n_sw} of 20 noisy and {n_det_sw} of 3 noise-free runs switched to fast\n" \
               f"states ({fsw.min():.0f}–{fsw.max():.0f} Hz, ≤3 gated pairs); {start_txt};\n" \
               f"other runs {f_oth.min():.1f}–{f_oth.max():.1f} Hz at 15–25 s"
    ax.text(0.03, 1.0, note, transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=C.MUTED, linespacing=1.1)
    ax.set_xlim(-0.25, 3.25); ax.set_ylim(-0.03, 0.47)
    ax.set_xticks(xs); ax.set_xticklabels([C.WIN_LABEL[w] for w in C.WINDOWS])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Confirmation window")
    ax.set_ylabel("S")
    C.panel_title(ax, f"{C.TAGS[tag]} · real network, run by run", y_pt=4)
    C.panel_label(ax, lab, x_pt=-30 if lab == "h" else -24, y_pt=4)
handles = [Line2D([], [], color=C.RUN, lw=0.6, label="Noisy run"),
           Line2D([], [], color=C.REAL, lw=0.75, ls=(0, (3, 1.6)), label="Noise-free run"),
           Line2D([], [], color=SW, lw=0.8, label="Run that switched"),
           Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=SW, mew=0.6, label="Fast state (≥3 Hz, ≤3 gated pairs)"),
           Line2D([], [], color=C.REAL, lw=1.5, marker="o", ms=2.6, mfc=C.REAL, mec="white", label="Mean of 20 noisy runs")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(12.0 / W, 0.4 / H), ncol=5, fontsize=5.6,
           handlelength=2.0, columnspacing=1.5, frameon=False, borderaxespad=0)
C.write_source(pd.DataFrame(pers_rows), "Fig2hi_real_runs_by_window",
               "Real networks at their frozen settings; S, dominant frequency and number of gated pairs per run and window.\n"
               "fast_state: dominant frequency >= 3 Hz and <= 3 gated pairs. switched_run: not fast at 15-25 s but fast in a later window.")

C.save(fig, "Fig2_rhythm_vs_coordination", dict(
    axes=[ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i],
    panel_ids=list("abcdefghi"),
    row_groups=[["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]],
    column_groups=[["a", "h"]],
    exemptions=[{"panels": ["a"], "checks": ["panel-width", "horizontal-gutter"],
                 "reason": "example-trace panel is a different modality and intentionally wider than the two scan scatter plots"}],
))
print("Fig. 2 written")
