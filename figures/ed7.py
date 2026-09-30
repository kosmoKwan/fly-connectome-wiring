"""Extended Data Fig. 7 | Persistence of the coordinated state over 5 minutes (round 2, experiment 3).

300-s simulations at the frozen settings of the real MANC network, its two strongest rewired networks (lineage block
1 and 0) and the real MaleCNS network; 40 noisy trajectories per network (streams A + B) plus 10 noise-free ones (D).
Coordinated window (pre-registered): S >= 0.10 with >= 6 modulated pairs, 10-s windows from 10 to 300 s.
a  fraction of noisy trajectories in the coordinated state; b  mean S (± s.e.m.); c  state of every trajectory;
d  switching rate (leaving the coordinated state for >= 3 windows) per minute spent coordinated, exact Poisson 95% CI;
e  S of every noisy trajectory in the last window (290-300 s).
All values from derived/r2_exp3_*.csv, recomputed from long_scalar.parquet and checked against analysis/exp3_*.csv.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import common as C

C.setup_style()
W, H = 183.0, 140.0
fig = plt.figure(figsize=(W * C.MM, H * C.MM))
A = lambda l, b, w, h: C.add_axes_mm(fig, l, b, w, h, H, W)
TR = pd.read_csv(C.DERIVED / "r2_exp3_trajectory.csv")
SU = pd.read_csv(C.DERIVED / "r2_exp3_summary.csv")
WN = pd.read_csv(C.DERIVED / "r2_exp3_windows.csv")
NETS = [("theta50", "real", 0, "MaleCNS real", C.REAL, "-", 1.4),
        ("manc50", "real", 0, "MANC real", C.REAL_MID, "-", 1.4),
        ("manc50", "hl", 1, "MANC lineage block 1", C.FAM_COLOR["hl"], "-", 1.0),
        ("manc50", "hl", 0, "MANC lineage block 0", "#D9826B", (0, (3, 1.6)), 1.0)]
ax_a = A(14.0, 96.0, 68.0, 34.0)
ax_b = A(104.0, 96.0, 68.0, 34.0)
RAS = [A(14.0 + i * 42.0, 42.0, 35.0, 34.0) for i in range(4)]
ax_d = A(40.0, 8.0, 72.0, 20.0)
ax_e = A(124.0, 8.0, 56.0, 20.0)

# ---------------- a, b: ensemble trajectories
rows_ab = []
for tag, g, s, lab, col, ls, lw in NETS:
    q = TR[(TR.connectome == tag) & (TR.graph == g) & (TR.gseed == s)].sort_values("t_start")
    tmid = q.t_start + 5
    ax_a.plot(tmid, 100 * q.frac_coord, color=col, ls=ls, lw=lw, label=lab, zorder=3 if g == "real" else 2)
    ax_b.fill_between(tmid, q.mean_S - q.sem_S, q.mean_S + q.sem_S, color=col, alpha=0.18, lw=0, zorder=1)
    ax_b.plot(tmid, q.mean_S, color=col, ls=ls, lw=lw, zorder=3 if g == "real" else 2)
    for _, r in q.iterrows():
        rows_ab.append(dict(connectome=tag, graph=g, gseed=s, t_start=r.t_start, frac_coord=r.frac_coord, mean_S=r.mean_S, sem_S=r.sem_S, n_runs=int(r.n)))
    last = SU[(SU.connectome == tag) & (SU.graph == g) & (SU.gseed == s)].iloc[0]
    ax_a.text(302, 100 * last.frac_coord_last, f"{100 * last.frac_coord_last:.0f}%", fontsize=5.4, va="center", ha="left", color=col)
    ax_b.text(302, last.mean_S_last, C.fmt(last.mean_S_last), fontsize=5.4, va="center", ha="left", color=col)
# de-overlap the labels at the right edge of b (MANC values are close): nudge vertically by rank
texts = [t for t in ax_b.texts]
ys = sorted([(t.get_position()[1], t) for t in texts], key=lambda z: z[0])
for i in range(1, len(ys)):
    if ys[i][0] - ys[i - 1][0] < 0.021:
        y_new = ys[i - 1][0] + 0.021
        ys[i][1].set_position((302, y_new)); ys[i] = (y_new, ys[i][1])
for ax in (ax_a, ax_b):
    ax.set_xlim(0, 300); ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_xlabel("Time (s)")
    ax.axvspan(290, 300, color=C.SAND, lw=0, zorder=0)
ax_a.set_ylim(-3, 105); ax_a.set_yticks([0, 25, 50, 75, 100]); ax_a.set_ylabel("Trajectories coordinated (%)")
ax_a.axhline(50, color=C.LIGHT, lw=0.5, ls=(0, (2, 2)), zorder=0)
ax_b.set_ylim(-0.01, 0.37); ax_b.set_yticks([0, 0.1, 0.2, 0.3]); ax_b.set_ylabel("Mean S (± s.e.m.)")
C.panel_title(ax_a, "Fraction of 40 noisy trajectories in the coordinated state", y_pt=4)
C.panel_title(ax_b, "Ensemble mean S", y_pt=4)
C.panel_label(ax_a, "a", x_pt=-30, y_pt=4); C.panel_label(ax_b, "b", x_pt=-28, y_pt=4)
leg = [Line2D([], [], color=c, ls=ls, lw=lw, label=l) for _, _, _, l, c, ls, lw in NETS]
fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, (96.0 + 34.0 + 6.5) / H), ncol=4, fontsize=5.8, frameon=False,
           handlelength=2.2, columnspacing=1.8, borderaxespad=0)
C.write_source(pd.DataFrame(rows_ab), "ED7ab_ensemble_trajectories",
               "Streams A+B (40 noisy trajectories per network), 10-s windows (t_start = window start). coordinated: S (undefined -> 0) >= 0.10 and\n"
               ">= 6 modulated pairs. mean_S over trajectories with undefined S counted as 0 (as in the pre-registered analysis). Sand band: last window.")

# ---------------- c: state rasters
rows_c = []
for ax, (tag, g, s, lab, col, ls, lw) in zip(RAS, NETS):
    q = WN[(WN.connectome == tag) & (WN.graph == g) & (WN.gseed == s)]
    order = []
    for st in ["A", "B", "D"]:
        qs = q[q.stream == st]
        for run in sorted(qs.run.unique()):
            order.append((st, run))
    ts = sorted(q.t_start.unique())
    M = np.zeros((len(order), len(ts)))
    for i, (st, run) in enumerate(order):
        r = q[(q.stream == st) & (q.run == run)].set_index("t_start").reindex(ts)
        M[i] = r.coord.astype(float).to_numpy()
    cmap = ListedColormap([C.SAND, col])
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest", extent=(10, 300, len(order), 0))
    n_noisy = sum(1 for o in order if o[0] != "D")
    ax.axhline(n_noisy, color=C.INK, lw=0.6)
    ax.set_xlim(10, 300); ax.set_xticks([10, 100, 200, 300])
    ax.set_yticks([n_noisy / 2, n_noisy + (len(order) - n_noisy) / 2]); ax.set_yticklabels(["noisy\n(40)", "noise-\nfree (10)"], fontsize=5.2,
                                                                                            linespacing=1.3)
    ax.tick_params(axis="y", length=0)
    if ax is not RAS[0]:
        ax.set_yticklabels([])
    ax.set_xlabel("Time (s)")
    su = SU[(SU.connectome == tag) & (SU.graph == g) & (SU.gseed == s)].iloc[0]
    C.panel_title(ax, lab, y_pt=4, fontsize=6.0, color=col)
    for i, (st, run) in enumerate(order):
        rows_c.append(dict(connectome=tag, graph=g, gseed=s, stream=st, run=run, coordinated_windows=int(M[i].sum()), n_windows=len(ts)))
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(C.LIGHT)
C.panel_label(RAS[0], "c", x_pt=-30, y_pt=4)
fig.text(14.0 / W, (42.0 + 34.0 + 7.5) / H, "Every trajectory, window by window (filled: coordinated state)", fontsize=6.3, ha="left", va="bottom")
C.write_source(pd.DataFrame(rows_c), "ED7c_state_rasters",
               "Rows: trajectories (streams A, B = noisy, 20 each; D = noise-free, 10). Columns: 10-s windows from 10 to 300 s.\n"
               "Filled = coordinated window (S >= 0.10 and >= 6 modulated pairs). Per-window states are in derived/r2_exp3_windows.csv.")

# ---------------- d: switching rates
rows_d = []
for i, (tag, g, s, lab, col, ls, lw) in enumerate(NETS):
    su = SU[(SU.connectome == tag) & (SU.graph == g) & (SU.gseed == s)].iloc[0]
    y = 3 - i
    if su.switch_events > 0:
        ax_d.plot([su.rate_ci_lo, su.rate_ci_hi], [y, y], color=col, lw=1.0, solid_capstyle="butt")
        ax_d.scatter([su.rate_per_run_min], [y], s=14, color=col, zorder=3)
        txt = f"{su.rate_per_run_min:.2g} ({su.switch_events} events)"
    else:
        ax_d.annotate("", (0.012, y), xytext=(su.rate_ci_hi, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, mutation_scale=6))
        txt = f"0 events (95% CI < {su.rate_ci_hi:.2f})"
    ax_d.text(12.0, y, txt, fontsize=5.2, va="center", ha="left", color=C.INK)
    rows_d.append(dict(connectome=tag, graph=g, gseed=s, switch_events=int(su.switch_events), minutes_coordinated=su.exposure_min,
                       rate_per_min=su.rate_per_run_min, ci_lo=su.rate_ci_lo, ci_hi=su.rate_ci_hi, mean_dwell_min=su.mean_dwell_min))
ax_d.set_xscale("log"); ax_d.set_xlim(0.01, 150); ax_d.set_ylim(-0.6, 3.6)
ax_d.set_xticks([0.01, 0.1, 1, 10]); ax_d.set_xticklabels(["0.01", "0.1", "1", "10"]); ax_d.minorticks_off()
ax_d.set_yticks([3, 2, 1, 0]); ax_d.set_yticklabels([n[3] for n in NETS], fontsize=5.3)
ax_d.tick_params(axis="y", length=0); ax_d.spines["left"].set_visible(False)
ax_d.set_xlabel("Switches per minute coordinated")
C.panel_title(ax_d, "Leaving the coordinated state", y_pt=4)
C.panel_label(ax_d, "d", x_pt=-74, y_pt=4)
C.write_source(pd.DataFrame(rows_d), "ED7d_switching_rates",
               "Switch: a coordinated window followed by >= 3 non-coordinated windows. Rate = switches / minutes spent coordinated, streams A+B;\n"
               "exact Poisson 95% CI (chi-square). mean_dwell_min = minutes coordinated / switches.")

# ---------------- e: last-window distribution
rows_e = []
last = WN[(WN.t_start == WN.t_start.max()) & (WN.stream != "D")]
for i, (tag, g, s, lab, col, ls, lw) in enumerate(NETS):
    q = last[(last.connectome == tag) & (last.graph == g) & (last.gseed == s)]
    y = 3 - i
    jit = (np.arange(len(q)) % 7 - 3) * 0.06
    ax_e.scatter(q.S0, y + jit, s=4, color=col, alpha=0.8, lw=0, zorder=2)
    m = float(q.S0.mean())
    ax_e.plot([m, m], [y - 0.36, y + 0.36], color=C.INK, lw=1.0, zorder=3)
    for _, r in q.iterrows():
        rows_e.append(dict(connectome=tag, graph=g, gseed=s, stream=r.stream, run=r.run, S_290_300=r.S0, coordinated=bool(r.coord)))
ax_e.set_ylim(-0.6, 3.6); ax_e.set_yticks([3, 2, 1, 0]); ax_e.set_yticklabels([])
ax_e.tick_params(axis="y", length=0); ax_e.spines["left"].set_visible(False)
ax_e.set_xlim(-0.05, 0.42); ax_e.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax_e.set_xlabel("S at 290–300 s")
C.panel_title(ax_e, "Last window", y_pt=4)
C.panel_label(ax_e, "e", x_pt=-10, y_pt=4)
C.write_source(pd.DataFrame(rows_e), "ED7e_last_window",
               "S of every noisy trajectory (streams A+B) in the 290-300-s window (undefined -> 0); bar = mean (pre-registered readout 1).")

C.save(fig, "EDFig7_persistence_300s", dict(
    axes=[ax_a, ax_b] + RAS + [ax_d, ax_e], panel_ids=["a", "b", "c1", "c2", "c3", "c4", "d", "e"],
    row_groups=[["a", "b"], ["c1", "c2", "c3", "c4"], ["d", "e"]],
))
print("ED Fig. 7 written")
