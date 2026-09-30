"""Round 2 analysis for experiment 2 (dose-matched strength-controlled surgery, amendment 1) and experiment 3
(300-s trajectories). Pre-registration: round2_20260927/预注册_第二轮_20260927.md.
usage: analyze_exp2_3.py RESULTS_DIR OUT_DIR [PREFIX]
"""
import sys, os
import numpy as np, polars as pl
from scipy.stats import chi2

RES, OUT = sys.argv[1], sys.argv[2]; PRE = sys.argv[3] if len(sys.argv) > 3 else ""
ARCH = "/Volumes/ninishares/Experiments/fly-connectome-research-20260926/final_heldout_20260927_121507_B9ue3i/results/"
os.makedirs(OUT, exist_ok=True)
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
REDIS = {"theta50": {"mn_shuffle_joint_w": 0.139, "mn_r2_cross_lo": 0.217, "mn_r2_cross_hi": 0.273, "mn_shuffle_joint": 0.272},
         "manc50": {"mn_shuffle_joint_w": 0.131, "mn_r2_cross_lo": 0.202, "mn_r2_cross_hi": 0.252, "mn_shuffle_joint": 0.265}}


def gmeans(S):
    st = S.filter(pl.col("mode") == "stochastic")
    return st.group_by("graph", "gseed", "rule", "window").agg(pl.col("S_all").fill_nan(None).mean(), pl.col("rhythm").mean(), pl.col("osc_frac").mean(), pl.col("n_gated_all").mean())


def gated_frac(P):
    pa = P.filter((pl.col("mode") == "stochastic") & (pl.col("readout") == "all"))
    return pa.group_by("graph", "gseed", "rule", "window", "pair").agg(pl.col("gated").mean().alias("fg"))


# ---------------- experiment 2 ----------------
rows2, dose = [], []
for tag in ["theta50", "manc50"]:
    f2 = os.path.join(RES, f"{PRE}pairs_scalar_{tag}_surg2.parquet"); f1 = os.path.join(RES, f"{PRE}pairs_scalar_{tag}.parquet")
    if not (os.path.exists(f2) and os.path.exists(f1)):
        continue
    S2 = pl.read_parquet(f2).filter(pl.col("mode") != "none"); S1 = pl.read_parquet(f1).filter(pl.col("mode") != "none")
    G = pl.concat([gmeans(S1), gmeans(S2)]); F = pl.concat([gated_frac(pl.read_parquet(os.path.join(RES, f"{PRE}pairs_pairs_{tag}.parquet"))),
                                                          gated_frac(pl.read_parquet(os.path.join(RES, f"{PRE}pairs_pairs_{tag}_surg2.parquet")))])
    for w in ["w15_25", "w25_35", "w35_45", "w50_60"]:
        R = G.filter((pl.col("graph") == "real") & (pl.col("window") == w)).row(0, named=True)
        contrib = set(F.filter((pl.col("graph") == "real") & (pl.col("window") == w) & (pl.col("fg") >= 0.5))["pair"].to_list())
        for xg, rule in [("mn_r2_cross_hi", "own"), ("mn_r2_cross_hi", "real_setting"), ("mn_r2_cross_lo", "real_setting")]:
            X = G.filter((pl.col("graph") == xg) & (pl.col("rule") == rule) & (pl.col("window") == w)).sort("gseed")
            W = G.filter((pl.col("graph") == "mn_r2_within") & (pl.col("rule") == rule) & (pl.col("window") == w)).sort("gseed")
            if X.height < 5 or W.height < 5:
                continue
            xv, wv = X["S_all"].to_numpy(), W["S_all"].to_numpy(); r = R["S_all"]; Xm, Wm = float(np.median(xv)), float(np.median(wv))
            Xrh, Wrh = float(np.median(X["rhythm"].to_numpy())), float(np.median(W["rhythm"].to_numpy()))
            ret = []
            for s in range(5):
                kept = set(F.filter((pl.col("graph") == xg) & (pl.col("gseed") == s) & (pl.col("rule") == rule) & (pl.col("window") == w) & (pl.col("fg") >= 0.5))["pair"].to_list())
                ret.append(len(contrib & kept) / max(len(contrib), 1))
            oscr = float(np.median(X["osc_frac"].to_numpy())) / R["osc_frac"]
            c = dict(C1=Wm >= 0.8 * r, C2=(Xm <= 0.5 * r) and (r - Xm >= 0.10), C3=(Xm <= 0.5 * Wm) and (Wm - Xm >= 0.10),
                     C4=bool((xv < r).all() and xv.max() < wv.min()), C5=(Xrh >= 0.8 * R["rhythm"]) and (Xrh >= 0.8 * Wrh),
                     C6=(float(np.median(ret)) >= 0.8) and (oscr >= 0.8))
            rows2.append(dict(connectome=tag, window=w, arm=xg, rule=rule, R=r, W=Wm, X=Xm, X_over_R=Xm / r, R_minus_X=r - Xm, W_minus_X=Wm - Xm,
                              X_graphs=",".join(f"{v:.3f}" for v in xv), W_graphs=",".join(f"{v:.3f}" for v in wv),
                              R_rhythm=R["rhythm"], W_rhythm=Wrh, X_rhythm=Xrh, pair_retention=float(np.median(ret)), osc_retention=oscr, **c,
                              all_six=all(c.values()), collapse=(Xm <= 0.05 and c["C1"] and c["C5"] and c["C6"])))
        for xg in ["mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi", "mn_shuffle_joint"]:
            X = G.filter((pl.col("graph") == xg) & (pl.col("rule") == "real_setting") & (pl.col("window") == w))
            if X.height:
                dose.append(dict(connectome=tag, window=w, arm=xg, redistribution=REDIS[tag][xg], X_median=float(X["S_all"].median()), X_over_R=float(X["S_all"].median()) / R["S_all"]))
if rows2:
    T2 = pl.DataFrame(rows2); T2.write_csv(os.path.join(OUT, "exp2_conditions.csv")); pl.DataFrame(dose).write_csv(os.path.join(OUT, "exp2_dose_response.csv"))
    print(T2.filter(pl.col("window").is_in(["w15_25", "w50_60"])).select("connectome", "window", "arm", "rule", "R", "W", "X", "X_over_R", "W_minus_X", "C1", "C2", "C3", "C4", "C5", "C6", "all_six", "collapse", "pair_retention"))
    print(pl.DataFrame(dose).filter(pl.col("window").is_in(["w15_25", "w50_60"])).sort("connectome", "window", "redistribution"))

# ---------------- experiment 3 ----------------
fl = os.path.join(RES, f"{PRE}long_scalar.parquet")
if os.path.exists(fl):
    L = pl.read_parquet(fl)
    t = L.filter(pl.col("window").str.starts_with("t")).with_columns(((pl.col("S_all").fill_nan(0) >= 0.10) & (pl.col("n_gated_all") >= 6)).alias("coord")).sort("connectome", "graph", "gseed", "stream", "run", "t_start")
    runs = []
    for key, g in t.group_by(["connectome", "graph", "gseed", "stream", "run"]):
        c = g["coord"].to_list(); ts = g["t_start"].to_list(); ev, first, exposure = 0, None, 0
        for i in range(len(c)):
            if c[i]:
                exposure += 1   # 10-s windows spent in the coordinated state (time at risk)
                if i + 3 < len(c) and not (c[i + 1] or c[i + 2] or c[i + 3]):   # leaves and stays out >= 3 windows
                    ev += 1; first = ts[i + 1] if first is None else first
        runs.append(dict(connectome=key[0], graph=key[1], gseed=key[2], stream=key[3], run=key[4], switches=ev, first_switch_t=first,
                         coord_windows=exposure, n_windows=len(c), coord_at_end=bool(c[-1]), S_end=float(g["S_all"].fill_nan(0)[-1])))
    RU = pl.DataFrame(runs); RU.write_csv(os.path.join(OUT, "exp3_runs.csv"))
    summ = []
    for key, g in RU.filter(pl.col("stream") != "D").group_by(["connectome", "graph", "gseed"]):
        ev = int(g["switches"].sum()); expo_min = float(g["coord_windows"].sum()) * 10 / 60
        lo = chi2.ppf(0.025, 2 * ev) / 2 / expo_min if ev > 0 else 0.0; hi = chi2.ppf(0.975, 2 * ev + 2) / 2 / expo_min if expo_min > 0 else float("nan")
        tmax = float(t["t_start"].max())   # last 10-s window (290-300 s in the full run)
        end = t.filter((pl.col("connectome") == key[0]) & (pl.col("graph") == key[1]) & (pl.col("gseed") == key[2]) & (pl.col("stream") != "D") & (pl.col("t_start") == tmax))
        summ.append(dict(connectome=key[0], graph=key[1], gseed=key[2], n_runs=g.height, last_window_start=tmax, mean_S_last=float(end["S_all"].fill_nan(0).mean()),
                         frac_coord_last=float(end["coord"].mean()), switch_events=ev, exposure_min=expo_min,
                         rate_per_run_min=ev / expo_min if expo_min else float("nan"), rate_ci_lo=lo, rate_ci_hi=hi))
    SU = pl.DataFrame(summ).sort("connectome", "graph", "gseed"); SU.write_csv(os.path.join(OUT, "exp3_summary.csv"))
    traj = t.filter(pl.col("stream") != "D").group_by("connectome", "graph", "gseed", "t_start").agg(pl.col("S_all").fill_nan(0).mean().alias("mean_S"), pl.col("coord").mean().alias("frac_coord")).sort("connectome", "graph", "gseed", "t_start")
    traj.write_csv(os.path.join(OUT, "exp3_ensemble_trajectory.csv"))
    # consistency: stream A first 60 s vs archive
    A = pl.concat([pl.read_parquet(ARCH + f"heldout_stat_{tg}.parquet").filter(pl.col("mode") == "stochastic").with_columns(pl.lit(tg).alias("connectome")) for tg in ["theta50", "manc50"]])
    jA = L.filter((pl.col("stream") == "A") & pl.col("window").str.starts_with("w")).join(A.select("connectome", "graph", "gseed", "run", "window", pl.col("gmean").alias("arch")), on=["connectome", "graph", "gseed", "run", "window"])
    print("exp3 consistency (stream A vs archive): rows", jA.height, "max|d|", (jA["S_all"] - jA["arch"]).abs().max())
    print(SU)
    print(RU.filter(pl.col("switches") > 0).select("connectome", "graph", "gseed", "stream", "run", "switches", "first_switch_t", "coord_at_end", "S_end"))
