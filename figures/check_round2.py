"""Independent recomputation of the round-2 results from the raw parquet files, checked against
(1) the pre-registered analysis tables in round2_20260927/analysis/ and (2) the reference numbers of the update request.

Rules are re-implemented from round2_20260927/src/analyze_exp1.py and analyze_exp2_3.py (pandas instead of polars):
  exp 1  graph means over the 20 stochastic runs (S_all skips undefined runs); §3.23 levels on the six main families;
         C6 by pair identity (pairs gated in >= 50% of the real network's runs).
  exp 2  R from the exp-1 file, W and X arms from *_surg2; C1-C6 exactly as pre-registered.
  exp 3  coordinated window: S_all (NaN -> 0) >= 0.10 and >= 6 gated pairs; a switch = leaving the coordinated state and
         staying out >= 3 windows; rate = switches / minutes spent coordinated (streams A+B), exact Poisson 95% CI.
Writes qa/round2_check.csv and derived/r2_*.csv (tables reused by the figures). Exits with an error on any mismatch.
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import chi2
import common as C

RES = C.R2 / "results"; AN = C.R2 / "analysis"
TAGS = ["theta50", "manc50"]; WINS = C.WINDOWS
READ = ["S_all", "S_exact", "S_nonThC", "S_ThC", "S_CTr", "S_FTi", "S_TiTa"]
rows = []


def check(section, item, computed, reference, source, tol=1e-9):
    if isinstance(reference, str) or isinstance(computed, str):
        ok = str(computed) == str(reference)
    elif isinstance(reference, (bool, np.bool_)):
        ok = bool(computed) == bool(reference)
    else:
        ok = (np.isnan(computed) and np.isnan(reference)) or abs(float(computed) - float(reference)) <= tol
    rows.append(dict(section=section, item=item, computed=computed, reference=reference, reference_source=source, match=bool(ok)))


def level(r, fams):
    gt = {f: bool((r > v).all()) for f, v in fams.items()}; med = {f: float(np.median(v)) for f, v in fams.items()}
    if all(gt.values()) and all(r - m >= 0.15 for m in med.values()):
        return "STRONG"
    if sum(gt.values()) >= 5 and all(r > m for m in med.values()):
        return "MODERATE"
    return "WEAK"


def gmeans(S):
    st = S[S["mode"] == "stochastic"]
    g = st.groupby(["graph", "gseed", "rule", "window"])
    out = g[READ].mean()                                   # pandas mean skips NaN, as polars mean over fill_nan(None)
    out["rhythm"] = g["rhythm"].mean(); out["osc_frac"] = g["osc_frac"].mean(); out["n_gated_all"] = g["n_gated_all"].mean()
    sem = g[READ].agg(lambda v: v.std(ddof=1) / np.sqrt(v.notna().sum())); sem.columns = [c + "_sem" for c in READ]
    return out.join(sem).reset_index()


def gated_frac(P):
    pa = P[(P["mode"] == "stochastic") & (P.readout == "all")].copy()
    pa["contrib"] = np.where(pa.gated, -pa["corr"], 0.0)
    return pa.groupby(["graph", "gseed", "rule", "window", "pair"]).agg(fg=("gated", "mean"), contrib=("contrib", "mean"),
                                                                        corr=("corr", "mean")).reset_index()


# ============================================================================== exp 1
E1, F1 = {}, {}
cl = pd.read_csv(AN / "exp1_claim_levels.csv")
c6 = pd.read_csv(AN / "exp1_C6_pair_identity.csv")
lev_rows = []
for tag in TAGS:
    S = pd.read_parquet(RES / f"pairs_scalar_{tag}.parquet"); S = S[S["mode"] != "none"]
    P = pd.read_parquet(RES / f"pairs_pairs_{tag}.parquet")
    gm = gmeans(S); fr = gated_frac(P); E1[tag] = gm; F1[tag] = fr
    gm.to_csv(C.DERIVED / f"r2_exp1_graph_means_{tag}.csv", index=False); fr.to_csv(C.DERIVED / f"r2_exp1_pair_scores_{tag}.csv", index=False)
    for w in WINS:
        x = gm[gm.window == w]; real = x[x.graph == "real"].iloc[0]
        for rd in READ:
            fams = {f: x[(x.graph == f) & (x.rule == "own")][rd].to_numpy() for f in C.MAIN}
            lv = level(float(real[rd]), fams)
            best_f = max(fams, key=lambda f: fams[f].max())
            ref = cl[(cl.connectome == tag) & (cl.window == w) & (cl.readout == rd)].iloc[0]
            check("exp1", f"{tag} {w} {rd} real", float(real[rd]), float(ref.real), "exp1_claim_levels.csv")
            check("exp1", f"{tag} {w} {rd} level", lv, ref.level, "exp1_claim_levels.csv")
            check("exp1", f"{tag} {w} {rd} strongest rewired", float(fams[best_f].max()), float(ref.strongest_null), "exp1_claim_levels.csv")
            lev_rows.append(dict(connectome=tag, window=w, readout=rd, real=float(real[rd]), real_sem=float(real[rd + "_sem"]), level=lv,
                                 strongest_rewired=float(fams[best_f].max()), strongest_family=best_f,
                                 families_all_below=sum(bool((float(real[rd]) > v).all()) for v in fams.values()),
                                 min_margin_over_median=min(float(real[rd]) - float(np.median(v)) for v in fams.values())))
        fw = fr[(fr.graph == "real") & (fr.window == w)]
        contrib = set(fw[fw.fg >= 0.5].pair)
        real_osc = float(x[x.graph == "real"].osc_frac.iloc[0])
        for fam in ["mn_shuffle_joint", "mn_shuffle_pool", "mn_shuffle_joint_w", "mn_shuffle_pool_w"]:
            ret, oscr = [], []
            for s in range(5):
                q = fr[(fr.graph == fam) & (fr.gseed == s) & (fr.rule == "real_setting") & (fr.window == w)]
                ret.append(len(contrib & set(q[q.fg >= 0.5].pair)) / max(len(contrib), 1))
                oscr.append(float(x[(x.graph == fam) & (x.gseed == s) & (x.rule == "real_setting")].osc_frac.iloc[0]) / real_osc)
            ref = c6[(c6.connectome == tag) & (c6.window == w) & (c6.surgery == fam)].iloc[0]
            check("exp1", f"{tag} {w} {fam} pair retention", float(np.median(ret)), float(ref.pair_retention_median), "exp1_C6_pair_identity.csv")
            check("exp1", f"{tag} {w} {fam} osc retention", float(np.median(oscr)), float(ref.osc_retention_median), "exp1_C6_pair_identity.csv")
            check("exp1", f"{tag} {w} contributing pairs", ",".join(sorted(contrib)), ref.contributing_pairs, "exp1_C6_pair_identity.csv")
pd.DataFrame(lev_rows).to_csv(C.DERIVED / "r2_exp1_levels.csv", index=False)
# reference numbers of the update request (MANC, 15-25 s)
L = pd.DataFrame(lev_rows).set_index(["connectome", "window", "readout"])
for rd, val, lvl_, extra in [("S_ThC", 0.271, "STRONG", 0.105), ("S_exact", 0.122, "MODERATE", None), ("S_nonThC", 0.131, "MODERATE", None)]:
    r_ = L.loc[("manc50", "w15_25", rd)]
    check("request", f"MANC {rd} 15-25 s", round(r_.real, 3), val, "update request", tol=5e-4)
    check("request", f"MANC {rd} 15-25 s level", r_.level, lvl_, "update request")
    if extra is not None:
        check("request", f"MANC {rd} strongest rewired", round(r_.strongest_rewired, 3), extra, "update request", tol=5e-4)

# ============================================================================== exp 2
cond_ref = pd.read_csv(AN / "exp2_conditions.csv"); dose_ref = pd.read_csv(AN / "exp2_dose_response.csv")
REDIS = {}
import json
rep = json.loads((C.R2 / "experiment2_graphs/surgery_r2_report.json").read_text())
aud = C.intervention_audit()
for tag in TAGS:
    REDIS[tag] = {"mn_shuffle_joint_w": float(aud[(aud.connectome == tag) & (aud.graph == "mn_shuffle_joint_w")].pool_allocation_redistribution_fraction.median()),
                  "mn_shuffle_joint": float(aud[(aud.connectome == tag) & (aud.graph == "mn_shuffle_joint")].pool_allocation_redistribution_fraction.median())}
    for arm in ["cross_lo", "cross_hi"]:
        REDIS[tag][f"mn_r2_{arm}"] = float(np.median([v["redistr"] for k, v in rep.items() if k.startswith(f"{tag}/mn_r2_{arm}_s")]))
cond_rows, dose_rows = [], []
E2 = {}
for tag in TAGS:
    S2 = pd.read_parquet(RES / f"pairs_scalar_{tag}_surg2.parquet"); S2 = S2[S2["mode"] != "none"]
    P2 = pd.read_parquet(RES / f"pairs_pairs_{tag}_surg2.parquet")
    G = pd.concat([E1[tag], gmeans(S2)], ignore_index=True); F = pd.concat([F1[tag], gated_frac(P2)], ignore_index=True)
    E2[tag] = G
    G.to_csv(C.DERIVED / f"r2_exp2_graph_means_{tag}.csv", index=False); gated_frac(P2).to_csv(C.DERIVED / f"r2_exp2_pair_scores_{tag}.csv", index=False)
    for w in WINS:
        R = G[(G.graph == "real") & (G.window == w)].iloc[0]
        contrib = set(F[(F.graph == "real") & (F.window == w) & (F.fg >= 0.5)].pair)
        for xg, rule in [("mn_r2_cross_hi", "own"), ("mn_r2_cross_hi", "real_setting"), ("mn_r2_cross_lo", "real_setting")]:
            X = G[(G.graph == xg) & (G.rule == rule) & (G.window == w)].sort_values("gseed")
            Wd = G[(G.graph == "mn_r2_within") & (G.rule == rule) & (G.window == w)].sort_values("gseed")
            xv, wv = X.S_all.to_numpy(), Wd.S_all.to_numpy(); r = float(R.S_all)
            Xm, Wm = float(np.median(xv)), float(np.median(wv))
            Xrh, Wrh = float(np.median(X.rhythm)), float(np.median(Wd.rhythm))
            ret = []
            for s in range(5):
                kept = set(F[(F.graph == xg) & (F.gseed == s) & (F.rule == rule) & (F.window == w) & (F.fg >= 0.5)].pair)
                ret.append(len(contrib & kept) / max(len(contrib), 1))
            oscr = float(np.median(X.osc_frac)) / float(R.osc_frac)
            wosc = float(np.median(Wd.osc_frac)) / float(R.osc_frac)
            c = dict(C1=Wm >= 0.8 * r, C2=(Xm <= 0.5 * r) and (r - Xm >= 0.10), C3=(Xm <= 0.5 * Wm) and (Wm - Xm >= 0.10),
                     C4=bool((xv < r).all() and xv.max() < wv.min()), C5=(Xrh >= 0.8 * float(R.rhythm)) and (Xrh >= 0.8 * Wrh),
                     C6=(float(np.median(ret)) >= 0.8) and (oscr >= 0.8))
            row = dict(connectome=tag, window=w, arm=xg, rule=rule, R=r, W=Wm, X=Xm, X_over_R=Xm / r, W_over_R=Wm / r, pair_retention=float(np.median(ret)),
                       osc_retention=oscr, W_osc_retention=wosc, R_rhythm=float(R.rhythm), W_rhythm=Wrh, X_rhythm=Xrh, **c, all_six=all(c.values()),
                       collapse=(Xm <= 0.05 and c["C1"] and c["C5"] and c["C6"]), n_contrib=len(contrib))
            cond_rows.append(row)
            ref = cond_ref[(cond_ref.connectome == tag) & (cond_ref.window == w) & (cond_ref.arm == xg) & (cond_ref.rule == rule)].iloc[0]
            for k in ["R", "W", "X", "X_over_R", "pair_retention", "osc_retention", "R_rhythm", "W_rhythm", "X_rhythm"]:
                check("exp2", f"{tag} {w} {xg} {rule} {k}", row[k], float(ref[k]), "exp2_conditions.csv")
            for k in ["C1", "C2", "C3", "C4", "C5", "C6", "all_six", "collapse"]:
                check("exp2", f"{tag} {w} {xg} {rule} {k}", bool(row[k]), bool(ref[k]), "exp2_conditions.csv")
        for xg in ["mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi", "mn_shuffle_joint"]:
            X = G[(G.graph == xg) & (G.rule == "real_setting") & (G.window == w)]
            xm = float(X.S_all.median()); dose_rows.append(dict(connectome=tag, window=w, arm=xg, redistribution=REDIS[tag][xg], X_median=xm,
                                                                   X_over_R=xm / float(R.S_all)))
            ref = dose_ref[(dose_ref.connectome == tag) & (dose_ref.window == w) & (dose_ref.arm == xg)].iloc[0]
            check("exp2", f"{tag} {w} {xg} X/R", xm / float(R.S_all), float(ref.X_over_R), "exp2_dose_response.csv")
            check("exp2", f"{tag} {w} {xg} redistribution (3 dp)", round(REDIS[tag][xg], 3), float(ref.redistribution), "exp2_dose_response.csv", tol=5e-4)
CD = pd.DataFrame(cond_rows); CD.to_csv(C.DERIVED / "r2_exp2_conditions.csv", index=False)
DR = pd.DataFrame(dose_rows); DR.to_csv(C.DERIVED / "r2_exp2_dose.csv", index=False)
for tag, R_, W_, X_ in [("theta50", 0.312, 0.277, -0.076), ("manc50", 0.173, 0.135, -0.096)]:
    r_ = CD[(CD.connectome == tag) & (CD.window == "w15_25") & (CD.arm == "mn_r2_cross_hi") & (CD.rule == "real_setting")].iloc[0]
    for k, v in [("R", R_), ("W", W_), ("X", X_)]:
        check("request", f"exp2 {tag} 15-25 s {k}", round(r_[k], 3), v, "update request", tol=5e-4)
    if tag == "theta50":
        check("request", "exp2 MaleCNS all six conditions", bool(r_.all_six), True, "update request")
    else:
        check("request", "exp2 MANC condition 6 fails", bool(r_.C6), False, "update request")
        check("request", "exp2 MANC oscillating coverage", round(r_.osc_retention, 2), 0.69, "update request", tol=5e-3)
for tag, vals in [("theta50", (0.55, 0.13, -0.24, -0.23)), ("manc50", (0.54, 0.14, -0.56, -0.32))]:
    for arm, v in zip(["mn_shuffle_joint_w", "mn_r2_cross_lo", "mn_r2_cross_hi", "mn_shuffle_joint"], vals):
        x = DR[(DR.connectome == tag) & (DR.window == "w15_25") & (DR.arm == arm)].X_over_R.iloc[0]
        check("request", f"dose {tag} {arm} X/R", round(x, 2), v, "update request", tol=5e-3)

# ============================================================================== exp 3
Lg = pd.read_parquet(RES / "long_scalar.parquet")
t = Lg[Lg.window.str.startswith("t")].copy()
t["S0"] = t.S_all.fillna(0.0)
t["coord"] = (t.S0 >= 0.10) & (t.n_gated_all >= 6)
t = t.sort_values(["connectome", "graph", "gseed", "stream", "run", "t_start"])
t.to_csv(C.DERIVED / "r2_exp3_windows.csv", index=False,
         columns=["connectome", "graph", "gseed", "stream", "run", "t_start", "S_all", "S0", "n_gated_all", "freq", "rhythm", "osc_frac", "coord"])
run_rows = []
for key, g in t.groupby(["connectome", "graph", "gseed", "stream", "run"]):
    c = g.coord.to_list(); ts = g.t_start.to_list(); ev, first, expo = 0, None, 0
    for i in range(len(c)):
        if c[i]:
            expo += 1
            if i + 3 < len(c) and not (c[i + 1] or c[i + 2] or c[i + 3]):
                ev += 1; first = ts[i + 1] if first is None else first
    run_rows.append(dict(connectome=key[0], graph=key[1], gseed=key[2], stream=key[3], run=key[4], switches=ev, first_switch_t=first,
                         coord_windows=expo, n_windows=len(c), coord_at_end=bool(c[-1]), S_end=float(g.S0.iloc[-1])))
RU = pd.DataFrame(run_rows); RU.to_csv(C.DERIVED / "r2_exp3_runs.csv", index=False)
ru_ref = pd.read_csv(AN / "exp3_runs.csv")
mm = RU.merge(ru_ref, on=["connectome", "graph", "gseed", "stream", "run"], suffixes=("", "_ref"))
check("exp3", "runs matched", len(mm), len(ru_ref), "exp3_runs.csv", tol=0)
for k in ["switches", "coord_windows", "n_windows"]:
    check("exp3", f"per-run {k} identical", int((mm[k] != mm[k + "_ref"]).sum()), 0, "exp3_runs.csv", tol=0)
check("exp3", "per-run S_end max |diff|", float((mm.S_end - mm.S_end_ref).abs().max()), 0.0, "exp3_runs.csv", tol=1e-12)
summ = []
tmax = t.t_start.max()
for key, g in RU[RU.stream != "D"].groupby(["connectome", "graph", "gseed"]):
    ev = int(g.switches.sum()); expo_min = float(g.coord_windows.sum()) * 10 / 60
    lo = chi2.ppf(0.025, 2 * ev) / 2 / expo_min if ev > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * ev + 2) / 2 / expo_min if expo_min > 0 else np.nan
    end = t[(t.connectome == key[0]) & (t.graph == key[1]) & (t.gseed == key[2]) & (t.stream != "D") & (t.t_start == tmax)]
    summ.append(dict(connectome=key[0], graph=key[1], gseed=key[2], n_runs=len(g), mean_S_last=float(end.S0.mean()), frac_coord_last=float(end.coord.mean()),
                     switch_events=ev, exposure_min=expo_min, rate_per_run_min=ev / expo_min if expo_min else np.nan, rate_ci_lo=lo, rate_ci_hi=hi,
                     mean_dwell_min=(expo_min / ev) if ev else np.nan))
SU = pd.DataFrame(summ); SU.to_csv(C.DERIVED / "r2_exp3_summary.csv", index=False)
su_ref = pd.read_csv(AN / "exp3_summary.csv")
for _, r in SU.iterrows():
    ref = su_ref[(su_ref.connectome == r.connectome) & (su_ref.graph == r.graph) & (su_ref.gseed == r.gseed)].iloc[0]
    for k in ["mean_S_last", "frac_coord_last", "switch_events", "exposure_min", "rate_per_run_min", "rate_ci_lo", "rate_ci_hi"]:
        check("exp3", f"{r.connectome} {r.graph} s{r.gseed} {k}", float(r[k]), float(ref[k]), "exp3_summary.csv")
traj = t[t.stream != "D"].groupby(["connectome", "graph", "gseed", "t_start"]).agg(mean_S=("S0", "mean"), frac_coord=("coord", "mean"),
                                                                                  sem_S=("S0", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                                                                  n=("S0", "size")).reset_index()
traj.to_csv(C.DERIVED / "r2_exp3_trajectory.csv", index=False)
tr_ref = pd.read_csv(AN / "exp3_ensemble_trajectory.csv")
mt = traj.merge(tr_ref, on=["connectome", "graph", "gseed", "t_start"], suffixes=("", "_ref"))
check("exp3", "trajectory rows matched", len(mt), len(tr_ref), "exp3_ensemble_trajectory.csv", tol=0)
check("exp3", "trajectory mean_S max |diff|", float((mt.mean_S - mt.mean_S_ref).abs().max()), 0.0, "exp3_ensemble_trajectory.csv", tol=1e-12)
check("exp3", "trajectory frac_coord max |diff|", float((mt.frac_coord - mt.frac_coord_ref).abs().max()), 0.0, "exp3_ensemble_trajectory.csv", tol=1e-12)
for (tag, g, s), vals in {("manc50", "real", 0): (0.088, 0.45, 0.24), ("manc50", "hl", 1): (0.076, 0.10, 3.7),
                          ("manc50", "hl", 0): (0.072, 0.0, 5.6), ("theta50", "real", 0): (0.312, 1.0, 0.0)}.items():
    r = SU[(SU.connectome == tag) & (SU.graph == g) & (SU.gseed == s)].iloc[0]
    check("request", f"exp3 {tag} {g} s{s} mean S 290-300 s", round(r.mean_S_last, 3), vals[0], "update request", tol=5e-4)
    check("request", f"exp3 {tag} {g} s{s} fraction coordinated", round(r.frac_coord_last, 2), vals[1], "update request", tol=5e-3)
    check("request", f"exp3 {tag} {g} s{s} switch rate", round(r.rate_per_run_min, 2 if vals[2] < 1 else 1), vals[2], "update request", tol=5e-3 if vals[2] < 1 else 0.05)
# consistency of stream A (first 60 s) with the archived confirmation
arch = pd.concat([C.heldout(tg).assign(connectome=tg) for tg in TAGS])
arch = arch[arch["mode"] == "stochastic"]
A = Lg[(Lg.stream == "A") & Lg.window.str.startswith("w")].merge(arch[["connectome", "graph", "gseed", "run", "window", "gmean"]],
                                                                  on=["connectome", "graph", "gseed", "run", "window"])
check("exp3", "stream A vs archive rows", len(A), 320, "progress log (320 rows)", tol=0)
d = (A.S_all - A.gmean).abs()
check("exp3", "stream A vs archive max |diff|", float(d.max()), 0.0, "progress log (<= 1.1e-16)", tol=1e-12)

out = pd.DataFrame(rows)
out.to_csv(C.QA / "round2_check.csv", index=False)
bad = out[~out.match]
print(out.groupby("section").match.agg(["size", "sum"]))
if len(bad):
    print(bad.to_string())
    sys.exit(f"{len(bad)} mismatches - see qa/round2_check.csv")
print("round-2 check: all", len(out), "items match")
