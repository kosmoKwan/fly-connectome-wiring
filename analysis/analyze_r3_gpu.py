"""Round 3 analysis of the GPU experiments A (relu_sat), B (25 networks per competitive family), C (rescue),
D2 (strength-capped reassignment) and F (sign errors). Written before any of these results were looked at.
Pre-registration: NAS round3_20260929/预注册_第三轮_20260929.md (+ addendum 1). Conventions copied from round 2:
network value = mean of S_all over the 20 stochastic runs (undefined runs excluded), graphs without an eligible setting
(mode "none") dropped, level() identical to analyze_exp1.py, condition 6 identical to analyze_exp2_3.py.
Each section runs only if its result files exist.
usage: analyze_r3_gpu.py ROUND3_RESULTS_DIR OUT_DIR
"""
import sys, os, json
import numpy as np, polars as pl

R3, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
R2 = "/Volumes/ninishares/Experiments/fly-connectome-research-20260926/round2_20260927/results/"
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]
WINS = ["w15_25", "w50_60"]
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
verdicts = {}


def level(r, fams):   # identical to analyze_exp1.py (§3.23)
    gt = {f: bool((r > v).all()) for f, v in fams.items()}
    med = {f: float(np.median(v)) for f, v in fams.items()}
    if all(gt.values()) and all(r - m >= 0.15 for m in med.values()):
        return "STRONG"
    if sum(gt.values()) >= 5 and all(r > m for m in med.values()):
        return "MODERATE"
    return "WEAK"


def gmeans(S):   # identical to analyze_exp2_3.py
    st = S.filter((pl.col("mode") == "stochastic"))
    return st.group_by("graph", "gseed", "rule", "window").agg(pl.col("S_all").fill_nan(None).mean(), pl.col("rhythm").mean(), pl.col("osc_frac").mean(), pl.col("n_gated_all").mean())


def gated_frac(P):
    pa = P.filter((pl.col("mode") == "stochastic") & (pl.col("readout") == "all"))
    return pa.group_by("graph", "gseed", "rule", "window", "pair").agg(pl.col("gated").mean().alias("fg"))


def load_scalar(path):
    return pl.read_parquet(path).filter(pl.col("mode") != "none") if os.path.exists(path) else None


def r2_main(tag):
    return gmeans(load_scalar(R2 + f"pairs_scalar_{tag}.parquet").filter(pl.col("rule") == "own"))


def val(G, graph, seed, w):
    x = G.filter((pl.col("graph") == graph) & (pl.col("gseed") == seed) & (pl.col("window") == w))
    return float(x["S_all"][0]) if x.height else np.nan


def fam_vals(G, fam, w, seeds=None):
    x = G.filter((pl.col("graph") == fam) & (pl.col("window") == w))
    if seeds is not None:
        x = x.filter(pl.col("gseed").is_in(seeds))
    return x["S_all"].drop_nulls().drop_nans().to_numpy()


rows = []
# ---------------- A: relu_sat ----------------
for tag in ["theta50", "manc50"]:
    S = load_scalar(os.path.join(R3, f"pairs_scalar_{tag}_relu.parquet"))
    if S is None:
        continue
    G = gmeans(S)
    scan = pl.read_parquet(os.path.join(R3, f"nullscan_sets_relu_{tag}.parquet"))
    rmax = scan.filter(pl.col("rhythmic_stat")).group_by("graph", "seed").agg(pl.col("rhythm_stat").max().alias("rmax"))
    real_rmax = float(rmax.filter(pl.col("graph") == "real")["rmax"][0]) if rmax.filter(pl.col("graph") == "real").height else np.nan
    n_more_rhythmic = int(rmax.filter(pl.col("graph").is_in(MAIN) & (pl.col("rmax") > real_rmax)).height)   # paper's rule: maxR(rewired) > maxR(real), eligible settings
    n_null_elig = int(rmax.filter(pl.col("graph").is_in(MAIN)).height)
    for w in WINS:
        r = val(G, "real", 0, w); fams = {f: fam_vals(G, f, w) for f in MAIN}; fams = {f: v for f, v in fams.items() if len(v)}
        allv = np.concatenate(list(fams.values())) if fams else np.array([])
        strongest = max(fams, key=lambda f: fams[f].max()) if fams else None
        rows.append(dict(exp="A_relu_sat", connectome=tag, window=w, real=r, level=level(r, fams) if len(fams) == 6 else f"n/a ({len(fams)} families)",
                         families_all_below=sum(bool((r > v).all()) for v in fams.values()), strongest_null=float(allv.max()) if len(allv) else np.nan,
                         strongest_family=strongest, min_margin_over_family_median=float(min(r - np.median(v) for v in fams.values())) if fams else np.nan,
                         n_nulls=int(len(allv)), real_scan_max_rhythm=real_rmax, n_nulls_more_rhythmic=n_more_rhythmic, n_nulls_with_eligible_setting=n_null_elig))
# ---------------- B: 25 networks for hl, degree, leg_hl ----------------
for tag in ["theta50", "manc50"]:
    S = load_scalar(os.path.join(R3, f"pairs_scalar_{tag}_n25.parquet"))
    if S is None:
        continue
    G3 = gmeans(S); G2 = r2_main(tag)
    for w in WINS:
        r = val(G2, "real", 0, w); fams, pvals = {}, {}
        for f in MAIN:
            v = fam_vals(G2, f, w, seeds=[0, 1, 2, 3, 4])
            if f in ("hl", "degree", "leg_hl"):
                v = np.concatenate([v, fam_vals(G3, f, w, seeds=list(range(5, 25)))])
            fams[f] = v; pvals[f] = (1 + int((v >= r).sum())) / (len(v) + 1)
        allv = np.concatenate(list(fams.values()))
        rows.append(dict(exp="B_n25", connectome=tag, window=w, real=r, level=level(r, fams), families_all_below=sum(bool((r > v).all()) for v in fams.values()),
                         strongest_null=float(allv.max()), strongest_family=max(fams, key=lambda f: fams[f].max()),
                         min_margin_over_family_median=float(min(r - np.median(v) for v in fams.values())), n_nulls=int(len(allv)),
                         n_per_family=json.dumps({f: int(len(v)) for f, v in fams.items()}), empirical_p=json.dumps({f: round(p, 4) for f, p in pvals.items()}),
                         n_nulls_at_or_above_real=int((allv >= r).sum())))
# ---------------- C: rescue ----------------
PAR = {"theta50": ["hl", "degree"], "manc50": ["hl", "leg_hl"]}
crow = []
for tag in ["theta50", "manc50"]:
    S = load_scalar(os.path.join(R3, f"pairs_scalar_{tag}_rescue.parquet"))
    if S is None:
        continue
    G3 = gmeans(S); G2 = r2_main(tag)
    for w in WINS:
        sr = val(G2, "real", 0, w)
        per = []
        for arm in ["C1", "C1p", "C0"]:
            for fam in PAR[tag]:
                for k in range(5):
                    sp_, sa = val(G2, fam, k, w), val(G3, f"{arm}_{fam}", k, w)
                    per.append(dict(connectome=tag, window=w, arm=arm, parent=f"{fam}_s{k}", S_parent=sp_, S_arm=sa, S_real=sr,
                                    RF=(sa - sp_) / (sr - sp_) if np.isfinite(sa) and np.isfinite(sp_) else np.nan))
        P = pl.DataFrame(per); crow.append(P)
        summ = {}
        for arm in ["C1", "C1p", "C0"]:
            x = P.filter(pl.col("arm") == arm)
            rf = float(x["RF"].drop_nans().drop_nulls().median()); nab = int((x["S_arm"] > x["S_parent"]).sum())
            cat = "恢复" if (rf >= 0.5 and nab >= 8) else ("部分恢复" if rf >= 0.2 else "不恢复")
            summ[arm] = dict(median_RF=rf, n_above_parent=nab, median_S_arm=float(x["S_arm"].median()), category=cat)
        a = P.filter(pl.col("arm") == "C1p").sort("parent")["S_arm"].to_numpy(); b = P.filter(pl.col("arm") == "C0").sort("parent")["S_arm"].to_numpy()
        n_c1p_gt_c0 = int((a > b).sum())
        attrib = bool(summ["C0"]["median_RF"] < 0.2 and n_c1p_gt_c0 >= 8)
        for arm, d in summ.items():
            rows.append(dict(exp=f"C_rescue_{arm}", connectome=tag, window=w, real=sr, level=d["category"], median_RF=d["median_RF"], n_above_parent=d["n_above_parent"],
                             median_S_arm=d["median_S_arm"], n_C1p_gt_C0=n_c1p_gt_c0, attributable_to_allocation=attrib))
if crow:
    pl.concat(crow).write_csv(os.path.join(OUT, "C_rescue_per_network.csv"))
# ---------------- D2: strength-capped reassignment at the real setting ----------------
for tag in ["theta50", "manc50"]:
    S = load_scalar(os.path.join(R3, f"pairs_scalar_{tag}_cap.parquet"))
    if S is None:
        continue
    G3 = gmeans(S); G2 = r2_main(tag)
    Gs = gmeans(load_scalar(R2 + f"pairs_scalar_{tag}_surg2.parquet"))
    F2 = gated_frac(pl.read_parquet(R2 + f"pairs_pairs_{tag}.parquet")); F3 = gated_frac(pl.read_parquet(os.path.join(R3, f"pairs_pairs_{tag}_cap.parquet")))
    for w in WINS:
        R = G2.filter((pl.col("graph") == "real") & (pl.col("window") == w)).row(0, named=True); r = R["S_all"]
        X = G3.filter((pl.col("graph") == "mn_r3_cap") & (pl.col("window") == w)).sort("gseed")
        W = Gs.filter((pl.col("graph") == "mn_r2_within") & (pl.col("rule") == "real_setting") & (pl.col("window") == w)).sort("gseed")
        xv, wv = X["S_all"].to_numpy(), W["S_all"].to_numpy(); Xm, Wm = float(np.median(xv)), float(np.median(wv))
        contrib = set(F2.filter((pl.col("graph") == "real") & (pl.col("rule") == "own") & (pl.col("window") == w) & (pl.col("fg") >= 0.5))["pair"].to_list())
        ret = [len(contrib & set(F3.filter((pl.col("gseed") == s) & (pl.col("window") == w) & (pl.col("fg") >= 0.5))["pair"].to_list())) / max(len(contrib), 1) for s in range(5)]
        oscr = float(np.median(X["osc_frac"].to_numpy())) / R["osc_frac"]
        Xrh, Wrh = float(np.median(X["rhythm"].to_numpy())), float(np.median(W["rhythm"].to_numpy()))
        c = dict(C1=Wm >= 0.8 * r, C2=(Xm <= 0.5 * r) and (r - Xm >= 0.10), C3=(Xm <= 0.5 * Wm) and (Wm - Xm >= 0.10),
                 C4=bool((xv < r).all() and xv.max() < wv.min()), C5=(Xrh >= 0.8 * R["rhythm"]) and (Xrh >= 0.8 * Wrh), C6=(float(np.median(ret)) >= 0.8) and (oscr >= 0.8))
        rows.append(dict(exp="D2_cap", connectome=tag, window=w, real=r, level="成立" if (c["C2"] and c["C3"] and c["C6"]) else "不成立", X_median=Xm, W_median=Wm,
                         X_graphs=",".join(f"{v:.3f}" for v in xv), pair_retention=float(np.median(ret)), osc_retention=oscr, X_rhythm=Xrh,
                         **{k: bool(v) for k, v in c.items()}))
# ---------------- F: sign errors ----------------
var_file = os.path.join(R3, "signerr_variants.txt")
if os.path.exists(var_file):
    frows = []
    for line in open(var_file):
        p = line.split()
        if len(p) < 4:
            continue
        tag, ntag, f1, f2 = p
        S = load_scalar(os.path.join(R3, f"pairs_scalar_{tag}_{ntag}.parquet"))
        if S is None:
            continue
        G = gmeans(S)
        for w in WINS:
            r = val(G, "real", 0, w); nv = np.concatenate([fam_vals(G, f1, w), fam_vals(G, f2, w)])
            frows.append(dict(connectome=tag, variant=ntag, kind="glu25" if "glu25" in ntag else "se9", window=w, real=r, null_max=float(nv.max()) if len(nv) else np.nan,
                              n_nulls=int(len(nv)), real_above_all=bool(np.isfinite(r) and len(nv) == 10 and (r > nv).all())))
    if frows:
        FR = pl.DataFrame(frows); FR.write_csv(os.path.join(OUT, "F_signerr_per_variant.csv"))
        for (tag, kind, w), x in FR.group_by(["connectome", "kind", "window"]):
            npass = int(x["real_above_all"].sum())
            rows.append(dict(exp=f"F_{kind}", connectome=tag, window=w, real=float(x["real"].median()), level=("稳健" if npass >= 2 else "不稳健") + f" ({npass}/{x.height})",
                             strongest_null=float(x["null_max"].max())))
if rows:
    T = pl.DataFrame(rows, infer_schema_length=None).sort("exp", "connectome", "window")
    T.write_csv(os.path.join(OUT, "round3_decisions.csv")); print(T)
