"""Round 2, experiment 1 analysis (pre-registered in round2_20260927/预注册_第二轮_20260927.md).

Inputs (local copies): pairs_scalar_<tag>.parquet, pairs_pairs_<tag>.parquet, archive heldout_stat_<tag>.parquet.
Outputs (to OUT dir): consistency, claim levels per readout (§3.23), per-joint table, surgery C6 by pair identity,
pair-level losses; plus a short markdown summary.
usage: analyze_exp1.py RESULTS_DIR OUT_DIR [PREFIX]
"""
import sys, os
import numpy as np, polars as pl

RES, OUT = sys.argv[1], sys.argv[2]; PRE = sys.argv[3] if len(sys.argv) > 3 else ""
ARCH = "/Volumes/ninishares/Experiments/fly-connectome-research-20260926/final_heldout_20260927_121507_B9ue3i/results/"
os.makedirs(OUT, exist_ok=True)
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]
READ = ["S_all", "S_exact", "S_nonThC", "S_ThC", "S_CTr", "S_FTi", "S_TiTa"]
DECISION = ["S_exact", "S_nonThC"]
WINS = ["w15_25", "w25_35", "w35_45", "w50_60"]


def level(r, fams):
    """§3.23: fams = {family: array of graph means}."""
    gt = {f: bool((r > v).all()) for f, v in fams.items()}
    med = {f: float(np.median(v)) for f, v in fams.items()}
    if all(gt.values()) and all(r - m >= 0.15 for m in med.values()):
        return "STRONG"
    if sum(gt.values()) >= 5 and all(r > m for m in med.values()):
        return "MODERATE"
    return "WEAK"


md = ["# 实验 1 结果（自动生成，判定规则见预注册）", ""]
cons, claims, c6rows, pairloss = [], [], [], []
for tag in ["theta50", "manc50"]:
    fs = os.path.join(RES, f"{PRE}pairs_scalar_{tag}.parquet")
    if not os.path.exists(fs):
        continue
    S = pl.read_parquet(fs).filter(pl.col("mode") != "none")
    P = pl.read_parquet(os.path.join(RES, f"{PRE}pairs_pairs_{tag}.parquet"))
    # --- consistency with the archived 60-s confirmation (same trajectories)
    A = pl.read_parquet(ARCH + f"heldout_stat_{tag}.parquet").select("graph", "gseed", "rule", "mode", "run", "window", pl.col("gmean").alias("arch"))
    j = S.join(A, on=["graph", "gseed", "rule", "mode", "run", "window"], how="left")
    jj = j.with_columns((pl.col("S_all").fill_nan(None) - pl.col("arch").fill_nan(None)).abs().alias("d"))
    g = jj.filter(pl.col("mode") == "stochastic").group_by("graph", "gseed", "rule", "window").agg(
        (pl.col("S_all").fill_nan(None).mean() - pl.col("arch").fill_nan(None).mean()).abs().alias("dmean"))
    cons.append(dict(connectome=tag, rows=S.height, matched=int(jj["arch"].is_not_null().sum()), max_abs_run_diff=float(jj["d"].max() or 0),
                     max_abs_graph_mean_diff=float(g["dmean"].max() or 0), graphs_flagged=int((g["dmean"] > 0.01).sum())))
    # --- graph-level stochastic means (sub-readouts are always defined; S_all uses finite values like the archive)
    st = S.filter(pl.col("mode") == "stochastic")
    gm = st.group_by("graph", "gseed", "rule", "window").agg([pl.col(r).fill_nan(None).mean().alias(r) for r in READ] +
                                                             [pl.col("n_gated_all").mean(), pl.col("n_gated_exact").mean(), pl.col("osc_frac").mean(), pl.col("rhythm").mean()])
    gm.write_csv(os.path.join(OUT, f"exp1_graph_means_{tag}.csv"))
    for w in WINS:
        x = gm.filter(pl.col("window") == w)
        real = x.filter((pl.col("graph") == "real"))
        if real.height == 0:
            continue
        for rd in READ:
            r = float(real[rd][0])
            fams = {f: x.filter((pl.col("graph") == f) & (pl.col("rule") == "own"))[rd].to_numpy() for f in MAIN}
            fams = {f: v for f, v in fams.items() if len(v)}
            if len(fams) < 6:
                continue
            best_f = max(fams, key=lambda f: fams[f].max())
            claims.append(dict(connectome=tag, window=w, readout=rd, decision=rd in DECISION, real=r, level=level(r, fams),
                               families_all_below=sum(bool((r > v).all()) for v in fams.values()),
                               min_margin_over_median=min(r - float(np.median(v)) for v in fams.values()),
                               strongest_null=float(fams[best_f].max()), strongest_family=best_f, real_minus_strongest=r - float(fams[best_f].max())))
    # --- C6 by pair identity at the real frozen setting (real graph rule 'own' IS the real setting)
    pa = P.filter((pl.col("mode") == "stochastic") & (pl.col("readout") == "all"))
    frac = pa.group_by("graph", "gseed", "rule", "window", "pair").agg(pl.col("gated").mean().alias("fg"),
                                                                        pl.when(pl.col("gated")).then(-pl.col("corr")).otherwise(0.0).mean().alias("contrib"))
    for w in WINS:
        fr = frac.filter((pl.col("graph") == "real") & (pl.col("window") == w))
        if fr.height == 0:
            continue
        contrib = set(fr.filter(pl.col("fg") >= 0.5)["pair"].to_list())
        real_osc = float(gm.filter((pl.col("graph") == "real") & (pl.col("window") == w))["osc_frac"][0])
        for fam in ["mn_shuffle_joint", "mn_shuffle_pool", "mn_shuffle_joint_w", "mn_shuffle_pool_w"]:
            ret, oscr = [], []
            for s in range(5):
                q = frac.filter((pl.col("graph") == fam) & (pl.col("gseed") == s) & (pl.col("rule") == "real_setting") & (pl.col("window") == w))
                if q.height == 0:
                    continue
                kept = set(q.filter(pl.col("fg") >= 0.5)["pair"].to_list())
                ret.append(len(contrib & kept) / max(len(contrib), 1))
                oscr.append(float(gm.filter((pl.col("graph") == fam) & (pl.col("gseed") == s) & (pl.col("rule") == "real_setting") & (pl.col("window") == w))["osc_frac"][0]) / real_osc)
            if ret:
                c6rows.append(dict(connectome=tag, window=w, surgery=fam, n_contributing_pairs=len(contrib), contributing_pairs=",".join(sorted(contrib)),
                                   pair_retention_median=float(np.median(ret)), osc_retention_median=float(np.median(oscr)),
                                   C6_pair_identity=bool(np.median(ret) >= 0.8 and np.median(oscr) >= 0.8)))
        # pair-level contribution: real vs surgery family medians (descriptive)
        base = fr.select("pair", pl.col("contrib").alias("real_contrib"), pl.col("fg").alias("real_gated_frac"))
        for fam in ["mn_shuffle_joint", "mn_shuffle_joint_w", "mn_shuffle_pool_w"]:
            q = frac.filter((pl.col("graph") == fam) & (pl.col("rule") == "real_setting") & (pl.col("window") == w)).group_by("pair").agg(pl.col("contrib").median().alias("fam_contrib"))
            b = base.join(q, on="pair", how="left").with_columns((pl.col("real_contrib") - pl.col("fam_contrib")).alias("loss"))
            for row in b.iter_rows(named=True):
                pairloss.append(dict(connectome=tag, window=w, surgery=fam, **row))
    # real vs null per pair (descriptive): which pairs carry the real-graph advantage
    fo = frac.filter((pl.col("rule") == "own") & pl.col("window").is_in(["w15_25", "w50_60"]))
    rp = fo.filter(pl.col("graph") == "real").select("window", "pair", pl.col("contrib").alias("real"))
    npair = fo.filter(pl.col("graph").is_in(MAIN)).group_by("window", "pair").agg(pl.col("contrib").max().alias("null_max"), pl.col("contrib").median().alias("null_median"))
    rp.join(npair, on=["window", "pair"]).with_columns((pl.col("real") - pl.col("null_max")).alias("real_minus_nullmax")).sort("window", "pair").write_csv(os.path.join(OUT, f"exp1_pair_real_vs_nulls_{tag}.csv"))

pl.DataFrame(cons).write_csv(os.path.join(OUT, "exp1_consistency.csv"))
C = pl.DataFrame(claims); C.write_csv(os.path.join(OUT, "exp1_claim_levels.csv"))
if c6rows:
    pl.DataFrame(c6rows).write_csv(os.path.join(OUT, "exp1_C6_pair_identity.csv"))
if pairloss:
    pl.DataFrame(pairloss).write_csv(os.path.join(OUT, "exp1_pair_losses.csv"))
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
print(pl.DataFrame(cons))
if C.height:
    print(C.filter(pl.col("window").is_in(["w15_25", "w50_60"])).select("connectome", "window", "readout", "decision", "real", "level", "families_all_below", "min_margin_over_median", "strongest_null", "strongest_family", "real_minus_strongest"))
if c6rows:
    print(pl.DataFrame(c6rows).filter(pl.col("window").is_in(["w15_25", "w50_60"])).drop("contributing_pairs"))
