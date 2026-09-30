"""Post hoc: thorax-coxa (ThC) pair audit on the round-2 pair-level confirmation runs (own settings, stochastic runs):
per ThC pair, the real network's contribution to S versus the 30 main rewired networks, for the all-MN and exact-MN readouts.
Reproduces round2_20260927/analysis/thc_audit_pairs.csv.
usage: posthoc_thc_audit.py ROUND2_RESULTS_DIR OUT_CSV
"""
import sys, os
import polars as pl
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]
out = []
for tag in ["theta50", "manc50"]:
    P = pl.read_parquet(os.path.join(sys.argv[1], f"pairs_pairs_{tag}.parquet")).filter((pl.col("mode") == "stochastic") & (pl.col("rule") == "own") & (pl.col("joint") == "ThC") & pl.col("window").is_in(["w15_25", "w50_60"]))
    c = P.with_columns(pl.when(pl.col("gated")).then(-pl.col("corr")).otherwise(0.0).alias("contrib"))
    g = c.group_by("readout", "graph", "gseed", "window", "pair").agg(pl.col("contrib").mean().alias("contrib"), pl.col("gated").mean().alias("fg"), pl.col("corr").mean().alias("corr_mean"))
    for (ro, w, pr), d in g.group_by(["readout", "window", "pair"]):
        real = d.filter(pl.col("graph") == "real")
        nul = d.filter(pl.col("graph").is_in(MAIN))
        if real.height == 0:
            continue
        fam_med = nul.group_by("graph").agg(pl.col("contrib").median()).get_column("contrib")
        out.append(dict(connectome=tag, readout=ro, window=w, pair=pr, real=float(real["contrib"][0]), real_gated=float(real["fg"][0]), real_corr=float(real["corr_mean"][0]),
                        null_max=float(nul["contrib"].max()), null_median=float(nul["contrib"].median()), max_family_median=float(fam_med.max()),
                        n_nulls_ge_real=int((nul["contrib"] >= real["contrib"][0]).sum())))
pl.DataFrame(out).sort("connectome", "readout", "window", "pair").write_csv(sys.argv[2])
