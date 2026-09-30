"""Post hoc (R1): in each network, an antagonistic pair is active when it passes the amplitude gate in >= 50% of the
20 stochastic runs; is an active pair antiphasic (mean correlation over its gated runs < 0)? Real networks versus
the 30 main rewired networks pooled. Reproduces round2_20260927/analysis/r1_active_pairs_{real,nulls}_exploratory.csv.
usage: posthoc_r1_active_pairs.py ROUND2_RESULTS_DIR OUT_DIR
"""
import sys, os
import polars as pl
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]
rows, nullrows = [], []
for tag in ["theta50", "manc50"]:
    for w in ["w15_25", "w50_60"]:
        P = pl.read_parquet(os.path.join(sys.argv[1], f"pairs_pairs_{tag}.parquet")).filter((pl.col("mode") == "stochastic") & (pl.col("rule") == "own") & (pl.col("readout") == "all") & (pl.col("window") == w))
        g = P.group_by("graph", "gseed", "pair", "joint").agg(pl.col("gated").mean().alias("frac_runs_active"), pl.col("corr").filter(pl.col("gated")).mean().alias("mean_corr_when_active"))
        real = g.filter((pl.col("graph") == "real") & (pl.col("frac_runs_active") >= 0.5))
        for r in real.iter_rows(named=True):
            rows.append(dict(connectome=tag, window=w, pair=r["pair"], joint=r["joint"], frac_runs_active=r["frac_runs_active"], mean_corr_when_active=r["mean_corr_when_active"], antiphase=r["mean_corr_when_active"] < 0))
        nul = g.filter(pl.col("graph").is_in(MAIN) & (pl.col("frac_runs_active") >= 0.5))
        nullrows.append(dict(connectome=tag, window=w, graph_set="30 main nulls", n_active_pairs=nul.height, n_antiphase=int((nul["mean_corr_when_active"] < 0).sum()),
                             frac_antiphase=float((nul["mean_corr_when_active"] < 0).mean()), median_corr=float(nul["mean_corr_when_active"].median())))
pl.DataFrame(rows).sort("connectome", "window", "pair").write_csv(os.path.join(sys.argv[2], "r1_active_pairs_real_exploratory.csv"))
pl.DataFrame(nullrows).write_csv(os.path.join(sys.argv[2], "r1_active_pairs_nulls_exploratory.csv"))
