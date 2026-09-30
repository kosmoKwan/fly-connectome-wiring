"""Post hoc (descriptive, not pre-registered): bootstrap of the difference in mean S at 290-300 s between the real MANC
network and its two strongest rewired networks (streams A+B, 40 runs per graph; undefined S counted as 0).
Reproduces round2_20260927/analysis/exp3_posthoc_bootstrap.txt.
usage: posthoc_exp3_bootstrap.py ROUND2_RESULTS_DIR > exp3_posthoc_bootstrap.txt
"""
import sys, os
import polars as pl, numpy as np
print("POST HOC, descriptive only (not pre-registered). 290-300 s window, streams A+B (40 runs per graph), S_all NaN->0.")
L = pl.read_parquet(os.path.join(sys.argv[1], "long_scalar.parquet"))
tmax = L.filter(pl.col("window").str.starts_with("t"))["t_start"].max()
last = L.filter((pl.col("t_start") == tmax) & (pl.col("stream") != "D") & (pl.col("connectome") == "manc50"))
rng = np.random.default_rng(0)
g = {k: last.filter((pl.col("graph") == k[0]) & (pl.col("gseed") == k[1]))["S_all"].fill_nan(0).to_numpy() for k in [("real", 0), ("hl", 1), ("hl", 0)]}
for k, v in g.items():
    print(k, "n", len(v), "mean", round(float(v.mean()), 4), "median", round(float(np.median(v)), 4))
for nk in [("hl", 1), ("hl", 0)]:
    a, b = g[("real", 0)], g[nk]
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(20000)]
    print("real minus", nk, "diff", round(float(a.mean() - b.mean()), 4), "bootstrap 95% CI (20000 resamples, seed 0)", np.percentile(bs, [2.5, 97.5]).round(4).tolist())
