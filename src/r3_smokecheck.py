"""Round 3 smoke check: tanh_rect scan + confirmation of the real graph must reproduce the archive bitwise."""
import sys, polars as pl, numpy as np
R = "/root/fly-connectome/results/"
tag = sys.argv[1]
arch_scan = {"theta50": "nullscan_sets_stat_theta50_part1.parquet", "manc50": "nullscan_sets_stat_manc50.parquet"}[tag]
a = pl.read_parquet(R + arch_scan).filter(pl.col("graph") == "real")
b = pl.read_parquet(R + f"round3/nullscan_sets_smoke_tanh_{tag}.parquet").filter(pl.col("graph") == "real")
cols = [c for c in ["rhythm_stat", "freq_stat", "osc_stat", "antiphase_stat", "gmean_stat"] if c in a.columns and c in b.columns]
j = a.join(b, on=["g", "beta", "drive"], suffix="_n")
d = max(float((j[c] - j[c + "_n"]).abs().fill_null(0).fill_nan(0).max()) for c in cols)
print(tag, "scan rows", a.height, b.height, "joined", j.height, "max diff", d, "rhythmic equal", bool((j["rhythmic_stat"] == j["rhythmic_stat_n"]).all()))
a2 = pl.read_parquet(R + f"round2/pairs_scalar_{tag}.parquet").filter((pl.col("graph") == "real") & (pl.col("rule") == "own"))
b2 = pl.read_parquet(R + f"round3/pairs_scalar_{tag}_smoke_tanh.parquet").filter(pl.col("graph") == "real")
j2 = a2.join(b2, on=["mode", "run", "window"], suffix="_n")
d2 = max(float((j2[c] - j2[c + "_n"]).abs().fill_null(0).fill_nan(0).max()) for c in ["S_all_raw", "rhythm", "osc_frac", "S_exact", "S_nonThC"])
print(tag, "confirm rows", a2.height, b2.height, "joined", j2.height, "max diff", d2, "setting", b2.select("g", "beta", "drive").unique().rows())
ok = (j.height == 132 and d == 0.0 and j2.height == 92 and d2 == 0.0)
print(tag, "SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
