"""Run the same open-loop DN-drive scan on the real graph and every null graph.

Per graph we report the best rhythm found anywhere in the (g, beta, drive) grid, so
every graph gets the same chance. A parameter set counts as rhythmic when >20% of MNs
oscillate at 1-50 Hz. Scores:
  rhythm      spectral concentration at the dominant frequency (median over MNs)
  antiphase   -median flexor/extensor correlation (1 = perfect alternation)
  combined    rhythm * max(antiphase, 0)
"""
import sys, json, time, os
import numpy as np
import polars as pl
from pathlib import Path
sys.path.insert(0, "/root/fly-connectome/src")
from openloop import load_graph, build_W, simulate, analyse, tripod_score

G = Path("/root/fly-connectome/graphs")
OUT = Path("/root/fly-connectome/results"); OUT.mkdir(exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
kind = sys.argv[2] if len(sys.argv) > 2 else "tanh_rect"
drive_types = (sys.argv[3] if len(sys.argv) > 3 else "DNg100").split(",")
T = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

nodes, edges_real, mnmap = load_graph(tag)
mn_idx = mnmap["idx"].to_numpy()
if drive_types == ["none"]:
    drive_idx = np.array([], np.int64)
else:
    drive_idx = nodes.filter(pl.col("type").is_in(drive_types))["idx"].to_numpy()
gs = np.logspace(-4, -1.5, 11); betas = [0.5, 1.0, 2.0]; drives = [0.5, 1.0, 2.0, 4.0]
params = np.array([(g, b, d) for g in gs for b in betas for d in drives], np.float32)

LEVELS = os.environ.get("NULL_LEVELS", "er,role,degree,leg,hl,leg_hl").split(",")
NULLDIR = Path(os.environ.get("NULL_DIR", str(G / "nulls" / tag)))
NS = int(os.environ.get("NULL_N", "5"))
graphs = [("real", 0, None)] + [(lvl, s, NULLDIR / f"{lvl}_s{s}.parquet")
                                for lvl in LEVELS for s in range(NS) if (NULLDIR / f"{lvl}_s{s}.parquet").exists()]
rows, per_set = [], []
t0 = time.time()
for name, s, f in graphs:
    e = edges_real if f is None else pl.read_parquet(f)
    WE, WI = build_W(nodes, e)
    res = []
    for c in range(0, len(params), 66):
        P = params[c:c + 66]
        X = simulate(WE, WI, drive_idx if len(drive_idx) else np.array([0]), P if len(drive_idx) else P * np.array([1, 1, 0], np.float32),
                     kind=kind, T=T, record_idx=mn_idx)
        for i, p in enumerate(P):
            r = analyse(X[i], fs=500.0, mnmap=mnmap)
            res.append({"graph": name, "seed": s, "g": float(p[0]), "beta": float(p[1]), "drive": float(p[2]),
                        "mn_mean": r["mn_mean"], "mn_osc_frac": r["mn_osc_frac"], "rhythm": r.get("rhythm", 0.0),
                        "freq": r.get("freq", np.nan), "fe_corr": r.get("flex_ext_corr_median", np.nan),
                        "tripod": tripod_score(r.get("leg_phase_rel_T1L")),
                        "alt_amp": r.get("alt_amp", np.nan), "n_pairs_amp": r.get("n_pairs_amp", 0),
                        "alt_amp_strict": r.get("alt_amp_strict", np.nan), "alt_amp_mean": r.get("alt_amp_mean", np.nan)})
    df = pl.DataFrame(res).with_columns(
        (pl.col("mn_osc_frac") > 0.2) & (pl.col("freq") >= 1.0) & (pl.col("freq") <= 50.0)).rename({"literal": "rhythmic"}) \
        if False else pl.DataFrame(res)
    df = df.with_columns(((pl.col("mn_osc_frac") > 0.2) & (pl.col("freq") >= 1.0) & (pl.col("freq") <= 50.0)).alias("rhythmic"))
    df = df.with_columns((-pl.col("fe_corr")).fill_nan(0).alias("antiphase"))
    df = df.with_columns((pl.col("rhythm") * pl.col("antiphase").clip(lower_bound=0)).alias("combined"))
    per_set.append(df)
    rh = df.filter(pl.col("rhythmic"))
    best = rh.sort("combined", descending=True).head(1)
    row = {"graph": name, "seed": s, "n_rhythmic_sets": rh.height,
           "best_rhythm": float(rh["rhythm"].max()) if rh.height else 0.0,
           "best_antiphase": float(rh["antiphase"].max()) if rh.height else 0.0,
           "best_combined": float(rh["combined"].max()) if rh.height else 0.0,
           "best_tripod": float(rh["tripod"].max()) if rh.height else 0.0,
           "best_alt_amp": float(rh["alt_amp"].fill_nan(0).max()) if rh.height else 0.0,
           "best_alt_amp_strict": float(rh["alt_amp_strict"].fill_nan(0).max()) if rh.height else 0.0,
           "best_alt_amp_mean": float(rh["alt_amp_mean"].fill_nan(0).max()) if rh.height else 0.0,
           "best_combined_amp": float((rh["rhythm"] * rh["alt_amp"].fill_nan(0).clip(lower_bound=0)).max()) if rh.height else 0.0,
           "freq_at_best": float(best["freq"][0]) if rh.height else np.nan}
    rows.append(row)
    print(json.dumps(row), f"t={time.time()-t0:.0f}s", flush=True)

label = os.environ.get("LABEL_PREFIX", "") + f"{tag}_{kind}_{'+'.join(drive_types)}_T{T:g}"
pl.concat(per_set).write_parquet(OUT / f"nullscan_sets_{label}.parquet")
summ = pl.DataFrame(rows)
summ.write_parquet(OUT / f"nullscan_{label}.parquet")
agg = summ.group_by("graph").agg(
    pl.len().alias("n"), *[pl.col(c).mean().alias(c) for c in ["n_rhythmic_sets", "best_rhythm", "best_antiphase", "best_combined", "best_alt_amp", "best_alt_amp_strict", "best_alt_amp_mean", "freq_at_best"]])
order = {"real": 0, "er": 1, "role": 2, "degree": 3, "leg": 4, "hl": 5, "leg_hl": 6, "degree_recip": 7, "leg_hl_recip": 8}
agg = agg.with_columns(pl.col("graph").replace_strict(order, default=99).alias("o")).sort("o").drop("o")
print(agg.to_pandas().to_string())
real = summ.filter(pl.col("graph") == "real")
for c in ["best_alt_amp", "best_alt_amp_strict", "best_alt_amp_mean", "best_antiphase", "best_rhythm"]:
    rv = float(real[c][0])
    for lvl in LEVELS:
        nv = summ.filter(pl.col("graph") == lvl)[c].to_numpy()
        print(f"{c:16s} real={rv:.3f}  {lvl:7s} mean={nv.mean():.3f} max={nv.max():.3f}  real>all:{bool(rv > nv.max())}")
