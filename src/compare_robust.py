"""Robustness pack for the open-loop comparison (same grid and simulator as compare_nulls.py).

From ONE simulation per (graph, parameter set) we score the antagonist endpoints under
several read-outs, so real-vs-null contrasts are computed on identical activity:
  all        every mapped antagonist pair (the main analysis)
  exact      only MNs whose joint/direction mapping is not flagged approximate (drops the
             sternal rotators and long-tendon-muscle MNs, 76 of 381 in MaleCNS)
  <joint>    only pairs of one joint (ThC, CTr, FTi, TiTa)
Per graph we store the best value over rhythmic sets for each read-out, plus the fraction of
rhythmic sets whose gated mean exceeds 0.1 and 0.2.
Env: NULL_DIR, NULL_LEVELS, NULL_N, LABEL_PREFIX as in compare_nulls.py.
"""
import sys, json, time, os
import numpy as np
import polars as pl
from pathlib import Path
sys.path.insert(0, "/root/fly-connectome/src")
from openloop import load_graph, build_W, simulate, analyse, JOINTS

G = Path("/root/fly-connectome/graphs")
OUT = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
kind = sys.argv[2] if len(sys.argv) > 2 else "tanh_rect"
T = 5.0

nodes, edges_real, mnmap = load_graph(tag)
mn_idx = mnmap["idx"].to_numpy()
drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
gs = np.logspace(-4, -1.5, 11); betas = [0.5, 1.0, 2.0]; drives = [0.5, 1.0, 2.0, 4.0]
params = np.array([(g, b, d) for g in gs for b in betas for d in drives], np.float32)

# read-out variants: same MN columns, antagonist membership changed through direction/joint
readouts = {"all": mnmap,
            "exact": mnmap.with_columns(pl.when(pl.col("approx")).then(0).otherwise(pl.col("direction")).alias("direction"))}
for J in JOINTS:
    readouts[J] = mnmap.with_columns(pl.when(pl.col("joint") == J).then(pl.col("joint")).otherwise(None).alias("joint"))

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
        X = simulate(WE, WI, drive_idx, P, kind=kind, T=T, record_idx=mn_idx, seed=0)
        for i, p in enumerate(P):
            r0 = analyse(X[i], fs=500.0, mnmap=mnmap)
            d = {"graph": name, "seed": s, "g": float(p[0]), "beta": float(p[1]), "drive": float(p[2]),
                 "mn_osc_frac": r0["mn_osc_frac"], "rhythm": r0.get("rhythm", 0.0), "freq": r0.get("freq", np.nan)}
            for k, mm in readouts.items():
                r = r0 if k == "all" else analyse(X[i], fs=500.0, mnmap=mm)
                d[f"antiphase_{k}"] = -r.get("flex_ext_corr_median", np.nan)
                d[f"gmean_{k}"] = r.get("alt_amp_mean", np.nan)
            res.append(d)
    df = pl.DataFrame(res).fill_nan(None)
    df = df.with_columns(((pl.col("mn_osc_frac") > 0.2) & (pl.col("freq") >= 1.0) & (pl.col("freq") <= 50.0)).fill_null(False).alias("rhythmic"))
    per_set.append(df)
    rh = df.filter(pl.col("rhythmic"))
    row = {"graph": name, "seed": s, "n_rhythmic_sets": rh.height,
           "best_rhythm": float(rh["rhythm"].max()) if rh.height else 0.0}
    for k in readouts:
        for m in ("antiphase", "gmean"):
            v = rh[f"{m}_{k}"].fill_null(0.0)
            row[f"best_{m}_{k}"] = float(v.max()) if rh.height else 0.0
    for thr in (0.1, 0.2):
        row[f"frac_gmean_gt{thr}"] = float((rh["gmean_all"].fill_null(0.0) > thr).mean()) if rh.height else 0.0
    rows.append(row)
    print(json.dumps(row), f"t={time.time()-t0:.0f}s", flush=True)

label = os.environ.get("LABEL_PREFIX", "robust_") + f"{tag}_{kind}"
pl.concat(per_set, how="diagonal_relaxed").write_parquet(OUT / f"nullscan_sets_{label}.parquet")
summ = pl.DataFrame(rows)
summ.write_parquet(OUT / f"nullscan_{label}.parquet")
real = summ.filter(pl.col("graph") == "real")
for c in [c for c in summ.columns if c.startswith("best_") or c.startswith("frac_")]:
    rv = float(real[c][0]); nulls = summ.filter(pl.col("graph") != "real")
    if nulls.height == 0:
        continue
    worst = nulls.group_by("graph").agg(pl.col(c).max().alias("mx"), pl.col(c).median().alias("md")).sort("mx", descending=True)
    top = worst.row(0, named=True)
    n_ge = int((nulls[c].to_numpy() >= rv).sum())
    print(f"{c:22s} real={rv:.3f}  max-null={top['mx']:.3f} ({top['graph']})  nulls>=real: {n_ge}/{nulls.height}")
