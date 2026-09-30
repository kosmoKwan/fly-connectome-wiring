"""Held-out confirmation of the stationary scan (notebook 3.26).

For every graph in the stationary scan of a connectome (real, all nulls, all surgeries), freeze the setting
that maximised the primary endpoint (gated mean on 15-25 s among rhythmic settings) and re-run it with
  20 fresh noise seeds (seed stream 3000, never used before), 60-s runs, and
   3 noise-free runs from different random initial states (seed stream 4000).
Surgery graphs are additionally run at the REAL graph's frozen setting (fixed-regime disruption).
Scored in windows 15-25 (confirmation), 25-35, 35-45 and 50-60 s (late persistence), with the scan's
metrics plus plv_anti and coverage (number of gated antagonist pairs, oscillating-MN fraction).
usage: heldout_stat.py TAG SETS_PARQUET[,SETS_PARQUET...]
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
from pathlib import Path
from openloop import load_graph, build_W, simulate, analyse, LEGS, JOINTS

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag = sys.argv[1]; SETS = [R / f for f in sys.argv[2].split(",")]
T, FS, N_STOCH, N_DET = float(os.environ.get("HO_T", 60)), 500.0, 20, 3
WINDOWS = {k: v for k, v in {"w15_25": (15, 25), "w25_35": (25, 35), "w35_45": (35, 45), "w50_60": (50, 60)}.items() if v[1] <= T}
nodes, e_real, mnmap = load_graph(tag)
drive = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy(); mn_idx = mnmap["idx"].to_numpy()
sets = pl.concat([pl.read_parquet(f) for f in SETS], how="diagonal_relaxed").unique(subset=["graph", "seed", "g", "beta", "drive"], keep="first")
leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); dd = mnmap["direction"].to_numpy()
PAIRS = [((leg == L) & (joint == J) & (dd > 0), (leg == L) & (joint == J) & (dd < 0)) for L in LEGS for J in JOINTS]
PAIRS = [p for p in PAIRS if p[0].any() and p[1].any()]
SDIR = G / ("surgery" if tag == "theta50" else f"surgery_{tag}")


def path_of(graph, seed):
    if graph == "real":
        return None
    if graph.endswith("_w") and graph.startswith("mn_shuffle"):
        return G / f"surgery_w_{tag}" / f"{graph}_s{seed}.parquet"
    if graph.startswith("mn_shuffle"):
        return SDIR / f"{graph}_s{seed}.parquet"
    for d in [G / "nulls" / tag, G / "nulls" / f"{tag}_n100"]:
        if (d / f"{graph}_s{seed}.parquet").exists():
            return d / f"{graph}_s{seed}.parquet"
    raise FileNotFoundError(graph, seed)


def analytic(x):
    n = len(x); F = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, 1 / FS)
    F[(f < 0.5) | (f > 20)] = 0
    full = np.zeros(n, complex); full[:len(F)] = F; full[1:(n + 1) // 2] *= 2
    return np.fft.ifft(full)


def pair_stats(X):
    Xc = X - X.mean(0); vals, n_gated = [], 0
    for e, f in PAIRS:
        a, b = Xc[:, e].mean(1), Xc[:, f].mean(1); se, sf = a.std(), b.std()
        if se <= 1e-4 or sf <= 1e-4:
            continue
        if not (min(se, sf) / max(se, sf) >= 0.2 and min(se, sf) >= 0.001):
            vals.append(0.0); continue
        n_gated += 1
        z = analytic(a) * np.conj(analytic(b)); m = (z / (np.abs(z) + 1e-12)).mean()
        vals.append(float(-np.abs(m) * np.cos(np.angle(m))))
    return (float(np.mean(vals)) if vals else 0.0), n_gated


def best_setting(graph, seed):
    s = sets.filter((pl.col("graph") == graph) & (pl.col("seed") == seed) & pl.col("rhythmic_stat"))
    if s.height == 0:
        return None
    # Freeze a deterministic tie-break before running fresh validation seeds.
    return s.sort(["gmean_stat", "g", "beta", "drive"], descending=[True, False, False, False], nulls_last=True).head(1).select("g", "beta", "drive", "gmean_stat").row(0, named=True)


graphs = sets.select("graph", "seed").unique().sort("graph", "seed").rows()
graphs = ([("real", 0)] + [g for g in graphs if g[0] != "real"])[:int(os.environ.get("HO_LIMIT", 10**9))]
real_p = best_setting("real", 0)
print(tag, "graphs", len(graphs), "real setting", real_p, flush=True)
out = R / (f"heldout_stat_{tag}.parquet" if "HO_T" not in os.environ else f"smoke_heldout_stat_{tag}.parquet")
rows = pl.read_parquet(out).to_dicts() if out.exists() else []
completed = {(row["graph"], row["gseed"]) for row in rows}
t0 = time.time()
print("resuming", len(completed), "completed graphs from", out, flush=True)
for graph, seed in graphs:
    if (graph, seed) in completed:
        continue
    f = path_of(graph, seed)
    WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
    rules = [("own", best_setting(graph, seed))]
    if graph.startswith("mn_shuffle"):
        rules.append(("real_setting", real_p))
    for rule, p in rules:
        if p is None:
            rows.append({"connectome": tag, "graph": graph, "gseed": seed, "rule": rule, "mode": "none", "run": -1}); continue
        for mode, n, noise, sstream in [("stochastic", N_STOCH, 0.01, 3000), ("deterministic", N_DET, 0.0, 4000)]:
            P = np.tile(np.array([[p["g"], p["beta"], p["drive"]]], np.float32), (n, 1))
            X = simulate(WE, WI, drive, P, kind="tanh_rect", T=T, record_idx=mn_idx, noise=noise, seed=sstream + seed)
            for k in range(n):
                for w, (a, b) in WINDOWS.items():
                    Xw = X[k, int(a * FS):int(b * FS)]
                    r = analyse(Xw, fs=FS, mnmap=mnmap, t_discard=0.0)
                    plv, ng = pair_stats(Xw)
                    rows.append({"connectome": tag, "graph": graph, "gseed": seed, "rule": rule, "g": p["g"], "beta": p["beta"], "drive": p["drive"],
                                 "scan_gmean": float(p["gmean_stat"] or 0), "mode": mode, "run": k, "window": w,
                                 "rhythm": r.get("rhythm", 0.0), "freq": r.get("freq", np.nan), "osc_frac": r["mn_osc_frac"],
                                 "antiphase": -r.get("flex_ext_corr_median", np.nan), "gmean": r.get("alt_amp_mean", np.nan),
                                 "plv_anti": plv, "n_gated_pairs": ng})
    temp = out.with_suffix(".tmp.parquet")
    pl.DataFrame(rows).write_parquet(temp)
    os.replace(temp, out)
    print(graph, seed, f"t={time.time() - t0:.0f}s", flush=True)
df = pl.DataFrame(rows)
s = (df.filter((pl.col("mode") == "stochastic")).group_by("graph", "gseed", "rule", "window")
     .agg(pl.col("gmean").fill_nan(0).fill_null(0).mean().alias("gmean"), pl.col("plv_anti").mean(), pl.col("rhythm").mean(), pl.col("n_gated_pairs").mean()))
with pl.Config(tbl_rows=200, tbl_width_chars=200):
    print(s.pivot(on="window", index=["graph", "gseed", "rule"], values="gmean").sort("graph", "gseed", "rule"))
