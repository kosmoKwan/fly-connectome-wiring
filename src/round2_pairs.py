"""Round 2, experiment 1: pair-resolved re-run of the 60-s frozen-setting confirmation.

Pre-registration: NAS round2_20260927/预注册_第二轮_20260927.md.
Same frozen settings (argmax of the 25-s stationary scan, as in heldout_stat_resume.py), same noise streams
(stochastic 3000+gseed, B=20; noise-free 4000+gseed, B=3), same 60-s runs and four windows, so trajectories
reproduce the archived confirmation. Adds pair-level statistics under the full and the exact MN map.
Graphs: real + the 30 main-family nulls at their own setting; the 20 motor-input intervention graphs at the
real graph's frozen setting.
usage: round2_pairs.py TAG     env: R2_SMOKE=1 -> real graph only, output prefixed smoke_
       R2_JOBSET=surg2 -> experiment 2 graphs (amendment 1): X_hi and W at own (new scan) and real setting, X_lo at real setting
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
from pathlib import Path
from openloop import load_graph, build_W, simulate
from round2_common import make_pairs, window_record, PAIR_COLS

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results"); OUTD = R / "round2"; OUTD.mkdir(exist_ok=True)
tag = sys.argv[1]
SETS = {"theta50": ["nullscan_sets_stat_theta50_part1.parquet", "nullscan_sets_stat_part2_theta50.parquet"],
        "manc50": ["nullscan_sets_stat_manc50.parquet"]}[tag]
T, FS, N_STOCH, N_DET = 60.0, 500.0, 20, 3
WINDOWS = {"w15_25": (15, 25), "w25_35": (25, 35), "w35_45": (35, 45), "w50_60": (50, 60)}
SMOKE = os.environ.get("R2_SMOKE") == "1"
JOBSET = os.environ.get("R2_JOBSET", "main")
if JOBSET == "surg2":
    SETS = SETS + [f"nullscan_sets_stat_r2_{tag}.parquet"]   # originals first: unique(keep="first") keeps the original real rows
nodes, e_real, mnmap = load_graph(tag)
drive = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy(); mn_idx = mnmap["idx"].to_numpy()
sets = pl.concat([pl.read_parquet(R / f) for f in SETS], how="diagonal_relaxed").unique(subset=["graph", "seed", "g", "beta", "drive"], keep="first")
PA, PX = make_pairs(mnmap, False), make_pairs(mnmap, True)
assert len(PA) == 20 and len(PX) == 16, (len(PA), len(PX))
SDIR = G / ("surgery" if tag == "theta50" else f"surgery_{tag}")


def path_of(graph, seed):
    if graph == "real":
        return None
    if graph.startswith("mn_r2_"):
        return G / f"surgery_r2_{tag}" / f"{graph}_s{seed}.parquet"
    if graph.endswith("_w") and graph.startswith("mn_shuffle"):
        return G / f"surgery_w_{tag}" / f"{graph}_s{seed}.parquet"
    if graph.startswith("mn_shuffle"):
        return SDIR / f"{graph}_s{seed}.parquet"
    for d in [G / "nulls" / tag, G / "nulls" / f"{tag}_n100"]:
        if (d / f"{graph}_s{seed}.parquet").exists():
            return d / f"{graph}_s{seed}.parquet"
    raise FileNotFoundError(graph, seed)


def best_setting(graph, seed):
    s = sets.filter((pl.col("graph") == graph) & (pl.col("seed") == seed) & pl.col("rhythmic_stat"))
    if s.height == 0:
        return None
    return s.sort(["gmean_stat", "g", "beta", "drive"], descending=[True, False, False, False], nulls_last=True).head(1).select("g", "beta", "drive", "gmean_stat").row(0, named=True)


MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]
SURG = ["mn_shuffle_joint", "mn_shuffle_pool", "mn_shuffle_joint_w", "mn_shuffle_pool_w"]
jobs = [("real", 0, "own")] + [(f, s, "own") for f in MAIN for s in range(5)] + [(f, s, "real_setting") for f in SURG for s in range(5)]
if JOBSET == "surg2":
    jobs = [(f, s, "own") for f in ["mn_r2_cross_hi", "mn_r2_within"] for s in range(5)] + \
           [(f, s, "real_setting") for f in ["mn_r2_cross_hi", "mn_r2_cross_lo", "mn_r2_within"] for s in range(5)]
if SMOKE:
    jobs = jobs[:1]
real_p = best_setting("real", 0)
pre = "smoke_" if SMOKE else ""
suf = "" if JOBSET == "main" else f"_{JOBSET}"
out_s, out_p = OUTD / f"{pre}pairs_scalar_{tag}{suf}.parquet", OUTD / f"{pre}pairs_pairs_{tag}{suf}.parquet"
S = pl.read_parquet(out_s).to_dicts() if out_s.exists() else []
P = pl.read_parquet(out_p).to_dicts() if out_p.exists() else []
done = {(r["graph"], r["gseed"], r["rule"]) for r in S}
print(tag, "jobs", len(jobs), "resuming", len(done), "real setting", real_p, flush=True)
t0 = time.time()
for graph, seed, rule in jobs:
    if (graph, seed, rule) in done:
        continue
    f = path_of(graph, seed)
    WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
    p = real_p if rule == "real_setting" else best_setting(graph, seed)
    if p is None:   # no eligible setting in the scan: record and skip (same convention as heldout_stat.py)
        S.append({"connectome": tag, "graph": graph, "gseed": seed, "rule": rule, "mode": "none", "run": -1, "window": "none"})
        continue
    for mode, n, noise, sstream in [("stochastic", N_STOCH, 0.01, 3000), ("deterministic", N_DET, 0.0, 4000)]:
        Pm = np.tile(np.array([[p["g"], p["beta"], p["drive"]]], np.float32), (n, 1))
        X = simulate(WE, WI, drive, Pm, kind="tanh_rect", T=T, record_idx=mn_idx, noise=noise, seed=sstream + seed)
        for k in range(n):
            for w, (a, b) in WINDOWS.items():
                rec, prs = window_record(X[k, int(a * FS):int(b * FS)], FS, mnmap, PA, PX)
                key = {"connectome": tag, "graph": graph, "gseed": seed, "rule": rule, "mode": mode, "run": k, "window": w}
                S.append({**key, "g": p["g"], "beta": p["beta"], "drive": p["drive"], **rec})
                P.extend({**key, **dict(zip(PAIR_COLS, pr))} for pr in prs)
    for df, path in [(pl.DataFrame(S), out_s), (pl.DataFrame(P), out_p)]:
        tmp = path.with_suffix(".tmp.parquet"); df.write_parquet(tmp); os.replace(tmp, path)
    print(graph, seed, rule, f"t={time.time() - t0:.0f}s", flush=True)
(OUTD / f"{pre}pairs_{tag}{suf}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
print(tag, "complete", flush=True)
