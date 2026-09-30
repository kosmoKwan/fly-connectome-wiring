"""Round 3 (stress tests; pre-registration written before running): 60-s confirmation at each graph's own
stationary-best setting, with the round-2 pair-level readouts. Generalises round2_pairs.py through env vars.
env:
  R3_GRAPHS  comma list of graph files relative to graphs/ without .parquet (e.g. nulls/theta50_n100/hl_s5); "real" = real graph
  R3_SETS    comma list of scan files in results/ used to pick each graph's best setting (first file wins on duplicates)
  R3_KIND    neuron nonlinearity (default tanh_rect)
  R3_NODES   node table tag (default = TAG; e.g. theta50gluE for glutamate-excitatory signs)
  R3_SUFFIX  output suffix (required), outputs results/round3/pairs_{scalar,pairs}_{TAG}_{SUFFIX}.parquet
  R3_AT_REAL =1: run every graph at the real graph's best setting from R3_SETS (rule "real_setting")
usage: round3_pairs.py TAG
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
from pathlib import Path
from openloop import load_graph, build_W, simulate
from round2_common import make_pairs, window_record, PAIR_COLS

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results"); OUTD = R / "round3"; OUTD.mkdir(exist_ok=True)
tag = sys.argv[1]
GRAPHS = [x for x in os.environ["R3_GRAPHS"].split(",") if x]
SETS = [x for x in os.environ["R3_SETS"].split(",") if x]
KIND = os.environ.get("R3_KIND", "tanh_rect"); NTAG = os.environ.get("R3_NODES", tag); SUF = os.environ["R3_SUFFIX"]
T, FS, N_STOCH, N_DET = 60.0, 500.0, 20, 3
WINDOWS = {"w15_25": (15, 25), "w25_35": (25, 35), "w35_45": (35, 45), "w50_60": (50, 60)}
nodes, e_real, mnmap = load_graph(tag)
if NTAG != tag:
    nodes = pl.read_parquet(G / f"nodes_{NTAG}.parquet")
drive = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy(); mn_idx = mnmap["idx"].to_numpy()
sets = pl.concat([pl.read_parquet(R / f) for f in SETS], how="diagonal_relaxed").unique(subset=["graph", "seed", "g", "beta", "drive"], keep="first")
PA, PX = make_pairs(mnmap, False), make_pairs(mnmap, True)
assert len(PA) == 20 and len(PX) == 16, (len(PA), len(PX))


def best_setting(graph, seed):   # identical rule to round2_pairs.py
    s = sets.filter((pl.col("graph") == graph) & (pl.col("seed") == seed) & pl.col("rhythmic_stat"))
    if s.height == 0:
        return None
    return s.sort(["gmean_stat", "g", "beta", "drive"], descending=[True, False, False, False], nulls_last=True).head(1).select("g", "beta", "drive", "gmean_stat").row(0, named=True)


AT_REAL = os.environ.get("R3_AT_REAL") == "1"   # D2: every graph at the real graph's frozen setting (round-2 rule "real_setting")


def ident(stem):
    if stem == "real":
        return "real", 0, None
    name = Path(stem).name; lvl, s = name.rsplit("_s", 1)
    return lvl, int(s), G / f"{stem}.parquet"


out_s, out_p = OUTD / f"pairs_scalar_{tag}_{SUF}.parquet", OUTD / f"pairs_pairs_{tag}_{SUF}.parquet"
S = pl.read_parquet(out_s).to_dicts() if out_s.exists() else []
P = pl.read_parquet(out_p).to_dicts() if out_p.exists() else []
done = {(r["graph"], r["gseed"]) for r in S}
print(tag, SUF, "kind", KIND, "nodes", NTAG, "graphs", len(GRAPHS), "resuming", len(done), flush=True)
t0 = time.time()
for stem in GRAPHS:
    graph, seed, f = ident(stem)
    if (graph, seed) in done:
        continue
    p = best_setting("real", 0) if AT_REAL else best_setting(graph, seed)
    if p is None:
        S.append({"connectome": tag, "graph": graph, "gseed": seed, "rule": "own", "mode": "none", "run": -1, "window": "none"})
    else:
        WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
        for mode, n, noise, sstream in [("stochastic", N_STOCH, 0.01, 3000), ("deterministic", N_DET, 0.0, 4000)]:
            Pm = np.tile(np.array([[p["g"], p["beta"], p["drive"]]], np.float32), (n, 1))
            X = simulate(WE, WI, drive, Pm, kind=KIND, T=T, record_idx=mn_idx, noise=noise, seed=sstream + seed)
            for k in range(n):
                for w, (a, b) in WINDOWS.items():
                    rec, prs = window_record(X[k, int(a * FS):int(b * FS)], FS, mnmap, PA, PX)
                    key = {"connectome": tag, "graph": graph, "gseed": seed, "rule": "real_setting" if AT_REAL else "own", "mode": mode, "run": k, "window": w}
                    S.append({**key, "g": p["g"], "beta": p["beta"], "drive": p["drive"], **rec})
                    P.extend({**key, **dict(zip(PAIR_COLS, pr))} for pr in prs)
    for df, path in [(pl.DataFrame(S), out_s), (pl.DataFrame(P), out_p)]:
        if df.height:
            tmp = path.with_suffix(".tmp.parquet"); df.write_parquet(tmp); os.replace(tmp, path)
    print(graph, seed, f"t={time.time() - t0:.0f}s", flush=True)
(OUTD / f"pairs_{tag}_{SUF}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
print(tag, SUF, "complete", flush=True)
