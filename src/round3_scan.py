"""Round 3 stationary scan: identical to stationary_scan.py (25-s runs, 15-25 s scored, batches of 33 settings,
noise seed 0 per batch, same metrics and eligibility), generalised through env vars.
env:
  R3_GRAPHS  comma list: "real" and/or graph files relative to graphs/ without .parquet
  R3_GRID    std (the 132-setting grid, default) | ext_new (only the settings added by the extended grid:
             g = logspace(-4, -1.25, 12), beta in {0.25,0.5,1,2,4}, drive in {0.25,0.5,1,2,4,8}, minus the 132 std settings)
  R3_KIND    neuron nonlinearity (default tanh_rect)
  R3_NODES   node table tag (default = TAG)
  R3_LABEL   output label (required): results/round3/nullscan_sets_{LABEL}.parquet
usage: round3_scan.py TAG
"""
import sys, json, time, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
import numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, "/root/fly-connectome/src")
from openloop import load_graph, build_W, simulate, analyse, LEGS, JOINTS

G = Path("/root/fly-connectome/graphs"); OUT = Path("/root/fly-connectome/results/round3"); OUT.mkdir(exist_ok=True)
tag = sys.argv[1]
T, FS, BATCH = 25.0, 500.0, 33
WINDOWS = {"stat": (15.0, 25.0), "early": (1.0, 5.0)}
KIND = os.environ.get("R3_KIND", "tanh_rect"); NTAG = os.environ.get("R3_NODES", tag); LABEL = os.environ["R3_LABEL"]
GRID = os.environ.get("R3_GRID", "std")
nodes, e_real, mnmap = load_graph(tag)
if NTAG != tag:
    nodes = pl.read_parquet(G / f"nodes_{NTAG}.parquet")
mn_idx = mnmap["idx"].to_numpy(); drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
std = np.array([(g, b, d) for g in np.logspace(-4, -1.5, 11) for b in [0.5, 1.0, 2.0] for d in [0.5, 1.0, 2.0, 4.0]], np.float32)
if GRID == "std":
    params = std
elif GRID == "ext_new":
    ext = np.array([(g, b, d) for g in np.logspace(-4, -1.25, 12) for b in [0.25, 0.5, 1.0, 2.0, 4.0] for d in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]], np.float32)
    isstd = np.array([np.any(np.all(np.isclose(std, p, rtol=1e-5, atol=0), axis=1)) for p in ext])
    params = ext[~isstd]
    assert len(ext) == 360 and isstd.sum() == 132 and len(params) == 228, (len(ext), isstd.sum(), len(params))
else:
    raise ValueError(GRID)
graphs = []
for x in [x for x in os.environ["R3_GRAPHS"].split(",") if x]:
    if x == "real":
        graphs.append(("real", 0, None))
    else:
        lvl, s = Path(x).name.rsplit("_s", 1); graphs.append((lvl, int(s), G / f"{x}.parquet"))
leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); dd = mnmap["direction"].to_numpy()
PAIRS = [((leg == L) & (joint == J) & (dd > 0), (leg == L) & (joint == J) & (dd < 0)) for L in LEGS for J in JOINTS]
PAIRS = [p for p in PAIRS if p[0].any() and p[1].any()]


def analytic(x):
    n = len(x); F = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, 1 / FS)
    F[(f < 0.5) | (f > 20)] = 0
    full = np.zeros(n, complex); full[:len(F)] = F; full[1:(n + 1) // 2] *= 2
    return np.fft.ifft(full)


def plv_anti(X):
    Xc = X - X.mean(0); vals = []
    for e, f in PAIRS:
        a, b = Xc[:, e].mean(1), Xc[:, f].mean(1); se, sf = a.std(), b.std()
        if se <= 1e-4 or sf <= 1e-4:
            continue
        if not (min(se, sf) / max(se, sf) >= 0.2 and min(se, sf) >= 0.001):
            vals.append(0.0); continue
        z = analytic(a) * np.conj(analytic(b)); m = (z / (np.abs(z) + 1e-12)).mean()
        vals.append(float(-np.abs(m) * np.cos(np.angle(m))))
    return float(np.mean(vals)) if vals else 0.0


part = OUT / f"nullscan_sets_{LABEL}_partial.parquet"
per_set = [pl.read_parquet(part)] if part.exists() else []
done = {(r[0], r[1]) for r in per_set[0].select("graph", "seed").unique().rows()} if per_set else set()
print(tag, LABEL, "kind", KIND, "nodes", NTAG, "grid", GRID, len(params), "settings;", len(graphs), "graphs; resuming", len(done), flush=True)
t0 = time.time()
for name, s, f in graphs:
    if (name, s) in done:
        continue
    WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
    res = []
    for c in range(0, len(params), BATCH):
        P = params[c:c + BATCH]
        X = simulate(WE, WI, drive_idx, P, kind=KIND, T=T, record_idx=mn_idx, seed=0)
        for i, p in enumerate(P):
            d = {"graph": name, "seed": s, "g": float(p[0]), "beta": float(p[1]), "drive": float(p[2])}
            for w, (a, b) in WINDOWS.items():
                Xw = X[i, int(a * FS):int(b * FS)]
                r = analyse(Xw, fs=FS, mnmap=mnmap, t_discard=0.0)
                d.update({f"rhythm_{w}": r.get("rhythm", 0.0), f"freq_{w}": r.get("freq", np.nan), f"osc_{w}": r["mn_osc_frac"],
                          f"antiphase_{w}": -r.get("flex_ext_corr_median", np.nan), f"gmean_{w}": r.get("alt_amp_mean", np.nan),
                          f"plv_{w}": plv_anti(Xw) if w == "stat" else np.nan})
            res.append(d)
    df = pl.DataFrame(res).fill_nan(None)
    for w in WINDOWS:
        df = df.with_columns(((pl.col(f"osc_{w}") > 0.2) & (pl.col(f"freq_{w}") >= 1.0) & (pl.col(f"freq_{w}") <= 50.0)).fill_null(False).alias(f"rhythmic_{w}"))
    per_set.append(df)
    tmp = part.with_suffix(".tmp.parquet"); pl.concat(per_set, how="diagonal_relaxed").write_parquet(tmp); os.replace(tmp, part)
    rh = df.filter(pl.col("rhythmic_stat"))
    print(json.dumps({"graph": name, "seed": s, "n_rhythmic_stat": rh.height, "best_gmean_stat": float(rh["gmean_stat"].fill_null(0).max()) if rh.height else 0.0}), f"t={time.time() - t0:.0f}s", flush=True)
pl.concat(per_set, how="diagonal_relaxed").write_parquet(OUT / f"nullscan_sets_{LABEL}.parquet")
print(tag, LABEL, "complete", flush=True)
