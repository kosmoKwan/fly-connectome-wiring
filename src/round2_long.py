"""Round 2, experiment 3: 300-s trajectories at frozen settings (MANC persistence and state switching).

Pre-registration: NAS round2_20260927/预注册_第二轮_20260927.md.
Graphs (own frozen setting): MANC real, MANC hl s1, MANC hl s0 (the two strongest MANC ordinary nulls in the
60-s confirmation) and MaleCNS real (stable reference).
Streams: A = 3000+gseed (B=20, noise 0.01; continues the archived confirmation trajectories),
         B = 5000+gseed (B=20, noise 0.01; independent), D = 6000+gseed (B=10, noise 0; initial states).
The dynamics are integrated in 30-s chunks that carry the full rate vector and PRNG key, which is identical to
openloop.simulate run in one piece (same initial state, same key splitting, same recording every 2nd step).
Windows: 10-s windows 10-20 ... 290-300 s plus the four original confirmation windows.
usage: round2_long.py      env: R2_SMOKE=1 -> MANC real, stream A, 60 s only (checks against the archive)
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
import jax, jax.numpy as jnp
from pathlib import Path
from openloop import load_graph, build_W, act
from round2_common import make_pairs, window_record, PAIR_COLS

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results"); OUTD = R / "round2"; OUTD.mkdir(exist_ok=True)
SMOKE = os.environ.get("R2_SMOKE") == "1"
FS, DT, TAU, REC_EVERY, CHUNK_S = 500.0, 1e-3, 0.02, 2, 30.0
T = 60.0 if SMOKE else 300.0
SETS = {"theta50": ["nullscan_sets_stat_theta50_part1.parquet", "nullscan_sets_stat_part2_theta50.parquet"],
        "manc50": ["nullscan_sets_stat_manc50.parquet"]}
JOBS = [("manc50", "real", 0), ("manc50", "hl", 1), ("manc50", "hl", 0), ("theta50", "real", 0)]
STREAMS = [("A", 20, 0.01, 3000), ("B", 20, 0.01, 5000), ("D", 10, 0.0, 6000)]
if SMOKE:
    JOBS, STREAMS = JOBS[:1], STREAMS[:1]
WINDOWS = {f"t{a:03d}_{a + 10:03d}": (a, a + 10) for a in range(10, int(T), 10)}
WINDOWS.update({k: v for k, v in {"w15_25": (15, 25), "w25_35": (25, 35), "w35_45": (35, 45), "w50_60": (50, 60)}.items() if v[1] <= T})


def simulate_chunked(WE, WI, drive_idx, params, T, rec_idx, noise, seed, kind="tanh_rect"):
    B = params.shape[0]; N = WE.shape[0]
    WE_j, WI_j = jnp.asarray(WE), jnp.asarray(WI)
    g = jnp.asarray(params[:, 0:1], jnp.float32); beta = jnp.asarray(params[:, 1:2], jnp.float32)
    I = jnp.zeros((B, N), jnp.float32).at[:, drive_idx].set(jnp.asarray(params[:, 2:3], jnp.float32))
    rec = jnp.asarray(rec_idx); a = DT / TAU
    key = jax.random.PRNGKey(seed)
    carry = (0.01 * jax.random.uniform(key, (B, N)), key)

    def step(c, k):
        r, kk = c
        kk, sub = jax.random.split(kk)
        inp = g * (r @ WE_j.T + beta * (r @ WI_j.T)) + I + noise * jax.random.normal(sub, (B, N))
        r = r + a * (-r + act(inp, kind))
        return (r, kk), r[:, rec]

    def chunk(c, _):
        c, ys = jax.lax.scan(step, c, jnp.arange(REC_EVERY))
        return c, ys[-1]

    # eager lax.scan with the same closure structure as openloop.simulate (no jit around closed-over matrices)
    n_rec_total = int(round(T / DT)) // REC_EVERY; n_rec_chunk = int(round(CHUNK_S / DT)) // REC_EVERY
    X = np.empty((B, n_rec_total, len(rec_idx)), np.float32); pos = 0
    while pos < n_rec_total:
        n = min(n_rec_chunk, n_rec_total - pos)
        carry, out = jax.lax.scan(chunk, carry, jnp.arange(n))
        X[:, pos:pos + n] = np.asarray(jnp.transpose(out, (1, 0, 2))); pos += n
    return X


pre = "smoke_" if SMOKE else ""
out_s, out_p = OUTD / f"{pre}long_scalar.parquet", OUTD / f"{pre}long_pairs.parquet"
S = pl.read_parquet(out_s).to_dicts() if out_s.exists() else []
P = pl.read_parquet(out_p).to_dicts() if out_p.exists() else []
done = {(r["connectome"], r["graph"], r["gseed"], r["stream"]) for r in S}
print("long jobs", JOBS, "T", T, "resuming", len(done), flush=True)
t0 = time.time(); cache = {}
for tag, graph, seed in JOBS:
    if tag not in cache:
        nodes, e_real, mnmap = load_graph(tag)
        sets = pl.concat([pl.read_parquet(R / f) for f in SETS[tag]], how="diagonal_relaxed").unique(subset=["graph", "seed", "g", "beta", "drive"], keep="first")
        cache[tag] = (nodes, e_real, mnmap, sets, make_pairs(mnmap, False), make_pairs(mnmap, True))
    nodes, e_real, mnmap, sets, PA, PX = cache[tag]
    drive = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy(); mn_idx = mnmap["idx"].to_numpy()
    p = (sets.filter((pl.col("graph") == graph) & (pl.col("seed") == seed) & pl.col("rhythmic_stat"))
         .sort(["gmean_stat", "g", "beta", "drive"], descending=[True, False, False, False], nulls_last=True).head(1).row(0, named=True))
    ef = None if graph == "real" else next(d / f"{graph}_s{seed}.parquet" for d in [G / "nulls" / tag, G / "nulls" / f"{tag}_n100"] if (d / f"{graph}_s{seed}.parquet").exists())
    WE, WI = build_W(nodes, e_real if ef is None else pl.read_parquet(ef))
    for sname, n, noise, sstream in STREAMS:
        if (tag, graph, seed, sname) in done:
            continue
        Pm = np.tile(np.array([[p["g"], p["beta"], p["drive"]]], np.float32), (n, 1))
        X = simulate_chunked(WE, WI, drive, Pm, T, mn_idx, noise, sstream + seed)
        for k in range(n):
            for w, (a, b) in WINDOWS.items():
                rec, prs = window_record(X[k, int(a * FS):int(b * FS)], FS, mnmap, PA, PX)
                key = {"connectome": tag, "graph": graph, "gseed": seed, "stream": sname, "run": k, "window": w, "t_start": float(a)}
                S.append({**key, "g": p["g"], "beta": p["beta"], "drive": p["drive"], **rec})
                P.extend({**key, **dict(zip(PAIR_COLS, pr))} for pr in prs)
        del X
        for df, path in [(pl.DataFrame(S), out_s), (pl.DataFrame(P), out_p)]:
            tmp = path.with_suffix(".tmp.parquet"); df.write_parquet(tmp); os.replace(tmp, path)
        print(tag, graph, seed, sname, f"t={time.time() - t0:.0f}s", flush=True)
(OUTD / f"{pre}long.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
print("long complete", flush=True)
