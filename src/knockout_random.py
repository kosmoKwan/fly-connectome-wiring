"""Size- and structure-matched random knockout control for the knockout screen.

Control A ("one per leg"): 6 random interneurons, one drawn from each of the six
leg hemineuromeres (T1-3 x L/R), mirroring a serially homologous cell type.
Control B ("same hemilineage, one per leg"): as A, but all six drawn from one
randomly chosen hemilineage, mirroring a typed class within a hemilineage.
Scored exactly as knockout.py (same 3 parameter sets, same metrics).
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
import jax, jax.numpy as jnp
from pathlib import Path
from openloop import load_graph, build_W, act, analyse

R = Path("/root/fly-connectome/results")
kind = "tanh_rect"; N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 500
T, dt, tau, rec_every, B = 5.0, 1e-3, 0.02, 2, 64
nodes, edges, mnmap = load_graph("theta50")
WE, WI = build_W(nodes, edges); N = nodes.height
mn_idx = mnmap["idx"].to_numpy()
drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
params = pl.read_parquet(R / "knockout_tanh_rect.parquet").filter(pl.col("unit") == "intact").sort("param_set").select("g", "beta", "drive").rows()

rng = np.random.default_rng(7)
IN = nodes.filter((pl.col("role") == "IN") & pl.col("leg_seg").is_not_null() & pl.col("side").is_in(["L", "R"]))
pools = {(s, d): IN.filter((pl.col("leg_seg") == s) & (pl.col("side") == d))["idx"].to_numpy() for s in ["T1", "T2", "T3"] for d in ["L", "R"]}
units = [("intact", "none", np.array([], np.int64))]
for i in range(N_EACH):
    units.append((f"randA_{i}", "random_one_per_leg", np.array([rng.choice(p) for p in pools.values()])))
hls = [h for h in IN["trumanHl"].unique().drop_nulls().to_list()
       if all(IN.filter((pl.col("trumanHl") == h) & (pl.col("leg_seg") == s) & (pl.col("side") == d)).height > 0 for (s, d) in pools)]
for i in range(N_EACH):
    h = hls[rng.integers(len(hls))]
    sel = [rng.choice(IN.filter((pl.col("trumanHl") == h) & (pl.col("leg_seg") == s) & (pl.col("side") == d))["idx"].to_numpy()) for (s, d) in pools]
    units.append((f"randB_{h}_{i}", "random_same_hl_one_per_leg", np.array(sel)))
print("units", len(units), "hemilineages usable for B:", len(hls), flush=True)

WE_j, WI_j = jnp.asarray(WE), jnp.asarray(WI); rec = jnp.asarray(mn_idx)


@jax.jit
def run_batch(mask, g, beta, drive, key):
    I = jnp.zeros((mask.shape[0], N), jnp.float32).at[:, drive_idx].set(drive)
    r0 = 0.01 * jax.random.uniform(key, mask.shape) * mask
    a = dt / tau

    def step(carry, _):
        r, key = carry
        key, sub = jax.random.split(key)
        inp = g * (r @ WE_j.T + beta * (r @ WI_j.T)) + I + 0.01 * jax.random.normal(sub, r.shape)
        return ((r + a * (-r + act(inp, kind))) * mask, key), None

    def chunk(carry, _):
        carry, _ = jax.lax.scan(step, carry, None, length=rec_every)
        return carry, carry[0][:, rec]

    _, out = jax.lax.scan(chunk, (r0, key), None, length=int(T / dt) // rec_every)
    return jnp.transpose(out, (1, 0, 2))


rows = []; t0 = time.time()
for pi, (g, beta, drive) in enumerate(params):
    for c in range(0, len(units), B):
        cu = units[c:c + B]
        mask = np.ones((len(cu), N), np.float32)
        for i, (_, _, idx) in enumerate(cu):
            mask[i, idx] = 0.0
        X = np.asarray(run_batch(jnp.asarray(mask), jnp.float32(g), jnp.float32(beta), jnp.float32(drive), jax.random.PRNGKey(pi)))
        for i, (name, uk, idx) in enumerate(cu):
            r = analyse(X[i], fs=500.0, mnmap=mnmap)
            rows.append({"param_set": pi, "unit": name, "unit_kind": uk, "n_neurons": int(len(idx)), "mn_mean": r["mn_mean"],
                         "mn_osc_frac": r["mn_osc_frac"], "rhythm": r.get("rhythm", 0.0), "freq": r.get("freq", np.nan),
                         "fe_corr": r.get("flex_ext_corr_median", np.nan)})
    print(f"param {pi} done t={time.time() - t0:.0f}s", flush=True)
df = pl.DataFrame(rows).with_columns((-pl.col("fe_corr")).fill_nan(0).alias("antiphase"))
base = df.filter(pl.col("unit") == "intact").select("param_set", pl.col("antiphase").alias("a0"), pl.col("rhythm").alias("r0"))
df = df.join(base, on="param_set").with_columns((pl.col("antiphase") - pl.col("a0")).alias("d_antiphase"), (pl.col("rhythm") - pl.col("r0")).alias("d_rhythm"))
df.write_parquet(R / "knockout_random_control.parquet")
s = df.filter(pl.col("unit") != "intact").group_by("unit", "unit_kind").agg(pl.col("d_antiphase").mean(), pl.col("mn_osc_frac").mean(),
                                                                          (pl.col("d_antiphase") < -0.2).sum().alias("n_sets_drop"))
for k in ["random_one_per_leg", "random_same_hl_one_per_leg"]:
    x = s.filter(pl.col("unit_kind") == k)
    print(k, "n", x.height, "mean d_antiphase", round(x["d_antiphase"].mean(), 3), "5th pct", round(x["d_antiphase"].quantile(0.05), 3),
          "1st pct", round(x["d_antiphase"].quantile(0.01), 3), "frac consistent drop (3/3 sets)", round((x["n_sets_drop"] == 3).mean(), 4))
