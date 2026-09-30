"""In-silico knockout screen on the untrained connectome (open loop, DN drive).

Every knockout unit (each IN cell type, each DN type, each hemilineage) is silenced
by clamping its neurons' rates to zero. Different knockouts ride along the batch
dimension, so one GPU call evaluates dozens of knockouts at a fixed parameter set.
We evaluate at the K best parameter sets of the intact real graph and report the
change in rhythm, flexor/extensor alternation and MN activity.
"""
import os, sys, time, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
import jax, jax.numpy as jnp
from pathlib import Path
from openloop import load_graph, build_W, act, analyse

R = Path("/root/fly-connectome/results")
kind = sys.argv[1] if len(sys.argv) > 1 else "tanh_rect"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T, dt, tau, rec_every, B = 5.0, 1e-3, 0.02, 2, 64

nodes, edges, mnmap = load_graph("theta50")
WE, WI = build_W(nodes, edges)
N = nodes.height
mn_idx = mnmap["idx"].to_numpy()
drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()

# parameter sets: best K for the intact real graph from the null scan
sets = pl.read_parquet(R / f"nullscan_sets_theta50_{kind}_DNg100_T5.parquet").filter(
    (pl.col("graph") == "real") & pl.col("rhythmic")).sort("combined", descending=True).head(K)
params = sets.select("g", "beta", "drive").rows()
print("param sets:", params, flush=True)

units = [("intact", "none", np.array([], np.int64))]
for role in ["IN", "DN"]:
    for t, grp in nodes.filter((pl.col("role") == role) & pl.col("type").is_not_null()).group_by("type"):
        units.append((f"{role}:{t[0]}", "type", grp["idx"].to_numpy()))
for h, grp in nodes.filter((pl.col("role") == "IN") & pl.col("trumanHl").is_not_null()).group_by("trumanHl"):
    units.append((f"HL:{h[0]}", "hemilineage", grp["idx"].to_numpy()))
units = [u for u in units if not np.intersect1d(u[2], drive_idx).size]  # never knock out the driver itself
print("knockout units:", len(units), flush=True)

WE_j, WI_j = jnp.asarray(WE), jnp.asarray(WI)
rec = jnp.asarray(mn_idx)


@jax.jit
def run_batch(mask, g, beta, drive, key):
    I = jnp.zeros((mask.shape[0], N), jnp.float32).at[:, drive_idx].set(drive)
    r0 = 0.01 * jax.random.uniform(key, mask.shape) * mask
    a = dt / tau

    def step(carry, _):
        r, key = carry
        key, sub = jax.random.split(key)
        inp = g * (r @ WE_j.T + beta * (r @ WI_j.T)) + I + 0.01 * jax.random.normal(sub, r.shape)
        r = (r + a * (-r + act(inp, kind))) * mask
        return (r, key), None

    def chunk(carry, _):
        carry, _ = jax.lax.scan(step, carry, None, length=rec_every)
        return carry, carry[0][:, rec]

    _, out = jax.lax.scan(chunk, (r0, key), None, length=int(T / dt) // rec_every)
    return jnp.transpose(out, (1, 0, 2))


rows = []
t0 = time.time()
for pi, (g, beta, drive) in enumerate(params):
    for c in range(0, len(units), B):
        chunk_units = units[c:c + B]
        mask = np.ones((len(chunk_units), N), np.float32)
        for i, (_, _, idx) in enumerate(chunk_units):
            mask[i, idx] = 0.0
        X = np.asarray(run_batch(jnp.asarray(mask), jnp.float32(g), jnp.float32(beta), jnp.float32(drive), jax.random.PRNGKey(pi)))
        for i, (name, ukind, idx) in enumerate(chunk_units):
            r = analyse(X[i], fs=500.0, mnmap=mnmap)
            rows.append({"param_set": pi, "g": g, "beta": beta, "drive": drive, "unit": name, "unit_kind": ukind,
                         "n_neurons": int(len(idx)), "mn_mean": r["mn_mean"], "mn_osc_frac": r["mn_osc_frac"],
                         "rhythm": r.get("rhythm", 0.0), "freq": r.get("freq", np.nan),
                         "fe_corr": r.get("flex_ext_corr_median", np.nan)})
        print(f"param {pi} units {c + len(chunk_units)}/{len(units)} t={time.time() - t0:.0f}s", flush=True)

df = pl.DataFrame(rows).with_columns((-pl.col("fe_corr")).fill_nan(0).alias("antiphase"))
df = df.with_columns((pl.col("rhythm") * pl.col("antiphase").clip(lower_bound=0)).alias("combined"))
base = df.filter(pl.col("unit") == "intact").select("param_set", pl.col("combined").alias("c0"), pl.col("antiphase").alias("a0"), pl.col("rhythm").alias("r0"))
df = df.join(base, on="param_set").with_columns(
    (pl.col("combined") - pl.col("c0")).alias("d_combined"), (pl.col("antiphase") - pl.col("a0")).alias("d_antiphase"),
    (pl.col("rhythm") - pl.col("r0")).alias("d_rhythm"))
df.write_parquet(R / f"knockout_{kind}.parquet")
summ = df.group_by("unit", "unit_kind", "n_neurons").agg(pl.col("d_combined").mean(), pl.col("d_antiphase").mean(), pl.col("d_rhythm").mean()).sort("d_combined")
summ.write_parquet(R / f"knockout_{kind}_summary.parquet")
print("most disruptive:"); print(summ.head(25).to_pandas().to_string())
print("most enhancing:"); print(summ.tail(8).to_pandas().to_string())
