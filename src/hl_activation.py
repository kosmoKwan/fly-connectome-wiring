"""In-silico hemilineage ACTIVATION, to compare with published activation phenotypes
(pre-registered in draft_v0.md section 3.11 before this script was run).

For each graph (real + nulls), each of its top-3 scan settings, two conditions (no DN drive =
"decapitated"; DNg100 drive) and two amplitudes, every hemilineage is activated in turn by adding
a constant input to all of its interneurons. Rows of one batch = baseline + one row per hemilineage,
all sharing the noise stream. Output: per (graph, setting, condition, amplitude, hemilineage, leg,
joint) the change versus baseline of mean extensor-pool rate minus mean flexor-pool rate (1-5 s).
usage: hl_activation.py TAG
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
import jax, jax.numpy as jnp
from pathlib import Path
from openloop import load_graph, build_W, act, LEGS, JOINTS

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
kind, K = "tanh_rect", 3
T, dt, tau, rec_every = 5.0, 1e-3, 0.02, 2
SETS = {"theta50": "nullscan_sets_amp_theta50_tanh_rect_DNg100_T5.parquet",
        "manc50": "nullscan_sets_manc_manc50_tanh_rect_DNg100_T5.parquet"}
LEVELS = ["er", "degree", "hl", "leg_hl"]
AMPS = [0.5, 1.0]

nodes, e_real, mnmap = load_graph(tag); N = nodes.height
mn_idx = mnmap["idx"].to_numpy()
drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
sets = pl.read_parquet(R / SETS[tag])
role = nodes["role"].to_numpy(); hl = nodes["trumanHl"].fill_null("NA").to_numpy()
hls = sorted(h for h in np.unique(hl[role == "IN"]) if h not in ("NA", "TBD", "None", ""))
members = {h: np.flatnonzero((role == "IN") & (hl == h) & ~np.isin(np.arange(N), drive_idx)) for h in hls}
leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); d = mnmap["direction"].to_numpy()
pools = [(L, J, np.flatnonzero((leg == L) & (joint == J) & (d > 0)), np.flatnonzero((leg == L) & (joint == J) & (d < 0)))
         for L in LEGS for J in JOINTS]
pools = [p for p in pools if len(p[2]) and len(p[3])]
print(tag, "hemilineages", len(hls), "antagonist pairs", len(pools), flush=True)


def make_run(WE, WI):
    WE_j, WI_j = jnp.asarray(WE), jnp.asarray(WI); rec = jnp.asarray(mn_idx)

    @jax.jit
    def run(I, g, beta, key):
        r0 = 0.01 * jax.random.uniform(key, I.shape)
        a = dt / tau

        def step(carry, _):
            r, key = carry
            key, sub = jax.random.split(key)
            inp = g * (r @ WE_j.T + beta * (r @ WI_j.T)) + I + 0.01 * jax.random.normal(sub, r.shape)
            return (r + a * (-r + act(inp, kind)), key), None

        def chunk(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=rec_every)
            return carry, carry[0][:, rec]

        _, out = jax.lax.scan(chunk, (r0, key), None, length=int(T / dt) // rec_every)
        return jnp.transpose(out, (1, 0, 2))
    return run


graphs = [("real", 0, None)] + [(lvl, s, G / "nulls" / tag / f"{lvl}_s{s}.parquet") for lvl in LEVELS for s in range(5)]
rows = []; t0 = time.time()
for gname, seed, f in graphs:
    if f is not None and not f.exists():
        continue
    WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
    run = make_run(WE, WI)
    top = sets.filter((pl.col("graph") == gname) & (pl.col("seed") == seed) & pl.col("rhythmic")).sort("combined", descending=True).head(K)
    for si, (g, beta, drive) in enumerate(top.select("g", "beta", "drive").rows()):
        for cond in ["nodrive", "DNg100"]:
            for amp in AMPS:
                I = np.zeros((1 + len(hls), N), np.float32)
                if cond == "DNg100":
                    I[:, drive_idx] = drive
                for k, h in enumerate(hls):
                    I[1 + k, members[h]] += amp
                X = np.asarray(run(jnp.asarray(I), jnp.float32(g), jnp.float32(beta), jax.random.PRNGKey(si)))
                m = X[:, 500:, :].mean(1)                                   # 1-5 s mean rate per MN
                ef = np.stack([m[:, e].mean(1) - m[:, fl].mean(1) for _, _, e, fl in pools], 1)   # rows x pairs
                dm = ef[1:] - ef[0]
                for k, h in enumerate(hls):
                    for pi, (L, J, _, _) in enumerate(pools):
                        rows.append((tag, gname, seed, si, cond, amp, h, len(members[h]), L, J, float(dm[k, pi]), float(ef[0, pi])))
    print(gname, seed, f"t={time.time() - t0:.0f}s", flush=True)
df = pl.DataFrame(rows, schema=["connectome", "graph", "seed", "setting", "cond", "amp", "hl", "n_hl", "leg", "joint", "d_ext_minus_flx", "base_ext_minus_flx"], orient="row")
df.write_parquet(R / f"hl_activation_{tag}.parquet")

# pre-registered primary score
LAB = {"01A": 1, "04B": 1, "10B": 1, "13B": 1, "12B": 1, "20A.22A": 1, "08A": -1, "14A": -1, "16B": -1, "17A": -1, "19A": -1, "21A": -1}
prim = (df.filter((pl.col("cond") == "nodrive") & (pl.col("amp") == 1.0) & pl.col("joint").is_in(["FTi", "CTr"]))
        .group_by("graph", "seed", "hl").agg(pl.col("d_ext_minus_flx").mean().alias("bias")))
prim = prim.filter(pl.col("hl").is_in(list(LAB))).with_columns(pl.col("hl").replace_strict(LAB, default=0).alias("label"))
score = prim.group_by("graph", "seed").agg(((pl.col("bias").sign() == pl.col("label")).sum()).alias("n_correct"), pl.len().alias("n_labeled")).sort("graph", "seed")
score.write_parquet(R / f"hl_activation_{tag}_score.parquet")
with pl.Config(tbl_rows=40):
    print(prim.filter(pl.col("graph") == "real").sort("label", "bias"))
    print(score)
