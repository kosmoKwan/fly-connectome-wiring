"""Hemilineage knockouts against size- and composition-matched random deletions.

For every hemilineage H (interneurons with that Truman hemilineage label) we silence all of
its neurons, and compare with random deletions of the SAME number of interneurons drawn from
outside H, stratified so that each deleted set matches H neuron-for-neuron in
  A: segment x side x transmitter sign
  B: segment x side x transmitter sign x premotor status (direct synapses onto a leg MN)
If a stratum is too small, the match falls back to a coarser stratum (B -> A -> sign only);
the number of fallbacks is recorded. Scored as knockout.py: the K best settings of the intact
real graph by rhythm x antiphase, same simulator, same metrics; deltas vs intact per setting.

usage: knockout_hl.py TAG N_CTRL [HL,HL,...]   (optional list restricts the hemilineages; stage 2)
"""
import os, sys, time, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
import jax, jax.numpy as jnp
from pathlib import Path
from openloop import load_graph, build_W, act, analyse

R = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
N_CTRL = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
kind, K = "tanh_rect", 3
T, dt, tau, rec_every, B = 5.0, 1e-3, 0.02, 2, 64
SETS = {"theta50": "nullscan_sets_theta50_tanh_rect_DNg100_T5.parquet",
        "manc50": "nullscan_sets_manc_manc50_tanh_rect_DNg100_T5.parquet"}

nodes, edges, mnmap = load_graph(tag)
WE, WI = build_W(nodes, edges); N = nodes.height
mn_idx = mnmap["idx"].to_numpy()
drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
sets = pl.read_parquet(R / SETS[tag]).filter((pl.col("graph") == "real") & pl.col("rhythmic")).sort("combined", descending=True).head(K)
params = sets.select("g", "beta", "drive").rows()
print("param sets:", params, flush=True)

role = nodes["role"].to_numpy(); sign = nodes["sign"].to_numpy()
seg = nodes["leg_seg"].fill_null("NA").to_numpy(); side = nodes["side"].fill_null("NA").to_numpy()
hl = nodes["trumanHl"].fill_null("NA").to_numpy()
is_mn = role == "MN"
premotor = np.zeros(N, bool); _pre, _post = edges["pre"].to_numpy(), edges["post"].to_numpy(); premotor[np.unique(_pre[is_mn[_post]])] = True
is_in = (role == "IN") & ~np.isin(np.arange(N), drive_idx)
keyA = np.array([f"{a}|{b}|{c}" for a, b, c in zip(seg, side, sign)], dtype=object)
keyB = np.array([f"{k}|{int(p)}" for k, p in zip(keyA, premotor)], dtype=object)
keyS = sign.astype(str).astype(object)

rng = np.random.default_rng(11 if tag == "theta50" else 12)
hls = sorted(h for h in np.unique(hl[is_in]) if h not in ("NA", "TBD", "None", ""))
if ONLY:
    hls = [h for h in hls if h in ONLY]


STRATA = {id(k): {v: np.flatnonzero(is_in & (k == v)) for v in np.unique(k[is_in])} for k in (keyA, keyB, keyS)}


def matched(members, keys_chain):
    """Draw a random set outside `members`, matching members stratum by stratum; fall back to coarser keys."""
    taken = set(members.tolist()); pick = []; need = members; fb = 0
    for li, key in enumerate(keys_chain):
        left = []
        for k in np.unique(key[need]) if len(need) else []:
            want = need[key[need] == k]
            cand = STRATA[id(key)].get(k, np.array([], np.int64))
            cand = cand[~np.isin(cand, list(taken))]
            take = min(len(want), len(cand))
            if take:
                sel = rng.choice(cand, take, replace=False); pick.extend(sel.tolist()); taken.update(sel.tolist())
            left.extend(want[take:].tolist())
        if li > 0:
            fb += len(need) - len(left)
        need = np.array(left, np.int64)
    return np.array(pick, np.int64), fb + len(need)


units = [("intact", "none", "", np.array([], np.int64), 0)]
for h in hls:
    mem = np.flatnonzero(is_in & (hl == h))
    units.append((f"HL:{h}", "hemilineage", h, mem, 0))
    for i in range(N_CTRL):
        idx, fb = matched(mem, [keyA, keyS]); units.append((f"ctrlA:{h}:{i}", "ctrl_A", h, idx, fb))
        idx, fb = matched(mem, [keyB, keyA, keyS]); units.append((f"ctrlB:{h}:{i}", "ctrl_B", h, idx, fb))
print("hemilineages", len(hls), "units", len(units), flush=True)

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
        for i, u in enumerate(cu):
            mask[i, u[3]] = 0.0
        X = np.asarray(run_batch(jnp.asarray(mask), jnp.float32(g), jnp.float32(beta), jnp.float32(drive), jax.random.PRNGKey(pi)))
        for i, (name, uk, h, idx, fb) in enumerate(cu):
            r = analyse(X[i], fs=500.0, mnmap=mnmap)
            rows.append({"param_set": pi, "unit": name, "unit_kind": uk, "hl": h, "n_neurons": int(len(idx)), "n_fallback": int(fb),
                         "mn_mean": r["mn_mean"], "mn_osc_frac": r["mn_osc_frac"], "rhythm": r.get("rhythm", 0.0),
                         "freq": r.get("freq", np.nan), "fe_corr": r.get("flex_ext_corr_median", np.nan),
                         "gmean": r.get("alt_amp_mean", np.nan)})
        if (c // B) % 20 == 0:
            print(f"param {pi} units {c + len(cu)}/{len(units)} t={time.time() - t0:.0f}s", flush=True)
df = pl.DataFrame(rows).with_columns((-pl.col("fe_corr")).fill_nan(0).alias("antiphase"), pl.col("gmean").fill_nan(0))
base = df.filter(pl.col("unit") == "intact").select("param_set", pl.col("antiphase").alias("a0"), pl.col("rhythm").alias("r0"), pl.col("gmean").alias("m0"))
df = df.join(base, on="param_set").with_columns((pl.col("antiphase") - pl.col("a0")).alias("d_antiphase"),
                                                (pl.col("rhythm") - pl.col("r0")).alias("d_rhythm"), (pl.col("gmean") - pl.col("m0")).alias("d_gmean"))
suffix = f"_{'-'.join(ONLY)}" if ONLY else ""
df.write_parquet(R / f"knockout_hl_{tag}_n{N_CTRL}{suffix}.parquet")

# per unit: mean delta over the K settings; per hemilineage: add-one lower-tail p vs its own matched controls
u = df.filter(pl.col("unit") != "intact").group_by("unit", "unit_kind", "hl", "n_neurons", "n_fallback").agg(
    pl.col("d_antiphase").mean(), pl.col("d_rhythm").mean(), pl.col("d_gmean").mean())
out = []
for h in hls:
    x = u.filter(pl.col("hl") == h); hv = x.filter(pl.col("unit_kind") == "hemilineage").row(0, named=True)
    row = {"hl": h, "n": hv["n_neurons"], "d_antiphase": hv["d_antiphase"], "d_rhythm": hv["d_rhythm"], "d_gmean": hv["d_gmean"]}
    for ck in ["ctrl_A", "ctrl_B"]:
        c = x.filter(pl.col("unit_kind") == ck)
        for m in ["d_antiphase", "d_gmean", "d_rhythm"]:
            v = c[m].to_numpy()
            row[f"{ck}_{m}_med"] = float(np.median(v))
            row[f"{ck}_{m}_p_low"] = float((1 + (v <= hv[m]).sum()) / (1 + len(v)))
        row[f"{ck}_fallback_mean"] = float(c["n_fallback"].mean())
    out.append(row)
S = pl.DataFrame(out).sort("ctrl_B_d_antiphase_p_low")
S.write_parquet(R / f"knockout_hl_{tag}_n{N_CTRL}{suffix}_summary.parquet")
with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250):
    print(S.select("hl", "n", "d_antiphase", "ctrl_B_d_antiphase_med", "ctrl_B_d_antiphase_p_low", "d_gmean", "ctrl_B_d_gmean_p_low",
                   "d_rhythm", "ctrl_B_d_rhythm_med", "ctrl_B_d_rhythm_p_low", "ctrl_B_fallback_mean"))
