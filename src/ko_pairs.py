"""Pair-level localisation and graded suppression for a candidate hemilineage (default 01A).

Same simulator, settings (real top-3 by rhythm x antiphase) and matching rule (B: segment x side x sign
x premotor) as knockout_hl.py. Units: intact; the hemilineage scaled to output gain m in {0.75, 0.5,
0.25, 0} (graded suppression: its rates are multiplied by m at every step); 20 fresh matched random
deletions (m = 0). For every unit, setting and antagonist pair we store the pool SDs, balance, correlation,
gates and pool mean rates, so a loss of antiphase can be attributed to (i) a pool going quiet, (ii) a
change of sign of the correlation with both pools still modulated, or (iii) the pair becoming unmeasurable.
usage: ko_pairs.py TAG [HL]
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
import jax, jax.numpy as jnp
from pathlib import Path
from openloop import load_graph, build_W, act, analyse, LEGS, JOINTS

R = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
H = sys.argv[2] if len(sys.argv) > 2 else "01A"
kind, K, N_CTRL = "tanh_rect", 3, 20
T, dt, tau, rec_every = 5.0, 1e-3, 0.02, 2
SETS = {"theta50": "nullscan_sets_theta50_tanh_rect_DNg100_T5.parquet", "manc50": "nullscan_sets_manc_manc50_tanh_rect_DNg100_T5.parquet"}
nodes, edges, mnmap = load_graph(tag); N = nodes.height
WE, WI = build_W(nodes, edges)
mn_idx = mnmap["idx"].to_numpy(); drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
params = pl.read_parquet(R / SETS[tag]).filter((pl.col("graph") == "real") & pl.col("rhythmic")).sort("combined", descending=True).head(K).select("g", "beta", "drive").rows()
role = nodes["role"].to_numpy(); sign = nodes["sign"].to_numpy()
seg = nodes["leg_seg"].fill_null("NA").to_numpy(); side = nodes["side"].fill_null("NA").to_numpy(); hl = nodes["trumanHl"].fill_null("NA").to_numpy()
is_mn = role == "MN"; pre, post = edges["pre"].to_numpy(), edges["post"].to_numpy()
premotor = np.zeros(N, bool); premotor[np.unique(pre[is_mn[post]])] = True
is_in = (role == "IN") & ~np.isin(np.arange(N), drive_idx)
keyA = np.array([f"{a}|{b}|{c}" for a, b, c in zip(seg, side, sign)], dtype=object)
keyB = np.array([f"{k}|{int(p)}" for k, p in zip(keyA, premotor)], dtype=object)
keyS = sign.astype(str).astype(object)
STRATA = {id(k): {v: np.flatnonzero(is_in & (k == v)) for v in np.unique(k[is_in])} for k in (keyA, keyB, keyS)}
rng = np.random.default_rng(222)


def matched(members, keys_chain):
    taken = set(members.tolist()); pick = []; need = members
    for key in keys_chain:
        left = []
        for k in np.unique(key[need]) if len(need) else []:
            want = need[key[need] == k]
            cand = STRATA[id(key)].get(k, np.array([], np.int64)); cand = cand[~np.isin(cand, list(taken))]
            take = min(len(want), len(cand))
            if take:
                sel = rng.choice(cand, take, replace=False); pick.extend(sel.tolist()); taken.update(sel.tolist())
            left.extend(want[take:].tolist())
        need = np.array(left, np.int64)
    return np.array(pick, np.int64)


mem = np.flatnonzero(is_in & (hl == H))
units = [("intact", 1.0, np.array([], np.int64))] + [(f"{H}_x{m}", m, mem) for m in (0.75, 0.5, 0.25, 0.0)] \
      + [(f"ctrl{i}", 0.0, matched(mem, [keyB, keyA, keyS])) for i in range(N_CTRL)]
leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); d = mnmap["direction"].to_numpy()
PAIRS = [(L, J, (leg == L) & (joint == J) & (d > 0), (leg == L) & (joint == J) & (d < 0)) for L in LEGS for J in JOINTS]
PAIRS = [p for p in PAIRS if p[2].any() and p[3].any()]
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


rows, summ = [], []
for pi, (g, beta, drive) in enumerate(params):
    mask = np.ones((len(units), N), np.float32)
    for i, (_, m, idx) in enumerate(units):
        mask[i, idx] = m
    X = np.asarray(run_batch(jnp.asarray(mask), jnp.float32(g), jnp.float32(beta), jnp.float32(drive), jax.random.PRNGKey(pi)))
    for i, (name, m, idx) in enumerate(units):
        Xw = X[i, 500:]; Xc = Xw - Xw.mean(0)
        r = analyse(X[i], fs=500.0, mnmap=mnmap)
        summ.append({"setting": pi, "unit": name, "gain": m, "n": int(len(idx)), "rhythm": r.get("rhythm", 0.0), "freq": r.get("freq", np.nan),
                     "antiphase": -r.get("flex_ext_corr_median", np.nan), "gmean": r.get("alt_amp_mean", np.nan), "mn_mean": r["mn_mean"]})
        for L, J, e, f in PAIRS:
            a, b = Xc[:, e].mean(1), Xc[:, f].mean(1); se, sf = float(a.std()), float(b.std())
            c = float(np.corrcoef(a, b)[0, 1]) if se > 1e-4 and sf > 1e-4 else np.nan
            bal = min(se, sf) / max(se, sf) if max(se, sf) > 0 else 0.0
            rows.append({"setting": pi, "unit": name, "gain": m, "leg": L, "joint": J, "sd_ext": se, "sd_flx": sf, "balance": bal, "corr": c,
                         "gate": bool(bal >= 0.2 and min(se, sf) >= 0.001), "mean_ext": float(Xw[:, e].mean()), "mean_flx": float(Xw[:, f].mean())})
    print("setting", pi, "done", flush=True)
P = pl.DataFrame(rows); S = pl.DataFrame(summ)
P.write_parquet(R / f"ko_pairs_{tag}_{H}.parquet"); S.write_parquet(R / f"ko_pairs_{tag}_{H}_summary.parquet")
with pl.Config(tbl_rows=40, tbl_width_chars=200):
    print(S.filter(~pl.col("unit").str.starts_with("ctrl")).group_by("unit", "gain").agg(pl.col("antiphase", "gmean", "rhythm", "mn_mean").mean()).sort("gain", descending=True))
    print(S.filter(pl.col("unit").str.starts_with("ctrl")).select(pl.col("antiphase", "gmean", "rhythm", "mn_mean").mean()))
    base = P.filter(pl.col("unit") == "intact").select("setting", "leg", "joint", pl.col("corr").alias("c0"), pl.col("gate").alias("g0"),
                                                      pl.col("sd_ext").alias("se0"), pl.col("sd_flx").alias("sf0"))
    ko = P.filter(pl.col("unit") == f"{H}_x0.0").join(base, on=["setting", "leg", "joint"])
    print(ko.filter(pl.col("g0") & (pl.col("c0") < -0.3)).select("setting", "leg", "joint", "c0", "corr", "se0", "sd_ext", "sf0", "sd_flx", "gate").sort("setting", "leg", "joint"))
