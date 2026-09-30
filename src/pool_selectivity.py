"""Neuron-level antagonist-pool selectivity of premotor interneurons.

For every premotor interneuron and every leg x joint whose two antagonist pools it contacts with >= 10
synapses in total, selectivity S = |e - f| / (e + f), where e and f are its synapse counts onto the
extensor- and flexor-direction pools divided by the number of MNs in each pool (so pool size does not
create selectivity). S = 1: contacts only one antagonist pool; S = 0: contacts both equally.
Reported per graph: mean S, fraction of neuron-joint pairs with S >= 0.8 ("pool-specific"), and the
same by transmitter class. Real graph vs null graphs (NULLDIR, levels, n each).
usage: pool_selectivity.py TAG NULLDIR N LEVELS
"""
import sys
import numpy as np, polars as pl
from pathlib import Path

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag, ndir, NN, LEVELS = sys.argv[1], G / sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
nodes = pl.read_parquet(G / f"nodes_{tag}.parquet"); m = pl.read_parquet(G / f"mn_joint_map_{tag}.parquet").filter(pl.col("joint").is_not_null())
N = nodes.height; role = nodes["role"].to_numpy(); sign = nodes["sign"].to_numpy()
pool_id = np.full(N, -1); pool_dir = np.zeros(N, int); keys = {}
for idx, lg, j, d in m.select("idx", "leg", "joint", "direction").iter_rows():
    k = keys.setdefault((lg, j), len(keys)); pool_id[idx] = k; pool_dir[idx] = d
npool = np.zeros((len(keys), 2))
for idx, d in zip(m["idx"].to_numpy(), m["direction"].to_numpy()):
    npool[pool_id[idx], 0 if d > 0 else 1] += 1
ok_pair = (npool > 0).all(1)


def selectivity(e):
    pre, post, w = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy().astype(float)
    s = (role[pre] == "IN") & (pool_id[post] >= 0)
    pre, post, w = pre[s], post[s], w[s]
    key = pre.astype(np.int64) * len(keys) + pool_id[post]
    ext = np.bincount(key, weights=w * (pool_dir[post] > 0), minlength=N * len(keys)).reshape(N, len(keys))
    flx = np.bincount(key, weights=w * (pool_dir[post] < 0), minlength=N * len(keys)).reshape(N, len(keys))
    tot = ext + flx
    sel_mask = (tot >= 10) & ok_pair[None, :]
    en, fn = ext / np.maximum(npool[:, 0], 1)[None, :], flx / np.maximum(npool[:, 1], 1)[None, :]
    S = np.abs(en - fn) / np.maximum(en + fn, 1e-12)
    ii, jj = np.nonzero(sel_mask)
    Sv = S[ii, jj]; sg = sign[ii]
    out = {"n_pairs": int(len(Sv)), "mean_S": float(Sv.mean()), "frac_specific": float((Sv >= 0.8).mean())}
    for lab, msk in [("exc", sg > 0), ("inh", sg < 0)]:
        out[f"mean_S_{lab}"] = float(Sv[msk].mean()); out[f"frac_specific_{lab}"] = float((Sv[msk] >= 0.8).mean())
    return out


rows = [{"graph": "real", "seed": 0, **selectivity(pl.read_parquet(G / f"edges_{tag}.parquet"))}]
for lv in LEVELS:
    for s in range(NN):
        f = ndir / f"{lv}_s{s}.parquet"
        if f.exists():
            rows.append({"graph": lv, "seed": s, **selectivity(pl.read_parquet(f))})
df = pl.DataFrame(rows); df.write_parquet(R / f"pool_selectivity_{tag}.parquet")
with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=220, float_precision=3):
    print(tag); print(df.group_by("graph").agg(pl.len(), *[pl.col(c).mean() for c in df.columns if c.startswith(("mean_S", "frac_"))],
                                              pl.col("frac_specific").max().alias("max_frac_specific")).sort("graph"))
