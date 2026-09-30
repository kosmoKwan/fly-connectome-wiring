"""Reciprocal-innervation index (as diag.py) for every current graph of one connectome.

For each leg x joint with both antagonist pools, the signed 1-step + 2-step influence of every
neuron on the extensor pool and on the flexor pool is computed; the index is minus their
correlation (positive = upstream neurons push antagonists in opposite directions). Per graph we
report the median over joints. Graph sets: real; every null family present in nulls/<tag>
(including leg_hl_realdyads and the weight-stratified *_w nulls when present); every surgery graph.
usage: diag_struct2.py TAG
"""
import sys, re
import numpy as np, polars as pl, scipy.sparse as sp
from pathlib import Path

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
nodes = pl.read_parquet(G / f"nodes_{tag}.parquet"); e_real = pl.read_parquet(G / f"edges_{tag}.parquet")
mnmap = pl.read_parquet(G / f"mn_joint_map_{tag}.parquet")
N = nodes.height; sign = nodes["sign"].to_numpy().astype(np.float64)


def recip_index(edges):
    pre, post, w = edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy().astype(np.float64)
    s = sign[pre]
    if "esign" in edges.columns:
        es = edges["esign"].to_numpy(); s = np.where(es != 0, es, s)
    W = sp.csr_matrix((w * s, (post, pre)), shape=(N, N))
    out = {}
    for (lg, j), grp in mnmap.filter(pl.col("joint").is_not_null()).group_by(["leg", "joint"]):
        ext = grp.filter(pl.col("direction") > 0)["idx"].to_numpy(); flx = grp.filter(pl.col("direction") < 0)["idx"].to_numpy()
        if not len(ext) or not len(flx):
            continue
        def infl(tgt):
            r1 = np.asarray(W[tgt].mean(axis=0)).ravel()
            r2 = np.asarray((W[tgt] @ W).mean(axis=0)).ravel()
            return r1 / (np.abs(r1).max() + 1e-12) + r2 / (np.abs(r2).max() + 1e-12)
        ie, if_ = infl(ext), infl(flx)
        m = (ie != 0) | (if_ != 0)
        out[f"{lg}_{j}"] = float(-np.corrcoef(ie[m], if_[m])[0, 1]) if m.sum() > 10 else np.nan
    v = np.array([x for x in out.values() if np.isfinite(x)])
    return float(np.median(v)), out


files = [("real", "real", None)]
ndir = G / "nulls" / tag
for f in sorted(ndir.glob("*_s[0-9].parquet")):
    files.append((f.stem, "null", f))
wdir = G / "nulls" / f"{tag}_w"
for f in sorted(wdir.glob("*_s[0-9].parquet")) if wdir.exists() else []:
    files.append((f.stem, "null", f))
sdir = G / ("surgery" if tag == "theta50" else f"surgery_{tag}")
for f in sorted(sdir.glob("*_s[0-9].parquet")):
    files.append((f.stem, "surgery", f))
rows, per_joint = [], []
for name, kind, f in files:
    med, per = recip_index(e_real if f is None else pl.read_parquet(f))
    rows.append({"connectome": tag, "graph": name, "kind": kind, "level": re.sub(r"_s\d+$", "", name), "recip_innervation": med})
    per_joint += [{"connectome": tag, "graph": name, "pair": k, "recip_innervation": v} for k, v in per.items()]
    print(name, round(med, 3), flush=True)
df = pl.DataFrame(rows); df.write_parquet(R / f"diag_recip_innervation_v2_{tag}.parquet")
pl.DataFrame(per_joint).write_parquet(R / f"diag_recip_innervation_v2_{tag}_perjoint.parquet")
with pl.Config(tbl_rows=40):
    print(df.group_by("kind", "level").agg(pl.len(), pl.col("recip_innervation").mean().alias("mean"), pl.col("recip_innervation").min().alias("min"),
                                            pl.col("recip_innervation").max().alias("max")).sort("mean", descending=True))
