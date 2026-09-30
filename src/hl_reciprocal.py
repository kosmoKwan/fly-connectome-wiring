"""Do hemilineages implement reciprocal innervation (Sherrington): direct drive to one antagonist pool
and disynaptic inhibition of the other?

For every hemilineage H and joint (T2+T3 legs; ThC, CTr, FTi):
  D  = log2((direct synapses H -> +1 pool + 1) / (direct synapses H -> -1 pool + 1))
  I2 = log2((disynaptic inhibition of -1 pool + 1) / (disynaptic inhibition of +1 pool + 1)),
       disynaptic inhibition of pool P = sum over inhibitory interneurons k of w(H->k) * w(k->P) / 1000
For an excitatory H, reciprocal innervation means D and I2 have the SAME sign (it excites one pool and,
through inhibitory interneurons, inhibits the other). For an inhibitory H the direct term already
inhibits, so reciprocal organisation means D and I2 have OPPOSITE signs. We report the signed alignment
A = sign(H) * D * I2 (positive = reciprocal), and z-scores of D, I2 and A against null graphs.
usage: hl_reciprocal.py TAG NULLDIR N LEVEL
"""
import sys
import numpy as np, polars as pl, scipy.sparse as sp
from pathlib import Path

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag, ndir, NN, LEVEL = sys.argv[1], G / sys.argv[2], int(sys.argv[3]), sys.argv[4]
nodes = pl.read_parquet(G / f"nodes_{tag}.parquet"); m = pl.read_parquet(G / f"mn_joint_map_{tag}.parquet")
N = nodes.height; sign = nodes["sign"].to_numpy(); role = nodes["role"].to_numpy()
hl = nodes["trumanHl"].fill_null("NA").to_numpy()
hls = sorted(h for h in np.unique(hl[role == "IN"]) if h not in ("NA", "TBD", "None", ""))
Hm = np.zeros((N, len(hls)), np.float32)
for k, h in enumerate(hls):
    Hm[(role == "IN") & (hl == h), k] = 1.0
Hs = sp.csr_matrix(Hm)
inh = sp.diags(((role == "IN") & (sign < 0)).astype(np.float32))
pools = {}
for J in ["ThC", "CTr", "FTi"]:
    for d in (1, -1):
        idx = m.filter((pl.col("seg") != "T1") & (pl.col("joint") == J) & (pl.col("direction") == d))["idx"].to_numpy()
        v = np.zeros(N, np.float32); v[idx] = 1.0; pools[(J, d)] = sp.csr_matrix(v)
hsign = np.array([np.sign(sign[(role == "IN") & (hl == h)].mean()) for h in hls])


def stats(e):
    W = sp.csr_matrix((e["weight"].to_numpy().astype(np.float32), (e["post"].to_numpy(), e["pre"].to_numpy())), shape=(N, N))  # unsigned, W[post, pre]
    direct = {k: np.asarray((v @ W @ Hs).todense()).ravel() for k, v in pools.items()}            # synapses H -> pool
    di = {k: np.asarray((v @ W @ inh @ W @ Hs).todense()).ravel() / 1000.0 for k, v in pools.items()}  # H -> inhibitory IN -> pool
    out = {}
    for J in ["ThC", "CTr", "FTi"]:
        D = np.log2((direct[(J, 1)] + 1) / (direct[(J, -1)] + 1))
        I2 = np.log2((di[(J, -1)] + 1) / (di[(J, 1)] + 1))
        out[J] = (D, I2, hsign * D * I2)
    tot = sum(direct.values())
    return out, tot


real, tot = stats(pl.read_parquet(G / f"edges_{tag}.parquet"))
nul = [stats(pl.read_parquet(ndir / f"{LEVEL}_s{s}.parquet"))[0] for s in range(NN) if (ndir / f"{LEVEL}_s{s}.parquet").exists()]
rows = []
for J in ["ThC", "CTr", "FTi"]:
    for q, name in enumerate(["D", "I2", "A"]):
        rv = real[J][q]; nv = np.stack([n[J][q] for n in nul]); mu, sd = nv.mean(0), nv.std(0) + 1e-9
        p = (1 + (np.abs(nv - mu) >= np.abs(rv - mu)[None]).sum(0)) / (1 + len(nv))
        for k, h in enumerate(hls):
            if tot[k] >= 1000:
                rows.append({"connectome": tag, "hl": h, "sign": int(hsign[k]), "mn_syn": int(tot[k]), "joint": J, "stat": name,
                             "real": float(rv[k]), "null_mean": float(mu[k]), "z": float((rv[k] - mu[k]) / sd[k]), "p": float(p[k]), "n_null": len(nv)})
df = pl.DataFrame(rows); df.write_parquet(R / f"hl_reciprocal_{tag}.parquet")
w = df.pivot(on=["stat", "joint"], index=["hl", "sign", "mn_syn"], values="real").sort("mn_syn", descending=True)
with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=260, float_precision=2):
    print(f"{tag}: real D (direct +1/-1), I2 (disynaptic inhibition of -1 vs +1), A = sign*D*I2 (>0 reciprocal); T2+T3 legs")
    print(w)
    a = df.filter(pl.col("stat") == "A")
    print("fraction of HL x joint cells with A>0 (reciprocal): real", round(float((a["real"] > 0).mean()), 2), " null mean", round(float((a["null_mean"] > 0).mean()), 2))
    big = a.filter(pl.col("real").abs() > 0.5)
    print("among cells with |A|>0.5: fraction reciprocal", round(float((big["real"] > 0).mean()), 2), "n", big.height)
