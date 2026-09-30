"""Hemilineage -> motor-pool synergy atlas (structural; direct premotor->MN synapses).

For every hemilineage H with >= 1000 direct synapses onto mapped leg MNs, and for each leg group
(T1; T2+T3) and joint (ThC, CTr, FTi), the directional bias is L = log2((syn onto +1 pool + 1) /
(syn onto -1 pool + 1)), where +1 = promotor (ThC), depressor/extensor (CTr), extensor (FTi).
The same quantity is computed in every null graph of the given null directory; per H and joint we
report the real value, null mean/sd, z and a two-sided empirical p (add-one). Stance-like synergy for
T2+T3 = ThC < 0 (remotion), CTr > 0, FTi > 0 relative to nulls.
usage: hl_synergy.py TAG NULLDIR N_PER_LEVEL LEVELS
"""
import sys
import numpy as np, polars as pl
from pathlib import Path

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag = sys.argv[1]; ndir = G / sys.argv[2]; NN = int(sys.argv[3]); LEVELS = sys.argv[4].split(",")
nodes = pl.read_parquet(G / f"nodes_{tag}.parquet"); m = pl.read_parquet(G / f"mn_joint_map_{tag}.parquet")
N = nodes.height
hl = nodes["trumanHl"].fill_null("NA").to_numpy(); role = nodes["role"].to_numpy()
hls = sorted(h for h in np.unique(hl[role == "IN"]) if h not in ("NA", "TBD", "None", ""))
hidx = {h: k for k, h in enumerate(hls)}
pre_h = np.array([hidx.get(h, -1) if r == "IN" else -1 for h, r in zip(hl, role)])
seg = np.full(N, "", dtype=object); jnt = np.full(N, "", dtype=object); dr = np.zeros(N, int)
for idx, s, j, d in m.select("idx", "seg", "joint", "direction").iter_rows():
    seg[idx] = "T1" if s == "T1" else "T23"; jnt[idx] = j or ""; dr[idx] = d
CELLS = [(g, j) for g in ("T1", "T23") for j in ("ThC", "CTr", "FTi")]


def table(e):
    pre, post, w = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy().astype(float)
    ok = (pre_h[pre] >= 0) & (jnt[post] != "") & (dr[post] != 0)
    pre, post, w = pre[ok], post[ok], w[ok]
    out = np.zeros((len(hls), len(CELLS), 2)); tot = np.zeros(len(hls))
    np.add.at(tot, pre_h[pre], w)
    for c, (g, j) in enumerate(CELLS):
        s = (seg[post] == g) & (jnt[post] == j)
        np.add.at(out[:, c, 0], pre_h[pre[s & (dr[post] > 0)]], w[s & (dr[post] > 0)])
        np.add.at(out[:, c, 1], pre_h[pre[s & (dr[post] < 0)]], w[s & (dr[post] < 0)])
    return np.log2((out[..., 0] + 1) / (out[..., 1] + 1)), tot


real, tot = table(pl.read_parquet(G / f"edges_{tag}.parquet"))
rows = []
for lv in LEVELS:
    nul = np.stack([table(pl.read_parquet(ndir / f"{lv}_s{s}.parquet"))[0] for s in range(NN) if (ndir / f"{lv}_s{s}.parquet").exists()])
    mu, sd = nul.mean(0), nul.std(0) + 1e-9
    p = (1 + (np.abs(nul - mu) >= np.abs(real - mu)[None]).sum(0)) / (1 + len(nul))
    for k, h in enumerate(hls):
        if tot[k] < 1000:
            continue
        for c, (g, j) in enumerate(CELLS):
            rows.append({"connectome": tag, "null": lv, "n_null": len(nul), "hl": h, "mn_syn": int(tot[k]), "legs": g, "joint": j,
                         "real_log2": float(real[k, c]), "null_mean": float(mu[k, c]), "null_sd": float(sd[k, c]),
                         "z": float((real[k, c] - mu[k, c]) / sd[k, c]), "p": float(p[k, c])})
df = pl.DataFrame(rows); df.write_parquet(R / f"hl_synergy_{tag}.parquet")
with pl.Config(tbl_rows=200, tbl_width_chars=200):
    x = df.filter((pl.col("null") == LEVELS[-1]) & (pl.col("legs") == "T23")).pivot(on="joint", index=["hl", "mn_syn"], values="z").sort("mn_syn", descending=True)
    print(f"{tag}: z vs {LEVELS[-1]} nulls, T2+T3 legs (ThC<0 & CTr>0 & FTi>0 = stance-like)"); print(x)
