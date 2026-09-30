"""Anatomical baseline for the pre-registered activation benchmark (section 3.11 of the notebook).

Tests the post-hoc explanation "tonic extension/flexion bias is set by transmitter sign x connection
number". For each graph and hemilineage H, the signed synaptic input from H's interneurons to each MN
pool is computed over 1 step (W) and 2 steps (W @ W), averaged per MN of the pool; the predicted bias
is the mean over legs and the FTi + CTr joints of [extensor-pool input - flexor-pool input]. Its sign is
scored against the 12 frozen labels exactly as the dynamical readout was, and compared with the
dynamical model's own bias (hl_activation_<tag>.parquet) by rank correlation across all hemilineages.
usage: act_baseline.py TAG
"""
import sys
import numpy as np, polars as pl, scipy.sparse as sp
from scipy.stats import spearmanr
from pathlib import Path

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
LAB = {"01A": 1, "04B": 1, "10B": 1, "13B": 1, "12B": 1, "20A.22A": 1, "08A": -1, "14A": -1, "16B": -1, "17A": -1, "19A": -1, "21A": -1}
nodes = pl.read_parquet(G / f"nodes_{tag}.parquet"); mnmap = pl.read_parquet(G / f"mn_joint_map_{tag}.parquet")
N = nodes.height; sign = nodes["sign"].to_numpy().astype(float); role = nodes["role"].to_numpy()
hl = nodes["trumanHl"].fill_null("NA").to_numpy()
hls = sorted(h for h in np.unique(hl[role == "IN"]) if h not in ("NA", "TBD", "None", ""))
leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); d = mnmap["direction"].to_numpy(); midx = mnmap["idx"].to_numpy()
pairs = [(midx[(leg == L) & (joint == J) & (d > 0)], midx[(leg == L) & (joint == J) & (d < 0)]) for L in np.unique(leg) for J in ["FTi", "CTr"]]
pairs = [p for p in pairs if len(p[0]) and len(p[1])]
Hm = np.zeros((N, len(hls)))
for k, h in enumerate(hls):
    Hm[(role == "IN") & (hl == h), k] = 1.0
act = pl.read_parquet(R / f"hl_activation_{tag}.parquet") if (R / f"hl_activation_{tag}.parquet").exists() else None

graphs = [("real", 0, G / f"edges_{tag}.parquet")] + [(lv, s, G / "nulls" / tag / f"{lv}_s{s}.parquet") for lv in ["er", "degree", "hl", "leg_hl"] for s in range(5)]
rows = []
for name, seed, f in graphs:
    if not f.exists():
        continue
    e = pl.read_parquet(f)
    W = sp.csr_matrix((e["weight"].to_numpy() * sign[e["pre"].to_numpy()], (e["post"].to_numpy(), e["pre"].to_numpy())), shape=(N, N))
    one = W @ Hm; two = W @ one                      # (N x H): summed signed input from each hemilineage
    for step, M in [("1step", one), ("2step", two), ("1+2step", one / (np.abs(one).max() + 1e-12) + two / (np.abs(two).max() + 1e-12))]:
        bias = np.array([np.mean([M[ex, k].mean() - M[fl, k].mean() for ex, fl in pairs]) for k in range(len(hls))])
        lab = np.array([LAB.get(h, 0) for h in hls])
        ncor = int(((np.sign(bias) == lab) & (lab != 0)).sum())
        rho = np.nan
        if act is not None:
            ab = (act.filter((pl.col("graph") == name) & (pl.col("seed") == seed) & (pl.col("cond") == "nodrive") & (pl.col("amp") == 1.0)
                             & pl.col("joint").is_in(["FTi", "CTr"])).group_by("hl").agg(pl.col("d_ext_minus_flx").mean().alias("b")))
            mb = dict(zip(ab["hl"].to_list(), ab["b"].to_list()))
            x = [bias[k] for k, h in enumerate(hls) if h in mb]; y = [mb[h] for h in hls if h in mb]
            rho = float(spearmanr(x, y).correlation) if len(x) > 3 else np.nan
        rows.append({"connectome": tag, "graph": name, "seed": seed, "predictor": step, "n_correct": ncor, "n_labeled": int((lab != 0).sum()),
                     "spearman_vs_model": rho})
df = pl.DataFrame(rows); df.write_parquet(R / f"act_baseline_{tag}.parquet")
with pl.Config(tbl_rows=80):
    print(df.group_by("graph", "predictor").agg(pl.col("n_correct").mean(), pl.col("n_correct").min().alias("min"), pl.col("n_correct").max().alias("max"),
                                               pl.col("spearman_vs_model").mean()).sort("predictor", "graph"))
