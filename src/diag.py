"""Diagnostics.
(1) noise: intact network, 256 noise realisations per parameter set -> spread of alternation metrics.
(2) reciprocal-innervation index: for every (leg, joint) antagonist pair, compute each upstream
    neuron's signed 1-step + 2-step influence on the extensor group and on the flexor group;
    index = -corr over upstream neurons (1 = influences on antagonists are opposite).
"""
import os, sys, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl, scipy.sparse as sp
from pathlib import Path
from openloop import load_graph, build_W, simulate, analyse

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
nodes, e_real, mnmap = load_graph("theta50"); N = nodes.height
sign = nodes["sign"].to_numpy().astype(np.float32)
which = sys.argv[1] if len(sys.argv) > 1 else "both"

if which in ("noise", "both"):
    drive = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
    params = pl.read_parquet(R / "knockout_tanh_rect.parquet").filter(pl.col("unit") == "intact").sort("param_set").select("g", "beta", "drive").rows()
    rows = []
    for gname, f in [("real", None), ("leg_hl_s0", G / "nulls/theta50/leg_hl_s0.parquet")]:
        WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
        for pi, p in enumerate(params):
            for chunk in range(4):
                P = np.tile(np.array(p, np.float32), (64, 1))
                X = simulate(WE, WI, drive, P, T=5.0, record_idx=mnmap["idx"].to_numpy(), seed=1000 * pi + chunk)
                for i in range(64):
                    r = analyse(X[i], fs=500.0, mnmap=mnmap)
                    rows.append({"graph": gname, "param_set": pi, "rep": chunk * 64 + i, "rhythm": r.get("rhythm", 0.0),
                                 "antiphase": -r.get("flex_ext_corr_median", np.nan), "alt_amp": r.get("alt_amp", np.nan),
                                 "n_pairs_amp": r.get("n_pairs_amp", 0), "mn_osc_frac": r["mn_osc_frac"], "freq": r.get("freq", np.nan)})
    df = pl.DataFrame(rows); df.write_parquet(R / "diag_noise.parquet")
    print(df.group_by("graph", "param_set").agg(pl.col("antiphase").fill_nan(0).mean().alias("alt_mean"), pl.col("antiphase").fill_nan(0).std().alias("alt_sd"),
          pl.col("antiphase").fill_nan(0).quantile(0.02).alias("alt_p2"), pl.col("alt_amp").fill_nan(0).mean().alias("altamp_mean"),
          pl.col("alt_amp").fill_nan(0).std().alias("altamp_sd"), pl.col("rhythm").mean()).sort("graph", "param_set"))


def recip_index(edges):
    pre, post, w = edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy().astype(np.float64)
    W = sp.csr_matrix((w * sign[pre], (post, pre)), shape=(N, N))          # W[post, pre], signed
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


if which in ("struct", "both"):
    rows = []
    med, per = recip_index(e_real); rows.append({"graph": "real", "recip_innervation": med})
    print("real", round(med, 3))
    for lvl in ["er", "degree", "leg", "hl", "leg_hl", "degree_recip", "leg_hl_recip"]:
        for s in range(5):
            f = G / "nulls/theta50" / f"{lvl}_s{s}.parquet"
            if f.exists():
                m, _ = recip_index(pl.read_parquet(f)); rows.append({"graph": f"{lvl}_s{s}", "recip_innervation": m})
    for f in sorted((G / "surgery").glob("*.parquet")):
        m, _ = recip_index(pl.read_parquet(f)); rows.append({"graph": f.stem, "recip_innervation": m})
    df = pl.DataFrame(rows).with_columns(pl.col("graph").str.replace(r"_s\d+$", "").alias("level"))
    df.write_parquet(R / "diag_recip_innervation.parquet")
    print(df.group_by("level").agg(pl.len(), pl.col("recip_innervation").mean(), pl.col("recip_innervation").max()).sort("recip_innervation", descending=True))
