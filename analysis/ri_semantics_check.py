"""Check what the reciprocal-innervation index measures (post hoc, CPU only, existing graphs).

Raised by GPT-6 Astra (2026-09-29): minus a Pearson correlation across neurons can be high when
antagonist pools are driven by *different* neurons (segregation), without any neuron pushing the
two pools in *opposite* directions. For every pair (leg x joint, as in diag_struct2.py) we compute,
with exactly the influence vectors of diag_struct2.py:
  index      = -corr(u_E, u_F) over neurons with non-zero influence (reproduces the paper's index)
  gamma      = -cos(u_E, u_F), uncentred
  f_opp      = sum|u_E u_F| [u_E u_F < 0] / sum|u_E u_F|, over neurons influencing both pools
  f_both     = share of influence mass (|u_E|+|u_F|) carried by neurons influencing both pools
  f_opp_1, f_opp_2: f_opp with one-step or two-step influence only (one-step is 0 by Dale's law)
usage: ri_semantics_check.py GRAPHS_DIR STORED_RESULTS_DIR OUT_DIR
"""
import sys, os
import numpy as np, polars as pl, scipy.sparse as sp

GD, SD, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]


def vectors(W, tgt):
    r1 = np.asarray(W[tgt].mean(axis=0)).ravel()
    r2 = np.asarray((W[tgt] @ W).mean(axis=0)).ravel()
    n1 = r1 / (np.abs(r1).max() + 1e-12)
    n2 = r2 / (np.abs(r2).max() + 1e-12)
    return n1, n2, n1 + n2


def fopp(a, b):
    p = a * b
    both = (a != 0) & (b != 0)
    tot = np.abs(p[both]).sum()
    return float(np.abs(p[both & (p < 0)]).sum() / tot) if tot > 0 else np.nan


rows = []
for tag in ["theta50", "manc50"]:
    nodes = pl.read_parquet(os.path.join(GD, f"nodes_{tag}.parquet"))
    mnmap = pl.read_parquet(os.path.join(GD, f"mn_joint_map_{tag}.parquet"))
    N = nodes.height; sign = nodes["sign"].to_numpy().astype(np.float64)
    graphs = [("real", "real", os.path.join(GD, f"edges_{tag}.parquet"))]
    graphs += [(f"{fam}_s{s}", fam, os.path.join(GD, "nulls", tag, f"{fam}_s{s}.parquet")) for fam in MAIN for s in range(5)]
    for name, fam, path in graphs:
        e = pl.read_parquet(path)
        pre, post, w = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy().astype(np.float64)
        s = sign[pre]
        if "esign" in e.columns:
            es = e["esign"].to_numpy(); s = np.where(es != 0, es, s)
        W = sp.csr_matrix((w * s, (post, pre)), shape=(N, N))
        for (lg, j), grp in mnmap.filter(pl.col("joint").is_not_null()).group_by(["leg", "joint"]):
            ext = grp.filter(pl.col("direction") > 0)["idx"].to_numpy(); flx = grp.filter(pl.col("direction") < 0)["idx"].to_numpy()
            if not len(ext) or not len(flx):
                continue
            e1, e2, ue = vectors(W, ext); f1, f2, uf = vectors(W, flx)
            m = (ue != 0) | (uf != 0)
            if m.sum() <= 10:
                continue
            index = float(-np.corrcoef(ue[m], uf[m])[0, 1])
            gamma = float(-(ue @ uf) / (np.linalg.norm(ue) * np.linalg.norm(uf)))
            both = (ue != 0) & (uf != 0)
            mass = np.abs(ue) + np.abs(uf)
            rows.append(dict(connectome=tag, graph=name, family=fam, pair=f"{lg}_{j}", index=index, gamma=gamma,
                             f_opp=fopp(ue, uf), f_opp_1=fopp(e1, f1), f_opp_2=fopp(e2, f2),
                             f_both=float(mass[both].sum() / mass.sum()), n_infl=int(m.sum())))
        print(tag, name, flush=True)

D = pl.DataFrame(rows)
D.write_csv(os.path.join(OUT, "ri_semantics_pairs.csv"))
G = D.group_by(["connectome", "graph", "family"]).agg([pl.col(c).median() for c in ["index", "gamma", "f_opp", "f_opp_1", "f_opp_2", "f_both"]])
G.write_csv(os.path.join(OUT, "ri_semantics_graphs.csv"))

# validation against the stored index (diag_struct2.py output)
for tag in ["theta50", "manc50"]:
    st = pl.read_parquet(os.path.join(SD, f"diag_recip_innervation_v2_{tag}.parquet")).select("graph", pl.col("recip_innervation").alias("stored"))
    v = G.filter(pl.col("connectome") == tag).join(st, on="graph", how="inner")
    print(tag, "validation: graphs matched", v.height, "max |index - stored|", float((v["index"] - v["stored"]).abs().max()))

with pl.Config(tbl_rows=40, tbl_width_chars=200, tbl_cols=12):
    fam = G.with_columns(pl.when(pl.col("family") == "real").then(pl.lit("real")).otherwise(pl.lit("rewired")).alias("kind"))
    print(fam.group_by(["connectome", "kind"]).agg([pl.col(c).min().alias(f"{c}_min") for c in ["index", "gamma", "f_opp", "f_both"]] +
                                                   [pl.col(c).max().alias(f"{c}_max") for c in ["index", "gamma", "f_opp", "f_both"]]).sort("connectome", "kind"))
    print(G.filter(pl.col("family") == "real"))
    # per-pair real: how many pairs have f_opp > 0.5 (majority of overlapping influence is opposite-signed)
    R = D.filter(pl.col("family") == "real")
    print(R.group_by("connectome").agg(pl.len().alias("pairs"), (pl.col("f_opp") > 0.5).sum().alias("pairs_fopp_gt_0.5"),
                                       (pl.col("gamma") > 0).sum().alias("pairs_gamma_pos"), pl.col("f_opp").median().alias("f_opp_med")))
    # per pair: real f_opp above every rewired network?
    NU = D.filter(pl.col("family") != "real").group_by(["connectome", "pair"]).agg(pl.col("f_opp").max().alias("null_max_fopp"), pl.col("gamma").max().alias("null_max_gamma"))
    P = R.join(NU, on=["connectome", "pair"]).with_columns((pl.col("f_opp") > pl.col("null_max_fopp")).alias("fopp_above_all"),
                                                           (pl.col("gamma") > pl.col("null_max_gamma")).alias("gamma_above_all"))
    P.write_csv(os.path.join(OUT, "ri_semantics_real_vs_nulls_by_pair.csv"))
    print(P.group_by("connectome").agg(pl.len(), pl.col("fopp_above_all").sum(), pl.col("gamma_above_all").sum()))
