"""Per-neuron influences behind the reciprocal-innervation index, for one antagonistic pair (Fig. 3a).

Re-implements the archived index (final_heldout/src/diag_struct2.py, recip_index) exactly: for each neuron k, the influence
on a pool is the mean signed one-step weight from k onto the pool's MNs plus the mean signed two-step weight, each normalized
by its maximum absolute value across neurons; the index is minus the Pearson correlation of the extensor-pool and flexor-pool
influences over neurons with non-zero influence. The recomputed index is checked against the archived per-pair value.
Writes ../derived/struct_example_<tag>_<graph>_<pair>.csv
"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp
import common as C


def influences(tag, edges_file, pair):
    nodes = pd.read_parquet(C.GRAPHS / f"nodes_{tag}.parquet"); mn = pd.read_parquet(C.GRAPHS / f"mn_joint_map_{tag}.parquet")
    e = pd.read_parquet(C.GRAPHS / (edges_file or f"edges_{tag}.parquet"))
    N = len(nodes); sign = nodes["sign"].to_numpy().astype(np.float64)
    pre, post, w = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy().astype(np.float64)
    W = sp.csr_matrix((w * sign[pre], (post, pre)), shape=(N, N))
    leg, joint = pair.split("_")
    g = mn[(mn.leg == leg) & (mn.joint == joint)]
    ext = g[g.direction > 0]["idx"].to_numpy(); flx = g[g.direction < 0]["idx"].to_numpy()

    def infl(tgt):
        r1 = np.asarray(W[tgt].mean(axis=0)).ravel()
        r2 = np.asarray((W[tgt] @ W).mean(axis=0)).ravel()
        return r1 / (np.abs(r1).max() + 1e-12) + r2 / (np.abs(r2).max() + 1e-12)

    ie, if_ = infl(ext), infl(flx)
    m = (ie != 0) | (if_ != 0)
    idx = float(-np.corrcoef(ie[m], if_[m])[0, 1])
    df = pd.DataFrame(dict(neuron_idx=np.arange(N)[m], role=nodes["role"].to_numpy()[m], sign=sign[m].astype(int),
                           infl_ext=ie[m], infl_flx=if_[m]))
    return df, idx, len(ext), len(flx)


if __name__ == "__main__":
    for tag, gname, efile, pair in [("theta50", "real", None, "T2L_ThC"), ("theta50", "leg_s0", "nulls/theta50/leg_s0.parquet", "T2L_ThC")]:
        df, idx, ne, nf = influences(tag, efile, pair)
        arch = C.recip_pairs(tag)
        a = float(arch[(arch.graph == gname) & (arch.pair == pair)].recip_innervation.iloc[0])
        df.insert(0, "graph", gname); df.insert(0, "pair", pair)
        df.to_csv(C.DERIVED / f"struct_example_{tag}_{gname}_{pair}.csv", index=False)
        q = np.quantile(np.abs(df[["infl_ext", "infl_flx"]].values), [0.5, 0.9, 0.99, 1.0])
        print(tag, gname, pair, f"n={len(df)} ext={ne} flx={nf} index={idx:.6f} archived={a:.6f} |infl| q50/90/99/max={np.round(q,3)}")
        print(df.groupby("role").size().to_dict(), "sign counts", df.sign.value_counts().to_dict())
