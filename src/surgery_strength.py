"""Strength-controlled versions of the cross-pool shuffle and the within-pool control.

As surgery.py, degree-preserving swaps among edges onto MNs, but swap blocks are additionally
restricted to the same presynaptic transmitter sign and the same synapse-weight decile. A swap
exchanges targets of two edges of (nearly) equal weight and equal sign, so every MN keeps (nearly)
its excitatory and its inhibitory input strength, while the identity of its presynaptic partners,
and hence which antagonist pool each premotor neuron targets, is scrambled (cross-pool) or kept at
the pool level (within-pool control). Residual per-MN strength changes are reported.
usage: surgery_strength.py TAG
"""
import sys, os, json
import numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, "/root/fly-connectome/src")
from nulls import node_groups, _swap_in_blocks

G = Path("/root/fly-connectome/graphs")
TAG = sys.argv[1] if len(sys.argv) > 1 else "theta50"
OUT = G / f"surgery_w_{TAG}"; OUT.mkdir(exist_ok=True)
nodes = pl.read_parquet(G / f"nodes_{TAG}.parquet"); N = nodes.height
edges = pl.read_parquet(G / f"edges_{TAG}.parquet")
mnmap = pl.read_parquet(G / f"mn_joint_map_{TAG}.parquet")
is_mn = (nodes["role"] == "MN").to_numpy(); sign = nodes["sign"].to_numpy()
leg_of = np.full(N, "NA", dtype=object); joint_of = np.full(N, "NA", dtype=object); dir_of = np.zeros(N, np.int64)
for idx, lg, j, dd in mnmap.select("idx", "leg", "joint", "direction").iter_rows():
    leg_of[idx] = lg; joint_of[idx] = j if j is not None else "unmapped"; dir_of[idx] = dd
pre_group = node_groups(nodes, "leg_hl")
pre, post, w = edges["pre"].to_numpy().astype(np.int64), edges["post"].to_numpy().astype(np.int64), edges["weight"].to_numpy()
sel = is_mn[post]
wbin = np.zeros(len(w), np.int64)
qs = np.quantile(w[sel], np.linspace(0, 1, 11)[1:-1]); wbin[sel] = np.searchsorted(qs, w[sel], side="right")


def strengths(p, q, ww):
    E = np.zeros(N); I = np.zeros(N)
    s = sign[p]; np.add.at(E, q[s > 0], ww[s > 0]); np.add.at(I, q[s < 0], ww[s < 0])
    return E, I


E0, I0 = strengths(pre[sel], post[sel], w[sel])
jointkey = np.array([f"{l}|{j}" for l, j in zip(leg_of, joint_of)], dtype=object)
poolkey = np.array([f"{l}|{j}|{dd}" for l, j, dd in zip(leg_of, joint_of, dir_of)], dtype=object)
report = {}
for name, key in [("mn_shuffle_joint_w", jointkey), ("mn_shuffle_pool_w", poolkey)]:
    for s in range(5):
        p, q, ww, wb = pre[sel].copy(), post[sel].copy(), w[sel].copy(), wbin[sel].copy()
        _, pk = np.unique(key[q], return_inverse=True)
        sg = (np.sign(sign[p]) + 1).astype(np.int64)
        bkey = ((pre_group[p] * 3 + sg) * 10 + wb) * (pk.max() + 1) + pk
        o = np.argsort(bkey, kind="stable"); p, q, ww, bkey = p[o], q[o], ww[o], bkey[o]
        st = np.flatnonzero(np.r_[True, bkey[1:] != bkey[:-1]]); ln = np.diff(np.r_[st, len(bkey)])
        acc = _swap_in_blocks(p, q, st.astype(np.int64), ln.astype(np.int64), N, 20, 5000 + s)
        E1, I1 = strengths(p, q, ww)
        m = is_mn & ((E0 + I0) > 0)
        # fraction of each MN's input edges whose presynaptic partner changed
        changed = np.mean([len(set(zip(pre[sel][post[sel] == i], w[sel][post[sel] == i])) ^ set(zip(p[q == i], ww[q == i]))) > 0 for i in np.flatnonzero(m)[:400]])
        report[f"{name}_s{s}"] = {"mn_edges": int(len(p)), "accepted": int(acc),
                                  "median_rel_dE": float(np.median(np.abs(E1[m] - E0[m]) / np.maximum(E0[m], 1))),
                                  "median_rel_dI": float(np.median(np.abs(I1[m] - I0[m]) / np.maximum(I0[m], 1))),
                                  "p95_rel_dE": float(np.quantile(np.abs(E1[m] - E0[m]) / np.maximum(E0[m], 1), 0.95)),
                                  "p95_rel_dI": float(np.quantile(np.abs(I1[m] - I0[m]) / np.maximum(I0[m], 1), 0.95)),
                                  "frac_MN_with_changed_inputs": float(changed)}
        pl.DataFrame({"pre": np.r_[pre[~sel], p], "post": np.r_[post[~sel], q], "weight": np.r_[w[~sel], ww]}).write_parquet(OUT / f"{name}_s{s}.parquet")
        print(name, s, json.dumps(report[f"{name}_s{s}"]), flush=True)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
