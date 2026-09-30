"""Graph surgery on the last (premotor -> MN) layer, to localise flexor/extensor alternation.

Remove (on the real graph, everything else untouched):
  mn_shuffle_leg    degree-preserving swaps among edges onto MNs, restricted to blocks
                    (presyn fine group, MN leg): which MN of a leg receives which input is randomised.
  mn_shuffle_joint  same, but swaps only between MNs of the same leg AND joint, i.e.
                    extensor vs flexor identity is scrambled while joint-level input is kept.
Add (on nulls): hybrid_<null>  = null edges onto non-MN targets + the REAL edges onto MNs.
Class-restricted add: hybrid_<null>_pre<CLASS> = like hybrid, but only real MN-input edges whose
                    presynaptic neuron belongs to CLASS are restored (others keep the null's MN inputs).
Every graph keeps each neuron's sign (Dale) because sign is a node property.
"""
import sys, os
import numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, "/root/fly-connectome/src")
from nulls import node_groups, _swap_in_blocks, graph_stats

G = Path("/root/fly-connectome/graphs")
TAG = os.environ.get("SURG_TAG", "theta50")
OUT = G / ("surgery" if TAG == "theta50" else f"surgery_{TAG}"); OUT.mkdir(exist_ok=True)
nodes = pl.read_parquet(G / f"nodes_{TAG}.parquet"); N = nodes.height
edges = pl.read_parquet(G / f"edges_{TAG}.parquet")
mnmap = pl.read_parquet(G / f"mn_joint_map_{TAG}.parquet")
role = nodes["role"].to_numpy()
is_mn = role == "MN"
leg_of = np.full(N, "NA", dtype=object); joint_of = np.full(N, "NA", dtype=object)
for idx, lg, j in mnmap.select("idx", "leg", "joint").iter_rows():
    leg_of[idx] = lg; joint_of[idx] = j if j is not None else "unmapped"
pre_group = node_groups(nodes, "leg_hl")          # role x seg x side x hemilineage


def swap_mn_inputs(e, post_key, seed, swaps=20):
    pre, post, w = e["pre"].to_numpy().astype(np.int64), e["post"].to_numpy().astype(np.int64), e["weight"].to_numpy()
    sel = is_mn[post]
    p, q, ww = pre[sel].copy(), post[sel].copy(), w[sel].copy()
    _, pk = np.unique(post_key[q], return_inverse=True)
    bkey = pre_group[p] * (pk.max() + 1) + pk
    o = np.argsort(bkey, kind="stable"); p, q, ww, bkey = p[o], q[o], ww[o], bkey[o]
    st = np.flatnonzero(np.r_[True, bkey[1:] != bkey[:-1]]); ln = np.diff(np.r_[st, len(bkey)])
    # the hash set only needs the MN-target edges: swaps never create edges onto non-MNs
    acc = _swap_in_blocks(p, q, st.astype(np.int64), ln.astype(np.int64), N, swaps, seed)
    out = pl.DataFrame({"pre": np.r_[pre[~sel], p], "post": np.r_[post[~sel], q], "weight": np.r_[w[~sel], ww]})
    return out, acc, len(p)


def hybrid(null_e, restrict_pre=None):
    rp, rq, rw = (edges[c].to_numpy() for c in ("pre", "post", "weight"))
    npre, npost, nw = (null_e[c].to_numpy() for c in ("pre", "post", "weight"))
    keep_null = ~is_mn[npost]
    real_mn = is_mn[rq]
    if restrict_pre is None:
        parts = [(npre[keep_null], npost[keep_null], nw[keep_null]), (rp[real_mn], rq[real_mn], rw[real_mn])]
    else:
        # restore real MN inputs only from presynaptic neurons in restrict_pre; null MN inputs from others kept
        rsel = real_mn & restrict_pre[rp]
        nsel = is_mn[npost] & ~restrict_pre[npre]
        parts = [(npre[keep_null], npost[keep_null], nw[keep_null]), (rp[rsel], rq[rsel], rw[rsel]), (npre[nsel], npost[nsel], nw[nsel])]
    p = np.concatenate([a for a, _, _ in parts]); q = np.concatenate([b for _, b, _ in parts]); ww = np.concatenate([c for _, _, c in parts])
    df = pl.DataFrame({"pre": p, "post": q, "weight": ww}).unique(subset=["pre", "post"], keep="first")
    return df


if __name__ == "__main__":
    report = {}
    legkey = np.array([f"{l}" for l in leg_of], dtype=object)
    jointkey = np.array([f"{l}|{j}" for l, j in zip(leg_of, joint_of)], dtype=object)
    dir_of = np.full(N, 0, dtype=np.int64)
    for idx, d in mnmap.select("idx", "direction").iter_rows():
        dir_of[idx] = d
    poolkey = np.array([f"{l}|{j}|{d}" for l, j, d in zip(leg_of, joint_of, dir_of)], dtype=object)
    only = os.environ.get("SURG_ONLY")
    plan = [("mn_shuffle_leg", legkey), ("mn_shuffle_joint", jointkey), ("mn_shuffle_pool", poolkey)]
    if only:
        plan = [p for p in plan if p[0] in only.split(",")]
    for name, key in plan:
        for s in range(5):
            e2, acc, n = swap_mn_inputs(edges, key, 3000 + s)
            e2.write_parquet(OUT / f"{name}_s{s}.parquet"); report[f"{name}_s{s}"] = {"mn_edges": n, "accepted": int(acc)}
    if only:
        raise SystemExit(0)
    hl = nodes["trumanHl"].fill_null("NA").to_numpy(); sign = nodes["sign"].to_numpy()
    classes = {"inhib": (sign < 0) & (role == "IN"), "excit": (sign > 0) & (role == "IN"), "DN": role == "DN", "SN": role == "SN"}
    for base in ["leg_hl", "degree"]:
        for s in range(5):
            ne = pl.read_parquet(G / "nulls" / TAG / f"{base}_s{s}.parquet")
            hybrid(ne).write_parquet(OUT / f"hybrid_{base}_s{s}.parquet")
            if base == "leg_hl":
                for cname, cmask in classes.items():
                    hybrid(ne, cmask).write_parquet(OUT / f"hybrid_{base}_pre{cname}_s{s}.parquet")
    import json
    (OUT / "surgery_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    # sanity: structural stats
    for f in ["mn_shuffle_joint_s0", "hybrid_leg_hl_s0", "hybrid_leg_hl_preinhib_s0"]:
        e = pl.read_parquet(OUT / f"{f}.parquet")
        print(f, e.height, graph_stats(nodes, e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy()))
