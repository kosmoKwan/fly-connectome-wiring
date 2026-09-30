"""Round 3, experiment C (rescue): build graphs (CPU). Pre-registration: NAS round3_20260929/预注册_第三轮_20260929.md §2C.

For each parent rewired network (MaleCNS: hl, degree; MANC: hl, leg_hl; seeds 0-4) remove every edge onto a motor
neuron (role == MN) and insert one of three motor-input layers:
  C1   the real network's edges onto MNs
  C1p  the edges onto MNs of round-2 mn_r2_within_s{k}   (real layer after within-pool exchange)
  C0   the edges onto MNs of round-2 mn_r2_cross_hi_s{k} (real layer after strength-matched cross-pool exchange)
All other edges are the parent's. Signs stay node properties. Writes graphs, list files and a manipulation report
(edge counts, out-degree changes, reciprocal-innervation index computed exactly as diag_struct2.py).
usage: build_rescue.py GRAPHS_DIR SURG2_DIR OUT_DIR
"""
import sys, os, json
import numpy as np, polars as pl, scipy.sparse as sp

GD, SD, OUT = sys.argv[1:4]
FAMS = {"theta50": ["hl", "degree"], "manc50": ["hl", "leg_hl"]}
ARMS = {"C1": None, "C1p": "mn_r2_within", "C0": "mn_r2_cross_hi"}


def recip_index(edges, sign, mnmap, N):   # identical to record_20260927_0202/src/diag_struct2.py
    pre, post, w = edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy().astype(np.float64)
    W = sp.csr_matrix((w * sign[pre], (post, pre)), shape=(N, N))
    out = {}
    for (lg, j), grp in mnmap.filter(pl.col("joint").is_not_null()).group_by(["leg", "joint"]):
        ext = grp.filter(pl.col("direction") > 0)["idx"].to_numpy(); flx = grp.filter(pl.col("direction") < 0)["idx"].to_numpy()
        if not len(ext) or not len(flx):
            continue
        def infl(t):
            r1 = np.asarray(W[t].mean(axis=0)).ravel(); r2 = np.asarray((W[t] @ W).mean(axis=0)).ravel()
            return r1 / (np.abs(r1).max() + 1e-12) + r2 / (np.abs(r2).max() + 1e-12)
        ie, if_ = infl(ext), infl(flx); m = (ie != 0) | (if_ != 0)
        out[f"{lg}_{j}"] = float(-np.corrcoef(ie[m], if_[m])[0, 1]) if m.sum() > 10 else np.nan
    v = np.array([x for x in out.values() if np.isfinite(x)])
    return float(np.median(v))


report = []
for tag, fams in FAMS.items():
    nodes = pl.read_parquet(os.path.join(GD, f"nodes_{tag}.parquet")); N = nodes.height
    sign = nodes["sign"].to_numpy().astype(np.float64); is_mn = (nodes["role"] == "MN").to_numpy()
    mnmap = pl.read_parquet(os.path.join(GD, f"mn_joint_map_{tag}.parquet"))
    real = pl.read_parquet(os.path.join(GD, f"edges_{tag}.parquet")).select("pre", "post", "weight")
    od = os.path.join(OUT, f"rescue_{tag}"); os.makedirs(od, exist_ok=True)
    stems = []
    for fam in fams:
        for k in range(5):
            parent = pl.read_parquet(os.path.join(GD, "nulls", tag, f"{fam}_s{k}.parquet")).select("pre", "post", "weight")
            keep = parent.filter(~pl.Series(is_mn[parent["post"].to_numpy()]))
            for arm, src in ARMS.items():
                layer_src = real if src is None else pl.read_parquet(os.path.join(SD, f"surgery_r2_{tag}", f"{src}_s{k}.parquet")).select("pre", "post", "weight")
                layer = layer_src.filter(pl.Series(is_mn[layer_src["post"].to_numpy()]))
                g = pl.concat([keep, layer]).with_columns(pl.col("pre").cast(parent["pre"].dtype), pl.col("post").cast(parent["post"].dtype), pl.col("weight").cast(parent["weight"].dtype))
                assert g.select("pre", "post").unique().height == g.height, (tag, fam, k, arm)
                name = f"{arm}_{fam}_s{k}"
                g.write_parquet(os.path.join(od, f"{name}.parquet"))
                stems.append(f"round3/rescue_{tag}/{name}")
                od_par = np.bincount(parent["pre"].to_numpy(), minlength=N); od_new = np.bincount(g["pre"].to_numpy(), minlength=N)
                report.append(dict(connectome=tag, graph=name, parent=f"{fam}_s{k}", arm=arm, edges=g.height, parent_edges=parent.height,
                                   replaced_parent_mn_edges=int(parent.height - keep.height), inserted_mn_edges=layer.height,
                                   frac_edges_mn_layer=layer.height / g.height, n_neurons_outdeg_changed=int((od_par != od_new).sum()),
                                   max_abs_outdeg_change=int(np.abs(od_par - od_new).max()),
                                   ri_index=recip_index(g, sign, mnmap, N)))
                print(tag, name, report[-1]["edges"], round(report[-1]["ri_index"], 3), flush=True)
            report.append(dict(connectome=tag, graph=f"PARENT_{fam}_s{k}", parent=f"{fam}_s{k}", arm="parent", edges=parent.height,
                               parent_edges=parent.height, ri_index=recip_index(parent, sign, mnmap, N)))
    with open(os.path.join(OUT, f"rescue_{tag}_list.txt"), "w") as f:
        f.write(",".join(stems))
    report.append(dict(connectome=tag, graph="REAL", arm="real", edges=real.height, ri_index=recip_index(real, sign, mnmap, N)))
pl.DataFrame(report, infer_schema_length=None).write_csv(os.path.join(OUT, "rescue_build_report.csv"))
print("done")
