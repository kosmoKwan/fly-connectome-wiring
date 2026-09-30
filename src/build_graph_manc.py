"""Leg-motor subgraph from MANC v1.0, built with the same rules as the MaleCNS graph.

Ports: leg MNs (class motor neuron, subclass fl/ml/hl), leg proprioceptive afferents
(modality proprioceptive, leg subclass), DNs and VNC interneurons with >= THETA of their
synapses in LegNp(T1-T3)(L/R) and >= 20 leg synapses. Edges: synapse counts summed over
ROIs, >= 5. Sign from predictedNt (ACh +1, GABA/Glu -1). DNg100 = MANC type DNxl058
(bodyIds 10093, 10339 via MaleCNS mancBodyid), renamed so the same drive code applies.
"""
import json, sys
import numpy as np, pandas as pd, polars as pl
from pathlib import Path

D = Path("/root/fly-connectome/data_manc"); G = Path("/root/fly-connectome/graphs")
THETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
TAG = f"manc{int(THETA * 100)}"
MIN_SYN, MIN_LEG = 5, 20
SEG = {"fl": "T1", "ml": "T2", "hl": "T3", "prothoracic": "T1", "mesothoracic": "T2", "metathoracic": "T3"}
DNG100 = {10093, 10339}

n = pd.read_feather(D / "manc-v1.0-neuron-properties.feather")
for c in n.columns:
    if isinstance(n[c].dtype, pd.CategoricalDtype): n[c] = n[c].astype(object)
cls = n["class"].fillna(""); sub = n["subclass"].fillna("")
role = np.where((cls == "motor neuron") & sub.isin(["fl", "ml", "hl"]), "MN",
       np.where(cls.isin(["sensory neuron", "sensory ascending"]) & (n["modality"] == "proprioceptive")
                & sub.str.contains("thoracic leg"), "SN",
       np.where(cls == "descending neuron", "DN",
       np.where(cls.isin(["intrinsic neuron", "ascending neuron"]), "IN", ""))))
n["role"] = role
cand = n[n.role != ""].copy()

conn = pd.read_csv(D / "traced-connections-per-roi.csv")
conn = conn[conn.bodyId_pre.isin(cand.bodyId) | conn.bodyId_post.isin(cand.bodyId)]
legroi = conn.roi.str.startswith("LegNp")
def per_body(col, mask):
    return conn[mask].groupby(col).weight.sum()
tot = per_body("bodyId_pre", slice(None)).add(per_body("bodyId_post", slice(None)), fill_value=0)
leg = per_body("bodyId_pre", legroi).add(per_body("bodyId_post", legroi), fill_value=0)
tot_out = per_body("bodyId_pre", slice(None)); leg_out = per_body("bodyId_pre", legroi)
cand["n_vnc"] = cand.bodyId.map(tot).fillna(0); cand["n_leg"] = cand.bodyId.map(leg).fillna(0)
cand["n_vnc_pre"] = cand.bodyId.map(tot_out).fillna(0); cand["n_leg_pre"] = cand.bodyId.map(leg_out).fillna(0)
cand["leg_frac"] = cand.n_leg / cand.n_vnc.clip(lower=1); cand["leg_frac_out"] = cand.n_leg_pre / cand.n_vnc_pre.clip(lower=1)
for s in ["T1", "T2", "T3"]:
    m = conn.roi.str.startswith(f"LegNp({s})")
    cand[f"n_{s}"] = cand.bodyId.map(per_body("bodyId_pre", m).add(per_body("bodyId_post", m), fill_value=0)).fillna(0)
keep = (cand.role.isin(["MN", "SN"])
        | ((cand.role == "DN") & (cand.leg_frac_out >= THETA) & (cand.n_leg_pre >= MIN_LEG))
        | ((cand.role == "IN") & (cand.leg_frac >= THETA) & (cand.n_leg >= MIN_LEG)))
nodes = cand[keep].copy()
# segment: MNs by subclass, SNs by subclass prefix, others by dominant leg neuropil
nodes["leg_seg"] = np.where(nodes.role == "MN", nodes.subclass.map(SEG),
                   np.where(nodes.role == "SN", nodes.subclass.str.split(" ").str[0].map(SEG),
                   np.array(["T1", "T2", "T3"])[nodes[["n_T1", "n_T2", "n_T3"]].to_numpy().argmax(1)]))
nodes.loc[(nodes.role.isin(["DN", "IN"])) & (nodes.n_leg == 0), "leg_seg"] = None
# MANC encodes sides as LHS/RHS; fall back to the nerve suffix (entry for afferents, exit for MNs)
SIDE = {"LHS": "L", "RHS": "R", "L": "L", "R": "R"}
soma = nodes.somaSide.map(SIDE); root = nodes.rootSide.map(SIDE)
nerve = nodes.entryNerve.fillna("").str.extract(r"_(L|R)$")[0].fillna(nodes.exitNerve.fillna("").str.extract(r"_(L|R)$")[0])
nodes["side"] = soma.fillna(root).fillna(nerve)
nodes["sign"] = nodes.predictedNt.map({"acetylcholine": 1, "gaba": -1, "glutamate": -1}).fillna(0).astype(int)
nodes["consensus_nt"] = nodes.predictedNt
nodes["trumanHl"] = nodes.hemilineage.where(~nodes.hemilineage.isin(["None", None]), None)
nodes.loc[nodes.bodyId.isin(DNG100), "type"] = "DNg100"
order = {"DN": 0, "IN": 1, "MN": 2, "SN": 3}
nodes = nodes.sort_values(["role", "leg_seg", "bodyId"], key=lambda s: s.map(order) if s.name == "role" else s).reset_index(drop=True)
nodes["idx"] = np.arange(len(nodes))
cols = ["idx", "bodyId", "type", "role", "subclass", "leg_seg", "side", "trumanHl", "sign", "consensus_nt", "leg_frac", "n_leg"]
nodes_pl = pl.from_pandas(nodes[cols])
nodes_pl.write_parquet(G / f"nodes_{TAG}.parquet")

idmap = dict(zip(nodes.bodyId, nodes.idx))
e = conn[conn.bodyId_pre.isin(idmap) & conn.bodyId_post.isin(idmap)].groupby(["bodyId_pre", "bodyId_post"]).weight.sum().reset_index()
e = e[(e.weight >= MIN_SYN) & (e.bodyId_pre != e.bodyId_post)]
edges = pl.DataFrame({"pre": e.bodyId_pre.map(idmap).to_numpy(), "post": e.bodyId_post.map(idmap).to_numpy(), "weight": e.weight.to_numpy()})
edges.write_parquet(G / f"edges_{TAG}.parquet")

sys.path.insert(0, "/root/fly-connectome/src")
TYPE_MAP = {
    "Tergopleural/Pleural promotor MN": ("ThC", +1, False), "Sternal anterior rotator MN": ("ThC", +1, True),
    "Sternal posterior rotator MN": ("ThC", -1, True), "Pleural remotor/abductor MN": ("ThC", -1, False),
    "Tr extensor MN": ("CTr", +1, False), "Sternotrochanter MN": ("CTr", +1, False), "Tergotr. MN": ("CTr", +1, False),
    "Tr flexor MN": ("CTr", -1, False), "Acc. tr flexor MN": ("CTr", -1, False),
    "Ti extensor MN": ("FTi", +1, False), "Ti flexor MN": ("FTi", -1, False), "Acc. ti flexor MN": ("FTi", -1, False),
    "Ta levator MN": ("TiTa", +1, False), "Ta depressor MN": ("TiTa", -1, False),
    "ltm MN": ("TiTa", -1, True), "ltm1-tibia MN": ("TiTa", -1, True), "ltm2-femur MN": ("TiTa", -1, True)}
mn = nodes[nodes.role == "MN"]
rows = [(int(r.idx), int(r.bodyId), r.type, f"{r.leg_seg}{r.side}", r.leg_seg, r.side, *TYPE_MAP.get(r.type, (None, 0, False))) for r in mn.itertuples()]
pl.DataFrame(rows, schema=["idx", "bodyId", "type", "leg", "seg", "side", "joint", "direction", "approx"], orient="row").write_parquet(G / f"mn_joint_map_{TAG}.parquet")
N = len(nodes)
summ = {"n_nodes": N, "n_edges": edges.height, "density": edges.height / (N * (N - 1)),
        "by_role": nodes.role.value_counts().to_dict(), "MN_by_seg": mn.leg_seg.value_counts().to_dict(),
        "DNg100_idx": nodes[nodes.type == "DNg100"].idx.tolist(), "n_hemilineages": int(nodes[nodes.role == "IN"].trumanHl.nunique()),
        "sign": nodes.sign.value_counts().to_dict(), "mapped_MN": int(sum(1 for r in rows if r[6] is not None)),
        "side_by_role": {f"{k[0]}_{k[1]}": int(v) for k, v in nodes.groupby(["role", "side"], dropna=False).size().items()}, "MN_legs": mn.assign(leg=mn.leg_seg + mn.side).leg.value_counts().to_dict()}
(G / f"summary_{TAG}.json").write_text(json.dumps(summ, indent=1, default=str)); print(json.dumps(summ, default=str))
