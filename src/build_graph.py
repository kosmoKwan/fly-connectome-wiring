"""Extract the leg-motor subgraph of the MaleCNS VNC.

Nodes: leg MNs (exit via ProLN/MesoLN/MetaLN), leg proprioceptive afferents,
DNs and VNC interneurons / ascending neurons whose VNC synapses are mostly in
leg neuropils. Edges: VNC-only synapse counts between selected nodes.
"""
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd

def read_small(path, cols):
    df = pd.read_feather(path, columns=cols)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype): df[c] = df[c].astype(object)
    return pl.from_pandas(df)

ROOT = Path("/root/fly-connectome")
DATA, OUT = ROOT / "data", ROOT / "graphs"
OUT.mkdir(exist_ok=True)
SYN = str(DATA / "syn-partners.feather")
LEG_NERVES = ["ProLN", "MesoLN", "MetaLN"]
# T1 coxal MNs and hair plates run in the prothoracic accessory / ventral nerves
LEG_SN_NERVES = LEG_NERVES + ["ProAN", "VProN"]
LEG_MN_SUBCLASS = {"fl": "T1", "ml": "T2", "hl": "T3"}
SN_NERVE_SEG = {"ProLN": "T1", "ProAN": "T1", "VProN": "T1", "MesoLN": "T2", "MetaLN": "T3"}
MIN_SYN = 5            # edge threshold (synapse count)
THETAS = [0.3, 0.5, 0.7]
MIN_LEG_SYN = 20       # a DN/IN must also have >= this many leg-neuropil synapses

ann = read_small(DATA / "body-annotations.feather", [
    "bodyId", "type", "superclass", "class", "subclass", "somaNeuromere", "somaSide",
    "trumanHl", "exitNerve", "entryNerve", "rootSide"])
nt = read_small(DATA / "body-neurotransmitters.feather", ["body", "consensus_nt"]).rename({"body": "bodyId"})

role = (
    pl.when((pl.col("superclass") == "vnc_motor") & pl.col("subclass").is_in(list(LEG_MN_SUBCLASS))).then(pl.lit("MN"))
    .when(pl.col("superclass").is_in(["vnc_sensory", "sensory_ascending"]) & pl.col("entryNerve").is_in(LEG_SN_NERVES)
          & (pl.col("class") == "mechanosensory_proprioceptive")).then(pl.lit("SN"))
    .when(pl.col("superclass") == "descending_neuron").then(pl.lit("DN"))
    .when(pl.col("superclass").is_in(["vnc_intrinsic", "ascending_neuron"])).then(pl.lit("IN"))
    .otherwise(None))
cand = ann.with_columns(role.alias("role")).filter(pl.col("role").is_not_null())
cand_ids = cand["bodyId"].to_list()
print("candidates:", cand.group_by("role").len().sort("role").rows(), flush=True)

# ---- pass 1: per-body synapse counts by ROI (VNC ROIs only) ----
lf = pl.scan_ipc(SYN).select("body_pre", "body_post", pl.col("primary_post").cast(pl.Utf8).alias("roi"))
all_rois = lf.select("roi").unique().collect(engine="streaming")["roi"].to_list()
BRAIN_LIKE = {"<unspecified>", "CentralBrain-unspecified", "GNG", "SAD", "PRW", "FLA(L)", "FLA(R)",
              "CAN(L)", "CAN(R)", "AMMC(L)", "AMMC(R)"}
def is_vnc(r):
    keys = ("LegNp", "mVAC", "Tct", "Ov(", "ANm", "VNC-unspecified", "LN(", "DMetaN", "ADMN", "PDMN",
            "AbN", "AbNT", "VProN", "DProN", "ProAN", "ProCN", "MesoAN", "MetaLN", "MesoLN", "ProLN", "CV-unspecified")
    return r not in BRAIN_LIKE and any(k in r for k in keys)
VNC_ROIS = sorted(r for r in all_rois if is_vnc(r))
LEG_ROIS = sorted(r for r in VNC_ROIS if r.startswith("LegNp"))
print("VNC rois:", len(VNC_ROIS), "leg rois:", LEG_ROIS, flush=True)

vnc = lf.filter(pl.col("roi").is_in(VNC_ROIS))
cnt_pre = (vnc.filter(pl.col("body_pre").is_in(cand_ids)).group_by("body_pre", "roi").len()
           .rename({"body_pre": "bodyId", "len": "n"}).with_columns(pl.lit("pre").alias("dir")))
cnt_post = (vnc.filter(pl.col("body_post").is_in(cand_ids)).group_by("body_post", "roi").len()
            .rename({"body_post": "bodyId", "len": "n"}).with_columns(pl.lit("post").alias("dir")))
counts = pl.concat([cnt_pre.collect(engine="streaming"), cnt_post.collect(engine="streaming")])
counts.write_parquet(OUT / "roi_counts.parquet")
print("pass1 done:", counts.height, flush=True)

leg = pl.col("roi").is_in(LEG_ROIS)
per = counts.group_by("bodyId").agg(
    pl.col("n").sum().alias("n_vnc"),
    pl.col("n").filter(leg).sum().alias("n_leg"),
    pl.col("n").filter(pl.col("dir") == "pre").sum().alias("n_vnc_pre"),
    pl.col("n").filter(leg & (pl.col("dir") == "pre")).sum().alias("n_leg_pre"),
    *[pl.col("n").filter(pl.col("roi").str.starts_with(f"LegNp({s})")).sum().alias(f"n_{s}") for s in ["T1", "T2", "T3"]],
)
cand = cand.join(per, on="bodyId", how="left").fill_null(0).join(nt, on="bodyId", how="left")
cand = cand.with_columns(
    (pl.col("n_leg") / pl.col("n_vnc").clip(lower_bound=1)).alias("leg_frac"),
    (pl.col("n_leg_pre") / pl.col("n_vnc_pre").clip(lower_bound=1)).alias("leg_frac_out"),
    pl.when(pl.col("consensus_nt") == "acetylcholine").then(1)
      .when(pl.col("consensus_nt").is_in(["gaba", "glutamate", "histamine"])).then(-1)
      .otherwise(0).alias("sign"),
)
# dominant leg segment by leg-synapse count
seg = pl.concat_list("n_T1", "n_T2", "n_T3").list.arg_max()
cand = cand.with_columns(pl.when(pl.col("n_leg") > 0).then(seg.map_elements(lambda i: ["T1", "T2", "T3"][i], return_dtype=pl.Utf8)).otherwise(None).alias("leg_seg"))
# ports get their segment from anatomy (nerve / MN subclass), not from synapse location
cand = cand.with_columns(
    pl.when(pl.col("role") == "MN").then(pl.col("subclass").replace_strict(LEG_MN_SUBCLASS, default=None))
      .when(pl.col("role") == "SN").then(pl.col("entryNerve").replace_strict(SN_NERVE_SEG, default=None))
      .otherwise(pl.col("leg_seg")).alias("leg_seg"))
# unified side: soma side when known, else the side of the nerve root (sensory afferents)
cand = cand.with_columns(pl.coalesce(pl.col("somaSide"), pl.col("rootSide")).alias("side"))
cand.write_parquet(OUT / "candidates.parquet")

summary = {}
for th in THETAS:
    keep = (
        pl.col("role").is_in(["MN", "SN"])
        | ((pl.col("role") == "DN") & (pl.col("leg_frac_out") >= th) & (pl.col("n_leg_pre") >= MIN_LEG_SYN))
        | ((pl.col("role") == "IN") & (pl.col("leg_frac") >= th) & (pl.col("n_leg") >= MIN_LEG_SYN))
    )
    nodes = cand.filter(keep).sort(["role", "leg_seg", "bodyId"]).with_row_index("idx")
    ids = nodes["bodyId"].to_list()
    # ---- pass 2: VNC-only edges among selected nodes ----
    e = (vnc.filter(pl.col("body_pre").is_in(ids) & pl.col("body_post").is_in(ids))
         .group_by("body_pre", "body_post").len().rename({"len": "weight"})
         .filter(pl.col("weight") >= MIN_SYN).collect(engine="streaming"))
    m = dict(zip(nodes["bodyId"].to_list(), nodes["idx"].to_list()))
    e = e.with_columns(pl.col("body_pre").replace_strict(m).alias("pre"),
                       pl.col("body_post").replace_strict(m).alias("post"))
    e = e.filter(pl.col("pre") != pl.col("post"))
    tag = f"theta{int(th*100)}"
    nodes.write_parquet(OUT / f"nodes_{tag}.parquet"); e.write_parquet(OUT / f"edges_{tag}.parquet")
    N = nodes.height
    indeg = np.bincount(e["post"].to_numpy(), minlength=N); outdeg = np.bincount(e["pre"].to_numpy(), minlength=N)
    s = {
        "n_nodes": N, "n_edges": e.height, "density": e.height / (N * (N - 1)),
        "by_role": dict(nodes.group_by("role").len().rows()),
        "by_role_seg": {f"{r}_{sg}": n for r, sg, n in nodes.group_by("role", "leg_seg").len().rows()},
        "sign": dict((str(k), v) for k, v in nodes.group_by("sign").len().rows()),
        "isolated": int(((indeg + outdeg) == 0).sum()),
        "MN_no_input": int((indeg[nodes["role"].to_numpy() == "MN"] == 0).sum()),
        "SN_no_output": int((outdeg[nodes["role"].to_numpy() == "SN"] == 0).sum()),
        "DN_no_output": int((outdeg[nodes["role"].to_numpy() == "DN"] == 0).sum()),
        "n_hemilineages": nodes.filter(pl.col("role") == "IN")["trumanHl"].n_unique(),
    }
    summary[tag] = s
    print(tag, json.dumps(s), flush=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
