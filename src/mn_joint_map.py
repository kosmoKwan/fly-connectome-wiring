"""Map leg motor neurons to (leg, joint, direction) from their MaleCNS type names.

Joints per leg (simplified hexapod / fly leg):
  ThC  body-coxa swing   (+ promotion / forward,  - remotion / backward)
  CTr  coxa-trochanter   (+ extension / depression, - flexion / levation)
  FTi  femur-tibia       (+ extension, - flexion)
  TiTa tibia-tarsus      (+ levation, - depression / claw flexion)
Types that act on DoFs we do not model (TrFe reductor, ThC adduction) or that are
untyped are kept in the table with joint = None so nothing is silently dropped.
The ThC assignment of the sternal rotators is an approximation (they rotate the
coxa about its long axis); it is flagged in the `approx` column.
"""
import polars as pl
from pathlib import Path

G = Path("/root/fly-connectome/graphs")

# type -> (joint, direction, approx)
TYPE_MAP = {
    "Tergopleural/Pleural promotor MN": ("ThC", +1, False),
    "Sternal anterior rotator MN":      ("ThC", +1, True),
    "Sternal posterior rotator MN":     ("ThC", -1, True),
    "Pleural remotor/abductor MN":      ("ThC", -1, False),
    "Tr extensor MN":                   ("CTr", +1, False),
    "Sternotrochanter MN":              ("CTr", +1, False),
    "Tergotr. MN":                      ("CTr", +1, False),
    "Tr flexor MN":                     ("CTr", -1, False),
    "Acc. tr flexor MN":                ("CTr", -1, False),
    "Ti extensor MN":                   ("FTi", +1, False),
    "Ti flexor MN":                     ("FTi", -1, False),
    "Acc. ti flexor MN":                ("FTi", -1, False),
    "Ta levator MN":                    ("TiTa", +1, False),
    "Ta depressor MN":                  ("TiTa", -1, False),
    "ltm MN":                           ("TiTa", -1, True),
    "ltm1-tibia MN":                    ("TiTa", -1, True),
    "ltm2-femur MN":                    ("TiTa", -1, True),
}

for tag in ["theta30", "theta50", "theta70"]:
    n = pl.read_parquet(G / f"nodes_{tag}.parquet").filter(pl.col("role") == "MN")
    rows = []
    for idx, bid, typ, seg, side in n.select("idx", "bodyId", "type", "leg_seg", "somaSide").iter_rows():
        j, d, ap = TYPE_MAP.get(typ, (None, 0, False))
        rows.append((idx, bid, typ, f"{seg}{side}", seg, side, j, d, ap))
    m = pl.DataFrame(rows, schema=["idx", "bodyId", "type", "leg", "seg", "side", "joint", "direction", "approx"], orient="row")
    m.write_parquet(G / f"mn_joint_map_{tag}.parquet")
    m.write_csv(G / f"mn_joint_map_{tag}.csv")
    if tag == "theta50":
        tab = (m.with_columns(pl.format("{}{}", pl.col("joint").fill_null("unmapped"),
                                        pl.when(pl.col("direction") > 0).then(pl.lit("+")).when(pl.col("direction") < 0).then(pl.lit("-")).otherwise(pl.lit(""))).alias("jd"))
               .group_by("leg", "jd").len().pivot(on="leg", index="jd", values="len").fill_null(0).sort("jd"))
        print(tab.select(["jd"] + sorted(c for c in tab.columns if c != "jd")).to_pandas().to_string())
        print("mapped:", m.filter(pl.col("joint").is_not_null()).height, "/", m.height,
              " approx:", m.filter(pl.col("approx")).height)
