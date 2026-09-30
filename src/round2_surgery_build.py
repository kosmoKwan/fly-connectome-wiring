"""Round 2, experiment 2: build the dose-matched strength-controlled surgery graphs (CPU only).
Pre-registration amendment 1 (round2_20260927/预注册_第二轮_20260927.md): arms X_hi, X_lo (cross-pool) and W (within-pool),
swap blocks = presynaptic sign x synapse-weight decile x (leg|joint) [or leg|joint|direction for W], no presynaptic
group restriction, construction seeds 7000-7004. Writes full edge tables (non-MN edges untouched) + report.json.
"""
import sys, json
import numpy as np, polars as pl
from pathlib import Path
from surgery_dose_explore import load, node_groups_leg_hl, swap_in_blocks, metrics

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ARMS = {"theta50": {"mn_r2_cross_hi": ("cross", 2.5), "mn_r2_cross_lo": ("cross", 1.0), "mn_r2_within": ("within", 20.0)},
        "manc50": {"mn_r2_cross_hi": ("cross", 3.0), "mn_r2_cross_lo": ("cross", 1.0), "mn_r2_within": ("within", 20.0)}}


def atlas(p, q, w, nodes, leg, joint, dr):
    hl = nodes["trumanHl"].to_numpy(); role = nodes["role"].to_numpy()
    ok = (role[p] == "IN") & np.array([h is not None for h in hl[p]]) & np.isin(leg[q], ["T2L", "T2R", "T3L", "T3R"]) & (dr[q] != 0) & (joint[q] != "unmapped")
    df = pl.DataFrame({"hl": hl[p][ok].astype(str), "joint": joint[q][ok].astype(str), "d": dr[q][ok], "w": w[ok]})
    t = df.group_by("hl", "joint").agg((pl.col("w") * (pl.col("d") > 0)).sum().alias("P"), (pl.col("w") * (pl.col("d") < 0)).sum().alias("M"))
    return t.with_columns(((pl.col("P") + 1) / (pl.col("M") + 1)).log(2).alias("lr"), (pl.col("P") + pl.col("M")).alias("tot"))


report = {}
for tag, arms in ARMS.items():
    ctx = load(tag); nodes, edges, sign, is_mn, leg, joint, dr = ctx; N = nodes.height
    pre, post = edges["pre"].to_numpy().astype(np.int64), edges["post"].to_numpy().astype(np.int64)
    w_orig = edges["weight"].to_numpy(); w = w_orig.astype(np.float64)
    sel = is_mn[post]
    qs = np.quantile(w[sel], np.linspace(0, 1, 11)[1:-1])
    (OUT / f"surgery_r2_{tag}").mkdir(exist_ok=True)
    A0 = atlas(pre[sel], post[sel], w[sel], nodes, leg, joint, dr)
    for name, (arm, spe) in arms.items():
        for s in range(5):
            p, q, ww, wo = pre[sel].copy(), post[sel].copy(), w[sel].copy(), w_orig[sel].copy()
            wb = np.searchsorted(qs, ww, side="right")
            k = np.array([f"{leg[x]}|{joint[x]}" if arm == "cross" else f"{leg[x]}|{joint[x]}|{dr[x]}" for x in q], dtype=object)
            pk = np.unique(k, return_inverse=True)[1]; sg = (np.sign(sign[p]) + 1).astype(np.int64)
            bkey = (sg * 11 + wb) * (pk.max() + 1) + pk
            o = np.argsort(bkey, kind="stable"); p, q, ww, wo, bkey = p[o], q[o], ww[o], wo[o], bkey[o]
            p0, q0 = p.copy(), q.copy()
            st = np.flatnonzero(np.r_[True, bkey[1:] != bkey[:-1]]); ln = np.diff(np.r_[st, len(bkey)])
            acc = swap_in_blocks(p, q, st.astype(np.int64), ln.astype(np.int64), N, float(spe), 7000 + s)
            m = metrics(p0, q0, ww, p, q, sign, is_mn, leg, joint, dr, N)
            A1 = atlas(p, q, ww, nodes, leg, joint, dr)
            j = A0.join(A1, on=["hl", "joint"], how="left", suffix="_s").filter(pl.col("tot") >= 50)
            dlr = (j["lr"] - j["lr_s"].fill_null(0)).abs()
            m.update(accepted=int(acc), atlas_mean_abs_dlog2=float(dlr.mean()), atlas_max_abs_dlog2=float(dlr.max()), atlas_cells=int(j.height), swaps_per_edge=spe, arm=arm, seed=7000 + s)
            full = pl.DataFrame({"pre": np.r_[pre[~sel], p], "post": np.r_[post[~sel], q], "weight": np.r_[w_orig[~sel], wo]})
            assert full.height == edges.height and full.select("pre", "post").unique().height == full.height
            full.write_parquet(OUT / f"surgery_r2_{tag}" / f"{name}_s{s}.parquet")
            report[f"{tag}/{name}_s{s}"] = m
            print(tag, name, s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
    # reference atlas change of the archived plain and _w surgeries (descriptive)
    for ref, d in [("mn_shuffle_joint", "surgery" if tag == "theta50" else f"surgery_{tag}"), ("mn_shuffle_joint_w", f"surgery_w_{tag}")]:
        e = pl.read_parquet(f"/Volumes/ninishares/Experiments/fly-connectome-research-20260926/final_heldout_20260927_121507_B9ue3i/graphs/{d}/{ref}_s0.parquet")
        pp, qq, wr = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy().astype(float); ss = is_mn[qq]
        A1 = atlas(pp[ss], qq[ss], wr[ss], nodes, leg, joint, dr)
        j = A0.join(A1, on=["hl", "joint"], how="left", suffix="_s").filter(pl.col("tot") >= 50); dlr = (j["lr"] - j["lr_s"].fill_null(0)).abs()
        report[f"{tag}/REF_{ref}_s0"] = {"atlas_mean_abs_dlog2": float(dlr.mean()), "atlas_max_abs_dlog2": float(dlr.max())}
        print(tag, "REF", ref, round(float(dlr.mean()), 3), round(float(dlr.max()), 3), flush=True)
(OUT / "surgery_r2_report.json").write_text(json.dumps(report, indent=1))
