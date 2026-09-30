"""Round 3, CPU-only analyses on existing round-2 data: D1 (input-strength audit) and E (common/differential framework).
Pre-registration: NAS round3_20260929/预注册_第三轮_20260929.md §2D (D1) and §2E (E1, P1-P3). Rules applied as written.
Operational definition fixed before running (logged): "W's active pairs" (P1) = pairs active (gated in >= 10 of 20 runs)
in at least 3 of the 5 W networks.
usage: analyze_r3_cpu.py OUT_DIR
"""
import sys, os, json
import numpy as np, polars as pl, scipy.sparse as sp
from scipy.stats import spearmanr

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ROOT = "/Volumes/ninishares/Experiments/fly-connectome-research-20260926/"
GD = ROOT + "final_heldout_20260927_121507_B9ue3i/graphs/"; SD = ROOT + "round2_20260927/experiment2_graphs/"; RD = ROOT + "round2_20260927/results/"
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]
out_json = {}


def mn_inputs(edges, sign, is_mn, N):
    pre, post, w = edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy().astype(np.float64)
    sel = is_mn[post]; s = np.sign(sign[pre[sel]])
    E = np.bincount(post[sel][s > 0], weights=w[sel][s > 0], minlength=N); I = np.bincount(post[sel][s < 0], weights=w[sel][s < 0], minlength=N)
    return E, I


def influence_vectors(edges, sign, N, ext, flx):
    pre, post, w = edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy().astype(np.float64)
    W = sp.csr_matrix((w * sign[pre], (post, pre)), shape=(N, N))
    def infl(t):
        r1 = np.asarray(W[t].mean(axis=0)).ravel(); r2 = np.asarray((W[t] @ W).mean(axis=0)).ravel()
        return r1 / (np.abs(r1).max() + 1e-12) + r2 / (np.abs(r2).max() + 1e-12)
    return infl(ext), infl(flx)


def pair_frame(p):
    """per run x pair: contribution, gate, Var(y+), Var(y-), D (primary window, stochastic, readout all)."""
    p = p.filter((pl.col("mode") == "stochastic") & (pl.col("window") == "w15_25") & (pl.col("readout") == "all"))
    se, sf, c = pl.col("se"), pl.col("sf"), pl.col("corr").fill_nan(0.0).fill_null(0.0)   # corr undefined only when a pool is flat (sd <= 1e-4): covariance ~ 0
    return p.with_columns(
        pl.when(pl.col("gated")).then(-pl.col("corr")).otherwise(0.0).alias("contrib"),
        ((se ** 2 + sf ** 2 + 2 * c * se * sf) / 2).alias("Vp"),
        ((se ** 2 + sf ** 2 - 2 * c * se * sf) / 2).alias("Vm"),
    ).with_columns((pl.col("Vm") / (pl.col("Vp") + pl.col("Vm"))).alias("D"))


for tag in ["theta50", "manc50"]:
    nodes = pl.read_parquet(GD + f"nodes_{tag}.parquet"); N = nodes.height
    sign = nodes["sign"].to_numpy().astype(np.float64); is_mn = (nodes["role"] == "MN").fill_null(False).to_numpy()
    mnmap = pl.read_parquet(GD + f"mn_joint_map_{tag}.parquet")
    real_e = pl.read_parquet(GD + f"edges_{tag}.parquet").select("pre", "post", "weight")
    E0, I0 = mn_inputs(real_e, sign, is_mn, N)
    pairs = {}
    for (lg, j), grp in mnmap.filter(pl.col("joint").is_not_null()).group_by(["leg", "joint"]):
        ext = grp.filter(pl.col("direction") > 0)["idx"].to_numpy(); flx = grp.filter(pl.col("direction") < 0)["idx"].to_numpy()
        if len(ext) and len(flx):
            pairs[f"{lg}_{j}"] = (ext, flx)
    assert len(pairs) == 20
    # ---------- data
    preal = pair_frame(pl.read_parquet(RD + f"pairs_pairs_{tag}.parquet").filter((pl.col("graph") == "real") & (pl.col("rule") == "own")))
    psurg = pair_frame(pl.read_parquet(RD + f"pairs_pairs_{tag}_surg2.parquet").filter(pl.col("rule") == "real_setting"))
    # ---------- D1: per-MN relative input change and low-change pairs
    dist, d1 = [], []
    for arm in ["mn_r2_cross_hi", "mn_r2_within"]:
        for k in range(5):
            g = pl.read_parquet(SD + f"surgery_r2_{tag}/{arm}_s{k}.parquet")
            E1, I1 = mn_inputs(g, sign, is_mn, N)
            act = np.flatnonzero(is_mn & ((E0 + I0) > 0))
            rE = np.abs(E1 - E0) / np.maximum(E0, 1); rI = np.abs(I1 - I0) / np.maximum(I0, 1)
            dist.append(dict(connectome=tag, arm=arm, seed=k, n_mn=len(act),
                             **{f"{nm}_{q}": float(np.quantile(v[act], qq)) for nm, v in [("rel_dE", rE), ("rel_dI", rI)] for q, qq in [("med", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1.0)]},
                             frac_mn_over_10pct=float(((rE[act] > 0.10) | (rI[act] > 0.10)).mean())))
            if arm != "mn_r2_cross_hi":
                continue
            low = [nm for nm, (e, f) in pairs.items() if max(rE[np.r_[e, f]].max(), rI[np.r_[e, f]].max()) <= 0.10]
            sx = psurg.filter((pl.col("graph") == arm) & (pl.col("gseed") == k) & pl.col("pair").is_in(low))
            sr = preal.filter(pl.col("pair").is_in(low))
            S_X = float(sx.group_by("pair").agg(pl.col("contrib").mean())["contrib"].mean()) if low else np.nan
            S_R = float(sr.group_by("pair").agg(pl.col("contrib").mean())["contrib"].mean()) if low else np.nan
            d1.append(dict(connectome=tag, seed=k, n_low_pairs=len(low), low_pairs=";".join(sorted(low)), S_X_low=S_X, S_R_low=S_R))
    D1 = pl.DataFrame(d1); pl.DataFrame(dist).write_csv(os.path.join(OUT, f"D1_mn_input_change_distribution_{tag}.csv")); D1.write_csv(os.path.join(OUT, f"D1_low_change_pairs_{tag}.csv"))
    nlow = float(D1["n_low_pairs"].median()); mSX = float(D1["S_X_low"].median()); mSR = float(D1["S_R_low"].median())
    verdict = "无法判定（低变化对少于 3 对）" if nlow < 3 else ("成立：S_X ≤ 0.5·S_R" if mSX <= 0.5 * mSR else "不成立")
    out_json[f"D1_{tag}"] = dict(median_n_low_pairs=nlow, median_S_X_low=mSX, median_S_R_low=mSR, verdict=verdict)
    # ---------- E1: gate decomposition (descriptive) for real, main nulls, W, X_hi
    pall = pair_frame(pl.read_parquet(RD + f"pairs_pairs_{tag}.parquet").filter(pl.col("rule") == "own"))
    frames = [pall.filter(pl.col("graph").is_in(["real"] + MAIN)).with_columns(pl.col("graph").alias("set")),
              psurg.filter(pl.col("graph").is_in(["mn_r2_within", "mn_r2_cross_hi"])).with_columns(pl.col("graph").alias("set"))]
    e1 = pl.concat(frames, how="diagonal_relaxed").group_by(["set", "gseed", "pair", "joint"]).agg(
        pl.col("gated").cast(pl.Float64).mean().alias("P_gate"),
        pl.when(pl.col("gated")).then(-pl.col("corr")).otherwise(None).mean().alias("antiphase_given_gate"),
        pl.col("contrib").mean().alias("contrib"))
    e1.write_csv(os.path.join(OUT, f"E1_gate_decomposition_pairs_{tag}.csv"))
    summ = e1.group_by(["set", "gseed"]).agg(pl.col("P_gate").mean().alias("mean_P_gate"), pl.col("antiphase_given_gate").mean().alias("mean_antiphase_given_gate"),
                                             pl.col("contrib").mean().alias("S")).group_by("set").agg(pl.col("mean_P_gate").median(), pl.col("mean_antiphase_given_gate").median(), pl.col("S").median())
    summ.write_csv(os.path.join(OUT, f"E1_gate_decomposition_summary_{tag}.csv"))
    # ---------- E: P1, P2, P3
    def net_pair(df):
        return df.group_by(["gseed", "pair"]).agg(pl.col("gated").sum().alias("n_gated"), pl.col("Vp").mean().alias("Vp_mean"),
                                                  pl.when(pl.col("gated")).then(pl.col("D")).otherwise(None).mean().alias("D_gated_mean"))
    nr = net_pair(preal).with_columns(pl.lit("real").alias("arm"))
    nw = net_pair(psurg.filter(pl.col("graph") == "mn_r2_within")).with_columns(pl.lit("W").alias("arm"))
    nx = net_pair(psurg.filter(pl.col("graph") == "mn_r2_cross_hi")).with_columns(pl.lit("X_hi").alias("arm"))
    NP = pl.concat([nr, nw, nx]).with_columns((pl.col("n_gated") >= 10).alias("active"))
    NP.write_csv(os.path.join(OUT, f"E_network_pair_{tag}.csv"))
    # P1
    w_act = nw.filter(pl.col("n_gated") >= 10).group_by("pair").agg(pl.len().alias("n")).filter(pl.col("n") >= 3)["pair"].to_list()
    vpw = nw.group_by("pair").agg(pl.col("Vp_mean").median().alias("W")); vpx = nx.group_by("pair").agg(pl.col("Vp_mean").median().alias("X"))
    r1 = vpw.join(vpx, on="pair").filter(pl.col("pair").is_in(w_act)).with_columns((pl.col("X") / pl.col("W")).alias("ratio"))
    P1 = float(r1["ratio"].median()) if r1.height else np.nan
    # P2
    def arm_D(df):
        a = df.filter(pl.col("active")).group_by("pair").agg(pl.col("D_gated_mean").median().alias("Dn"))
        return (float(a["Dn"].median()) if a.height else np.nan), a.height
    Dr, nr_ = arm_D(NP.filter(pl.col("arm") == "real")); Dw, nw_ = arm_D(NP.filter(pl.col("arm") == "W")); Dx, nx_ = arm_D(NP.filter(pl.col("arm") == "X_hi"))
    P2 = bool(Dr > 0.5 and Dw > 0.5 and (np.isfinite(Dx) and Dx < 0.5))
    # P3: structural gamma per network-pair
    gam = []
    for arm, src in [("real", None), ("W", "mn_r2_within"), ("X_hi", "mn_r2_cross_hi")]:
        for k in ([0] if arm == "real" else range(5)):
            g = real_e if src is None else pl.read_parquet(SD + f"surgery_r2_{tag}/{src}_s{k}.parquet")
            for nm, (e, f) in pairs.items():
                ue, uf = influence_vectors(g, sign, N, e, f)
                gam.append(dict(arm=arm, gseed=k, pair=nm, gamma=float(-(ue @ uf) / (np.linalg.norm(ue) * np.linalg.norm(uf)))))
    GM = pl.DataFrame(gam); GM.write_csv(os.path.join(OUT, f"E_gamma_{tag}.csv"))
    J = NP.filter(pl.col("active")).join(GM, on=["arm", "gseed", "pair"], how="inner").drop_nulls("D_gated_mean")
    rho = float(spearmanr(J["gamma"], J["D_gated_mean"]).statistic) if J.height >= 5 else np.nan
    out_json[f"E_{tag}"] = dict(P1_median_ratio_Vplus_X_over_W=P1, P1_n_pairs=r1.height, P1_holds=bool(P1 >= 0.8),
                                D_real=Dr, D_W=Dw, D_X_hi=Dx, n_active_pairs=dict(real=nr_, W=nw_, X_hi=nx_), P2_holds=P2,
                                P3_spearman_gamma_D=rho, P3_n=J.height, P3_holds=bool(rho >= 0.5))
    print(tag, json.dumps(out_json[f"D1_{tag}"], ensure_ascii=False), flush=True)
    print(tag, json.dumps(out_json[f"E_{tag}"], ensure_ascii=False), flush=True)
    print(summ.sort("set"), flush=True)
with open(os.path.join(OUT, "D1_E_verdicts.json"), "w") as f:
    json.dump(out_json, f, indent=1, ensure_ascii=False)
