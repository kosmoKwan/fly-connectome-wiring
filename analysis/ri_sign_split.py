"""Real networks only: who carries the opposite-signed influence on antagonist pools (post hoc, CPU)?
Share of the opposite-sign overlap mass sum|u_E u_F|[u_E u_F<0] contributed by excitatory vs inhibitory
presynaptic neurons (sign = transmitter sign; glutamate inhibitory), with the influence vectors of diag_struct2.py."""
import sys, os
import numpy as np, polars as pl, scipy.sparse as sp
GD, OUT = sys.argv[1:3]
rows = []
for tag in ["theta50", "manc50"]:
    nodes = pl.read_parquet(os.path.join(GD, f"nodes_{tag}.parquet")); mnmap = pl.read_parquet(os.path.join(GD, f"mn_joint_map_{tag}.parquet"))
    N = nodes.height; sign = nodes["sign"].to_numpy().astype(np.float64)
    e = pl.read_parquet(os.path.join(GD, f"edges_{tag}.parquet"))
    pre, post, w = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy().astype(np.float64)
    W = sp.csr_matrix((w * sign[pre], (post, pre)), shape=(N, N))
    def infl(t):
        r1 = np.asarray(W[t].mean(axis=0)).ravel(); r2 = np.asarray((W[t] @ W).mean(axis=0)).ravel()
        return r1 / (np.abs(r1).max() + 1e-12) + r2 / (np.abs(r2).max() + 1e-12)
    for (lg, j), grp in mnmap.filter(pl.col("joint").is_not_null()).group_by(["leg", "joint"]):
        ext = grp.filter(pl.col("direction") > 0)["idx"].to_numpy(); flx = grp.filter(pl.col("direction") < 0)["idx"].to_numpy()
        if not len(ext) or not len(flx):
            continue
        ue, uf = infl(ext), infl(flx); p = ue * uf; opp = p < 0
        tot = np.abs(p[opp]).sum()
        rows.append(dict(connectome=tag, pair=f"{lg}_{j}", opp_mass_from_exc=float(np.abs(p[opp & (sign > 0)]).sum() / tot) if tot > 0 else np.nan,
                         opp_mass_from_inh=float(np.abs(p[opp & (sign < 0)]).sum() / tot) if tot > 0 else np.nan,
                         n_opp_neurons=int(opp.sum()), n_opp_exc=int((opp & (sign > 0)).sum()), n_opp_inh=int((opp & (sign < 0)).sum())))
D = pl.DataFrame(rows); D.write_csv(os.path.join(OUT, "ri_semantics_real_sign_split.csv"))
print(D.group_by("connectome").agg(pl.col("opp_mass_from_exc").median(), pl.col("opp_mass_from_inh").median(), pl.col("n_opp_neurons").median(),
                                   pl.col("n_opp_exc").median(), pl.col("n_opp_inh").median()))
