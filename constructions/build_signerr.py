"""Round 3, experiment F: node tables with sign errors (CPU). Pre-registration §2F + addendum 1 (q = 0.09).
F1: each neuron with sign +-1 (except the two driven DNg100) flips sign independently with prob 0.09; seeds 11-13; both connectomes.
F2 (MaleCNS): each glutamatergic neuron (sign differs between nodes_theta50 and nodes_theta50gluE) becomes +1 with prob 0.25; seeds 21-23.
Sign-0 neurons are left unchanged (they carry no weight in the model). Writes nodes_<ntag>.parquet, variants list, report.
usage: build_signerr.py GRAPHS_DIR OUT_DIR"""
import sys, os
import numpy as np, polars as pl, scipy.sparse as sp
GD, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
FAMS = {"theta50": ("hl", "degree"), "manc50": ("hl", "leg_hl")}
rows, variants = [], []


def ri(nodes_sign, edges, mnmap, N):   # reciprocal-innervation index as diag_struct2.py (manipulation check)
    pre, post, w = edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy().astype(np.float64)
    W = sp.csr_matrix((w * nodes_sign[pre], (post, pre)), shape=(N, N)); out = []
    for (lg, j), grp in mnmap.filter(pl.col("joint").is_not_null()).group_by(["leg", "joint"]):
        ext = grp.filter(pl.col("direction") > 0)["idx"].to_numpy(); flx = grp.filter(pl.col("direction") < 0)["idx"].to_numpy()
        if not len(ext) or not len(flx):
            continue
        def infl(t):
            r1 = np.asarray(W[t].mean(axis=0)).ravel(); r2 = np.asarray((W[t] @ W).mean(axis=0)).ravel()
            return r1 / (np.abs(r1).max() + 1e-12) + r2 / (np.abs(r2).max() + 1e-12)
        ie, if_ = infl(ext), infl(flx); m = (ie != 0) | (if_ != 0)
        if m.sum() > 10:
            out.append(float(-np.corrcoef(ie[m], if_[m])[0, 1]))
    return float(np.median(out))


for tag in ["theta50", "manc50"]:
    nodes = pl.read_parquet(os.path.join(GD, f"nodes_{tag}.parquet")); N = nodes.height
    edges = pl.read_parquet(os.path.join(GD, f"edges_{tag}.parquet")); mnmap = pl.read_parquet(os.path.join(GD, f"mn_joint_map_{tag}.parquet"))
    sign = nodes["sign"].to_numpy(); dn = (nodes["type"] == "DNg100").fill_null(False).to_numpy().astype(bool); role = nodes["role"].fill_null("NA").to_numpy()
    jobs = [(f"{tag}_se9_d{s}", s, "se") for s in (11, 12, 13)]
    if tag == "theta50":
        glu = pl.read_parquet(os.path.join(GD, "nodes_theta50gluE.parquet"))["sign"].to_numpy() != sign
        assert glu.sum() == 1883
        jobs += [(f"theta50_glu25_d{s}", s, "glu") for s in (21, 22, 23)]
    for ntag, seed, kind in jobs:
        rng = np.random.default_rng(seed); u = rng.random(N)
        if kind == "se":
            flip = (u < 0.09) & (sign != 0) & ~dn; new = np.where(flip, -sign, sign)
        else:
            flip = glu & (u < 0.25) & ~dn; new = np.where(flip, 1, sign)
        nn = nodes.with_columns(pl.Series("sign", new.astype(sign.dtype)))
        nn.write_parquet(os.path.join(OUT, f"nodes_{ntag}.parquet"))
        variants.append(f"{tag} {ntag} {FAMS[tag][0]} {FAMS[tag][1]}")
        rows.append(dict(connectome=tag, ntag=ntag, n_flipped=int(flip.sum()), frac_of_signed=float(flip.sum() / (sign != 0).sum()),
                         flipped_IN=int((flip & (role == "IN")).sum()), flipped_MN=int((flip & (role == "MN")).sum()),
                         flipped_SN=int((flip & (role == "SN")).sum()), flipped_DN=int((flip & (role == "DN")).sum()),
                         exc_to_inh=int((flip & (sign > 0) & (new < 0)).sum()), inh_to_exc=int((flip & (sign < 0) & (new > 0)).sum()),
                         ri_index_real=ri(new.astype(np.float64), edges, mnmap, N)))
        print(rows[-1], flush=True)
    rows.append(dict(connectome=tag, ntag=f"{tag}_base", n_flipped=0, ri_index_real=ri(sign.astype(np.float64), edges, mnmap, N)))
with open(os.path.join(OUT, "signerr_variants.txt"), "w") as f:
    f.write("\n".join(variants) + "\n")
pl.DataFrame(rows, infer_schema_length=None).write_csv(os.path.join(OUT, "signerr_build_report.csv"))
