"""Round 2, experiment 2, step 0 (CPU only, no simulation): which strength-controlled cross-pool construction
reaches the plain shuffle's cross-pool redistribution while keeping per-MN E/I input strength?

Candidate constructions differ in the swap blocks for edges onto MNs:
  G+dec : presynaptic leg x hemilineage group x sign x weight decile x post block   (the archived _w)
  G+q5  : ... weight quintile        G+q3 : ... weight tercile
  dec   : sign x weight decile x post block (presynaptic group dropped)
Post block: leg|joint (cross-pool) or leg|joint|direction (within-pool control).
Swap intensity (attempts per edge) is varied to reach a target redistribution.
Metrics follow exploratory_analysis_20260927_060636/analyze_intervention_edges.py.
"""
import sys, json
import numpy as np, polars as pl, pandas as pd
from numba import njit

GR = "/Volumes/ninishares/Experiments/fly-connectome-research-20260926/final_heldout_20260927_121507_B9ue3i/graphs/"
HASHMUL = np.int64(-7046029254386353131)


@njit(cache=True)
def swap_in_blocks(pre, post, block_start, block_len, n_nodes, swaps_per_edge, seed):
    np.random.seed(seed)
    E = pre.shape[0]; cap = 1
    while cap < 4 * E:
        cap *= 2
    table = np.full(cap, -1, np.int64); mask = cap - 1
    for i in range(E):
        k = pre[i] * n_nodes + post[i]; s = (k * HASHMUL) & mask
        while table[s] != -1:
            s = (s + 1) & mask
        table[s] = k
    accepted = 0
    for b in range(block_start.shape[0]):
        L = block_len[b]
        if L < 2:
            continue
        o = block_start[b]
        for _ in range(int(swaps_per_edge * L)):
            i = o + np.random.randint(L); j = o + np.random.randint(L)
            a, bb, c, d = pre[i], post[i], pre[j], post[j]
            if i == j or a == c or bb == d or a == d or c == bb:
                continue
            k1 = a * n_nodes + d; k2 = c * n_nodes + bb
            found = False
            for kk in (k1, k2):
                s = (kk * HASHMUL) & mask
                while table[s] != -1:
                    if table[s] == kk:
                        found = True
                        break
                    s = (s + 1) & mask
                if found:
                    break
            if found:
                continue
            for t in range(2):
                kold = a * n_nodes + bb if t == 0 else c * n_nodes + d
                s = (kold * HASHMUL) & mask
                while table[s] != kold:
                    s = (s + 1) & mask
                table[s] = -2
            for t in range(2):
                knew = k1 if t == 0 else k2
                s = (knew * HASHMUL) & mask
                while table[s] >= 0:
                    s = (s + 1) & mask
                table[s] = knew
            post[i], post[j] = d, bb
            accepted += 1
    return accepted


def node_groups_leg_hl(nodes):
    role = nodes["role"].to_numpy(); seg = nodes["leg_seg"].fill_null("NA").to_numpy()
    side = (nodes["side"] if "side" in nodes.columns else nodes["somaSide"]).fill_null("NA").to_numpy()
    hl = nodes["trumanHl"].fill_null("NA").to_numpy()
    key = np.char.add(np.char.add(np.char.add(role.astype(str), seg.astype(str)), side.astype(str)), np.where(role == "IN", hl, "").astype(str))
    return np.unique(key, return_inverse=True)[1].astype(np.int64)


def load(tag):
    nodes = pl.read_parquet(GR + f"nodes_{tag}.parquet"); edges = pl.read_parquet(GR + f"edges_{tag}.parquet"); m = pl.read_parquet(GR + f"mn_joint_map_{tag}.parquet")
    N = nodes.height; sign = nodes["sign"].to_numpy(); is_mn = (nodes["role"] == "MN").to_numpy()
    leg = np.full(N, "NA", dtype=object); joint = np.full(N, "NA", dtype=object); dr = np.zeros(N, np.int64)
    for idx, lg, j, d in m.select("idx", "leg", "joint", "direction").iter_rows():
        leg[idx] = lg; joint[idx] = j if j is not None else "unmapped"; dr[idx] = d
    return nodes, edges, sign, is_mn, leg, joint, dr


def metrics(p0, q0, w, p1, q1, sign, is_mn, leg, joint, dr, N):
    """p0,q0: real MN-input edges; p1,q1: surgery (same order of weights w)."""
    def strengths(p, q):
        s = sign[p]; E = np.bincount(q[s > 0], weights=w[s > 0], minlength=N); I = np.bincount(q[s < 0], weights=w[s < 0], minlength=N); return E, I
    E0, I0 = strengths(p0, q0); E1, I1 = strengths(p1, q1)
    act = np.flatnonzero(is_mn & ((E0 + I0) > 0))
    rE = np.abs(E1[act] - E0[act]) / np.maximum(E0[act], 1); rI = np.abs(I1[act] - I0[act]) / np.maximum(I0[act], 1)
    ok = (dr[q0] != 0) & (joint[q0] != "unmapped") & (sign[p0] != 0)
    ok1 = (dr[q1] != 0) & (joint[q1] != "unmapped") & (sign[p1] != 0)
    def alloc(p, q, msk):
        return pd.Series(w[msk], index=pd.MultiIndex.from_arrays([p[msk], leg[q[msk]], joint[q[msk]], dr[q[msk]]])).groupby(level=[0, 1, 2, 3]).sum()
    A0 = alloc(p0, q0, ok); A1 = alloc(p1, q1, ok1)
    al = pd.concat([A0.rename("a"), A1.rename("b")], axis=1).fillna(0)
    redistr = float((al.a - al.b).abs().sum() / 2 / A0.sum())
    base = set(zip(p0.tolist(), q0.tolist())); novel = np.array([(a, b) not in base for a, b in zip(p1.tolist(), q1.tolist())])
    return dict(redistr=redistr, med_rel_dE=float(np.median(rE)), med_rel_dI=float(np.median(rI)), p95_rel_dE=float(np.quantile(rE, .95)),
                p95_rel_dI=float(np.quantile(rI, .95)), novel_edge_frac=float(novel.mean()))


def build(tag, scheme, postkey, spe, seed, ctx):
    nodes, edges, sign, is_mn, leg, joint, dr = ctx
    N = nodes.height; pg = node_groups_leg_hl(nodes)
    pre, post, w = edges["pre"].to_numpy().astype(np.int64), edges["post"].to_numpy().astype(np.int64), edges["weight"].to_numpy().astype(np.float64)
    sel = is_mn[post]; p, q, ww = pre[sel].copy(), post[sel].copy(), w[sel].copy()
    nb = {"dec": 10, "q5": 5, "q3": 3}[scheme.split("+")[-1]]
    qs = np.quantile(ww, np.linspace(0, 1, nb + 1)[1:-1]); wb = np.searchsorted(qs, ww, side="right")
    if postkey == "cross":
        k = np.array([f"{leg[x]}|{joint[x]}" for x in q], dtype=object)
    else:
        k = np.array([f"{leg[x]}|{joint[x]}|{dr[x]}" for x in q], dtype=object)
    pk = np.unique(k, return_inverse=True)[1]
    sg = (np.sign(sign[p]) + 1).astype(np.int64)
    gterm = pg[p] if scheme.startswith("G+") else np.zeros_like(p)
    bkey = ((gterm * 3 + sg) * (nb + 1) + wb) * (pk.max() + 1) + pk
    o = np.argsort(bkey, kind="stable"); p, q, ww, bkey = p[o], q[o], ww[o], bkey[o]
    p0, q0 = p.copy(), q.copy()
    st = np.flatnonzero(np.r_[True, bkey[1:] != bkey[:-1]]); ln = np.diff(np.r_[st, len(bkey)])
    acc = swap_in_blocks(p, q, st.astype(np.int64), ln.astype(np.int64), N, float(spe), seed)
    m = metrics(p0, q0, ww, p, q, sign, is_mn, leg, joint, dr, N)
    return dict(connectome=tag, scheme=scheme, arm=postkey, spe=spe, seed=seed, accepted=int(acc), **m)


if __name__ == "__main__":
    rows = []
    for tag in ["theta50", "manc50"]:
        ctx = load(tag)
        for scheme in ["G+dec", "G+q5", "G+q3", "dec"]:
            for arm in ["cross", "within"]:
                for spe in ([20] if scheme != "dec" else [0.25, 0.5, 1, 2, 20]):
                    r = build(tag, scheme, arm, spe, 5000, ctx); rows.append(r); print(json.dumps(r), flush=True)
    pl.DataFrame(rows).write_csv(sys.argv[1] if len(sys.argv) > 1 else "surgery_dose_explore.csv")
