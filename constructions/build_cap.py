"""Round 3, experiment D2: strength-capped cross-pool reassignment (CPU). Pre-registration: NAS round3_20260929/预注册_第三轮_20260929.md §2D.

Same swap blocks as round-2 X_hi (presynaptic sign x synapse-count decile x leg|joint, targets of edges onto MNs exchanged),
but a swap is accepted only if, for both affected motor neurons, the excitatory (for excitatory edges) or inhibitory
(for inhibitory edges) input stays within 5% of its original value, |X - X0| <= 0.05 * max(X0, 1) (the round-2 metric's
denominator). At most 10 attempts per edge; construction seeds 7100-7104. Feasible if the median net cross-pool
redistribution over the 5 networks is >= 0.20 (else "infeasible", no relaxation).
usage: build_cap.py OUT_DIR
"""
import sys, os, json
import numpy as np, polars as pl
from numba import njit
sys.path.insert(0, "/Volumes/ninishares/Experiments/fly-connectome-research-20260926/round2_20260927/experiment2_step0")
from surgery_dose_explore import load, metrics

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CAP, SPE, SEEDS = 0.05, 10.0, [7100, 7101, 7102, 7103, 7104]
HASHMUL = np.int64(-7046029254386353131)


@njit(cache=True)
def swap_capped(pre, post, w, chan, block_start, block_len, n_nodes, spe, seed, X0E, X0I, XE, XI, cap):
    np.random.seed(seed)
    E = pre.shape[0]; size = 1
    while size < 4 * E:
        size *= 2
    table = np.full(size, -1, np.int64); mask = size - 1
    for i in range(E):
        k = pre[i] * n_nodes + post[i]; s = (k * HASHMUL) & mask
        while table[s] != -1:
            s = (s + 1) & mask
        table[s] = k
    acc = 0; rej_cap = 0
    for b in range(block_start.shape[0]):
        L = block_len[b]
        if L < 2:
            continue
        o = block_start[b]
        for _ in range(int(spe * L)):
            i = o + np.random.randint(L); j = o + np.random.randint(L)
            a, bb, c, d = pre[i], post[i], pre[j], post[j]
            if i == j or a == c or bb == d or a == d or c == bb:
                continue
            dw = w[j] - w[i]            # bb loses w[i] gains w[j]; d loses w[j] gains w[i]
            if chan[i] > 0:
                nb = XE[bb] + dw; nd = XE[d] - dw
                if abs(nb - X0E[bb]) > cap * max(X0E[bb], 1.0) or abs(nd - X0E[d]) > cap * max(X0E[d], 1.0):
                    rej_cap += 1
                    continue
            else:
                nb = XI[bb] + dw; nd = XI[d] - dw
                if abs(nb - X0I[bb]) > cap * max(X0I[bb], 1.0) or abs(nd - X0I[d]) > cap * max(X0I[d], 1.0):
                    rej_cap += 1
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
            if chan[i] > 0:
                XE[bb] = nb; XE[d] = nd
            else:
                XI[bb] = nb; XI[d] = nd
            acc += 1
    return acc, rej_cap


report = {}
for tag in ["theta50", "manc50"]:
    nodes, edges, sign, is_mn, leg, joint, dr = load(tag); N = nodes.height
    pre, post = edges["pre"].to_numpy().astype(np.int64), edges["post"].to_numpy().astype(np.int64)
    w_orig = edges["weight"].to_numpy(); w = w_orig.astype(np.float64)
    sel = is_mn[post]
    qs = np.quantile(w[sel], np.linspace(0, 1, 11)[1:-1])
    s0 = np.sign(sign[pre[sel]])
    X0E = np.bincount(post[sel][s0 > 0], weights=w[sel][s0 > 0], minlength=N).astype(np.float64)
    X0I = np.bincount(post[sel][s0 < 0], weights=w[sel][s0 < 0], minlength=N).astype(np.float64)
    od = os.path.join(OUT, f"cap_{tag}"); os.makedirs(od, exist_ok=True); stems = []
    for s, seed in enumerate(SEEDS):
        p, q, ww, wo = pre[sel].copy(), post[sel].copy(), w[sel].copy(), w_orig[sel].copy()
        wb = np.searchsorted(qs, ww, side="right")
        k = np.array([f"{leg[x]}|{joint[x]}" for x in q], dtype=object)
        pk = np.unique(k, return_inverse=True)[1]; sg = (np.sign(sign[p]) + 1).astype(np.int64)
        bkey = (sg * 11 + wb) * (pk.max() + 1) + pk
        o = np.argsort(bkey, kind="stable"); p, q, ww, wo, bkey = p[o], q[o], ww[o], wo[o], bkey[o]
        chan = np.sign(sign[p]).astype(np.int64)
        p0, q0 = p.copy(), q.copy()
        st = np.flatnonzero(np.r_[True, bkey[1:] != bkey[:-1]]); ln = np.diff(np.r_[st, len(bkey)])
        XE, XI = X0E.copy(), X0I.copy()
        acc, rej = swap_capped(p, q, ww, chan, st.astype(np.int64), ln.astype(np.int64), N, SPE, seed, X0E, X0I, XE, XI, CAP)
        m = metrics(p0, q0, ww, p, q, sign, is_mn, leg, joint, dr, N)
        # exact max relative change over MNs with input (same denominator as metrics)
        E1 = np.bincount(q[chan > 0], weights=ww[chan > 0], minlength=N); I1 = np.bincount(q[chan < 0], weights=ww[chan < 0], minlength=N)
        actv = np.flatnonzero(is_mn & ((X0E + X0I) > 0))
        m.update(accepted=int(acc), rejected_by_cap=int(rej), max_rel_dE=float((np.abs(E1[actv] - X0E[actv]) / np.maximum(X0E[actv], 1)).max()),
                 max_rel_dI=float((np.abs(I1[actv] - X0I[actv]) / np.maximum(X0I[actv], 1)).max()), seed=seed, cap=CAP, attempts_per_edge=SPE)
        full = pl.DataFrame({"pre": np.r_[pre[~sel], p], "post": np.r_[post[~sel], q], "weight": np.r_[w_orig[~sel], wo]})
        assert full.height == edges.height and full.select("pre", "post").unique().height == full.height
        full.write_parquet(os.path.join(od, f"mn_r3_cap_s{s}.parquet")); stems.append(f"round3/cap_{tag}/mn_r3_cap_s{s}")
        report[f"{tag}/mn_r3_cap_s{s}"] = m
        print(tag, s, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()}), flush=True)
    med = float(np.median([report[f"{tag}/mn_r3_cap_s{s}"]["redistr"] for s in range(5)]))
    report[f"{tag}/FEASIBLE"] = {"median_redistr": med, "feasible": med >= 0.20}
    print(tag, "median redistribution", round(med, 4), "feasible", med >= 0.20, flush=True)
    with open(os.path.join(OUT, f"cap_{tag}_list.txt"), "w") as f:
        f.write(",".join(stems))
with open(os.path.join(OUT, "cap_report.json"), "w") as f:
    json.dump(report, f, indent=1)
