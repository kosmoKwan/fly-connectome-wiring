"""Reciprocity-preserving nulls.

Unidirectional edges are swapped among themselves (Maslov-Sneppen inside the
(group_pre, group_post) block) with an extra rejection of any swap that would
create a new reciprocal pair. Reciprocal dyads (a<->b) are swapped as units:
(a<->b, c<->d) -> (a<->d, c<->b), allowed only when group(b) == group(d) so
both directions stay in their blocks. Result: every node's in/out degree, every
block's edge count AND the number of reciprocal dyads are exactly preserved.

Levels: degree_recip (role blocks), leg_hl_recip (role x segment x side x hemilineage).
"""
import sys, json
import numpy as np
import polars as pl
from numba import njit
from pathlib import Path
sys.path.insert(0, "/root/fly-connectome/src")
from nulls import node_groups, graph_stats, HASHMUL

G = Path("/root/fly-connectome/graphs")


@njit(cache=True)
def _find(table, mask, k):
    s = (k * HASHMUL) & mask
    while table[s] != -1:
        if table[s] == k:
            return True
        s = (s + 1) & mask
    return False


@njit(cache=True)
def _remove(table, mask, k):
    s = (k * HASHMUL) & mask
    while table[s] != k:
        s = (s + 1) & mask
    table[s] = -2


@njit(cache=True)
def _insert(table, mask, k):
    s = (k * HASHMUL) & mask
    while table[s] >= 0:
        s = (s + 1) & mask
    table[s] = k


@njit(cache=True)
def _recip_swaps(pre, post, dy_a, dy_b, dy_start, dy_len, block_start, block_len, n_nodes, n_swaps, seed):
    np.random.seed(seed)
    E = pre.shape[0]
    cap = 1
    while cap < 4 * E:
        cap *= 2
    table = np.full(cap, -1, np.int64); mask = cap - 1
    for i in range(E):
        _insert(table, mask, pre[i] * n_nodes + post[i])
    acc_u = 0
    # unidirectional swaps within blocks (blocks index only unidirectional edges)
    for b in range(block_start.shape[0]):
        L = block_len[b]
        if L < 2:
            continue
        o = block_start[b]
        for _ in range(n_swaps * L):
            i = o + np.random.randint(L); j = o + np.random.randint(L)
            a, bb, c, d = pre[i], post[i], pre[j], post[j]
            if i == j or a == c or bb == d or a == d or c == bb:
                continue
            k1 = a * n_nodes + d; k2 = c * n_nodes + bb
            if _find(table, mask, k1) or _find(table, mask, k2):
                continue
            # must not create reciprocity
            if _find(table, mask, d * n_nodes + a) or _find(table, mask, bb * n_nodes + c):
                continue
            _remove(table, mask, a * n_nodes + bb); _remove(table, mask, c * n_nodes + d)
            _insert(table, mask, k1); _insert(table, mask, k2)
            post[i] = d; post[j] = bb
            acc_u += 1
    # reciprocal dyad swaps, done inside classes of dyads sharing (group(a), group(b));
    # dyads were oriented and sorted by class before this call
    acc_r = 0
    for b in range(dy_start.shape[0]):
        L = dy_len[b]
        if L < 2:
            continue
        o = dy_start[b]
        for _ in range(n_swaps * L):
            i = o + np.random.randint(L); j = o + np.random.randint(L)
            if i == j:
                continue
            a, bb, c, d = dy_a[i], dy_b[i], dy_a[j], dy_b[j]
            if a == d or c == bb or a == c or bb == d:
                continue
            if (_find(table, mask, a * n_nodes + d) or _find(table, mask, d * n_nodes + a)
                    or _find(table, mask, c * n_nodes + bb) or _find(table, mask, bb * n_nodes + c)):
                continue
            for k in (a * n_nodes + bb, bb * n_nodes + a, c * n_nodes + d, d * n_nodes + c):
                _remove(table, mask, k)
            for k in (a * n_nodes + d, d * n_nodes + a, c * n_nodes + bb, bb * n_nodes + c):
                _insert(table, mask, k)
            dy_b[i] = d; dy_b[j] = bb
            acc_r += 1
    return acc_u, acc_r


def make_recip_null(nodes, edges, level, seed, swaps=10):
    N = nodes.height
    base = {"degree_recip": "degree", "leg_hl_recip": "leg_hl"}[level]
    gid = node_groups(nodes, base)
    pre = edges["pre"].to_numpy().astype(np.int64); post = edges["post"].to_numpy().astype(np.int64)
    w = edges["weight"].to_numpy()
    key = pre * N + post
    rev = post * N + pre
    is_recip = np.isin(rev, key)
    # unidirectional pool
    upre, upost, uw = pre[~is_recip].copy(), post[~is_recip].copy(), w[~is_recip].copy()
    bkey = gid[upre] * (gid.max() + 1) + gid[upost]
    order = np.argsort(bkey, kind="stable")
    upre, upost, uw, bkey = upre[order], upost[order], uw[order], bkey[order]
    starts = np.flatnonzero(np.r_[True, bkey[1:] != bkey[:-1]]); lens = np.diff(np.r_[starts, len(bkey)])
    # dyads (a<b) with their two weights
    rp, rq, rw = pre[is_recip], post[is_recip], w[is_recip]
    fwd = rp < rq
    wmap = dict(zip((rp * N + rq).tolist(), rw.tolist()))
    dy_a, dy_b = rp[fwd].copy(), rq[fwd].copy()
    # orient so that group(a) <= group(b), then sort dyads by (group(a), group(b)) class
    flip = gid[dy_a] > gid[dy_b]
    dy_a[flip], dy_b[flip] = dy_b[flip].copy(), dy_a[flip].copy()
    ckey = gid[dy_a] * (gid.max() + 1) + gid[dy_b]
    o = np.argsort(ckey, kind="stable"); dy_a, dy_b, ckey = dy_a[o], dy_b[o], ckey[o]
    dy_start = np.flatnonzero(np.r_[True, ckey[1:] != ckey[:-1]]); dy_len = np.diff(np.r_[dy_start, len(ckey)])
    # weights travel with the (a-side) slot: w(a->partner), w(partner->a)
    w_ab = np.array([wmap[a * N + b] for a, b in zip(dy_a, dy_b)]); w_ba = np.array([wmap[b * N + a] for a, b in zip(dy_a, dy_b)])
    # hash table must contain all edges: pass the full edge list through the unidirectional arrays plus dyads
    allpre = np.concatenate([upre, dy_a, dy_b]); allpost = np.concatenate([upost, dy_b, dy_a])
    nu = len(upre)
    acc_u, acc_r = _recip_swaps(allpre, allpost, dy_a, dy_b, dy_start.astype(np.int64), dy_len.astype(np.int64), starts.astype(np.int64), lens.astype(np.int64), N, swaps, seed)
    p = np.concatenate([allpre[:nu], dy_a, dy_b]); q = np.concatenate([allpost[:nu], dy_b, dy_a])
    ww = np.concatenate([uw, w_ab, w_ba])
    return p, q, ww, {"acc_uni": int(acc_u), "acc_dyad": int(acc_r), "n_dyads": int(len(dy_a))}


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else G / "nulls" / tag
    nodes = pl.read_parquet(G / f"nodes_{tag}.parquet"); edges = pl.read_parquet(G / f"edges_{tag}.parquet")
    outdir.mkdir(parents=True, exist_ok=True)
    rep = {}
    for level in ["degree_recip", "leg_hl_recip"]:
        if level == "leg_hl_recip":
            # keep the earlier (dyads effectively frozen) version under a distinct name
            for s0 in range(n_samples):
                old = outdir / f"leg_hl_recip_s{s0}.parquet"
                if old.exists() and not (outdir / f"leg_hl_realdyads_s{s0}.parquet").exists():
                    old.rename(outdir / f"leg_hl_realdyads_s{s0}.parquet")
        for s in range(n_samples):
            p, q, w, info = make_recip_null(nodes, edges, level, 2000 + s)
            pl.DataFrame({"pre": p, "post": q, "weight": w}).write_parquet(outdir / f"{level}_s{s}.parquet")
            st = graph_stats(nodes, p, q, w); st.update(info); rep[f"{level}_s{s}"] = st
            print(level, s, st, flush=True)
    (outdir / "null_report_recip.json").write_text(json.dumps(rep, indent=1))
