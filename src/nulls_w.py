"""Nested null models for the leg-VNC connectome.

Each level preserves everything the previous one does, plus one more property.
Edge weights (synapse counts) travel with their edge; a node's sign is a node
property, so every null keeps Dale's law and each neuron's E/I identity.

  er         same N, same edge count, uniform random pairs (no self loops)
  role       + edge count of every (role_pre, role_post) block (SN/DN/IN/MN)
  degree     + every node's in/out degree within each role block
             (Maslov-Sneppen swaps restricted to a block)
  leg        + edge count between every (role, leg segment, side) block
  hl         + edge count between every (role, hemilineage) block
  leg_hl     + both: blocks = (role, segment, side, hemilineage)

`leg` and `hl` are sibling branches above `degree`; `leg_hl` sits above both.
Swaps are always done inside a block, so the block edge counts and every node's
in/out degree (and out-strength) are preserved exactly.
"""
import numpy as np
import polars as pl
from numba import njit
from pathlib import Path

G = Path("/root/fly-connectome/graphs")
HASHMUL = np.int64(-7046029254386353131)  # 0x9E3779B97F4A7C15 as int64
LEVELS = ["er", "role", "degree", "leg", "hl", "leg_hl"]


def node_groups(nodes: pl.DataFrame, level: str) -> np.ndarray:
    role = nodes["role"].to_numpy()
    seg = nodes["leg_seg"].fill_null("NA").to_numpy()
    side = (nodes["side"] if "side" in nodes.columns else nodes["somaSide"]).fill_null("NA").to_numpy()
    hl = nodes["trumanHl"].fill_null("NA").to_numpy()
    if level in ("role", "degree"):
        key = role
    elif level == "leg":
        key = np.char.add(np.char.add(role.astype(str), seg.astype(str)), side.astype(str))
    elif level == "hl":
        key = np.char.add(role.astype(str), np.where(role == "IN", hl, "").astype(str))
    elif level == "leg_hl":
        key = np.char.add(np.char.add(np.char.add(role.astype(str), seg.astype(str)), side.astype(str)),
                          np.where(role == "IN", hl, "").astype(str))
    else:
        raise ValueError(level)
    _, gid = np.unique(key, return_inverse=True)
    return gid.astype(np.int64)


@njit(cache=True)
def _swap_in_blocks(pre, post, block_start, block_len, n_nodes, n_swaps_per_edge, seed):
    np.random.seed(seed)
    E = pre.shape[0]
    # hash set of existing edges: open addressing on key = pre * n + post
    cap = 1
    while cap < 4 * E:
        cap *= 2
    table = np.full(cap, -1, np.int64)
    mask = cap - 1
    for i in range(E):
        k = pre[i] * n_nodes + post[i]
        s = (k * HASHMUL) & mask
        while table[s] != -1:
            s = (s + 1) & mask
        table[s] = k
    accepted = 0
    for b in range(block_start.shape[0]):
        L = block_len[b]
        if L < 2:
            continue
        o = block_start[b]
        for _ in range(n_swaps_per_edge * L):
            i = o + np.random.randint(L)
            j = o + np.random.randint(L)
            a, bb, c, d = pre[i], post[i], pre[j], post[j]
            if i == j or a == c or bb == d or a == d or c == bb:
                continue
            k1 = a * n_nodes + d
            k2 = c * n_nodes + bb
            # reject if either new edge already exists
            s = (k1 * HASHMUL) & mask
            found = False
            while table[s] != -1:
                if table[s] == k1:
                    found = True
                    break
                s = (s + 1) & mask
            if found:
                continue
            s = (k2 * HASHMUL) & mask
            while table[s] != -1:
                if table[s] == k2:
                    found = True
                    break
                s = (s + 1) & mask
            if found:
                continue
            # delete old keys (tombstone = -2), insert new ones
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


def _random_in_blocks(pre_g, post_g, gid, rng):
    """Place the same number of edges per (group_pre, group_post) block uniformly at random."""
    members = {g: np.flatnonzero(gid == g) for g in np.unique(gid)}
    pre_new, post_new = [], []
    blocks, counts = np.unique(np.stack([pre_g, post_g], 1), axis=0, return_counts=True)
    for (gp, gq), k in zip(blocks, counts):
        A, B = members[gp], members[gq]
        seen = set()
        tries = 0
        while len(seen) < k and tries < 50 * k:
            a = A[rng.integers(len(A), size=k)]
            b = B[rng.integers(len(B), size=k)]
            for x, y in zip(a, b):
                if x != y and (x, y) not in seen:
                    seen.add((x, y))
                    if len(seen) == k:
                        break
            tries += k
        if len(seen) < k:
            raise RuntimeError(f"block {gp}->{gq} too dense to sample {k} edges")
        s = np.array(sorted(seen))
        pre_new.append(s[:, 0]); post_new.append(s[:, 1])
    return np.concatenate(pre_new), np.concatenate(post_new)


def make_null(nodes: pl.DataFrame, edges: pl.DataFrame, level: str, seed: int, swaps_per_edge: int = 10):
    rng = np.random.default_rng(seed)
    N = nodes.height
    pre = edges["pre"].to_numpy().astype(np.int64).copy()
    post = edges["post"].to_numpy().astype(np.int64).copy()
    w = edges["weight"].to_numpy().copy()
    if level == "er":
        gid = np.zeros(N, np.int64)
        p2, q2 = _random_in_blocks(gid[pre], gid[post], gid, rng)
        w2 = rng.permutation(w)
        return p2, q2, w2, {"accepted_swaps": None}
    wbin = None
    if level.endswith("_w"):
        # input-strength-preserving variant: swaps only between edges in the same weight quantile bin,
        # so every neuron's excitatory/inhibitory in-strength is (nearly) preserved
        level = level[:-2]
        qs = np.quantile(w, np.linspace(0, 1, 21)[1:-1])
        wbin = np.searchsorted(qs, w, side="right").astype(np.int64)
    gid = node_groups(nodes, level)
    if level == "role":
        order = np.lexsort((gid[post], gid[pre]))
        p2, q2 = _random_in_blocks(gid[pre], gid[post], gid, rng)
        # weights shuffled within each block so block weight distributions are kept
        w2 = np.empty_like(w)
        bpre, bpost = gid[pre], gid[post]
        nb_pre, nb_post = gid[p2], gid[q2]
        for (gp, gq) in set(zip(bpre.tolist(), bpost.tolist())):
            src = np.flatnonzero((bpre == gp) & (bpost == gq))
            dst = np.flatnonzero((nb_pre == gp) & (nb_post == gq))
            w2[dst] = w[rng.permutation(src)]
        return p2, q2, w2, {"accepted_swaps": None}
    # degree-preserving swaps inside (group_pre, group_post) blocks
    bkey = gid[pre] * (gid.max() + 1) + gid[post]
    if wbin is not None:
        bkey = bkey * 32 + wbin
    order = np.argsort(bkey, kind="stable")
    pre, post, w, bkey = pre[order], post[order], w[order], bkey[order]
    starts = np.flatnonzero(np.r_[True, bkey[1:] != bkey[:-1]])
    lens = np.diff(np.r_[starts, len(bkey)])
    acc = _swap_in_blocks(pre, post, starts.astype(np.int64), lens.astype(np.int64), N, swaps_per_edge, seed)
    return pre, post, w, {"accepted_swaps": int(acc), "swap_rate": acc / max(1, swaps_per_edge * len(pre))}


def graph_stats(nodes, pre, post, w):
    N = nodes.height
    indeg = np.bincount(post, minlength=N); outdeg = np.bincount(pre, minlength=N)
    key = set((pre * N + post).tolist())
    recip = sum(1 for a, b in zip(pre.tolist(), post.tolist()) if b * N + a in key) / len(pre)
    role = nodes["role"].to_numpy(); seg = nodes["leg_seg"].fill_null("NA").to_numpy()
    side = (nodes["side"] if "side" in nodes.columns else nodes["somaSide"]).fill_null("NA").to_numpy()
    same_leg = np.mean((seg[pre] == seg[post]) & (side[pre] == side[post]))
    hl = nodes["trumanHl"].fill_null("NA").to_numpy()
    same_hl = np.mean((hl[pre] == hl[post]) & (role[pre] == "IN") & (role[post] == "IN"))
    return {"edges": int(len(pre)), "reciprocity": round(recip, 4), "same_leg_frac": round(float(same_leg), 4),
            "same_hl_IN_frac": round(float(same_hl), 4), "max_indeg": int(indeg.max()), "max_outdeg": int(outdeg.max())}


if __name__ == "__main__":
    import json, sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    nodes = pl.read_parquet(G / f"nodes_{tag}.parquet")
    edges = pl.read_parquet(G / f"edges_{tag}.parquet")
    import os
    out = Path(os.environ.get("NULL_OUTDIR", str(G / "nulls" / tag)))
    levels = os.environ.get("NULL_LEVELS", ",".join(LEVELS)).split(",")
    out.mkdir(parents=True, exist_ok=True)
    report = {"real": graph_stats(nodes, edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy())}
    print("real", report["real"], flush=True)
    for level in levels:
        for s in range(n_samples):
            p, q, w, info = make_null(nodes, edges, level, seed=1000 + s)
            pl.DataFrame({"pre": p, "post": q, "weight": w}).write_parquet(out / f"{level}_s{s}.parquet")
            st = graph_stats(nodes, p, q, w); st.update(info)
            report[f"{level}_s{s}"] = st
            print(level, s, st, flush=True)
    (out / "null_report.json").write_text(json.dumps(report, indent=1))
