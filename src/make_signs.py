"""Pathway-specific transmitter-sign flips with weight-matched controls, for any graph tag.

  <nt>E_toMN_s0      every edge from a <nt> neuron onto a leg MN made excitatory
  <nt>E_ctrl_s0..4   the same number of <nt> edges onto NON-MN targets made excitatory, each
                     matched to a target edge by synapse count (nearest available weight)
for nt in {glu, gaba}. Written as edge files with a per-edge `esign` column (+1 = flipped).
usage: make_signs.py TAG
"""
import sys, json
import numpy as np, polars as pl
from pathlib import Path

G = Path("/root/fly-connectome/graphs")
tag = sys.argv[1] if len(sys.argv) > 1 else "manc50"
OUT = G / f"signs_{tag}"; OUT.mkdir(exist_ok=True)
n = pl.read_parquet(G / f"nodes_{tag}.parquet"); e = pl.read_parquet(G / f"edges_{tag}.parquet")
nt = n["consensus_nt"].fill_null("").to_numpy(); role = n["role"].to_numpy()
pre, post, w = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy()
rep = {}
for short, name in [("glu", "glutamate"), ("gaba", "gaba")]:
    src = nt == name
    target = src[pre] & (role[post] == "MN")
    pool = np.flatnonzero(src[pre] & (role[post] != "MN"))
    e.with_columns(pl.Series("esign", np.where(target, 1, 0))).write_parquet(OUT / f"{short}E_toMN_s0.parquet")
    tw = np.sort(w[target])[::-1]
    rng = np.random.default_rng(5)
    rep[f"{short}E_toMN"] = {"edges": int(target.sum()), "synapses": int(w[target].sum()), "pool_edges": int(len(pool))}
    for s in range(5):
        avail = {}
        for i in rng.permutation(pool):
            avail.setdefault(int(w[i]), []).append(i)
        keys = np.array(sorted(avail)); pick = []
        for x in tw:
            k = keys[np.argmin(np.abs(keys - x))]
            while not avail.get(int(k)):
                keys = keys[keys != k]; k = keys[np.argmin(np.abs(keys - x))]
            pick.append(avail[int(k)].pop())
        m = np.zeros(len(pre), bool); m[np.array(pick)] = True
        e.with_columns(pl.Series("esign", np.where(m, 1, 0))).write_parquet(OUT / f"{short}E_ctrl_s{s}.parquet")
        rep[f"{short}E_ctrl_s{s}"] = {"edges": int(m.sum()), "synapses": int(w[m].sum())}
(OUT / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))
