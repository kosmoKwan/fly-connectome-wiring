"""Targeted cuts testing the Syed et al. (eLife 2026) inhibition/disinhibition motif, with matched controls.

Targets (edges removed from the real graph):
  cut_13A_13A    13A -> 13A edges (disinhibition among 13A classes)
  cut_13B_13A    13B -> 13A edges
  cut_13AB_IN    13A/13B -> any 13A/13B edges (both of the above plus 13A->13B, 13B->13B)
  cut_13A_MN     13A -> MN edges (direct premotor inhibition of MNs)
Matched controls (5 seeds each): remove the same number of edges, with the same total synapse
count (+-5%), drawn at random from inhibitory IN->IN (or inhibitory IN->MN) edges whose
presynaptic hemilineage is NOT 13A/13B.
"""
import json
import numpy as np, polars as pl
from pathlib import Path

G = Path("/root/fly-connectome/graphs"); OUT = G / "surgery2b"; OUT.mkdir(exist_ok=True)
nodes = pl.read_parquet(G / "nodes_theta50.parquet"); e = pl.read_parquet(G / "edges_theta50.parquet")
role = nodes["role"].to_numpy(); hl = nodes["trumanHl"].fill_null("NA").to_numpy(); sign = nodes["sign"].to_numpy()
pre, post, w = e["pre"].to_numpy(), e["post"].to_numpy(), e["weight"].to_numpy()
isIN = role == "IN"; isMN = role == "MN"
h13A, h13B = hl == "13A", hl == "13B"
targets = {
    "cut_13A_13A": h13A[pre] & h13A[post],
    "cut_13B_13A": h13B[pre] & h13A[post],
    "cut_13AB_IN": (h13A | h13B)[pre] & (h13A | h13B)[post],
    "cut_13A_MN": h13A[pre] & isMN[post],
}
pool_ININ = isIN[pre] & isIN[post] & (sign[pre] < 0) & ~(h13A | h13B)[pre]
pool_INMN = isIN[pre] & isMN[post] & (sign[pre] < 0) & ~(h13A | h13B)[pre]
rep = {}


def save(mask_remove, name):
    keep = ~mask_remove
    e.filter(pl.Series(keep)).write_parquet(OUT / f"{name}.parquet")


rng = np.random.default_rng(11)
for name, m in targets.items():
    save(m, f"{name}_s0")   # the targeted cut itself (deterministic; stored as seed 0 for the comparison tool)
    k, wsum = int(m.sum()), int(w[m].sum())
    pool = np.flatnonzero(pool_INMN if name.endswith("_MN") else pool_ININ)
    rep[name] = {"edges_removed": k, "synapses_removed": wsum, "pool": int(len(pool))}
    # weight-matched control: for every targeted edge, draw a pool edge with the same synapse count
    # (nearest available weight if exhausted), without replacement -> same edge count AND same synapse total
    tw = np.sort(w[m])[::-1]
    for s in range(5):
        avail = {}
        for i in rng.permutation(pool):
            avail.setdefault(int(w[i]), []).append(i)
        keys = np.array(sorted(avail))
        pick = []
        for x in tw:
            kk = keys[np.argmin(np.abs(keys - x))]
            while not avail.get(int(kk)):
                keys = keys[keys != kk]; kk = keys[np.argmin(np.abs(keys - x))]
            pick.append(avail[int(kk)].pop())
        pick = np.array(pick)
        mask = np.zeros(len(pre), bool); mask[pick] = True
        save(mask, f"{name}_ctrl_s{s}")
        rep[f"{name}_ctrl_s{s}"] = {"edges_removed": k, "synapses_removed": int(w[pick].sum())}
(OUT / "report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
