"""Example motor-pool traces for the figures (CPU re-simulation; NOT confirmation data).

The rate model is the one in the manuscript Methods, re-implemented with scipy sparse matrices
(same equations as the archived GPU code: tau dr/dt = -r + max(tanh(g (W_E + beta W_I) r + I + xi), 0),
tau = 20 ms, dt = 1 ms, noise SD 0.01 per step, r0 ~ U[0, 0.01), MN rates recorded at 500 Hz).
Noise comes from numpy, so the trajectories are not bit-identical to the GPU confirmation runs;
every example is therefore checked against the archived confirmation statistics of the same network
(see derived/example_runs_summary.csv) and labelled in the figures as an example simulation.

Inputs are read-only from the NAS archive; outputs go to ../derived/.
usage: sim_examples.py            (runs every example listed in EXAMPLES)
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = Path("/Volumes/ninishares/Experiments/fly-connectome-research-20260926")
GR = ROOT / "final_heldout_20260927_121507_B9ue3i/graphs"
HELD = ROOT / "final_heldout_20260927_121507_B9ue3i/results"
OUT = Path(__file__).resolve().parent.parent / "derived"
LEGS = ["T1L", "T2L", "T3L", "T1R", "T2R", "T3R"]; JOINTS = ["ThC", "CTr", "FTi", "TiTa"]
FS = 500.0

# (label, connectome tag, archive graph name, gseed, edge file relative to GR or None for the real graph, setting rule)
EXAMPLES = [
    ("malecns_real",        "theta50", "real",             0, None,                                   "own"),
    ("malecns_hl_s3",       "theta50", "hl",               3, "nulls/theta50/hl_s3.parquet",          "own"),
    ("malecns_cross_s0",    "theta50", "mn_shuffle_joint", 0, "surgery/mn_shuffle_joint_s0.parquet",  "real_setting"),
    ("manc_real",           "manc50",  "real",             0, None,                                   "own"),
    ("manc_hl_s1",          "manc50",  "hl",               1, "nulls/manc50/hl_s1.parquet",           "own"),
]


def load(tag, edges_file):
    nodes = pd.read_parquet(GR / f"nodes_{tag}.parquet")
    edges = pd.read_parquet(GR / (edges_file or f"edges_{tag}.parquet"))
    mn = pd.read_parquet(GR / f"mn_joint_map_{tag}.parquet")
    N = len(nodes); sign = nodes["sign"].to_numpy().astype(np.float32)
    pre, post, w = edges["pre"].to_numpy(), edges["post"].to_numpy(), edges["weight"].to_numpy().astype(np.float32)
    s = sign[pre]
    WE = sp.csr_matrix((w[s > 0], (post[s > 0], pre[s > 0])), shape=(N, N), dtype=np.float32)
    WI = sp.csr_matrix((-w[s < 0], (post[s < 0], pre[s < 0])), shape=(N, N), dtype=np.float32)
    drive = nodes.index[nodes["type"] == "DNg100"].to_numpy()
    assert (nodes["idx"].to_numpy() == np.arange(N)).all()
    return nodes, mn, WE, WI, drive


def simulate(WE, WI, drive, g, beta, d, T, seed, rec, dt=1e-3, tau=0.02, noise=0.01, rec_every=2):
    rng = np.random.default_rng(seed); N = WE.shape[0]; a = np.float32(dt / tau)
    W = ((WE + np.float32(beta) * WI) * np.float32(g)).tocsr()
    I = np.zeros(N, np.float32); I[drive] = d
    r = (0.01 * rng.random(N)).astype(np.float32)
    n = int(round(T / dt)); out = np.empty((n // rec_every, len(rec)), np.float32)
    for k in range(n):
        inp = W @ r + I + (noise * rng.standard_normal(N)).astype(np.float32)
        r += a * (-r + np.maximum(np.tanh(inp), 0))
        if (k + 1) % rec_every == 0:
            out[(k + 1) // rec_every - 1] = r[rec]
    return out                                              # (steps, n_MN), 500 Hz


def pair_table(Xw, mn):
    """Per-pair pool statistics exactly as in the archived readout (round2_common.pair_rows)."""
    leg = mn["leg"].to_numpy(); joint = mn["joint"].to_numpy(); dd = mn["direction"].to_numpy()
    Xc = Xw - Xw.mean(0); rows = []
    for L in LEGS:
        for J in JOINTS:
            e = (leg == L) & (joint == J) & (dd > 0); f = (leg == L) & (joint == J) & (dd < 0)
            if not (e.any() and f.any()):
                continue
            a, b = Xc[:, e].mean(1), Xc[:, f].mean(1); se, sf = float(a.std()), float(b.std())
            valid = se > 1e-4 and sf > 1e-4
            c = float(np.corrcoef(a, b)[0, 1]) if valid else np.nan
            gated = bool(valid and min(se, sf) / max(se, sf) >= 0.2 and min(se, sf) >= 0.001)
            rows.append(dict(pair=f"{L}_{J}", leg=L, joint=J, n_ext=int(e.sum()), n_flx=int(f.sum()),
                             sd_ext=se, sd_flx=sf, corr=c, gated=gated, contrib=(-c if gated else 0.0)))
    return pd.DataFrame(rows)


def rhythm_stats(Xw):
    """Rhythmicity / dominant frequency / oscillating fraction as in openloop.analyse (t_discard=0)."""
    Xc = Xw - Xw.mean(0); sd = Xc.std(0); osc = sd > 0.01
    if osc.sum() == 0:
        return 0.0, np.nan, 0.0
    F = np.fft.rfft(Xc[:, osc] * np.hanning(len(Xc))[:, None], axis=0); P = np.abs(F) ** 2
    f = np.fft.rfftfreq(len(Xc), 1 / FS); band = (f >= 0.5) & (f <= 50); Pb = P[band]; fb = f[band]
    pk = Pb.argmax(0)
    conc = np.array([Pb[max(0, p - 1):p + 2, i].sum() / (Pb[:, i].sum() + 1e-12) for i, p in enumerate(pk)])
    return float(np.median(conc)), float(np.median(fb[pk])), float(osc.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = []
    only = set(sys.argv[1:])
    for label, tag, gname, gseed, efile, rule in EXAMPLES:
        if only and label not in only:
            continue
        held = pd.read_parquet(HELD / f"heldout_stat_{tag}.parquet")
        h = held[(held.graph == gname) & (held.gseed == gseed) & (held.rule == rule)]
        g, beta, d = (float(h[c].iloc[0]) for c in ("g", "beta", "drive"))
        nodes, mn, WE, WI, drive = load(tag, efile)
        rec = mn["idx"].to_numpy()
        t0 = time.time()
        X = simulate(WE, WI, drive, g, beta, d, T=25.0, seed=20260928, rec=rec)
        win = X[int(15 * FS):int(25 * FS)]                   # 15-25 s scoring window
        pt = pair_table(win, mn); rh, fq, osc = rhythm_stats(win)
        S = float(pt["contrib"].mean()) if osc > 0 else np.nan
        hs = h[(h["mode"] == "stochastic") & (h.window == "w15_25")]
        summ.append(dict(example=label, connectome=tag, graph=gname, gseed=gseed, rule=rule, g=g, beta=beta, drive=d,
                         cpu_S=S, cpu_rhythm=rh, cpu_freq=fq, cpu_osc_frac=osc, cpu_n_gated=int(pt.gated.sum()),
                         gpu_S_mean=float(hs.gmean.mean()), gpu_S_min=float(hs.gmean.min()), gpu_S_max=float(hs.gmean.max()),
                         gpu_rhythm_mean=float(hs.rhythm.mean()), gpu_freq_median=float(hs.freq.median()),
                         gpu_osc_mean=float(hs.osc_frac.mean()), seconds=time.time() - t0))
        pt.insert(0, "example", label); pt.to_csv(OUT / f"example_pairs_{label}.csv", index=False)
        np.savez_compressed(OUT / f"example_traces_{label}.npz", X=win.astype(np.float32), fs=FS, t0=15.0,
                            mn_idx=rec, leg=np.array(mn["leg"].astype(str).tolist(), dtype="U8"),
                            joint=np.array(mn["joint"].astype(str).tolist(), dtype="U8"),
                            direction=mn["direction"].to_numpy())
        print(json.dumps(summ[-1]), flush=True)
    if summ:
        p = OUT / "example_runs_summary.csv"
        old = pd.read_csv(p) if p.exists() else pd.DataFrame()
        new = pd.DataFrame(summ)
        if len(old):
            old = old[~old.example.isin(new.example)]
        pd.concat([old, new]).to_csv(p, index=False)


if __name__ == "__main__":
    main()
