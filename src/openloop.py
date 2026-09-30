"""Open-loop VNC dynamics: untrained connectome rate model driven by DNs.

tau dr/dt = -r + f(g * (W_E + beta * W_I) r + I),  W[post, pre] = synapses * sign(pre)
No weights are trained. A grid over (g, beta, drive) is simulated in one GPU batch.
Outputs per parameter set: MN activity, rhythmicity, frequency, flexor/extensor
antiphase per joint, and inter-leg phases.
"""
import os, sys, json, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
import numpy as np
import polars as pl
import jax, jax.numpy as jnp
from pathlib import Path

G = Path("/root/fly-connectome/graphs")
LEGS = ["T1L", "T2L", "T3L", "T1R", "T2R", "T3R"]
JOINTS = ["ThC", "CTr", "FTi", "TiTa"]


def load_graph(tag="theta50", edges_file=None):
    nodes = pl.read_parquet(G / f"nodes_{tag}.parquet")
    edges = pl.read_parquet(edges_file or G / f"edges_{tag}.parquet")
    mnmap = pl.read_parquet(G / f"mn_joint_map_{tag}.parquet")
    return nodes, edges, mnmap


def build_W(nodes, edges):
    N = nodes.height
    sign = nodes["sign"].to_numpy().astype(np.float32)
    pre, post = edges["pre"].to_numpy(), edges["post"].to_numpy()
    w = edges["weight"].to_numpy().astype(np.float32)
    s = sign[pre]
    if "esign" in edges.columns:          # optional per-edge sign override (pathway-specific sign tests)
        es = edges["esign"].to_numpy()
        s = np.where(es != 0, es, s)
    WE = np.zeros((N, N), np.float32); WI = np.zeros((N, N), np.float32)
    np.add.at(WE, (post[s > 0], pre[s > 0]), w[s > 0])
    np.add.at(WI, (post[s < 0], pre[s < 0]), -w[s < 0])
    return WE, WI


def act(x, kind):
    if kind == "relu_sat":
        return jnp.clip(x, 0.0, 1.0)
    if kind == "tanh_rect":
        return jnp.maximum(jnp.tanh(x), 0.0)
    if kind == "relu":
        return jnp.maximum(x, 0.0)
    raise ValueError(kind)


def simulate(WE, WI, drive_idx, params, kind="tanh_rect", T=3.0, dt=1e-3, tau=0.02,
             record_idx=None, rec_every=2, noise=0.01, seed=0):
    """params: array (B, 3) of (g, beta, drive). Returns recorded rates (B, steps, R)."""
    B = params.shape[0]; N = WE.shape[0]
    WE_j, WI_j = jnp.asarray(WE), jnp.asarray(WI)
    g = jnp.asarray(params[:, 0:1], jnp.float32); beta = jnp.asarray(params[:, 1:2], jnp.float32)
    I = jnp.zeros((B, N), jnp.float32).at[:, drive_idx].set(jnp.asarray(params[:, 2:3], jnp.float32))
    rec = jnp.asarray(record_idx)
    n_steps = int(T / dt); a = dt / tau
    key = jax.random.PRNGKey(seed)
    r0 = 0.01 * jax.random.uniform(key, (B, N))

    def step(carry, k):
        r, key = carry
        key, sub = jax.random.split(key)
        inp = g * (r @ WE_j.T + beta * (r @ WI_j.T)) + I + noise * jax.random.normal(sub, (B, N))
        r = r + a * (-r + act(inp, kind))
        return (r, key), r[:, rec]

    def chunk(carry, _):
        carry, ys = jax.lax.scan(step, carry, jnp.arange(rec_every))
        return carry, ys[-1]

    (_, _), out = jax.lax.scan(chunk, (r0, key), jnp.arange(n_steps // rec_every))
    return np.asarray(jnp.transpose(out, (1, 0, 2)))  # (B, steps, R)


def analyse(X, fs, mnmap, t_discard=1.0):
    """X: (steps, n_MN) rates for one parameter set, columns ordered like mnmap."""
    X = X[int(t_discard * fs):]
    res = {"mn_mean": float(X.mean()), "mn_active_frac": float((X.mean(0) > 0.02).mean())}
    Xc = X - X.mean(0)
    sd = Xc.std(0)
    osc = sd > 0.01
    res["mn_osc_frac"] = float(osc.mean())
    if osc.sum() == 0:
        res.update(rhythm=0.0, freq=np.nan)
        return res
    F = np.fft.rfft(Xc[:, osc] * np.hanning(len(Xc))[:, None], axis=0)
    P = np.abs(F) ** 2
    f = np.fft.rfftfreq(len(Xc), 1 / fs)
    band = (f >= 0.5) & (f <= 50)
    Pb = P[band]; fb = f[band]
    pk = Pb.argmax(0)
    # power within +-1 bin of the peak / total band power
    conc = np.array([Pb[max(0, p - 1):p + 2, i].sum() / (Pb[:, i].sum() + 1e-12) for i, p in enumerate(pk)])
    res["rhythm"] = float(np.median(conc))
    res["freq"] = float(np.median(fb[pk]))
    fdom = res["freq"]; kdom = np.argmin(np.abs(f - fdom))
    # per-(leg, joint) extensor-minus-flexor signal and its phase at the dominant frequency
    leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); d = mnmap["direction"].to_numpy()
    anti, anti_amp, anti_amp2, phases = {}, {}, {}, {}
    for L in LEGS:
        for J in JOINTS:
            ext = (leg == L) & (joint == J) & (d > 0); flx = (leg == L) & (joint == J) & (d < 0)
            if ext.any() and flx.any():
                e, fl = Xc[:, ext].mean(1), Xc[:, flx].mean(1)
                se, sf = e.std(), fl.std()
                if se > 1e-4 and sf > 1e-4:
                    c = float(np.corrcoef(e, fl)[0, 1])
                    anti[f"{L}_{J}"] = c
                    # amplitude-aware: both antagonists must be substantially modulated
                    ratio = min(se, sf) / max(se, sf)
                    if ratio >= 0.2 and min(se, sf) >= 0.001:
                        anti_amp[f"{L}_{J}"] = c
                    if ratio >= 0.33 and min(se, sf) >= 0.002:
                        anti_amp2[f"{L}_{J}"] = c
        sel = (leg == L) & (joint == "FTi")
        if sel.any():
            sig = (Xc[:, sel] * d[sel]).mean(1)
            phases[L] = float(np.angle(np.fft.rfft(sig * np.hanning(len(sig)))[kdom]))
    res["flex_ext_corr_median"] = float(np.median(list(anti.values()))) if anti else np.nan
    # amplitude-aware alternation: pairs failing the amplitude criterion count as 0 (no functional alternation)
    n_pairs = sum(1 for L in LEGS for J in JOINTS if ((leg == L) & (joint == J) & (d > 0)).any() and ((leg == L) & (joint == J) & (d < 0)).any())
    vals = [-v for v in anti_amp.values()] + [0.0] * max(0, n_pairs - len(anti_amp))
    res["alt_amp"] = float(np.median(vals)) if n_pairs else np.nan
    res["n_pairs_amp"] = len(anti_amp)
    vals2 = [-v for v in anti_amp2.values()] + [0.0] * max(0, n_pairs - len(anti_amp2))
    res["alt_amp_strict"] = float(np.median(vals2)) if n_pairs else np.nan
    # mean (not median) over all antagonist pairs of the amplitude-gated antiphase: less all-or-none
    res["alt_amp_mean"] = float(np.mean(vals)) if n_pairs else np.nan
    res["flex_ext_corr"] = anti
    if "T1L" in phases:
        res["leg_phase_rel_T1L"] = {k: float(np.angle(np.exp(1j * (v - phases["T1L"])))) for k, v in phases.items()}
    return res


def tripod_score(ph):
    """1 = perfect tripod (T1L,T2R,T3L vs T1R,T2L,T3R in antiphase), from relative FTi phases."""
    if not ph or len(ph) < 6:
        return np.nan
    A = ["T1L", "T2R", "T3L"]; Bs = ["T1R", "T2L", "T3R"]
    z = np.array([np.exp(1j * ph[k]) for k in A] + [-np.exp(1j * ph[k]) for k in Bs])
    return float(np.abs(z.mean()))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
    edges_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "real" else None
    kind = sys.argv[3] if len(sys.argv) > 3 else "tanh_rect"
    drive_types = (sys.argv[4] if len(sys.argv) > 4 else "DNg100").split(",")
    out_name = sys.argv[5] if len(sys.argv) > 5 else f"openloop_{tag}_real_{kind}_{'+'.join(drive_types)}"
    nodes, edges, mnmap = load_graph(tag, edges_file)
    WE, WI = build_W(nodes, edges)
    drive_idx = nodes.filter(pl.col("type").is_in(drive_types))["idx"].to_numpy()
    mn_idx = mnmap["idx"].to_numpy()
    gs = np.logspace(-4, -1.5, 11); betas = [0.5, 1.0, 2.0]; drives = [0.5, 1.0, 2.0, 4.0]
    params = np.array([(g, b, d) for g in gs for b in betas for d in drives], np.float32)
    t0 = time.time()
    rows = []
    for c in range(0, len(params), 66):
        P = params[c:c + 66]
        X = simulate(WE, WI, drive_idx, P, kind=kind, record_idx=mn_idx)
        for i, p in enumerate(P):
            r = analyse(X[i], fs=500.0, mnmap=mnmap)
            ph = r.get("leg_phase_rel_T1L")
            rows.append({"g": float(p[0]), "beta": float(p[1]), "drive": float(p[2]),
                         **{k: v for k, v in r.items() if not isinstance(v, dict)},
                         "tripod": tripod_score(ph), "detail": json.dumps({k: v for k, v in r.items() if isinstance(v, dict)})})
    df = pl.DataFrame(rows)
    outdir = Path("/root/fly-connectome/results"); outdir.mkdir(exist_ok=True)
    df.write_parquet(outdir / f"{out_name}.parquet")
    print(f"drive {drive_types} -> idx {drive_idx.tolist()}  |  {len(params)} sets in {time.time()-t0:.1f}s")
    good = df.filter((pl.col("mn_osc_frac") > 0.2)).sort("rhythm", descending=True)
    print("sets with >20% MNs oscillating:", good.height)
    print(good.select("g", "beta", "drive", "mn_mean", "mn_osc_frac", "rhythm", "freq", "flex_ext_corr_median", "tripod").head(15).to_pandas().to_string())
