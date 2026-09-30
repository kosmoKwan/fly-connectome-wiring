"""Held-out evaluation: select a setting on the existing scan, freeze it, evaluate it on fresh noise.

For each graph we take two settings chosen ONLY from its existing scan (noise seed 0):
  sel_gmean     its best amplitude-gated mean among rhythmic settings
  sel_combined  its best rhythm x antiphase among rhythmic settings
and re-run each frozen setting for 25 s with
  stochastic    20 fresh noise realisations (seed stream 1000, never used by any scan)
  deterministic 5 noise-free runs from different random initial states (seed stream 2000)
Scores per run and window (early 1-13 s, late 13-25 s, full 1-25 s): the scan's own metrics (rhythm,
frequency, antiphase, gated mean) plus a phase-consistency diagnostic: for every antagonist pair,
the phase-locking value (PLV) of the pool-mean phase difference (0.5-20 Hz analytic signal) and its
mean angle; sustained antiphase pair = passes the lenient amplitude gate, PLV >= 0.5 and mean
phase difference within 45 deg of 180 deg; plv_anti = mean over all measurable pairs of
-PLV*cos(mean phase difference) with gate failures scored 0.
usage: heldout.py TAG
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
sys.path.insert(0, "/root/fly-connectome/src")
import numpy as np, polars as pl
from pathlib import Path
from openloop import load_graph, build_W, simulate, analyse, LEGS, JOINTS

G = Path("/root/fly-connectome/graphs"); R = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
T, FS, N_STOCH, N_DET = 25.0, 500.0, 20, 5
if tag == "theta50":
    main_sets, surg_sets, ndir, sdir = "nullscan_sets_amp_theta50_tanh_rect_DNg100_T5", "nullscan_sets_pool_theta50_tanh_rect_DNg100_T5", G / "nulls/theta50", G / "surgery"
else:
    main_sets, surg_sets, ndir, sdir = "nullscan_sets_manc_manc50_tanh_rect_DNg100_T5", "nullscan_sets_poolmanc_manc50_tanh_rect_DNg100_T5", G / f"nulls/{tag}", G / f"surgery_{tag}"
nodes, e_real, mnmap = load_graph(tag)
drive = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy(); mn_idx = mnmap["idx"].to_numpy()
M = pl.read_parquet(R / f"{main_sets}.parquet"); S = pl.read_parquet(R / f"{surg_sets}.parquet")
GRAPHS = [("real", 0, None, M)] + [(lv, s, ndir / f"{lv}_s{s}.parquet", M) for lv in ["degree", "hl", "leg_hl"] for s in (0, 1)] \
       + [(lv, s, sdir / f"{lv}_s{s}.parquet", S) for lv in ["mn_shuffle_joint", "mn_shuffle_pool"] for s in (0, 1)]
leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); d = mnmap["direction"].to_numpy()
PAIRS = [(L, J, (leg == L) & (joint == J) & (d > 0), (leg == L) & (joint == J) & (d < 0)) for L in LEGS for J in JOINTS]
PAIRS = [p for p in PAIRS if p[2].any() and p[3].any()]


def analytic(x):
    n = len(x); F = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, 1 / FS)
    F[(f < 0.5) | (f > 20)] = 0
    full = np.zeros(n, complex); full[:len(F)] = F; full[1:(n + 1) // 2] *= 2
    return np.fft.ifft(full)


def phase_metrics(X):
    Xc = X - X.mean(0); plv_anti, n_sust, n_meas = [], 0, 0
    for L, J, e, f in PAIRS:
        a, b = Xc[:, e].mean(1), Xc[:, f].mean(1); se, sf = a.std(), b.std()
        if se <= 1e-4 or sf <= 1e-4:
            continue
        n_meas += 1
        gate = min(se, sf) / max(se, sf) >= 0.2 and min(se, sf) >= 0.001
        z = analytic(a) * np.conj(analytic(b)); u = z / (np.abs(z) + 1e-12)
        m = u.mean(); plv, ang = float(np.abs(m)), float(np.angle(m))
        plv_anti.append(-plv * np.cos(ang) if gate else 0.0)
        if gate and plv >= 0.5 and abs(np.angle(np.exp(1j * (ang - np.pi)))) <= np.pi / 4:
            n_sust += 1
    return {"plv_anti": float(np.mean(plv_anti)) if plv_anti else 0.0, "n_sustained_anti": n_sust, "n_measurable": n_meas}


rows = []; t0 = time.time()
for name, seed, f, sets in GRAPHS:
    if f is not None and not f.exists():
        print("missing", f, flush=True); continue
    WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
    sel = sets.filter((pl.col("graph") == name) & (pl.col("seed") == seed) & pl.col("rhythmic"))
    for rule, col in [("sel_gmean", "alt_amp_mean"), ("sel_combined", "combined")]:
        p = sel.sort(col, descending=True, nulls_last=True).head(1).row(0, named=True)
        for mode, n, noise, sstream in [("stochastic", N_STOCH, 0.01, 1000), ("deterministic", N_DET, 0.0, 2000)]:
            P = np.tile(np.array([[p["g"], p["beta"], p["drive"]]], np.float32), (n, 1))
            X = simulate(WE, WI, drive, P, kind="tanh_rect", T=T, record_idx=mn_idx, noise=noise, seed=sstream + seed)
            for k in range(n):
                for win, (a, b) in {"early": (1, 13), "late": (13, 25), "full": (1, 25)}.items():
                    Xw = X[k, int(a * FS):int(b * FS)]
                    r = analyse(Xw, fs=FS, mnmap=mnmap, t_discard=0.0)
                    rows.append({"connectome": tag, "graph": name, "gseed": seed, "rule": rule, "sel_value": float(p[col]),
                                 "g": p["g"], "beta": p["beta"], "drive": p["drive"], "mode": mode, "run": k, "window": win,
                                 "rhythm": r.get("rhythm", 0.0), "freq": r.get("freq", np.nan), "osc_frac": r["mn_osc_frac"],
                                 "antiphase": -r.get("flex_ext_corr_median", np.nan), "gmean": r.get("alt_amp_mean", np.nan),
                                 **phase_metrics(Xw)})
        print(name, seed, rule, f"t={time.time() - t0:.0f}s", flush=True)
df = pl.DataFrame(rows); df.write_parquet(R / f"heldout_{tag}.parquet")
summ = (df.filter((pl.col("window") == "full") & (pl.col("mode") == "stochastic")).group_by("graph", "gseed", "rule")
        .agg(pl.col("sel_value").first(), pl.col("gmean").fill_nan(0).mean().alias("gmean_heldout"), pl.col("gmean").fill_nan(0).std().alias("gmean_sd"),
             pl.col("rhythm").mean(), pl.col("plv_anti").mean(), pl.col("n_sustained_anti").mean())
        .sort("rule", "graph", "gseed"))
with pl.Config(tbl_rows=60, tbl_width_chars=220):
    print(summ)
