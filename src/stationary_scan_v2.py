"""Stationary re-scan: the same 132-setting grid and simulator as compare_nulls.py, but 25-s runs
scored only after the transient (primary window 15-25 s; the old 1-5 s window is kept for reference).

Motivation (held-out test, 2026-09-26): at the real graph's best 5-s setting, antagonist alternation
was a ~12-s transient; null graphs were stationary. The 5-s scan therefore mixed transient and
stationary dynamics. Here every graph is scored on the same stationary window, with the scan's own
metrics plus a phase-locking diagnostic (plv_anti, see heldout.py). Rhythmic eligibility is evaluated
on the scored window.
Env as compare_nulls.py: NULL_DIR, NULL_LEVELS, NULL_N, LABEL_PREFIX, SURG (comma list of extra graph
files relative to graphs/, e.g. surgery/mn_shuffle_joint_s0).
usage: stationary_scan.py TAG
"""
import sys, json, time, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
import numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, "/root/fly-connectome/src")
from openloop import load_graph, build_W, simulate, analyse, LEGS, JOINTS

G = Path("/root/fly-connectome/graphs"); OUT = Path("/root/fly-connectome/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "theta50"
T, FS, BATCH = 25.0, 500.0, 33
KIND = os.environ.get("KIND", "tanh_rect"); SIM_SEED = int(os.environ.get("SIM_SEED", "0"))
WINDOWS = {"stat": (15.0, 25.0), "early": (1.0, 5.0)}
nodes, e_real, mnmap = load_graph(tag)
mn_idx = mnmap["idx"].to_numpy(); drive_idx = nodes.filter(pl.col("type") == "DNg100")["idx"].to_numpy()
gs = np.logspace(-4, -1.5, 11); betas = [0.5, 1.0, 2.0]; drives = [0.5, 1.0, 2.0, 4.0]
params = np.array([(g, b, d) for g in gs for b in betas for d in drives], np.float32)
LEVELS = [l for l in os.environ.get("NULL_LEVELS", "er,role,degree,leg,hl,leg_hl").split(",") if l]
NULLDIR = Path(os.environ.get("NULL_DIR", str(G / "nulls" / tag))); NS = int(os.environ.get("NULL_N", "5"))
graphs = [("real", 0, None)] + [(lvl, s, NULLDIR / f"{lvl}_s{s}.parquet") for lvl in LEVELS for s in range(NS) if (NULLDIR / f"{lvl}_s{s}.parquet").exists()]
for f in [x for x in os.environ.get("SURG", "").split(",") if x]:
    stem = Path(f).name; lvl, s = stem.rsplit("_s", 1)
    graphs.append((lvl, int(s), G / f"{f}.parquet"))
leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); dd = mnmap["direction"].to_numpy()
PAIRS = [((leg == L) & (joint == J) & (dd > 0), (leg == L) & (joint == J) & (dd < 0)) for L in LEGS for J in JOINTS]
PAIRS = [p for p in PAIRS if p[0].any() and p[1].any()]


def analytic(x):
    n = len(x); F = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, 1 / FS)
    F[(f < 0.5) | (f > 20)] = 0
    full = np.zeros(n, complex); full[:len(F)] = F; full[1:(n + 1) // 2] *= 2
    return np.fft.ifft(full)


def plv_anti(X):
    Xc = X - X.mean(0); vals = []
    for e, f in PAIRS:
        a, b = Xc[:, e].mean(1), Xc[:, f].mean(1); se, sf = a.std(), b.std()
        if se <= 1e-4 or sf <= 1e-4:
            continue
        if not (min(se, sf) / max(se, sf) >= 0.2 and min(se, sf) >= 0.001):
            vals.append(0.0); continue
        z = analytic(a) * np.conj(analytic(b)); m = (z / (np.abs(z) + 1e-12)).mean()
        vals.append(float(-np.abs(m) * np.cos(np.angle(m))))
    return float(np.mean(vals)) if vals else 0.0


label = os.environ.get("LABEL_PREFIX", "stat_") + tag + ("" if KIND == "tanh_rect" else f"_{KIND}") + (f"_seed{SIM_SEED}" if SIM_SEED else "")
rows, per_set = [], []; t0 = time.time()
for name, s, f in graphs:
    WE, WI = build_W(nodes, e_real if f is None else pl.read_parquet(f))
    res = []
    for c in range(0, len(params), BATCH):
        P = params[c:c + BATCH]
        X = simulate(WE, WI, drive_idx, P, kind=KIND, T=T, record_idx=mn_idx, seed=SIM_SEED)
        for i, p in enumerate(P):
            d = {"graph": name, "seed": s, "g": float(p[0]), "beta": float(p[1]), "drive": float(p[2])}
            for w, (a, b) in WINDOWS.items():
                Xw = X[i, int(a * FS):int(b * FS)]
                r = analyse(Xw, fs=FS, mnmap=mnmap, t_discard=0.0)
                d.update({f"rhythm_{w}": r.get("rhythm", 0.0), f"freq_{w}": r.get("freq", np.nan), f"osc_{w}": r["mn_osc_frac"],
                          f"antiphase_{w}": -r.get("flex_ext_corr_median", np.nan), f"gmean_{w}": r.get("alt_amp_mean", np.nan),
                          f"plv_{w}": plv_anti(Xw) if w == "stat" else np.nan})
            res.append(d)
    df = pl.DataFrame(res).fill_nan(None)
    for w in WINDOWS:
        df = df.with_columns(((pl.col(f"osc_{w}") > 0.2) & (pl.col(f"freq_{w}") >= 1.0) & (pl.col(f"freq_{w}") <= 50.0)).fill_null(False).alias(f"rhythmic_{w}"))
    per_set.append(df)
    rh = df.filter(pl.col("rhythmic_stat"))
    row = {"graph": name, "seed": s, "n_rhythmic_stat": rh.height}
    for m in ["rhythm", "antiphase", "gmean", "plv"]:
        row[f"best_{m}_stat"] = float(rh[f"{m}_stat"].fill_null(0).max()) if rh.height else 0.0
    re_ = df.filter(pl.col("rhythmic_early"))
    for m in ["rhythm", "antiphase", "gmean"]:
        row[f"best_{m}_early"] = float(re_[f"{m}_early"].fill_null(0).max()) if re_.height else 0.0
    rows.append(row); print(json.dumps(row), f"t={time.time() - t0:.0f}s", flush=True)
    pl.DataFrame(rows).write_parquet(OUT / f"nullscan_{label}_partial.parquet"); pl.concat(per_set, how="diagonal_relaxed").write_parquet(OUT / f"nullscan_sets_{label}_partial.parquet")
pl.concat(per_set, how="diagonal_relaxed").write_parquet(OUT / f"nullscan_sets_{label}.parquet")
summ = pl.DataFrame(rows); summ.write_parquet(OUT / f"nullscan_{label}.parquet")
real = summ.filter(pl.col("graph") == "real")
for c in ([] if summ.height < 2 else None) or [c for c in summ.columns if c.startswith("best_")]:
    rv = float(real[c][0]); nul = summ.filter(pl.col("graph") != "real")
    agg = nul.group_by("graph").agg(pl.col(c).median().alias("med"), pl.col(c).max().alias("mx")).sort("graph")
    print(f"{c:22s} real={rv:.3f} | " + "  ".join(f"{g}:{md:.2f}/{mx:.2f}" for g, md, mx in agg.rows()))
