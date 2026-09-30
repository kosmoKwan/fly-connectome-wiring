"""Shared helpers for round 2 (pre-registration: NAS round2_20260927/预注册_第二轮_20260927.md).

Pair-resolved antagonist statistics under two pool definitions:
  all   : the full MN->joint map (20 pairs; identical to openloop.analyse's alt_amp_mean when defined)
  exact : only MNs whose mapping is not flagged approximate (16 pairs: CTr 6, FTi 6, ThC 2, TiTa 2)
"""
import numpy as np, polars as pl
from openloop import LEGS, JOINTS, analyse

GATE_RATIO, GATE_MIN, VALID_MIN = 0.2, 0.001, 1e-4


def make_pairs(mnmap, use_exact):
    leg = mnmap["leg"].to_numpy(); joint = mnmap["joint"].to_numpy(); dd = mnmap["direction"].to_numpy()
    approx = mnmap["approx"].fill_null(False).to_numpy().astype(bool)
    out = []
    for L in LEGS:
        for J in JOINTS:
            base = (leg == L) & (joint == J)
            if use_exact:
                base = base & ~approx
            e = base & (dd > 0); f = base & (dd < 0)
            if e.any() and f.any():
                out.append((f"{L}_{J}", L, J, e, f))
    return out


def pair_rows(Xw, pairs):
    """Per-pair (name, leg, joint, se, sf, corr, gated) for one window of MN rates (steps, n_MN)."""
    Xc = Xw - Xw.mean(0)
    res = []
    for name, L, J, e, f in pairs:
        a = Xc[:, e].mean(1); b = Xc[:, f].mean(1)
        se, sf = float(a.std()), float(b.std())
        valid = se > VALID_MIN and sf > VALID_MIN
        c = float(np.corrcoef(a, b)[0, 1]) if valid else float("nan")
        gated = bool(valid and min(se, sf) / max(se, sf) >= GATE_RATIO and min(se, sf) >= GATE_MIN)
        res.append((name, L, J, se, sf, c, gated))
    return res


def score(rows, joints=None):
    """Amplitude-gated mean antiphase over the selected pairs; non-gated pairs contribute 0."""
    sel = [r for r in rows if joints is None or r[2] in joints]
    if not sel:
        return float("nan")
    return float(np.mean([-r[5] if r[6] else 0.0 for r in sel]))


def window_record(Xw, fs, mnmap, pairs_all, pairs_exact):
    """Scalar metrics + per-pair rows for one window."""
    r = analyse(Xw, fs=fs, mnmap=mnmap, t_discard=0.0)
    ra = pair_rows(Xw, pairs_all); rx = pair_rows(Xw, pairs_exact)
    s_all_raw = score(ra)
    osc = r["mn_osc_frac"]
    rec = {"rhythm": r.get("rhythm", 0.0), "freq": r.get("freq", np.nan), "osc_frac": osc,
           "antiphase": -r.get("flex_ext_corr_median", np.nan), "gmean_analyse": r.get("alt_amp_mean", np.nan),
           "S_all": (s_all_raw if osc > 0 else np.nan), "S_all_raw": s_all_raw, "S_exact": score(rx),
           "S_nonThC": score(ra, {"CTr", "FTi", "TiTa"}), "S_ThC": score(ra, {"ThC"}), "S_CTr": score(ra, {"CTr"}),
           "S_FTi": score(ra, {"FTi"}), "S_TiTa": score(ra, {"TiTa"}),
           "n_gated_all": int(sum(x[6] for x in ra)), "n_gated_exact": int(sum(x[6] for x in rx))}
    prs = [("all",) + x for x in ra] + [("exact",) + x for x in rx]
    return rec, prs


PAIR_COLS = ["readout", "pair", "leg", "joint", "se", "sf", "corr", "gated"]
