"""Shared style, palette, labels and data access for the figures_v3 set.

Every number drawn on a figure is computed here (or in the figure script) from the archived
simulation outputs on the NAS, which are opened read-only. Nothing in this module writes to the
archive; tidy per-panel tables are written to ../source_data/ by the figure scripts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- paths
ROOT = Path("/Volumes/ninishares/Experiments/fly-connectome-research-20260926")
HELD = ROOT / "final_heldout_20260927_121507_B9ue3i"
RES = HELD / "results"
GRAPHS = HELD / "graphs"
REC = ROOT / "record_20260927_0202/results"
R2 = ROOT / "round2_20260927"
AUDIT_TABLE = ROOT / "exploratory_analysis_20260927_060636/tables/intervention_edge_audit.csv"
R3 = ROOT / "round3_20260929"                                   # round 3 (pre-registered 2026-09-29)
R3A = R3 / "analysis"
RISEM = R2 / "analysis/ri_semantics_posthoc_20260929"          # sign structure of the reciprocal-innervation index (post hoc)
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SRC_DATA = BASE / "source_data"
DERIVED = BASE / "derived"
QA = BASE / "qa"
for _p in (SRC_DATA, DERIVED, QA):
    _p.mkdir(parents=True, exist_ok=True)

TAGS = {"theta50": "MaleCNS", "manc50": "MANC"}
WINDOWS = ["w15_25", "w25_35", "w35_45", "w50_60"]
WIN_LABEL = {"w15_25": "15–25 s", "w25_35": "25–35 s", "w35_45": "35–45 s", "w50_60": "50–60 s"}
WIN_MID = {"w15_25": 20.0, "w25_35": 30.0, "w35_45": 40.0, "w50_60": 55.0}
MAIN = ["er", "role", "degree", "leg", "hl", "leg_hl"]          # the six pre-specified families
EXTRA = ["degree_recip", "leg_hl_recip"]                          # MaleCNS-only secondary families
FAM_LABEL = {"er": "Density", "role": "Cell role", "degree": "Degree", "leg": "Leg block",
             "hl": "Lineage block", "leg_hl": "Leg × lineage",
             "degree_recip": "Degree + recip.", "leg_hl_recip": "Leg × lin. + recip."}
JOINTS = ["ThC", "CTr", "FTi", "TiTa"]
LEGS_ORDER = ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]
SURG = {"mn_shuffle_pool": "Within-pool", "mn_shuffle_joint": "Cross-pool",
        "mn_shuffle_pool_w": "Within-pool, strength-matched", "mn_shuffle_joint_w": "Cross-pool, strength-matched"}

# ----------------------------------------------------------------------------- palette (v2: one teal–rust scheme)
# Every data colour in the figure set comes from this block.
#   real networks (and anything measured on them)  -> teal family; coordination magnitude -> white-to-teal ramp
#   rewired networks                               -> warm ramp, light to dark = more structure preserved
#   extensor / flexor pools                         -> coral / sea green (used only for pool activity and schematics)
#   motor-input reassignment                        -> within-pool = mint (real-like), cross-pool = deep red (scrambled)
# Greys are used only for text, axes and schematic outlines.
REAL = "#005F73"          # real networks: lines, markers, 'strong'
REAL_MID = "#0A9396"
REAL_LIGHT = "#94D2BD"    # 'moderate', light fills
REAL_PALE = "#DDF0EA"     # ± s.e.m. bands, pale fills
INK = "#22313A"           # text, axes, medians
FAM_COLOR = {"er": "#E6D2A0", "role": "#F2B441", "degree": "#E08A1E", "leg": "#C85A1E",
             "hl": "#A3302A", "leg_hl": "#5C1A2E",
             "degree_recip": "#E08A1E", "leg_hl_recip": "#5C1A2E"}
EXT = "#E76F51"           # extensor pool (coral)
FLX = "#2A9D8F"           # flexor pool (sea green)
EXT_FADED = "#F5C8BC"
FLX_FADED = "#B3DDD7"
WITHIN = "#5FB49C"        # within-pool reassignment (structure kept)
CROSS = "#9B2226"         # cross-pool reassignment (pools scrambled)
SURG_COLOR = {"mn_shuffle_pool": WITHIN, "mn_shuffle_joint": CROSS, "mn_shuffle_pool_w": WITHIN, "mn_shuffle_joint_w": CROSS,
              "mn_r2_within": WITHIN, "mn_r2_cross_lo": CROSS, "mn_r2_cross_hi": CROSS}
# joints, proximal -> distal: the teal ramp
JOINT_COLOR = {"ThC": "#005F73", "CTr": "#0A9396", "FTi": "#6CC3AE", "TiTa": "#C9E8DE"}
COORD_COLORS = ["#FFFFFF", "#DDF0EA", "#94D2BD", "#0A9396", "#005F73", "#0B2E3A"]   # coordination magnitude (0 -> high)
LEVEL_FC = {"STRONG": REAL, "MODERATE": REAL_LIGHT, "WEAK": "#FFFFFF"}
PASS = "#E4F3EE"          # 'pre-specified criterion met' shading
SAND = "#F6EFE0"          # neutral warm shading (bands, pending areas)
NULL_RANGE = "#EADCC0"    # range of rewired networks
NULL_MED = "#B98A4A"      # median of rewired networks
SWITCH = "#E08A1E"        # runs that switched state / highlights
RUN = "#A9D6CB"           # individual runs of a real network
RING = "#E08A1E"          # highlight ring
BODY, BODY_EDGE, SEG = "#F5EFE3", "#CDBFA6", "#EADFCB"   # schematic anatomy
NODE = "#4A5A61"          # schematic neurons
SAND_DEEP = "#E9D8A6"; AMBER = "#EE9B00"; CLOUD = "#F3F9F7"
GRID = "#E3E9EA"
MUTED = "#5F6F76"
LIGHT = "#B9C7CB"
PENDING = "#8FA3A8"

MM = 1 / 25.4
FIG_W = 183 * MM          # Nature double-column width


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.labelpad": 2.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 0.9,
        "lines.solid_capstyle": "round",
        "mathtext.default": "regular",
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def panel_label(ax, label, x_pt=-18, y_pt=6, x=0.0, y=1.0, fig=None):
    """Bold lowercase 8-pt panel letter at a fixed physical offset from the axes corner."""
    fig = fig or ax.figure
    off = ScaledTranslation(x_pt / 72, y_pt / 72, fig.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + off, fontsize=8, fontweight="bold",
            ha="left", va="baseline", color=INK)


def panel_title(ax, text, x_pt=0, y_pt=6, loc="left", **kw):
    off = ScaledTranslation(x_pt / 72, y_pt / 72, ax.figure.dpi_scale_trans)
    xx = 0.0 if loc == "left" else (0.5 if loc == "center" else 1.0)
    ax.text(xx, 1.0, text, transform=ax.transAxes + off, fontsize=kw.pop("fontsize", 6.5),
            ha=loc, va="baseline", color=kw.pop("color", INK), **kw)


def add_axes_mm(fig, left, bottom, width, height, fig_h_mm, fig_w_mm=183.0, **kw):
    """Axes placed by physical position in mm (origin bottom-left of the figure)."""
    return fig.add_axes([left / fig_w_mm, bottom / fig_h_mm, width / fig_w_mm, height / fig_h_mm], **kw)


def fmt(x, nd=3):
    """Display rounding used for every number printed on the figures."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:.{nd}f}"
    return s.replace("-", "−")


def check_points_in_view(fig):
    """Data-integrity gate: every scatter point drawn in data coordinates must lie inside its axes' view limits
    (no silently clipped observations)."""
    bad = []
    for ax in fig.axes:
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        for coll in ax.collections:
            if coll.get_offset_transform() is not ax.transData:
                continue
            off = np.asarray(coll.get_offsets(), float)
            if off.size == 0:
                continue
            off = off[np.isfinite(off).all(axis=1)]
            out = (off[:, 0] < x0) | (off[:, 0] > x1) | (off[:, 1] < y0) | (off[:, 1] > y1)
            if out.any():
                bad.append((ax.get_label() or str(id(ax)), off[out][:5].tolist(), (x0, x1, y0, y1)))
    if bad:
        raise RuntimeError(f"points outside the view limits (would be clipped): {bad}")


def save(fig, name, alignment_kwargs=None, dpi_png=600):
    """Clipping gate and alignment gate (both blocking), then PDF/SVG/PNG export into the figures_v3 root."""
    import sys
    check_points_in_view(fig)
    sys.path.insert(0, str(HERE))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    kw = dict(tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    kw.update(alignment_kwargs or {})
    require_matplotlib_panel_alignment(fig, json_out=str(QA / f"{name}.alignment.json"),
                                       overlay_svg=str(QA / f"{name}.alignment.svg"), **kw)
    fig.savefig(BASE / f"{name}.pdf")              # vector, editable text (pdf.fonttype 42)
    fig.savefig(BASE / f"{name}.svg")              # vector, editable text (svg.fonttype none)
    fig.savefig(BASE / f"{name}.png", dpi=600)     # 600-dpi raster preview
    plt.close(fig)


def write_source(df: pd.DataFrame, name: str, note: str | None = None):
    p = SRC_DATA / f"{name}.csv"
    df.to_csv(p, index=False, float_format="%.6g")
    if note:
        (SRC_DATA / f"{name}.README.txt").write_text(note.strip() + "\n", encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- data access
def heldout(tag: str) -> pd.DataFrame:
    """60-s frozen-setting confirmation, one row per (graph, seed, rule, mode, run, window)."""
    df = pd.read_parquet(RES / f"heldout_stat_{tag}.parquet")
    return df


def graph_window_means(tag: str) -> pd.DataFrame:
    """Network-level values = mean over the 20 stochastic runs (undefined runs excluded, as in the
    archived analysis and the manuscript). n_undef counts runs without a defined score."""
    df = heldout(tag)
    st = df[df["mode"] == "stochastic"].copy()
    agg = (st.groupby(["graph", "gseed", "rule", "window"])
             .agg(S=("gmean", "mean"), S_sem=("gmean", lambda v: v.std(ddof=1) / np.sqrt(v.notna().sum())),
                  n_runs=("run", "size"), n_def=("gmean", lambda v: int(v.notna().sum())),
                  rhythm=("rhythm", "mean"), osc=("osc_frac", "mean"), freq=("freq", "mean"),
                  n_gated=("n_gated_pairs", "mean"))
             .reset_index())
    agg["n_undef"] = agg["n_runs"] - agg["n_def"]
    agg.insert(0, "connectome", tag)
    return agg


def scan(tag: str) -> pd.DataFrame:
    if tag == "theta50":
        df = pd.concat([pd.read_parquet(RES / "nullscan_sets_stat_theta50_part1.parquet"),
                        pd.read_parquet(RES / "nullscan_sets_stat_part2_theta50.parquet")], ignore_index=True)
    else:
        df = pd.read_parquet(RES / f"nullscan_sets_stat_{tag}.parquet")
    df = df.rename(columns={"seed": "gseed"})
    df["eligible"] = (df["osc_stat"] > 0.2) & (df["freq_stat"] >= 1) & (df["freq_stat"] <= 50)
    df.insert(0, "connectome", tag)
    return df


def recip_graph(tag: str) -> pd.DataFrame:
    return pd.read_parquet(REC / f"diag_recip_innervation_v2_{tag}.parquet")


def recip_pairs(tag: str) -> pd.DataFrame:
    return pd.read_parquet(REC / f"diag_recip_innervation_v2_{tag}_perjoint.parquet")


def round2_scalar(tag: str):
    p = R2 / "results" / f"pairs_scalar_{tag}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def round2_pairs(tag: str):
    p = R2 / "results" / f"pairs_pairs_{tag}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def exp2_results(tag: str):
    """Dose-matched surgery (round 2, experiment 2) simulation output, if it has arrived."""
    for name in (f"pairs_scalar_{tag}_surg2.parquet", f"surg2_scalar_{tag}.parquet"):
        p = R2 / "results" / name
        if p.exists():
            return pd.read_parquet(p)
    return None


def exp2_report() -> dict:
    return json.loads((R2 / "experiment2_graphs/surgery_r2_report.json").read_text())


def intervention_audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_TABLE)


def r3_gmeans(path) -> pd.DataFrame:
    """Round-3 network values exactly as analyze_r3_gpu.py: mean over the 20 stochastic runs of S_all (undefined runs
    excluded), rhythm, osc_frac and n_gated_all, per graph / seed / parameter rule / window."""
    d = pd.read_parquet(path)
    st = d[d["mode"] == "stochastic"]
    return (st.groupby(["graph", "gseed", "rule", "window"])
              .agg(S=("S_all", "mean"), rhythm=("rhythm", "mean"), osc=("osc_frac", "mean"), n_gated=("n_gated_all", "mean"),
                   n_runs=("S_all", "size"))
              .reset_index())


def level_rule(real, fams: dict) -> str:
    """§3.23 claim level (identical to analyze_exp1.py / analyze_r3_gpu.py): fams maps family -> array of network values."""
    gt = {f: bool((real > np.asarray(v)).all()) for f, v in fams.items()}
    med = {f: float(np.median(v)) for f, v in fams.items()}
    if all(gt.values()) and all(real - m >= 0.15 for m in med.values()):
        return "STRONG"
    if sum(gt.values()) >= 5 and all(real > m for m in med.values()):
        return "MODERATE"
    return "WEAK"


def split_pair(p: str):
    leg, joint = p.split("_")
    return leg, joint


def pair_sort_key(p: str):
    leg, joint = split_pair(p)
    return (JOINTS.index(joint), LEGS_ORDER.index(leg))
