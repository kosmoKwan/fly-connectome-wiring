"""Supplementary tables of the preprint, generated from the archived analysis tables (read-only).

Every number in these tables is taken from, or computed directly from, the files below; nothing is typed by hand.
  S1  claude_audit_20260927/recalc/out/claim_levels_3_23_on_heldout.csv       (60-s confirmation, section 3.23 levels)
  S2  claude_audit_20260927/recalc/out/surgery_conditions_3_26_on_heldout.csv (six surgery conditions, section 3.26)
  S3  round2_20260927/analysis/exp1_claim_levels.csv                           (round-2 pair-level readouts)
  S4  round2_20260927/analysis/exp1_C6_pair_identity.csv                       (condition 6 by pair identity)
  S5  round2_20260927/analysis/exp2_conditions.csv                             (dose-matched reassignment, six conditions)
  S6  round2_20260927/analysis/exp3_summary.csv, exp3_posthoc_bootstrap.txt    (300-s simulations)
  S7  round2_20260927/analysis/thc_audit_pairs.csv                             (thorax-coxa pairs, all vs exact pools; post hoc)
  S8  round2_20260927/analysis/r1_active_pairs_{real,nulls}_exploratory.csv     (active pairs are antiphasic; post hoc)
  S9  round2_20260927/analysis/ri_semantics_posthoc_20260929/                   (decomposition of the RI index; post hoc)
  S10 round3_20260929/analysis/A_relu_sat/                                      (second nonlinearity)
  S11 round3_20260929/analysis/B_n25/                                           (25 networks per competitive family)
  S12 round3_20260929/analysis/F_signerr/F_signerr_per_variant.csv, glu25        (partial glutamate sign flip)
  S13 round3_20260929/analysis/F_signerr/F_signerr_per_variant.csv, se9          (random sign errors, q = 0.09)
  S14 round3_20260929/analysis/C_D2/round3_decisions.csv                        (capped reassignment)
  S15 round3_20260929/analysis/D1_E_cpu/                                        (common and differential activity)
  S16 round3_20260929/analysis/C_D2/round3_decisions.csv, rescue_build_report.csv (final-layer rescue)

usage: python make_si_tables.py   -> ../generated/si_tables.tex
"""
from pathlib import Path
import json
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # fly-connectome-research-20260926/
OUT = HERE.parent / "generated" / "si_tables.tex"

FAM = {"er": "Density", "role": "Cell role", "degree": "Degree", "leg": "Leg block", "hl": "Lineage block",
       "leg_hl": "Leg $\\times$ lineage", "degree_recip": "Degree + recip.", "leg_hl_recip": "Leg $\\times$ lin. + recip."}
WIN = {"w15_25": "15--25", "w25_35": "25--35", "w35_45": "35--45", "w50_60": "50--60"}
CON = {"theta50": "MaleCNS", "manc50": "MANC"}
LEVEL = {"STRONG": "strong", "MODERATE": "moderate", "WEAK": "weak"}
SURG = {"plain": "Original", "strength_controlled": "Strength-matched"}
RULE = {"real_setting": "real", "own": "own"}
READ = {"S_all": "All pairs (20)", "S_exact": "Exact map (16)", "S_nonThC": "No ThC (14)", "S_ThC": "ThC (6)",
        "S_CTr": "CTr (6)", "S_FTi": "FTi (6)", "S_TiTa": "TiTa (2)"}
C6SURG = {"mn_shuffle_joint": "Original cross-pool", "mn_shuffle_pool": "Original within-pool",
          "mn_shuffle_joint_w": "Strength-matched cross-pool", "mn_shuffle_pool_w": "Strength-matched within-pool"}
YES, NO = "\\ding{51}", "\\ding{55}"


def f3(x):
    return f"${x:.3f}$" if x is not None else "--"


def f2(x):
    return f"${x:.2f}$" if x is not None else "--"


def mark(b):
    return YES if b else NO


def table_s1():
    d = pl.read_csv(ROOT / "claude_audit_20260927/recalc/out/claim_levels_3_23_on_heldout.csv")
    rows = []
    for con in ["theta50", "manc50"]:
        for w in WIN:
            g = d.filter((pl.col("connectome") == con) & (pl.col("window") == w))
            lev = g.filter(pl.col("family").str.starts_with("__LEVEL__"))["family"][0].split(":")[1]
            p = g.filter(pl.col("primary") & ~pl.col("family").str.starts_with("__LEVEL__"))
            top = p.sort("fam_max", descending=True).row(0, named=True)
            low = p.sort("real_minus_median").row(0, named=True)
            rows.append(f"{CON[con]} & {WIN[w]} & {f3(p['real'][0])} & {f3(top['fam_max'])} ({FAM[top['family']]}) & "
                        f"{f3(low['real_minus_median'])} ({FAM[low['family']]}) & {int(p['real_gt_all'].sum())}/6 & {LEVEL[lev]} \\\\")
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return ("\\begin{table}[h]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Pre-specified claim levels in the 60-s confirmation runs.} Network values are means of 20 stochastic runs at each "
            "network's frozen setting; family values are medians over the five networks of a family. Strongest rewired network and smallest "
            "margin (real minus family median) are taken over the six main families. Strong: real above all 30 networks and every margin "
            "$\\geq0.15$; moderate: real above every network in at least five families and above every family median.}\n"
            "\\label{tab:levels}\n\\begin{tabular}{llccccl}\\toprule\n"
            "Connectome & Window (s) & Real $S$ & \\shortstack{Strongest rewired\\\\(family)} & \\shortstack{Smallest margin\\\\(family)} & \\shortstack{Families\\\\all below} & Level \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


def table_s2():
    d = pl.read_csv(ROOT / "claude_audit_20260927/recalc/out/surgery_conditions_3_26_on_heldout.csv")
    rows = []
    for con in ["theta50", "manc50"]:
        for s in SURG:
            for r in RULE:
                for w in WIN:
                    x = d.filter((pl.col("connectome") == con) & (pl.col("surgery") == s) & (pl.col("rule") == r) & (pl.col("window") == w)).row(0, named=True)
                    cs = " & ".join(mark(x[f"C{i}"]) for i in range(1, 6)) + " & " + mark(x["C6_count_proxy"])
                    rows.append(f"{CON[con]} & {SURG[s]} & {RULE[r]} & {WIN[w]} & {f3(x['R'])} & {f3(x['W_median'])} & {f3(x['X_median'])} & "
                                f"{f2(x['X_over_R'])} & {cs} & {mark(x['all_six'])} \\\\")
            rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return ("\\begin{table}[p]\\centering\\scriptsize\n"
            "\\caption{\\textbf{Pre-specified conditions for a selective loss after motor-input reassignment.} $R$, real network; $W$ and $X$, "
            "medians over the five within-pool and five cross-pool networks (each a mean of 20 runs). Parameter rule: real, every network at "
            "the real network's frozen setting; own, every network at its own selected setting. Conditions: (1) $W\\geq0.8R$; (2) $X\\leq0.5R$ "
            "and $R-X\\geq0.10$; (3) $X\\leq0.5W$ and $W-X\\geq0.10$; (4) every cross-pool network below the real network and below every "
            "within-pool network; (5) cross-pool rhythmicity at least 80\\% of both references; (6) at least 80\\% retention of modulated "
            "antagonistic pairs and of oscillating motor neurons, here assessed from the number of modulated pairs (the 60-s runs did not "
            "store pair identities; see Supplementary Table~\\ref{tab:c6} for the pair-identity assessment).}\n"
            "\\label{tab:surgery}\n\\setlength{\\tabcolsep}{3.2pt}\n\\begin{tabular}{llllccccccccccc}\\toprule\n"
            "Connectome & Reassignment & Rule & Window (s) & $R$ & $W$ & $X$ & $X/R$ & (1) & (2) & (3) & (4) & (5) & (6) & All six \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


def table_s3():
    d = pl.read_csv(ROOT / "round2_20260927/analysis/exp1_claim_levels.csv")
    rows = []
    for con in [c for c in ["theta50", "manc50"] if c in d["connectome"].unique().to_list()]:
        for rd in READ:
            for w in WIN:
                x = d.filter((pl.col("connectome") == con) & (pl.col("readout") == rd) & (pl.col("window") == w))
                if x.height == 0:
                    continue
                x = x.row(0, named=True)
                tag = "" if rd in ("S_exact", "S_nonThC") else "$^{\\dagger}$"
                rows.append(f"{CON[con]} & {READ[rd]}{tag} & {WIN[w]} & {f3(x['real'])} & {f3(x['strongest_null'])} ({FAM[x['strongest_family']]}) & "
                            f"{f3(x['min_margin_over_median'])} & {x['families_all_below']}/6 & {LEVEL[x['level']]} \\\\")
            rows.append("\\addlinespace[2pt]")
    body = "\n".join(rows[:-1])
    pending = "" if "manc50" in d["connectome"].unique().to_list() else " The same analysis for MANC is in progress."
    return ("\\begin{table}[p]\\centering\\scriptsize\n"
            "\\caption{\\textbf{Pair-level readouts of the confirmation runs (second-round pre-registration).} Same trajectories as the 60-s "
            "confirmation (reproduced to within rounding error), every network at its own frozen setting; pairs failing the amplitude gate count as zero. "
            "Readouts without a dagger were pre-specified decision readouts; $^{\\dagger}$, descriptive. Level: the rule of Supplementary "
            "Table~\\ref{tab:levels} applied to each readout." + pending + "}\n"
            "\\label{tab:readouts}\n\\renewcommand{\\arraystretch}{0.86}\n\\begin{tabular}{lllcccll}\\toprule\n"
            "Connectome & Readout & Window (s) & Real & \\shortstack{Strongest rewired\\\\(family)} & \\shortstack{Smallest\\\\margin} & \\shortstack{Families\\\\all below} & Level \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


def table_s4():
    d = pl.read_csv(ROOT / "round2_20260927/analysis/exp1_C6_pair_identity.csv")
    rows = []
    for con in [c for c in ["theta50", "manc50"] if c in d["connectome"].unique().to_list()]:
        for s in C6SURG:
            for w in WIN:
                x = d.filter((pl.col("connectome") == con) & (pl.col("surgery") == s) & (pl.col("window") == w))
                if x.height == 0:
                    continue
                x = x.row(0, named=True)
                rows.append(f"{CON[con]} & {C6SURG[s]} & {WIN[w]} & {x['n_contributing_pairs']} & {x['pair_retention_median']:.2f} & "
                            f"{x['osc_retention_median']:.2f} & {mark(x['C6_pair_identity'])} \\\\")
            rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    pending = "" if "manc50" in d["connectome"].unique().to_list() else " The MANC assessment is in progress."
    return ("\\begin{table}[p]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Condition 6 assessed by pair identity (reassigned networks at the real frozen setting).} Contributing "
            "pairs: pairs that pass the amplitude gate in at least half of the real network's 20 runs. Pair retention: fraction of these "
            "pairs that still pass the gate in at least half of a reassigned network's runs (median over five networks). Oscillation "
            "retention: fraction of oscillating motor neurons relative to the real network (median over five networks). Condition met: "
            "both medians $\\geq0.8$." + pending + "}\n"
            "\\label{tab:c6}\n\\begin{tabular}{lllcccc}\\toprule\n"
            "Connectome & Reassignment & Window (s) & \\shortstack{Contributing\\\\pairs} & \\shortstack{Pair\\\\retention} & \\shortstack{Oscillation\\\\retention} & Met \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


ARM = {("mn_r2_cross_hi", "real_setting"): "X$_{\\mathrm{hi}}$, real$^{\\ast}$", ("mn_r2_cross_hi", "own"): "X$_{\\mathrm{hi}}$, own",
       ("mn_r2_cross_lo", "real_setting"): "X$_{\\mathrm{lo}}$, real"}


def table_s5():
    d = pl.read_csv(ROOT / "round2_20260927/analysis/exp2_conditions.csv")
    rows = []
    for con in ["theta50", "manc50"]:
        for (arm, rule), lab in ARM.items():
            for w in WIN:
                x = d.filter((pl.col("connectome") == con) & (pl.col("arm") == arm) & (pl.col("rule") == rule) & (pl.col("window") == w))
                if x.height == 0:
                    continue
                x = x.row(0, named=True)
                cs = " & ".join(mark(x[f"C{i}"]) for i in range(1, 7))
                rows.append(f"{CON[con]} & {lab} & {WIN[w]} & {f3(x['R'])} & {f3(x['W'])} & {f3(x['X'])} & {f2(x['X_over_R'])} & "
                            f"{f2(x['X_rhythm'] / x['R_rhythm'])} & {x['pair_retention']:.2f} & {x['osc_retention']:.2f} & {cs} & {mark(x['all_six'])} \\\\")
            rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return ("\\begin{table}[p]\\centering\\scriptsize\n"
            "\\caption{\\textbf{Dose-matched reassignment: pre-specified conditions (second-round pre-registration).} Cross-pool arms "
            "X$_{\\mathrm{hi}}$ (dose-matched) and X$_{\\mathrm{lo}}$ (lower dose, matched to the within-pool control W in the fraction of "
            "new edges), five networks each, exchanges restricted to edges of the same sign and synapse-count decile. Rule: real, every "
            "network at the real network's frozen setting; own, every network at its own re-optimised setting; $^{\\ast}$, primary test. $R$, real network; "
            "$W$ and $X$, medians of five network means (20 runs each). Rhythm ratio, cross-pool rhythmicity relative to the real network. "
            "Pairs kept and oscillation kept: condition 6 by pair identity (medians of five networks, relative to the real network). "
            "Conditions (1)--(6) as in Supplementary Table~\\ref{tab:surgery}.}\n"
            "\\label{tab:exp2}\n\\setlength{\\tabcolsep}{2.1pt}\n\\begin{tabular}{lllcccccccccccccc}\\toprule\n"
            "Connectome & Arm, rule & Window (s) & $R$ & $W$ & $X$ & $X/R$ & \\shortstack{Rhythm\\\\ratio} & \\shortstack{Pairs\\\\kept} & "
            "\\shortstack{Osc.\\\\kept} & (1) & (2) & (3) & (4) & (5) & (6) & All six \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


NET3 = {("theta50", "real", 0): "MaleCNS real", ("manc50", "real", 0): "MANC real",
        ("manc50", "hl", 1): "MANC lineage block 1", ("manc50", "hl", 0): "MANC lineage block 0"}


def table_s6():
    d = pl.read_csv(ROOT / "round2_20260927/analysis/exp3_summary.csv")
    boot = (ROOT / "round2_20260927/analysis/exp3_posthoc_bootstrap.txt").read_text()
    import re
    m = re.search(r"real minus \('hl', 1\) diff ([\d.]+) bootstrap 95% CI \(20000 resamples, seed 0\) \[(-?[\d.]+), (-?[\d.]+)\]", boot)
    diff, lo, hi = (float(v) for v in m.groups())
    rows = []
    for key, lab in NET3.items():
        x = d.filter((pl.col("connectome") == key[0]) & (pl.col("graph") == key[1]) & (pl.col("gseed") == key[2])).row(0, named=True)
        rows.append(f"{lab} & {x['n_runs']} & {f3(x['mean_S_last'])} & {100 * x['frac_coord_last']:.0f}\\% & {x['switch_events']} & "
                    f"{x['exposure_min']:.1f} & {x['rate_per_run_min']:.2f} ({x['rate_ci_lo']:.2f}--{x['rate_ci_hi']:.2f}) \\\\")
    body = "\n".join(rows)
    return ("\\begin{table}[h]\\centering\\footnotesize\n"
            "\\caption{\\textbf{300-s simulations (second-round pre-registration).} Noisy trajectories (20 continuing the confirmation runs, "
            "20 new) at each network's frozen setting. Coordinated: a 10-s window with $S\\geq0.10$ and at least six modulated pairs. "
            "Switch: a coordinated window followed by at least three non-coordinated windows. Rate: switches per minute spent coordinated, "
            "exact Poisson 95\\% interval. Post hoc (not pre-registered): the difference in mean $S$ at 290--300~s between the real MANC "
            f"network and lineage-block network 1 was {diff:.3f} (bootstrap 95\\% interval {lo:.3f} to {hi:.3f}).}}\n"
            "\\label{tab:exp3}\n\\begin{tabular}{lcccccc}\\toprule\n"
            "Network & Runs & \\shortstack{Mean $S$\\\\290--300 s} & \\shortstack{Coordinated\\\\at 290--300 s} & Switches & "
            "\\shortstack{Minutes\\\\coordinated} & \\shortstack{Switches per minute\\\\coordinated (95\\% CI)} \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


def table_s7():
    d = pl.read_csv(ROOT / "round2_20260927/analysis/thc_audit_pairs.csv")
    rows = []
    for con in ["theta50", "manc50"]:
        for rd, lab in (("all", "All assigned"), ("exact", "Exact only")):
            for w in ("w15_25", "w50_60"):
                for x in d.filter((pl.col("connectome") == con) & (pl.col("readout") == rd) & (pl.col("window") == w)).sort("pair").iter_rows(named=True):
                    rows.append(f"{CON[con]} & {lab} & {WIN[w]} & {x['pair'].replace('_', ' ')} & {f3(x['real'])} & {f3(x['null_max'])} & {x['n_nulls_ge_real']}/30 \\\\")
            rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return ("\\begin{table}[p]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Thorax--coxa pairs with all assigned motor neurons or with exactly assigned motor neurons only (post hoc).} "
            "Pair score of the real network (mean of 20 runs) and of the 30 main rewired networks (each at its own frozen setting). In both "
            "connectomes the promotor-side pool of the front legs contains pleural promotor (exact) and sternal anterior rotator "
            "(approximate) motor neurons, and that of the middle and hind legs only sternal anterior rotator motor neurons; exact-only "
            "pairs therefore exist only in the front legs.}\n"
            "\\label{tab:thc}\n\\begin{tabular}{llllccc}\\toprule\n"
            "Connectome & Pools & Window (s) & Pair & Real & \\shortstack{Rewired\\\\maximum} & \\shortstack{Rewired\\\\$\\geq$ real} \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


def table_s8():
    r = pl.read_csv(ROOT / "round2_20260927/analysis/r1_active_pairs_real_exploratory.csv")
    n = pl.read_csv(ROOT / "round2_20260927/analysis/r1_active_pairs_nulls_exploratory.csv")
    rows = []
    for con in ["theta50", "manc50"]:
        for w in ("w15_25", "w50_60"):
            x = r.filter((pl.col("connectome") == con) & (pl.col("window") == w))
            y = n.filter((pl.col("connectome") == con) & (pl.col("window") == w)).row(0, named=True)
            rows.append(f"{CON[con]} & {WIN[w]} & {x.height} & {int(x['antiphase'].sum())} & {y['n_active_pairs']} & {y['n_antiphase']} & "
                        f"{100 * y['frac_antiphase']:.0f}\\% \\\\")
    body = "\n".join(rows)
    pr = []
    for con in ["theta50", "manc50"]:
        x = r.filter((pl.col("connectome") == con) & (pl.col("window") == "w15_25")).sort("joint", "pair")
        pr.append(f"{CON[con]}: " + "; ".join(f"{q['pair'].replace('_', ' ')} ${q['mean_corr_when_active']:.2f}$" for q in x.iter_rows(named=True)))
    return ("\\begin{table}[h]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Antagonistic pairs that are active are antiphasic in the real networks (post hoc).} Active pair: both pools "
            "pass the amplitude gate in at least half of a network's 20 runs (the contributing pairs of condition 6). Antiphasic: mean "
            "extensor--flexor correlation over the gated runs below zero. Rewired: pairs pooled over the 30 main rewired networks, each at "
            "its own frozen setting. Mean correlation of each active pair of the real networks at 15--25~s: " + "; ".join(pr) + ".}\n"
            "\\label{tab:r1}\n\\begin{tabular}{llccccc}\\toprule\n"
            " & & \\multicolumn{2}{c}{Real network} & \\multicolumn{3}{c}{Rewired networks (30)} \\\\\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
            "Connectome & Window (s) & Active pairs & Antiphasic & Active pairs & Antiphasic & Fraction \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")


def table_s9():
    """Decomposition of the reciprocal-innervation index, pair by pair (post hoc, v7)."""
    D = ROOT / "round2_20260927/analysis/ri_semantics_posthoc_20260929"
    d = pl.read_csv(D / "ri_semantics_real_vs_nulls_by_pair.csv")
    g = pl.read_csv(D / "ri_semantics_graphs.csv")
    pairs = {con: set(d.filter(pl.col("connectome") == con)["pair"]) for con in ("theta50", "manc50")}
    if pairs["theta50"] != pairs["manc50"] or len(pairs["theta50"]) != 20:
        raise SystemExit("f_opp table: the two connectomes must have the same 20 pairs")
    order = sorted(pairs["theta50"], key=lambda q: (["ThC", "CTr", "FTi", "TiTa"].index(q.split("_")[1]), q))
    rows = []
    for q in order:
        cells = [q.replace("_", " ")]
        for con in ("theta50", "manc50"):
            x = d.filter((pl.col("connectome") == con) & (pl.col("pair") == q)).row(0, named=True)
            cells += [f2(x["f_both"]), f2(x["f_opp"]) + ("$^{\\ast}$" if x["fopp_above_all"] else "\\phantom{$^{\\ast}$}"), f2(x["null_max_fopp"])]
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    net = []
    for con in ("theta50", "manc50"):
        r = g.filter((pl.col("connectome") == con) & (pl.col("family") == "real")).row(0, named=True)
        nu = g.filter((pl.col("connectome") == con) & (pl.col("family") != "real"))
        n_above = int(d.filter(pl.col("connectome") == con)["fopp_above_all"].sum())
        n_half = int((d.filter(pl.col("connectome") == con)["f_opp"] > 0.5).sum())
        net.append(f"{CON[con]}, real network {r['f_both']:.2f} and {r['f_opp']:.2f}, rewired networks $f^{{\\mathrm{{opp}}}}$ "
                   f"{nu['f_opp'].min():.2f}--{nu['f_opp'].max():.2f} ({nu.height} networks); $f^{{\\mathrm{{opp}}}}>0.5$ at {n_half} of 20 pairs "
                   f"and above all rewired networks at {n_above} of 20")
    return ("\\begin{table}[p]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Opposite-signed influence behind the reciprocal-innervation index (post hoc).} For each antagonistic pair of the "
            "real networks: $f^{\\mathrm{both}}$, the share of the influence on the two pools carried by neurons that influence both "
            "(Eq.~(\\ref{eq:fboth})); $f^{\\mathrm{opp}}$, the opposite-signed share of their influence (Eq.~(\\ref{eq:fopp})); and the largest "
            "$f^{\\mathrm{opp}}$ among the 30 main rewired networks. $^{\\ast}$, real value above all 30 rewired networks. Influence as in the "
            "reciprocal-innervation index (one- plus two-step, signed synapse counts). Network-level values (medians over pairs): "
            + "; ".join(net) + ".}\n"
            "\\label{tab:fopp}\n\\begin{tabular}{lcccccc}\\toprule\n"
            " & \\multicolumn{3}{c}{MaleCNS} & \\multicolumn{3}{c}{MANC} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Pair & $f^{\\mathrm{both}}$ & $f^{\\mathrm{opp}}$ & \\shortstack{Rewired\\\\max. $f^{\\mathrm{opp}}$} & $f^{\\mathrm{both}}$ & "
            "$f^{\\mathrm{opp}}$ & \\shortstack{Rewired\\\\max. $f^{\\mathrm{opp}}$} \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")



R3A = ROOT / "round3_20260929/analysis"
LV = {"STRONG": "strong", "MODERATE": "moderate", "WEAK": "weak"}
FAMN = {"er": "density", "role": "cell role", "degree": "degree", "leg": "leg block", "hl": "lineage block", "leg_hl": "leg $\\times$ lineage"}


def table_s10():
    d = pl.read_csv(R3A / "A_relu_sat/round3_decisions.csv"); r = pl.read_csv(R3A / "A_relu_sat/A_rhythm_by_family.csv")
    rows = [f"{CON[x['connectome']]} & {WIN[x['window']]} & {f3(x['real'])} & {f3(x['strongest_null'])} ({FAMN[x['strongest_family']]}) & {f2(x['min_margin_over_family_median'])} & {LV[x['level']]} \\\\"
            for x in d.sort("connectome", "window", descending=[True, False]).iter_rows(named=True)]
    rr = []
    for con in ("theta50", "manc50"):
        x = r.filter(pl.col("connectome") == con)
        rr.append(f"{CON[con]} ({f2(x['real_max_rhythm'][0])}) & " + " & ".join(
            f"{f2(y['median_max_rhythm'])}{'$^{\\ast}$' if y['n_above_real'] else ''}" for y in x.iter_rows(named=True)) + " \\\\")
    return ("\\begin{table}[p]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Second neuron nonlinearity: saturating rectifier $f(x)=\\min(\\max(x,0),1)$.} Top: antagonist score of the real network, "
            "strongest of the 30 main rewired networks (family), smallest margin over the family medians and pre-specified level, 60-s confirmation at each "
            "network's own frozen setting. Bottom: maximum rhythmicity in the 132-setting scan, median over the five networks of each family, with the real "
            "network's value in parentheses; $^{\\ast}$, at least one network of the family exceeded the real network. All 30 rewired networks had eligible settings.}\n"
            "\\label{tab:A}\n\\begin{tabular}{llcccc}\\toprule\n"
            "Connectome & Window (s) & Real & Strongest rewired & Smallest margin & Level \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\n\n\\vspace{8pt}\n\\begin{tabular}{lcccccc}\\toprule\n"
            "Connectome (real) & " + " & ".join(FAMN[f] for f in ("er", "role", "degree", "leg", "hl", "leg_hl")) + " \\\\\\midrule\n" + "\n".join(rr) +
            "\n\\bottomrule\\end{tabular}\\end{table}\n")


def table_s11():
    d = pl.read_csv(R3A / "B_n25/round3_decisions.csv").filter(pl.col("exp") == "B_n25")
    rows = []
    for x in d.sort("connectome", "window", descending=[True, False]).iter_rows(named=True):
        pv = json.loads(x["empirical_p"])
        rows.append(f"{CON[x['connectome']]} & {WIN[x['window']]} & {f3(x['real'])} & {f3(x['strongest_null'])} ({FAMN[x['strongest_family']]}) & {int(x['n_nulls_at_or_above_real'])}/90 & "
                    f"{pv['hl']:.3f} & {pv['degree']:.3f} & {pv['leg_hl']:.3f} & {LV[x['level']]} \\\\")
    return ("\\begin{table}[p]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Competitive ensembles: 25 networks in each of the three most competitive families.} Lineage-block, degree and leg $\\times$ lineage "
            "families with 25 networks each and the other three families with five (90 rewired networks per connectome); each network at its own frozen setting. "
            "Empirical $p=(1+n_{\\geq})/(n+1)$ for the expanded families; level by the pre-specified rule.}\n"
            "\\label{tab:B}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{llccccccc}\\toprule\n"
            " & & & & & \\multicolumn{3}{c}{Empirical $p$} & \\\\\\cmidrule(lr){6-8}\n"
            "Connectome & Window (s) & Real & Strongest rewired & $\\geq$ real & lineage & degree & leg $\\times$ lin. & Level \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}\n")


def table_s12():
    d = pl.read_csv(R3A / "F_signerr/F_signerr_per_variant.csv").filter(pl.col("kind") == "glu25").sort("variant", "window")
    rows = [f"{x['variant'][-3:]} & {WIN[x['window']]} & {f3(x['real'])} & {f3(x['null_max'])} & {mark(x['real_above_all'])} \\\\" for x in d.iter_rows(named=True)]
    return ("\\begin{table}[h]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Partial glutamate sign flip (MaleCNS).} Each of the 1,883 glutamatergic neurons made excitatory with probability 0.25 (three draws). "
            "Real network and ten rewired networks (lineage block and degree, five each), each scanned, frozen and confirmed.}\n"
            "\\label{tab:F2}\n\\begin{tabular}{llccc}\\toprule\n"
            "Draw & Window (s) & Real & Strongest of 10 rewired & Real above all \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}\n")


def table_f1():
    d = (pl.read_csv(R3A / "F_signerr/F_signerr_per_variant.csv").filter(pl.col("kind") == "se9")
         .sort("connectome", "variant", "window", descending=[True, False, False]))
    rows = [f"{CON[x['connectome']]} & {x['variant'][-3:]} & {WIN[x['window']]} & {f3(x['real'])} & {f3(x['null_max'])} & {mark(x['real_above_all'])} \\\\"
            for x in d.iter_rows(named=True)]
    npass = {c: int(d.filter((pl.col("connectome") == c) & (pl.col("window") == "w15_25"))["real_above_all"].sum()) for c in ("theta50", "manc50")}
    verdict = "; ".join(f"{CON[c]} {npass[c]} of 3, criterion {'met' if npass[c] >= 2 else 'not met'}" for c in ("theta50", "manc50"))
    return ("\\begin{table}[h]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Random sign errors.} Every neuron except the driven DNg100 neurons had its sign flipped with probability $q=0.09$ "
            "(three draws per connectome), a deliberately harsh upper bound (Supplementary Note~4). Real network and ten rewired networks (MaleCNS lineage "
            "block and degree; MANC lineage block and leg $\\times$ lineage; five each), each scanned, frozen and confirmed. Pre-specified criterion: the real "
            f"network above all ten in the primary window in at least two of the three draws ({verdict}).}}\n"
            "\\label{tab:F1}\n\\begin{tabular}{lllccc}\\toprule\n"
            "Connectome & Draw & Window (s) & Real & Strongest of 10 rewired & Real above all \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}\n")


def table_s13():
    d = pl.read_csv(R3A / "C_D2/round3_decisions.csv").filter(pl.col("exp") == "D2_cap").sort("connectome", "window", descending=[True, False])
    ok = lambda v: mark(str(v).lower() == "true")
    rows = [f"{CON[x['connectome']]} & {WIN[x['window']]} & {f3(x['real'])} & {f3(x['W_median'])} & {f3(x['X_median'])} & {f2(x['pair_retention'])} & {f2(x['osc_retention'])} & "
            + " & ".join(ok(x[c]) for c in ("C1", "C2", "C3", "C4", "C5", "C6")) + " \\\\" for x in d.iter_rows(named=True)]
    return ("\\begin{table}[h]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Capped reassignment: no motor neuron's excitatory or inhibitory input changed by more than 5\\%.} Five networks per connectome at the "
            "real network's frozen setting; $R$ and $W$ are the archived values. Pairs kept: median fraction of the real network's contributing pairs that "
            "still contribute (by pair identity); oscillation kept: median oscillating fraction relative to the real network; condition 6 requires both "
            "to be at least 0.8. The pre-specified criterion was conditions 2, 3 and 6 in the primary window.}\n"
            "\\label{tab:D2}\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llccccccccccc}\\toprule\n"
            "Connectome & Window (s) & $R$ & $W$ & $X$ & Pairs kept & Osc.\\ kept & (1) & (2) & (3) & (4) & (5) & (6) \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}\n")


def table_s14():
    ev = json.loads((R3A / "D1_E_cpu/D1_E_verdicts.json").read_text())
    p3 = pl.read_csv(R3A / "D1_E_cpu/E_P3_within_arm.csv")
    rows = []
    for con in ("theta50", "manc50"):
        e = ev[f"E_{con}"]; w = p3.filter(pl.col("connectome") == con)
        g = {a: w.filter(pl.col("arm") == a)["spearman_gamma_D"][0] for a in ("real", "W", "X_hi", "pooled")}
        rows.append(f"{CON[con]} & {f2(e['D_real'])} & {f2(e['D_W'])} & {f2(e['D_X_hi'])} & {f2(e['P1_median_ratio_Vplus_X_over_W'])} & {f2(g['pooled'])} & "
                    f"{f2(g['real'])} & {f2(g['W'])} & {f2(g['X_hi'])} \\\\")
    sets = ["real", "mn_r2_within", "mn_r2_cross_hi", "er", "role", "degree", "leg", "hl", "leg_hl"]
    lab = {"real": "real network", "mn_r2_within": "within-pool control (W)", "mn_r2_cross_hi": "dose-matched cross-pool (X$_{\\mathrm{hi}}$)", **FAMN}

    def fx(v):                                    # exact values are multiples of 1/400; strip floating-point noise before rounding
        out = f"{round(float(v), 10):.2f}"
        return "$" + ("0.00" if out == "-0.00" else out) + "$"
    S = {con: pl.read_csv(R3A / f"D1_E_cpu/E1_gate_decomposition_summary_{con}.csv") for con in ("theta50", "manc50")}
    gv = lambda con, k, c: S[con].filter(pl.col("set") == k)[c][0]
    gr = [f"{lab[k]} & " + " & ".join(f"{fx(gv(con, k, 'mean_P_gate'))} & {fx(gv(con, k, 'mean_antiphase_given_gate'))}" for con in ("theta50", "manc50")) + " \\\\" for k in sets]
    return ("\\begin{table}[p]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Common and differential activity.} Top: differential share $D$ (Eq.~(\\ref{eq:Dshare})) of the real network, the within-pool controls (W) and "
            "the dose-matched cross-pool arm (X$_{\\mathrm{hi}}$); ratio of the common-mode variance of X$_{\\mathrm{hi}}$ to that of W (P1, pre-set $\\geq0.8$); Spearman "
            "correlation between $\\gamma$ and $D$ over the active pairs of all 11 networks (P3, pre-set $\\geq0.5$) and within each group. Bottom: gate decomposition "
            "(Eq.~(\\ref{eq:gatedec}); descriptive): fraction of pairs gated per run and mean antiphase once gated, 15--25~s; rewired families, medians over five networks.}\n"
            "\\label{tab:E}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lcccccccc}\\toprule\n"
            " & \\multicolumn{3}{c}{$D$} & P1 & \\multicolumn{4}{c}{Spearman $\\gamma$--$D$} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){6-9}\n"
            "Connectome & real & W & X$_{\\mathrm{hi}}$ & ratio & pooled & real & W & X$_{\\mathrm{hi}}$ \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\n\n\\vspace{8pt}\n\\begin{tabular}{lcccc}\\toprule\n"
            " & \\multicolumn{2}{c}{MaleCNS} & \\multicolumn{2}{c}{MANC} \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Network set & Gated & Antiphase once gated & Gated & Antiphase once gated \\\\\\midrule\n" + "\n".join(gr) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def table_s15():
    d = pl.read_csv(R3A / "C_D2/round3_decisions.csv").filter(pl.col("exp").str.starts_with("C_rescue")).sort("connectome", "exp", "window", descending=[True, False, False])
    rb = pl.read_csv(R3A / "rescue_build_report.csv")
    cat = {"不恢复": "none", "部分恢复": "partial", "恢复": "recovery"}
    rows = []
    for x in d.iter_rows(named=True):
        arm = x["exp"].split("_")[-1]
        ri = float(rb.filter((pl.col("connectome") == x["connectome"]) & (pl.col("arm") == arm))["ri_index"].median())
        rows.append(f"{CON[x['connectome']]} & {arm} & {WIN[x['window']]} & {f2(x['median_RF'])} & {int(x['n_above_parent'])}/10 & {f3(x['median_S_arm'])} & {int(x['n_C1p_gt_C0'])}/10 & {cat[x['level']]} & {f3(ri)} \\\\")
    ref = "; ".join(f"{CON[c]}: parents ${float(rb.filter((pl.col('connectome') == c) & (pl.col('arm') == 'parent'))['ri_index'].median()):.3f}$, real ${float(rb.filter((pl.col('connectome') == c) & (pl.col('arm') == 'real'))['ri_index'].median()):.2f}$"
                    for c in ("theta50", "manc50"))
    return ("\\begin{table}[p]\\centering\\footnotesize\n"
            "\\caption{\\textbf{Final-layer rescue.} In the two most competitive rewired families (MaleCNS lineage block and degree; MANC lineage block and leg $\\times$ lineage; "
            "five networks each), the edges onto motor neurons were replaced by those of the real network (C1), of a within-pool control (C1p) or of an X$_{\\mathrm{hi}}$ network (C0). "
            "RF, Eq.~(\\ref{eq:RF}), median over ten networks; C1p $>$ C0, pairs in which the C1p network scored above its C0 counterpart; category by the pre-specified rule; "
            "RI, reciprocal-innervation index (median over networks; " + ref + ").}\n"
            "\\label{tab:C}\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lllcccccc}\\toprule\n"
            "Connectome & Arm & Window (s) & RF & $>$ parent & $S$ & C1p $>$ C0 & Category & RI \\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table}\n")


if __name__ == "__main__":
    tex = "% Generated by src/make_si_tables.py from the archived analysis tables -- do not edit by hand.\n\n"
    tex += "\n".join([table_s1(), table_s2(), table_s3(), table_s4(), table_s5(), table_s6(), table_s7(), table_s8(), table_s9(), table_s10(), table_s11(), table_s12(), table_f1(), table_s13(), table_s14(), table_s15()])
    OUT.write_text(tex, encoding="utf-8")
    print(f"wrote {OUT}")
