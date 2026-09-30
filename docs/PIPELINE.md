# Pipeline: result → script → output

File names refer to our archive. Folders: `results/` holds simulation outputs; `analysis/` holds decision tables.

## Models and networks

| Result | Script | Output |
|---|---|---|
| Leg subnetworks (MaleCNS θ = 0.5: 10,173 neurons; MANC: 10,440) | `src/build_graph.py`, `src/build_graph_manc.py` | `graphs/nodes_*.parquet`, `graphs/edges_*.parquet` |
| Motor-neuron → joint/pool map (20 antagonistic pairs; exact vs approximate) | `src/mn_joint_map.py` | `graphs/mn_joint_map_*.parquet` |
| Transmitter signs (ACh +1; GABA/Glu −1) | `src/make_signs.py` | `graphs/signs_*` |
| Six rewired families, five networks each; reciprocity-preserving families | `src/nulls.py`, `src/nulls_recip.py` | `graphs/nulls/{tag}/` |
| 25 networks per competitive family (s5–s24) | `src/nulls.py` | `graphs/nulls/{tag}_n100/` |

## Round 1: main comparison and reassignment

| Result | Script | Output |
|---|---|---|
| Rate model (τ = 20 ms, Euler 1 ms, rectified tanh; `relu_sat` option) | `src/openloop.py` | — |
| 132-setting scan, 25 s runs, scored at 15–25 s | `src/stationary_scan.py` (+ `_resume`, `_v2`) | `results/nullscan_sets_stat_*.parquet` |
| 60-s confirmation (20 noisy + 3 noise-free runs, four windows) | `src/heldout_stat.py`, `src/heldout_stat_resume.py` | `results/heldout_stat_*.parquet` |
| Claim levels (strong/moderate/weak) and six reassignment conditions | `analysis/recalc_all.py` | `recalc/out/claim_levels_3_23_on_heldout.csv`, `surgery_conditions_3_26_on_heldout.csv` |
| Motor-input surgeries (cross-pool, within-pool, strength-matched) | `src/surgery.py`, `src/surgery_strength.py` | `graphs/surgery*/` |
| Reciprocal-innervation index | `src/diag_struct2.py` | `results/diag_recip_innervation_v2_*.parquet` |
| Supplementary Note 1 analyses (knockouts, hemilineage benchmark, early sign tests) | `src/knockout*.py`, `src/ko_pairs.py`, `src/hl_*.py`, `src/compare_*.py`, `src/pool_selectivity.py` | `results/` |

## Round 2: pre-registered follow-ups

| Result | Script | Output |
|---|---|---|
| Pair-level re-run (bitwise reproduction of round 1) | `src/round2_pairs.py`, `src/round2_common.py` | `results/pairs_{scalar,pairs}_*.parquet` |
| S_exact, S_nonThC, per-joint readouts; condition 6 by pair identity | `analysis/analyze_exp1.py` | `analysis/exp1_*.csv` |
| Dose-matched reassignment (X_hi, X_lo, W) | `src/round2_surgery_build.py`, `src/surgery_dose_explore.py` | `experiment2_graphs/`, `surgery_r2_report.json` |
| Six conditions and dose response | `analysis/analyze_exp2_3.py` | `analysis/exp2_conditions.csv`, `exp2_dose_response.csv` |
| 300-s runs | `src/round2_long.py` | `results/long_*.parquet`; `analysis/exp3_*.csv` |
| Post hoc: active pairs alternate (R1) | `analysis/posthoc_r1_active_pairs.py` | `analysis/r1_active_pairs_{real,nulls}_exploratory.csv` |
| Post hoc: ThC pair audit | `analysis/posthoc_thc_audit.py` | `analysis/thc_audit_pairs.csv` |
| Post hoc: bootstrap at 290–300 s | `analysis/posthoc_exp3_bootstrap.py` | `analysis/exp3_posthoc_bootstrap.txt` |
| Post hoc: opposite-signed influence (f_opp, f_both) and excitatory/inhibitory routes | `analysis/ri_semantics_check.py`, `analysis/ri_sign_split.py` | `analysis/ri_semantics_posthoc_20260929/` |

## Round 3: pre-registered robustness and mechanism analyses

| Result | Script | Output |
|---|---|---|
| A. Second nonlinearity (saturating linear) | `src/round3_scan.py`, `src/round3_pairs.py` (`R3_KIND=relu_sat`) | `results/*relu*`; `analysis/A_relu_sat/` |
| B. 25 networks per competitive family | `src/round3_scan.py`, `src/round3_pairs.py` | `results/*n25*`; `analysis/B_n25/` |
| C. Rescue (final layer restored into rewired networks) | `constructions/build_rescue.py` + scan/confirm | `results/*rescue*`; `analysis/C_D2/` |
| D2. Per-neuron ≤5% capped reassignment | `constructions/build_cap.py` + `round3_pairs.py` (`R3_AT_REAL=1`) | `results/*cap*`; `analysis/C_D2/` |
| D1, E. Input-strength audit; common/differential decomposition; gate decomposition | `analysis/analyze_r3_cpu.py` | `analysis/D1_E_cpu/` |
| F. Sign errors (9% random flips; 25% of glutamatergic neurons excitatory) | `constructions/build_signerr.py` + scan/confirm | `results/*se9*`, `*glu25*`; `analysis/F_signerr/` |
| All round-3 decisions | `analysis/analyze_r3_gpu.py` | `analysis/全部判定/round3_decisions.csv` |

## Figures and manuscript

| Step | Script |
|---|---|
| All figures | `figures/run_all.sh` (`fig1.py`–`fig6.py`, `ed1.py`–`ed7.py`; checks in `check_round2.py`, `check_round3.py`) |
| Supplementary tables | `manuscript/make_si_tables.py` |
| Number check (377 checks against the data) and NC length limits | `manuscript/check_numbers.py`, `manuscript/nc_limits.py` |
| Build | `manuscript/build.sh` |
