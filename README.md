# How much of fly walking is written in the wiring? — code

Code for the manuscript *How much of fly walking is written in the wiring?* by I. Guan, Y. Zhao, D. Zhang, S. Lyu and I.-M. Chen (2026).

The code builds fixed-weight rate models of the adult *Drosophila* leg motor system from two connectomes, MaleCNS v1.0 and MANC v1.0. It compares each real network with six families of rewired networks under pre-specified criteria. It also runs the motor-input reassignment, rescue, capped-reassignment and sign-perturbation experiments, together with the structural and dynamical analyses reported in the paper.

Status: a versioned copy with a DOI will be deposited in Zenodo on publication, together with the derived graphs and simulation outputs. Until then, these data are available from the corresponding author on request.

## Layout

| Folder | Contents |
|---|---|
| `src/` | Runtime modules, run from one flat directory (`/root/fly-connectome/src` in our setup): subnetwork extraction, the motor-neuron-to-joint map, transmitter signs, rewired networks, motor-input surgeries, the rate-model simulator (`openloop.py`, JAX), parameter scans, 60-s confirmations and the round-2/round-3 drivers |
| `constructions/` | Graph constructions for the third round: rescue graphs, strength-capped reassignment, sign-error node tables |
| `analysis/` | Decision and summary scripts: claim levels and surgery conditions (`recalc_all.py`), round-2 analyses, round-3 analyses, reciprocal-innervation decompositions, post hoc analyses |
| `queues/` | Shell queues used on the GPU workstation, with locks, resume and smoke checks |
| `figures/` | Figure scripts for the submitted figures (`run_all.sh`) |
| `manuscript/` | Manuscript build and number checks. `check_numbers.py` recomputes every reported number from the data and fails the build on any mismatch. The folder also holds the SI table generator, the NC length check and `split_si.py`, which writes the main text and the Supplementary Information as separate PDFs |
| `docs/PIPELINE.md` | Map from each result in the paper to the script that produces it and the file it writes |

## Environments

- **Simulation** (GPU; RTX 5090, driver 591.86; WSL Ubuntu 22.04): Python 3.12.14, jax/jaxlib 0.11.2 (CUDA), numpy 2.5.3, polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, scipy 1.18.1, numba 0.67.0. JAX runs with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`.
- **Analysis** (CPU): Python 3.9.6, numpy 2.0.2, polars 1.36.1, pandas 2.3.3, pyarrow 21.0.0, scipy 1.13.1, numba 0.60.0.
- **Manuscript**: tectonic.

See `requirements-simulation.txt` and `requirements-analysis.txt`.

## Data

The connectomes are public; they are not redistributed here.

- **MaleCNS v1.0** (Berg et al., *Cell* 189, 5504–5526, 2026). Put the neuPrint flat-file export (neuron properties and `syn-partners.feather`) in `data/`.
- **MANC v1.0** (Takemura et al., *eLife* 13, RP97769, 2024). Put the flat files (`manc-v1.0-neuron-properties.feather` and connectivity) in `data_manc/`.

`src/build_graph.py` and `src/build_graph_manc.py` write the leg subnetworks, node tables and motor-neuron-to-joint maps to `graphs/`.

## Running

The scripts use absolute paths from our setup: `/root/fly-connectome/{data,data_manc,graphs,results,src}` on the GPU machine, and `/Volumes/ninishares/...` for the archived results used by the analysis scripts. Either reproduce this layout (symbolic links are enough) or edit the `ROOT`/path constants at the top of each script.

The order of steps is in `docs/PIPELINE.md`. In outline:

1. Build the subnetworks and maps: `build_graph.py`, `build_graph_manc.py`, `mn_joint_map.py`, `make_signs.py`.
2. Build the rewired networks and surgeries: `nulls.py`, `nulls_recip.py`, `surgery*.py`, `round2_surgery_build.py`.
3. Scan the 132 parameter settings (25 s runs) with `stationary_scan*.py`, and freeze each network's best eligible setting.
4. Run the 60-s confirmations with fresh noise: `heldout_stat*.py` (round 1) and `round2_pairs.py` (pair-level re-run, reproducing round 1 to within rounding error).
5. Round 3: `round3_scan.py`, `round3_pairs.py` and the `constructions/` scripts, driven by `queues/round3_queue.sh`.
6. Analyses in `analysis/`, figures in `figures/`, then `manuscript/build.sh`. The build fails if any reported number does not match the data.

## Reproducibility

- Noise streams are fixed (stochastic runs use seed 3000 + network seed; noise-free runs use seed 4000 + network seed).
- Re-running a confirmation reproduces the archived run-window records to within 1e-16.
- `r3_smokecheck.py` and `round2_smokecheck.py` check this before new experiments run.
- Decision rules are coded exactly as pre-specified. Examples: `level()` in `analyze_exp1.py` and `analyze_r3_gpu.py`, and the six reassignment conditions in `analyze_exp2_3.py`.
- The time-stamped pre-registration documents for each round are available from the corresponding author on request.

## License

MIT (see `LICENSE`).

## Contact

Isabel Guan, The Hong Kong University of Science and Technology — eeguan@ust.hk
