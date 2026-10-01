# Synaptic Connectivity Change Prediction in Developmental Connectomes

Predicting *when* and *where* synaptic connections change across *C. elegans* development, using structural motif discovery and trajectory reconstruction — extending [DevoTG](https://github.com/Jayadratha/DevoTG_GSoC).

---

## Overview

Developmental connectomics can describe *what* a nervous system's wiring looks like at a few sparse points in time, but not the finer-grained process of *how* it gets there. [DevoTG](https://github.com/Jayadratha/DevoTG_GSoC) (Gayen, 2026) applied Temporal Graph Networks to *C. elegans* cell lineage events, but treated the connectome itself only descriptively — classifying existing connections as stable, developmental, or variable, without predicting future connectivity change. The authors explicitly name this as an open next step.

This project takes up that step: given the eight real Witvliet et al. (2021) connectome reconstructions spanning birth to adulthood, we predict **which synaptic connections change** (form, strengthen, weaken, disappear) and **approximately when**, by discovering recurring structural motifs along a reconstructed developmental trajectory and using them to condition a continuous-time event model.

**We are explicit about what is and isn't new here:** the underlying techniques (motif-conditioned event prediction, trajectory reconstruction from sparse snapshots) are adapted from prior work in other domains (see [Related Work](#related-work)). The contribution is applying and rigorously validating this combination on structural connectome data, where no predictive model of this kind currently exists.

Full research proposal, architecture specification, and test suite design are in [`docs/`](./docs).

---

## Research Questions

- **H1 (primary):** Does conditioning connectivity-change prediction on learned structural motifs improve performance over structure-only prediction?
- **H2 (secondary):** Can a blind, structure-only method recover the true developmental ordering of the 8 snapshots, validated against their real known collection timestamps?
- **H3 (exploratory):** Do motifs preceding connectivity change near known hub neurons (AVA, AVB, AVE) align with independently reported biological findings?

All three are open empirical questions. A rigorously demonstrated negative result on any of them is a valid, reportable finding — see [`docs/Research_Proposal.md`](./docs/Research_Proposal.md) for full details.

---

## Repository Structure

```
.
├── docs/                        # Full proposal, architecture spec, roadmap, test design
├── external/
│   └── devotg/                  # DevoTG cloned here for Phase 1 reproduction (not committed — see .gitignore)
├── data/
│   ├── loader.py                 # Data loading, temporal splitting
│   └── test_loader.py
├── baseline/
│   ├── tgn.py                    # Plain TGN baseline (Phase 1–2)
│   └── test_tgn.py
├── pcdgm/                        # The proposed model
│   ├── module_a_trajectory.py    # Trajectory reconstruction (A1 interpolation, A2 structural inference)
│   ├── module_b_encoder.py       # Structural encoder
│   ├── module_c_motifs.py        # Motif discovery (learned dictionary)
│   ├── module_d_fusion.py        # Graph fusion
│   ├── module_e_predictor.py     # Which + when predictor
│   └── test_*.py
├── training/
│   ├── loop.py
│   └── test_loop.py
├── ablations/
│   └── run_ablations.py
├── conftest.py                    # Shared pytest fixtures (synthetic data)
├── requirements.txt
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.10+, conda or venv recommended.

```bash
git clone https://github.com/<your-username>/synaptic-connectivity-prediction.git
cd synaptic-connectivity-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**For Phase 1 (DevoTG reproduction), clone their repo separately** (kept out of version control here — see `.gitignore`):

```bash
git clone https://github.com/Jayadratha/DevoTG_GSoC.git external/devotg
cd external/devotg
conda env create -f environment.yml
conda activate devotg
pip install -e .
```

---

## Data

**Witvliet et al. (2021)** *C. elegans* developmental connectome — 8 reconstructions:

| Stage | Timepoint (hours post-hatching) |
|---|---|
| L1 | 0, 5, 8, 16 |
| L2 | 23 |
| L3 | 27 |
| Young adult | 45, 45 (biological replicates) |

~225 neurons, 858–2,496 annotated connections per timepoint. No neural activity data is used in this project.

**Access options:**
- Pre-parsed: via `external/devotg` once cloned (see above).
- Independent cross-check: `pip install cect` → `from cect.WitvlietDataReader import WitvlietDataReader1` (through `8`).
- Original archive: [Zenodo, DOI 10.5281/zenodo.5637219](https://doi.org/10.5281/zenodo.5637219).

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Environment setup, loader verification, node consistency, and non-leakage temporal split tests | ✅ Complete (17 tests passing) |
| 1 | Reproduce DevoTG's lineage-prediction baseline (target AUC 0.839 ± 0.007) — pipeline correctness gate | ✅ Complete (Mean AUC 0.8398 across 5 seeds) |
| 2 | Plain structure-only connectivity-change baseline (novel in isolation — DevoTG never built this) | Up next |
| 3 | Trajectory reconstruction (A1 interpolation, A2 blind structural inference), H2 validated | Pending |
| 4 | Motif discovery module, synthetic sanity-checked before real data | Pending |
| 5 | Full model assembly (Modules A–E) | Pending |
| 6 | Full baseline ladder and ablations, H1 tested | Pending |
| 7 | Interpretability case study (AVA/AVB/AVE), H3 tested | Pending |
| 8 | Write-up | Pending |

Full detail, equations, and hyperparameters for every module: [`docs/Final_Detailed_Architecture.md`](./docs/Final_Detailed_Architecture.md).

---

## Testing

This project is built test-first. Every module has a corresponding test file asserting the specific failure modes we've identified as high-risk for this kind of temporal graph pipeline (temporal leakage, dictionary collapse, broadcast shape errors, and more — full list in [`docs/Complete_Build_Book_With_Tests.md`](./docs/Complete_Build_Book_With_Tests.md)).

```bash
# Run entire test suite
pytest

# Run Phase 1 DevoTG reproduction gate
pytest test_phase1_reproduction.py -v

# Run Phase 0 data loader and temporal split tests
pytest data/test_loader.py -v

# Run Phase 1 & 2 baseline tests
pytest baseline/test_tgn.py -v
```

**Gate Rule:** Phase 1 gate (`test_phase1_reproduction.py`) must pass with mean test AUC within $0.839 \pm 0.02$ before any novel PCDGM module is trusted.

---

## Contributing (for the two of us)

- One branch per module/phase (`phase0-data`, `module-a-trajectory`, etc.), merged into `main` via PR after tests pass.
- Review each other's PRs against the anomaly checklist for that module before merging.
- If using AI-assisted coding, paste the Strict AI Instructions block from `docs/` at the start of every session, and build one module at a time — never the whole pipeline in one prompt. See [`docs/Master_Tech_Roadmap_AI_Guide.md`](./docs/Master_Tech_Roadmap_AI_Guide.md) for the full rationale.

---

## Related Work

- **Temporal Graph Networks** (Rossi et al., 2020) — standard toolkit, used as baselines.
- **DevoTG** (Gayen, 2026, [arXiv:2606.21940](https://arxiv.org/abs/2606.21940)) — direct predecessor; same dataset, predicts lineage events, names connectome-level prediction as future work.
- **STEP** ([arXiv:2603.05874](https://arxiv.org/abs/2603.05874), 2026) — motif-transition-based continuous-time event prediction; closest prior art for the which+when mechanism, adapted here.
- **Temporal motif representation learning** (2026) and **TempME** — establish learned temporal motifs as informative features for dynamic graph link prediction; required as a baseline.
- **Pseudotime-informed temporal graph learning** ([arXiv:2605.28659](https://arxiv.org/abs/2605.28659), 2026, applied to gene regulatory networks) — closest prior art for trajectory reconstruction from sparse biological snapshots.

Full honest novelty positioning: [`docs/Research_Proposal.md`](./docs/Research_Proposal.md), Section 2.

---

## Citation

If referencing this work, please also cite the foundational sources:

```
Witvliet, D., et al. "Connectomes across development reveal principles of
brain maturation." Nature 596.7871 (2021): 257-261.

Gayen, J. "DevoTG: Temporal Graph Neural Networks for Modeling C. elegans
Developmental Connectomics." arXiv:2606.21940 (2026).
```

---

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgments

Built on data and infrastructure from the Witvliet et al. (2021) connectome reconstruction, the OpenWorm ConnectomeToolbox, and DevoTG (Gayen, 2026; DevoWorm/OpenWorm Foundation).
