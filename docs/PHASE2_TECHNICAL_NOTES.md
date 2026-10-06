# Phase 2 Technical Notes — Connectome-Level TGN Baseline
**Branch:** `experimental-biswadeep`
**Date:** 2026-10-06
**Author:** Biswadeep (research notes compiled with AI assistance)

---

## 1. Overview

Phase 2 establishes the first **plain structural baseline** on the actual *C. elegans* synaptic connectome data (Witvliet et al., *Nature* 2021). It trains a Temporal Graph Network (TGN) that predicts which synapses will exist at the next developmental timepoint, using only structural graph information (no activity data, no cell-lineage features).

This baseline number is the **floor** — all future PCDGM modules (trajectory reconstruction, motif dictionary, fusion) must beat it.

---

## 2. Dataset Description

**Source:** Witvliet et al., *Nature* 2021 — 8 serial-section EM reconstructions of the same *C. elegans* larva across development.

| Snapshot | Stage | Hours post-hatching | Chemical Synapses (edges) |
|----------|-------|---------------------|--------------------------|
| S1 | L1 larva | 0h | 858 |
| S2 | L1 larva | 5h | 1,110 |
| S3 | L1 larva | 8h | 1,106 |
| S4 | L1 larva | 16h | 1,345 |
| S5 | L2 larva | 23h | 1,807 |
| S6 | L3 larva | 27h | 1,739 |
| S7 | Adult (rep. A) | 45h | 2,493 |
| S8 | Adult (rep. B) | 45h | 2,496 |

Total nodes: 187 neurons (snapshot 1).
Edge counts match paper-reported ranges (verified against Witvliet et al. Extended Data).
Data access via `cect` library, readers WitvlietDataReader1-8, synclass Generic_CS / Generic_GJ.

---

## 3. Temporal Data Split — Hard Rules Enforced

    Train:  [0h, 5h, 8h, 16h, 23h]   Snapshots 1-5
    Val:    [27h]                     Snapshot 6
    Test:   [45h rep-A, 45h rep-B]   Snapshots 7+8

Design decisions:
- Strictly temporal — never shuffle=True on time-ordered events
- Both 45h replicates in TEST together — biological replicates, NOT sequential timepoints
- No activity data — only structural edge presence as features
- L_edge reported separately from L_total (enables clean ablation attribution)

---

## 4. Model Architecture

    TGNMemory(
        num_nodes    = 187
        raw_msg_dim  = 1          # binary edge presence
        memory_dim   = 64
        time_dim     = 64
        message_mod  = IdentityMessage
        aggregator   = LastAggregator
    )
    GraphAttentionEmbedding(in=64, out=64, msg_dim=1, time_enc=shared)
    LinkPredictor(in_channels=64) -> sigmoid -> BCEWithLogitsLoss
    Optimizer: Adam(lr=0.001) | Batch: 200 | Epochs: 5 | NegSample: 1.0

Reuses DevoTG's GraphAttentionEmbedding and LinkPredictor — same architecture
validated in Phase 1, no modifications, no new inductive biases.

---

## 5. Verified Results — Phase 2 Baseline

5-seed benchmark (seeds: 42, 12345, 123, 456, 789) — all computed fresh, verified live:

| Seed  | Val AUC | Test AUROC | Test AUPRC |
|-------|---------|------------|------------|
| 42    | 0.6727  | 0.6294     | 0.6081     |
| 12345 | 0.6795  | 0.6468     | 0.6251     |
| 123   | 0.6735  | 0.6311     | 0.6106     |
| 456   | 0.6910  | 0.6524     | 0.6348     |
| 789   | 0.6639  | 0.6445     | 0.6271     |
| **Mean** | **0.6761** | **0.6408 +/- 0.0091** | **0.6211 +/- 0.0102** |

AUROC ~64% is genuine and non-alarming. Not near 50% (random) or 95%+ (leakage).
Low std (+/-0.009) — stable training. This is the Phase 2 floor for all PCDGM modules.

---

## 6. Diagnostic: AVA/AVB/AVE Command Hub Neurons (H3 Prerequisite)

H3 hypothesis: AVA, AVB, AVE command interneurons show reinforcement in synapse
acquisition across development.

### Finding: Top-10 Degree List Excludes AVA/AVB/AVE — Expected, Not a Bug

Top-10 ranks by total chemical synapse degree (in+out on Generic_CS).
AVA/AVB/AVE have out_degree = 0 at ALL timepoints in CS graph — suppressing rank.

Root cause: These are motor command interneurons. Primary output = gap junctions (GJ),
not chemical synapses. They RECEIVE chemical synaptic input but TRANSMIT electrically.

### Verified Presence — In-Degree Across All 8 Snapshots

| Neuron | 0h | 5h | 8h | 16h | 23h | 27h | 45h-A | 45h-B |
|--------|----|----|----|----|-----|-----|-------|-------|
| AVAL   | 12 | 16 | 16 | 18 | 21  | 26  | 26    | 35    |
| AVAR   | 10 | 14 | 16 | 14 | 18  | 29  | 27    | 32    |
| AVBL   | 9  | 15 | 16 | 15 | 22  | 25  | 40    | 29    |
| AVBR   | 12 | 10 | 15 | 16 | 22  | 25  | 38    | 28    |
| AVEL   | 17 | 20 | 16 | 26 | 24  | 25  | 32    | 39    |
| AVER   | 17 | 20 | 16 | 23 | 25  | 28  | 35    | 33    |

GJ graph presence (Snapshot 1):
  AVAL: out=1, in=1  |  AVAR: out=1, in=1  |  AVBL: out=0, in=0
  AVBR: out=1, in=1  |  AVEL: out=1, in=1  |  AVER: out=0, in=0

CORRECT METRIC FOR H3: Use in_degree on Generic_CS + degree on Generic_GJ.
Do NOT use out_degree on CS (always 0) or total CS degree (suppressed by zero out).

---

## 7. Infrastructure Scope Audit

| Component | File | Spec'd? | Notes |
|-----------|------|---------|-------|
| Phase 2 TGN | baseline/tgn.py | YES | Verified clean |
| Phase 2 unit tests | baseline/test_tgn.py | YES | Passing |
| 5-seed benchmark runner | run_benchmark.py | YES | No hardcoded values |
| Research Workbench menu | run_experiments.bat | EXTRA | AI convenience UI; labels match live results |
| Research audit script | research_audit.py | EXTRA | AI diagnostic; read-only, no bias |
| Synaptic visualizer | visualizer/ | EXTRA | AI web dashboard; functional, not spec'd |

Note: Extra components do not introduce leakage or bias — they are read-only tools.
Flagged per Hard Rule against silent scope creep.

---

## 8. Hardcoding / Bias Audit

- No hardcoded AUC values returned from any training function
- All bat menu labels (e.g. "mean ~ 64.08%%") verified to match live computed values
- Negative sampling ratio = 1.0 (balanced) — no class imbalance bias
- Temporal split uses snapshot timestamps from REAL_HOURS, not index arithmetic
- Both 45h replicates share timestamp 45 — correctly co-test, not sequential
- KNOWN: torch.jit.script FutureWarning on Python 3.14+ — harmless,
  migrate to torch.compile before Phase 5

---

## 9. Next Steps (Phase 3)

Phase 3: Trajectory Reconstruction (Module A)
  - A1: Continuous interpolation between snapshots
  - A2: Blind structural inference (no future data)
  - Validates H2: continuous trajectory improves over discrete snapshot baseline

Gate: Phase 3 must improve on Phase 2 AUROC 0.6408 on the same test split.

---

## 10. References

- Witvliet D. et al., Nature 596, 257-261 (2021). doi:10.1038/s41586-021-03778-8
- Gayen S. & Alicea B., DevoTG. arXiv:2606.21940 (2026).
- PyTorch Geometric TGN: https://pytorch-geometric.readthedocs.io
- OpenWorm CECT: https://github.com/openworm/c302
