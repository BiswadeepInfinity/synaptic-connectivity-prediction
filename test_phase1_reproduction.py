"""
test_phase1_reproduction.py — Phase 1 gate (root-level).

Must pass before ANY novel PCDGM module is trusted.
Target: DevoTG lineage-prediction AUC 0.839 ± 0.007.

Paper: Gayen & Alicea, arXiv:2606.21940
Protocol: Temporal Graph Network on C. elegans lineage division CTDG across 5 seeds.
"""

import pytest
from baseline.tgn import run_devotg_reproduction


def test_devotg_baseline_auc_matches_paper():
    """
    PHASE 1 GATE. Must pass before any Part III module is trusted,
    regardless of how well its own synthetic tests pass.
    Target: AUC 0.839 ± 0.007 on DevoTG's lineage-prediction task.
    """
    reported_auc = run_devotg_reproduction(epochs=8)
    print(f"\nReported DevoTG Reproduction Mean AUC (5 seeds): {reported_auc:.4f}")

    assert abs(reported_auc - 0.839) < 0.02, (
        f"Phase 1 GATE FAILED: AUC={reported_auc:.4f}. "
        f"Expected 0.839 ± 0.007. Diagnose root cause — do NOT tune to match."
    )
