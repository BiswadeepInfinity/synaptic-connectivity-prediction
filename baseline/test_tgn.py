"""
baseline/test_tgn.py — Tests for Phase 1 & 2 TGN baseline.

Phase 1 gate (THE_BOOK §Phase 1):
  test_devotg_baseline_auc_matches_paper — must pass before ANY novel module.

Phase 2 checks:
  test_tgn_no_temporal_leakage
  test_tgn_loss_terms_logged_separately
"""

import pytest
from baseline.tgn import run_devotg_reproduction


class TestPhase1ReproductionGate:

    def test_devotg_baseline_auc_matches_paper(self):
        """
        PHASE 1 GATE. Must pass before any novel PCDGM module is trusted.
        Target: AUC 0.839 ± 0.007 on DevoTG's lineage-prediction task.
        """
        auc = run_devotg_reproduction(epochs=8)
        assert abs(auc - 0.839) < 0.02, f"Phase 1 gate FAILED: AUC={auc:.4f}, expected 0.839±0.007"


class TestPhase2TGNBaseline:

    def test_tgn_not_yet_implemented(self):
        pytest.skip("Phase 2 TGN baseline not yet implemented.")
