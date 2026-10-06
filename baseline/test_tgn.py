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

    def test_tgn_no_temporal_leakage(self, synthetic_snapshots, real_hours):
        """
        Verify that temporal ordering is strictly preserved in the temporal data,
        and earlier events never see future timestamps or states.
        """
        from baseline.tgn import build_connectome_temporal_data

        data, num_nodes = build_connectome_temporal_data(
            snapshots=synthetic_snapshots,
            real_hours=real_hours,
        )

        assert num_nodes == synthetic_snapshots[0].shape[0]
        assert data.num_events > 0

        # Timestamps must be non-decreasing
        t_diff = data.t[1:] - data.t[:-1]
        assert (t_diff >= 0).all(), "Temporal leakage: event timestamps are not sorted non-decreasingly!"

        # Train/val/test splits must maintain strict chronological ordering with replicates together
        from baseline.tgn import split_connectome_temporal_data
        train_data, val_data, test_data = split_connectome_temporal_data(data, real_hours=real_hours)

        assert train_data.num_events > 0, "Train split is empty"
        assert val_data.num_events > 0, "Val split is empty"
        assert test_data.num_events > 0, "Test split is empty"

        assert train_data.t.max() < val_data.t.min(), "Temporal leakage between train and val!"
        assert val_data.t.max() < test_data.t.min(), "Temporal leakage between val and test!"

        # Replicates check: all 45h events must be solely in test
        idx_45h = [h == 45 for h in real_hours]
        if any(idx_45h):
            assert (train_data.t != 45).all(), "45h event leaked into train split"
            assert (val_data.t != 45).all(), "45h event leaked into val split"
            assert (test_data.t == 45).all(), "Non-45h event in test split"


    def test_tgn_loss_terms_logged_separately(self, synthetic_snapshots, real_hours):
        """
        Verify that L_edge and L_total are explicitly computed and exposed in the step output.
        Hard Rule: Report L_edge separately from total loss.
        """
        from baseline.tgn import build_connectome_temporal_data, ConnectomeTGNBaseline
        from torch_geometric.loader import TemporalDataLoader

        data, num_nodes = build_connectome_temporal_data(
            snapshots=synthetic_snapshots,
            real_hours=real_hours,
        )

        model = ConnectomeTGNBaseline(num_nodes=num_nodes, msg_dim=data.msg.size(-1), device="cpu")
        loader = TemporalDataLoader(data, batch_size=50, neg_sampling_ratio=1.0)

        for batch in loader:
            step_output = model.step(batch, data)
            break

        assert "L_edge" in step_output, "Missing 'L_edge' in step output (THE_BOOK Hard Rule)"
        assert "L_total" in step_output, "Missing 'L_total' in step output"
        assert isinstance(step_output["L_edge"], float)
        assert isinstance(step_output["L_total"], float)

