"""
pcdgm/test_module_b_encoder.py — Phase 5 tests for Module B.

Tests (THE_BOOK Part V §Module B):
  - output shape [N, 64]
  - no BatchNorm used anywhere
  - isolated node (degree=0) produces no NaN
"""

import pytest
import torch
from pcdgm.module_b_encoder import StructuralEncoder


class TestModuleB:

    def test_output_shape(self, synthetic_snapshots):
        pytest.skip("Phase 5 not yet implemented.")
        enc = StructuralEncoder(hidden_dim=64)
        H = enc(synthetic_snapshots[0])
        assert H.shape == (20, 64), f"Expected [20, 64], got {H.shape}"

    def test_no_batchnorm_used(self):
        """
        BatchNorm is unstable at small N (20–225 nodes per graph).
        LayerNorm must be used instead. THE_BOOK anomaly table.
        """
        pytest.skip("Phase 5 not yet implemented.")
        enc = StructuralEncoder(hidden_dim=64)
        for name, m in enc.named_modules():
            assert not isinstance(m, torch.nn.BatchNorm1d), (
                f"BatchNorm found at '{name}'. Use LayerNorm instead."
            )
            assert not isinstance(m, torch.nn.BatchNorm2d), (
                f"BatchNorm2d found at '{name}'. Use LayerNorm instead."
            )

    def test_isolated_node_no_nan(self):
        """
        Fully isolated graph (no edges) must not produce NaN.
        GAT with zero-degree nodes can produce NaN if softmax over empty set.
        """
        pytest.skip("Phase 5 not yet implemented.")
        isolated_graph = torch.zeros(20, 20)
        enc = StructuralEncoder(hidden_dim=64)
        H = enc(isolated_graph)
        assert not torch.isnan(H).any(), (
            "NaN in encoder output for isolated graph — check GAT attention "
            "softmax handling for zero-degree nodes."
        )

    def test_no_activity_features(self):
        """
        Encoder must only use structural features (degree, clustering coeff,
        k-hop neighbourhood). No calcium / electrophysiology features.
        This is a code-review check, not auto-verifiable — left as reminder.
        """
        pytest.skip("Phase 5 not yet implemented — review manually at implementation.")
