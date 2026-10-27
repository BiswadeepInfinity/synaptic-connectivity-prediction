"""
pcdgm/test_module_d_fusion.py — Phase 5 tests for Module D.

Tests (THE_BOOK Part V §Module D):
  - z_per_node broadcast shape is correct [N, 64]
  - output shape [N, 100]
"""

import pytest
import torch
from pcdgm.module_d_fusion import GraphFusion


class TestGraphFusion:

    def test_z_per_node_shape_correct(self, synthetic_snapshots):
        """
        z_s must be broadcast to [N, 64] before fusion.
        Passing a [64] graph-level vector directly is a hard error
        (THE_BOOK anomaly: broadcast confusion).
        """
        fusion = GraphFusion(out_dim=100)
        H = torch.randn(20, 64)
        z_per_node = torch.randn(20, 64)   # already broadcast to N rows
        out = fusion(H, z_per_node, synthetic_snapshots[0])
        assert out.shape == (20, 100), f"Expected [20, 100], got {out.shape}"

    def test_broadcast_assertion_fires_on_wrong_shape(self, synthetic_snapshots):
        """
        GraphFusion must assert z_per_node.shape[0] == N and raise if not.
        This assertion is MANDATORY per THE_BOOK §Module D.
        """
        fusion = GraphFusion(out_dim=100)
        H = torch.randn(20, 64)
        z_wrong = torch.randn(1, 64)  # graph-level, not broadcast
        with pytest.raises((AssertionError, RuntimeError)):
            fusion(H, z_wrong, synthetic_snapshots[0])

