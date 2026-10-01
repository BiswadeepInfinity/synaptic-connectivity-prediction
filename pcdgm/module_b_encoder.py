"""
pcdgm/module_b_encoder.py — Phase 5: Structural Encoder.

Input:  node features (degree, clustering coefficient, k-hop neighbourhood).
        No activity features — none exist and none are used.
Output: H_s  Tensor[N, 64]

Architecture:
  2-layer GATConv, 2 heads, hidden_dim=100 (matches DevoTG for comparability),
  output 64-dim. LayerNorm (NOT BatchNorm — unstable at N~20–225).

Hard rules:
  - NO BatchNorm anywhere. LayerNorm only.
  - No activity / calcium / electrophysiology features.
  - Hidden dim 100, output 64 — matches DevoTG scale.

TODO (Phase 5): implement StructuralEncoder.
"""

import torch.nn as nn


class StructuralEncoder(nn.Module):
    """
    2-layer GATConv structural encoder.
    hidden_dim=100 (DevoTG-matched), output=64.
    Uses LayerNorm — NOT BatchNorm (THE_BOOK Hard Rule, anomaly table).

    TODO (Phase 5): implement forward().
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # DESIGN CHOICE (not in spec): hidden_dim param allows test-time
        # override; production always uses 64 as per THE_BOOK §Module B.
        self.hidden_dim = hidden_dim
        raise NotImplementedError("Module B (Phase 5) not yet implemented.")

    def forward(self, adjacency):
        raise NotImplementedError("Module B (Phase 5) not yet implemented.")
