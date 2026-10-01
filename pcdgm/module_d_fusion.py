"""
pcdgm/module_d_fusion.py — Phase 5: Graph Fusion.

Fuses the structural node embeddings H_s with the motif representation z_s.

Equation (THE_BOOK §Module D):
  H_s' = GATConv2( concat(H_s, broadcast(z_s)), edge_index=E_hat_s )   # [N, 100]

Critical: z_s is graph-level ([64]) and must be broadcast to node-level
  ([N, 64]) before concatenation. Shape must be asserted every forward pass
  to catch the "one pattern per graph vs. one per node" confusion
  (THE_BOOK anomaly: broadcast confusion → test_z_per_node_shape_correct).

TODO (Phase 5): implement GraphFusion.
"""

import torch
import torch.nn as nn


class GraphFusion(nn.Module):
    """
    GATConv fusion of node embeddings and motif representation.
    Output: H_s' Tensor[N, out_dim=100]

    TODO (Phase 5): implement forward().
    """

    def __init__(self, out_dim: int = 100):
        super().__init__()
        self.out_dim = out_dim
        raise NotImplementedError("Module D (Phase 5) not yet implemented.")

    def forward(
        self,
        H:           torch.Tensor,   # [N, 64] node embeddings
        z_per_node:  torch.Tensor,   # [N, 64] motif rep broadcast to nodes
        adjacency:   torch.Tensor,   # [N, N]  current graph
    ) -> torch.Tensor:               # [N, 100]
        # MANDATORY assertion (THE_BOOK §Module D):
        assert z_per_node.shape[0] == H.shape[0], (
            f"Broadcast confusion: z_per_node has {z_per_node.shape[0]} rows "
            f"but H has {H.shape[0]} nodes. z_s must be broadcast to N rows "
            f"before fusion."
        )
        raise NotImplementedError("Module D (Phase 5) not yet implemented.")
