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

    Formula:
      H_s' = GATConv2( concat(H_s, broadcast(z_s)), edge_index=E_hat_s )

    Ablation support:
      - If z_per_node is None or use_motif=False (or mode="no_motif"),
        the motif vector is omitted/zeroed and the layer adapts to [N, node_dim].
    """

    def __init__(
        self,
        node_dim: int = 64,
        motif_dim: int = 64,
        out_dim: int = 100,
        heads: int = 2,
        use_motif: bool = True,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.motif_dim = motif_dim if use_motif else 0
        self.out_dim = out_dim
        self.heads = heads
        self.use_motif = use_motif

        in_channels = node_dim + (motif_dim if use_motif else 0)
        per_head_dim = out_dim // heads

        from torch_geometric.nn import GATConv
        self.conv = GATConv(
            in_channels=in_channels,
            out_channels=per_head_dim,
            heads=heads,
            concat=True,
            add_self_loops=True,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        H: torch.Tensor,                  # [N, 64] node embeddings
        z_per_node: torch.Tensor = None,  # [N, 64] motif rep broadcast to nodes
        adjacency: torch.Tensor = None,   # [N, N] current graph adjacency
    ) -> torch.Tensor:                    # [N, 100]
        N = H.shape[0]
        device = H.device

        if self.use_motif and z_per_node is not None:
            # MANDATORY assertion (THE_BOOK §Module D):
            assert z_per_node.shape[0] == N, (
                f"Broadcast confusion: z_per_node has {z_per_node.shape[0]} rows "
                f"but H has {N} nodes. z_s must be broadcast to N rows "
                f"before fusion."
            )
            assert z_per_node.dim() == 2, f"z_per_node must be 2D [N, motif_dim], got {z_per_node.shape}"
            fused_input = torch.cat([H, z_per_node], dim=-1)
        else:
            if self.use_motif and z_per_node is None:
                # Fallback zero-pad if initialized with motif but none provided
                zeros_motif = torch.zeros(N, self.motif_dim, device=device, dtype=H.dtype)
                fused_input = torch.cat([H, zeros_motif], dim=-1)
            else:
                fused_input = H

        # Extract edge_index from adjacency
        if adjacency is not None:
            nonzero = torch.nonzero(adjacency > 0)
            if nonzero.numel() > 0:
                edge_index = nonzero.t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        h_out = self.conv(fused_input, edge_index)
        h_out = self.norm(h_out)
        h_out = nn.functional.elu(h_out)
        return h_out

