"""
pcdgm/module_b_encoder.py — Phase 5: Structural Encoder.

Input:  Purely topological node features:
          1. Normalized in-degree & out-degree (d / (N-1))
          2. Local clustering coefficient
          3. 2-hop neighborhood expansion ratio
        CRITICAL: All features are strictly normalized to eliminate raw scalar edge density/count
        bias, forcing the encoder to represent local network geometry and motifs rather than density.
        No activity / calcium / electrophysiology features — none exist and none are used.
Output: H_s  Tensor[N, 64]

Architecture:
  2-layer GATConv, 2 heads, hidden_dim=100 (matches DevoTG for comparability),
  output 64-dim. LayerNorm (NOT BatchNorm — unstable at N~20–225).

Hard rules:
  - NO BatchNorm anywhere. LayerNorm only.
  - No activity / calcium / electrophysiology features.
  - Hidden dim 100, output 64 — matches DevoTG scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def extract_structural_features(adj: torch.Tensor) -> torch.Tensor:
    """
    Computes purely topological node features invariant to global edge scaling:
      - in_degree / (N - 1)
      - out_degree / (N - 1)
      - local clustering coefficient
      - 2-hop reachability fraction
    Returns: Tensor[N, 4]
    """
    N = adj.shape[0]
    if N <= 1:
        return torch.zeros((N, 4), dtype=torch.float, device=adj.device)

    # 1. Degree features normalized by max possible degree (N-1)
    out_deg = adj.sum(dim=-1).float() / (N - 1)
    in_deg = adj.sum(dim=-2).float() / (N - 1)

    # 2. Local clustering coefficient
    # For directed graphs: triangles / (d_tot * (d_tot - 1) - 2 * d_recip)
    # Fast vectorized approximation using adj^3 diagonal:
    adj_bin = (adj > 0).float()
    adj_2 = torch.matmul(adj_bin, adj_bin)
    triangles = torch.diagonal(torch.matmul(adj_2, adj_bin))
    total_deg = adj_bin.sum(dim=-1) + adj_bin.sum(dim=-2)
    possible_triangles = total_deg * (total_deg - 1.0)
    clustering = torch.where(possible_triangles > 0, (2.0 * triangles) / possible_triangles, torch.zeros_like(triangles))

    # 3. 2-hop reachability fraction
    two_hop = (adj_2 > 0).float().sum(dim=-1) / (N - 1)

    features = torch.stack([in_deg, out_deg, clustering, two_hop], dim=-1)
    return features


class StructuralEncoder(nn.Module):
    """
    2-layer GATConv structural encoder.
    in_dim=4 -> hidden_dim=100 (DevoTG-matched, 2 heads x 50) -> output=64 (1 head x 64).
    Uses LayerNorm — NOT BatchNorm (THE_BOOK Hard Rule, anomaly table).
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 100, out_dim: int = 64, heads: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads

        # Layer 1: GATConv mapping in_dim -> hidden_dim (split over heads)
        per_head_dim = hidden_dim // heads
        self.conv1 = GATConv(in_channels=in_dim, out_channels=per_head_dim, heads=heads, concat=True, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Layer 2: GATConv mapping hidden_dim -> out_dim
        self.conv2 = GATConv(in_channels=hidden_dim, out_channels=out_dim, heads=1, concat=False, add_self_loops=True)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adjacency: Tensor[N, N] adjacency matrix
        Returns:
            H_s: Tensor[N, 64] structural node embeddings
        """
        device = adjacency.device
        N = adjacency.shape[0]

        # Extract density-normalized topological features
        x = extract_structural_features(adjacency).to(device)

        # Extract edge index
        nonzero = torch.nonzero(adjacency > 0)
        if nonzero.numel() > 0:
            edge_index = nonzero.t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        # Forward GATConv Layer 1
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.elu(h)

        # Forward GATConv Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)

        return h
