"""
pcdgm/module_e_predictor.py — Phase 5: Which + When Predictor.

Equations (THE_BOOK §Module E):
  which:
    p_ij = sigmoid( W3 · ReLU( W1·h_i' + W2·h_j' + W4·z_s ) )

  when:
    t_ij = s + Δ_ij
    Δ_ij ~ learned stochastic delay distribution conditioned on (h_i', h_j', z_s)

Resolution honesty rule (THE_BOOK §Module E):
  Ground truth is 8 discrete snapshot intervals — NOT continuous hours.
  Primary timing metric: correct_snapshot_interval (yes/no).
  Fine-grained hour MAE is ILLUSTRATIVE ONLY and must be explicitly caveated.
  when_interval() returns an int in range(8), never a float hour.

TODO (Phase 5): implement ChangePredictor.
"""

import torch
import torch.nn as nn
from typing import List


class ChangePredictor(nn.Module):
    """
    Predicts which edges change and which snapshot interval they change in.

    Exposes:
      which(H, H, z)                    → Tensor[N, N] probabilities
      when_interval(H, H, z, real_hours) → int in range(8)

    TODO (Phase 5): implement.
    """

    def __init__(
        self,
        dim: int = 100,
        motif_dim: int = 64,
        hidden_dim: int = 64,
        use_motif: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.motif_dim = motif_dim if use_motif else 0
        self.hidden_dim = hidden_dim
        self.use_motif = use_motif

        # Which predictor: p_ij = sigmoid( W3 · ReLU( W1·h_i' + W2·h_j' + W4·z_s ) )
        self.W1 = nn.Linear(dim, hidden_dim)
        self.W2 = nn.Linear(dim, hidden_dim)
        if self.use_motif:
            self.W4 = nn.Linear(motif_dim, hidden_dim)
        else:
            self.W4 = None
        self.W3 = nn.Linear(hidden_dim, 1)

        # When predictor: distribution over intervals
        self.when_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # 8 discrete snapshot intervals
        )

    def which(
        self,
        H_i: torch.Tensor,                # [N, dim] — source node embeddings
        H_j: torch.Tensor,                # [N, dim] — target node embeddings
        z_s: torch.Tensor = None,         # [N, motif_dim] or [motif_dim] motif rep
    ) -> torch.Tensor:                    # [N, N]   — change probabilities
        # h1: [N, 1, hidden_dim], h2: [1, N, hidden_dim]
        h1 = self.W1(H_i).unsqueeze(1)  # [N, 1, hidden_dim]
        h2 = self.W2(H_j).unsqueeze(0)  # [1, N, hidden_dim]
        h_comb = h1 + h2                # [N, N, hidden_dim]

        if self.use_motif and z_s is not None and self.W4 is not None:
            if z_s.dim() == 1:
                # Graph-level [motif_dim] broadcast across all pairs
                z_term = self.W4(z_s).view(1, 1, self.hidden_dim)
            elif z_s.dim() == 2:
                # Node-level [N, motif_dim] broadcast over source node
                z_term = self.W4(z_s).unsqueeze(1)  # [N, 1, hidden_dim]
            else:
                z_term = self.W4(z_s)
            h_comb = h_comb + z_term

        logits = self.W3(torch.relu(h_comb)).squeeze(-1)  # [N, N]
        return torch.sigmoid(logits)

    def when_interval(
        self,
        H_i:        torch.Tensor,
        H_j:        torch.Tensor,
        z_s:        torch.Tensor = None,
        real_hours: List[int] = None,
    ) -> int:
        """
        Return predicted snapshot interval index (int in range(len(real_hours))).
        NEVER return a raw float hour — resolution honesty rule.
        """
        num_intervals = len(real_hours) if real_hours is not None else 8
        h1 = self.W1(H_i).mean(dim=0, keepdim=True)
        h2 = self.W2(H_j).mean(dim=0, keepdim=True)
        h_pool = h1 + h2

        if self.use_motif and z_s is not None and self.W4 is not None:
            if z_s.dim() == 1:
                z_term = self.W4(z_s).unsqueeze(0)
            else:
                z_term = self.W4(z_s).mean(dim=0, keepdim=True)
            h_pool = h_pool + z_term

        logits = self.when_head(torch.relu(h_pool))
        logits = logits[:, :num_intervals]
        interval_idx = int(torch.argmax(logits, dim=-1).item())
        return interval_idx

