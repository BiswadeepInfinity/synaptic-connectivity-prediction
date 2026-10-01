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

    def __init__(self, dim: int = 100):
        super().__init__()
        self.dim = dim
        raise NotImplementedError("Module E (Phase 5) not yet implemented.")

    def which(
        self,
        H_i: torch.Tensor,   # [N, dim] — source node embeddings
        H_j: torch.Tensor,   # [N, dim] — target node embeddings
        z_s: torch.Tensor,   # [N, 64]  — motif representation (per node)
    ) -> torch.Tensor:       # [N, N]   — change probabilities
        raise NotImplementedError("Module E (Phase 5) not yet implemented.")

    def when_interval(
        self,
        H_i:        torch.Tensor,
        H_j:        torch.Tensor,
        z_s:        torch.Tensor,
        real_hours: List[int],
    ) -> int:
        """
        Return predicted snapshot interval index (int in range(len(real_hours))).
        NEVER return a raw float hour — resolution honesty rule.
        """
        raise NotImplementedError("Module E (Phase 5) not yet implemented.")
