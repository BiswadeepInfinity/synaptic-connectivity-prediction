"""
pcdgm/module_c_motifs.py — Phase 4: Structural Motif Discovery.

Core novel-application component. Learns a dictionary D ∈ R^{K×64} of
recurring structural patterns across the developmental trajectory.

Key equations (THE_BOOK §Module C):
  pool_s   = mean_pool(H_{s-k:s})                        # [64]
  a_s      = softmax( (pool_s · D^T) / sqrt(64) )        # [K]
  z_s      = a_s · D                                     # [64]
  L_pattern  = MSE( decoder(z_s), pool_s )               # full window
  L_diversity = -sum_k( H( mean_s(a_s)_k ) )            # entropy bonus

Parameters:
  K      : dictionary slots — sweep {5, 10, 20, 50}. Default 50 to start.
  window : k=3 (NOT 10 — only 8 total snapshots).
  mode   : "learned" (default) or "random_untrained" (ablation variant).

Hard rules:
  - a_s MUST be a real, retrievable output — never compute and discard.
  - Reconstruction target = full window, NOT last-step-only.
  - Dictionary init is random in primary run; data-derived init is an
    explicit ablation variant only.
  - K sweep is REQUIRED, not optional (collapse risk with n=8 snapshots).
  - window=3, not 10.

Phase 4 gate: synthetic injected-motif recovery must pass before real data.

TODO (Phase 4): implement MotifDictionary.
"""

import torch
import torch.nn as nn


class MotifDictionary(nn.Module):
    """
    Learned motif dictionary with soft attention over K slots.

    Exposes:
      forward(window_encodings) → (z_s, a_s, D)
      slot_usage_entropy(a_s)   → scalar entropy (collapse detector)
      reconstruct(window)       → full-window reconstruction for L_pattern

    TODO (Phase 4): implement all methods.
    """

    def __init__(
        self,
        K:      int = 50,
        dim:    int = 64,
        window: int = 3,
        mode:   str = "learned",   # "learned" | "random_untrained"
    ):
        super().__init__()
        self.K      = K
        self.dim    = dim
        self.window = window
        self.mode   = mode
        # DESIGN CHOICE: D stored as nn.Parameter so it's included in
        # model.parameters() and saved in checkpoints automatically.
        # random_untrained mode freezes D after random init.
        self.D = nn.Parameter(torch.randn(K, dim))
        raise NotImplementedError("Module C (Phase 4) not yet implemented.")

    def forward(self, window_encodings):
        """
        Args:
            window_encodings: list of k adjacency tensors or encoded tensors.
        Returns:
            z_s : Tensor[64]   — motif-weighted representation
            a_s : Tensor[K]    — attention weights (MUST be returned)
            D   : Tensor[K,64] — current dictionary
        """
        raise NotImplementedError("Module C (Phase 4) not yet implemented.")

    def slot_usage_entropy(self, a_s: torch.Tensor) -> float:
        """
        Compute per-slot usage entropy over a batch of attention weights.
        Low entropy → dictionary collapse.
        Returns a scalar float.

        Args:
            a_s: Tensor[batch, K] — attention weights across multiple steps.
        """
        raise NotImplementedError("Module C (Phase 4) not yet implemented.")

    def reconstruct(self, window: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the full window (not last-step-only) from z_s.
        Used for L_pattern loss.

        Args:
            window: Tensor[k, N, dim]
        Returns:
            recon: Tensor[k, N, dim]
        """
        raise NotImplementedError("Module C (Phase 4) not yet implemented.")
