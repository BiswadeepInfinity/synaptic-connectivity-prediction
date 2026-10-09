"""
pcdgm/module_c_motifs.py — Phase 4: Structural Motif Discovery.

Core novel-application component. Learns a dictionary D in R^{K x 64} of
recurring structural patterns across the developmental trajectory.

Key equations (THE_BOOK Section Module C):
  pool_s      = mean_pool(H_{s-k:s})                         # [dim] or [B, dim]
  a_s         = softmax( (pool_s . D^T) / sqrt(dim) )        # [K] or [B, K]
  z_s         = a_s . D                                      # [dim] or [B, dim]
  L_pattern   = MSE( decoder(z_s), pool_s )                  # full window
  L_diversity = -sum_k( H( mean_s(a_s)_k ) )                 # entropy bonus

Parameters:
  K      : dictionary slots -- sweep {5, 10, 20, 50}. Default 50 to start.
  window : k=3 (NOT 10 -- only 8 total snapshots).
  mode   : "learned" (default) or "random_untrained" (ablation variant).

Hard rules:
  - a_s MUST be a real, retrievable output -- never compute and discard.
  - Reconstruction target = full window, NOT last-step-only.
  - Dictionary init is random in primary run; data-derived init is an
    explicit ablation variant only.
  - K sweep is REQUIRED, not optional (collapse risk with n=8 snapshots).
  - window=3, not 10.

Phase 4 gate: synthetic injected-motif recovery must pass before real data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifDictionary(nn.Module):
    """
    Learned motif dictionary with soft attention over K slots.

    Key equations:
      pool_s      = mean_pool(H_{s-k:s})                         # [dim] or [B, dim]
      a_s         = softmax( (pool_s . D^T) / sqrt(dim) )        # [K] or [B, K]
      z_s         = a_s . D                                      # [dim] or [B, dim]
      L_pattern   = MSE( decoder(z_s), pool_s )                  # full window target
      L_diversity = -sum_k( H( mean_s(a_s)_k ) )                 # entropy bonus

    Exposes:
      forward(window_encodings) -> (z_s, a_s, D)
      slot_usage_entropy(a_s)   -> scalar entropy (collapse detector)
      reconstruct(window)       -> full-window reconstruction for L_pattern
    """

    def __init__(
        self,
        K: int = 50,
        dim: int = 64,
        window: int = 3,
        mode: str = "learned",  # "learned" | "random_untrained"
    ):
        super().__init__()
        self.K = K
        self.dim = dim
        self.window = window
        self.mode = mode

        # Dictionary matrix D: [K, dim]
        # In learned mode, D is optimized via gradients.
        # In random_untrained mode, D is frozen (requires_grad=False).
        if mode == "learned":
            self.D = nn.Parameter(torch.randn(K, dim) / math.sqrt(dim))
        elif mode == "random_untrained":
            self.D = nn.Parameter(torch.randn(K, dim) / math.sqrt(dim), requires_grad=False)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'learned' or 'random_untrained'.")

        # Lightweight MLP decoder mapping motif representation z_s to full window / pool representation
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def _pool_window(self, window_encodings) -> torch.Tensor:
        """
        Pool a sequence or window of representations into a summary vector of dimension [dim].
        Topological normalization is applied so that raw edge density/scale is removed,
        forcing the model to discover structural connectivity patterns rather than edge counts.
        Supports:
          - Tensor of shape [k, N, dim] or [k, dim] or [B, k, N, dim]
          - List/tuple of k Tensors of shape [N, dim], [N, N] (adjacency), or [dim]
        """
        if isinstance(window_encodings, (list, tuple)):
            processed = []
            for item in window_encodings:
                if item.dim() == 2:
                    if item.shape[-1] == self.dim:
                        # [N, dim] node embeddings -> mean pool over nodes -> [dim]
                        node_pool = item.mean(dim=0)
                        processed.append(node_pool)
                    elif item.shape[0] == item.shape[1]:
                        # Adjacency matrix [N, N] -> normalized degree distribution
                        # Dividing by sum(degrees) removes raw global density/edge count.
                        # Multiplying by N scales the relative distribution so mean = 1.0,
                        # preserving structural variations (hubs, cliques) without density bias.
                        N_nodes = item.shape[0]
                        raw_deg = item.sum(dim=-1).float()  # [N]
                        deg_sum = raw_deg.sum()
                        if deg_sum > 0:
                            norm_deg = (raw_deg / deg_sum) * N_nodes
                        else:
                            norm_deg = raw_deg
                        # Standardize/scale to standard feature dimension
                        if norm_deg.shape[0] == self.dim:
                            deg_feat = norm_deg
                        elif norm_deg.shape[0] < self.dim:
                            deg_feat = F.pad(norm_deg, (0, self.dim - norm_deg.shape[0]))
                        else:
                            deg_feat = norm_deg[:self.dim]
                        processed.append(deg_feat)
                    else:
                        processed.append(item.mean(dim=0))
                elif item.dim() == 1:
                    if item.shape[0] == self.dim:
                        processed.append(item.float())
                    else:
                        padded = F.pad(item.float(), (0, max(0, self.dim - item.shape[0])))[:self.dim]
                        processed.append(padded)
                else:
                    processed.append(item.mean(dim=tuple(range(item.dim() - 1))))
            stacked = torch.stack(processed, dim=0)  # [k, dim]
            return stacked.mean(dim=0)               # [dim]

        elif isinstance(window_encodings, torch.Tensor):
            if window_encodings.dim() == 3:
                # [k, N, dim] -> mean pool over window k and nodes N -> [dim]
                return window_encodings.mean(dim=(0, 1))
            elif window_encodings.dim() == 2:
                # [k, dim] -> mean pool over window k -> [dim]
                return window_encodings.mean(dim=0)
            elif window_encodings.dim() == 4:
                # [B, k, N, dim] -> [B, dim]
                return window_encodings.mean(dim=(1, 2))
            elif window_encodings.dim() == 1:
                return window_encodings.float()
            else:
                return window_encodings.mean(dim=tuple(range(window_encodings.dim() - 1)))
        else:
            raise TypeError(f"Unsupported window_encodings type: {type(window_encodings)}")

    def forward(self, window_encodings):
        """
        Args:
            window_encodings: list of k adjacency/embedding tensors, or Tensor [k, N, dim] / [k, dim].
        Returns:
            z_s : Tensor[dim] or [B, dim]  -- motif-weighted representation
            a_s : Tensor[K] or [B, K]      -- attention weights (MUST be returned, never discarded)
            D   : Tensor[K, dim]           -- current dictionary
        """
        pool_s = self._pool_window(window_encodings)  # [dim] or [B, dim] (already normalized)

        # Scaled dot-product attention against dictionary slots:
        # pool_s: [dim] or [B, dim]
        # D: [K, dim]
        scaling = math.sqrt(self.dim)
        if pool_s.dim() == 1:
            logits = torch.matmul(self.D, pool_s) / scaling  # [K]
            a_s = F.softmax(logits, dim=-1)                   # [K]
            z_s = torch.matmul(a_s, self.D)                  # [dim]
        else:
            logits = torch.matmul(pool_s, self.D.t()) / scaling  # [B, K]
            a_s = F.softmax(logits, dim=-1)                       # [B, K]
            z_s = torch.matmul(a_s, self.D)                      # [B, dim]

        return z_s, a_s, self.D

    def slot_usage_entropy(self, a_s: torch.Tensor, eps: float = 1e-9) -> float:
        """
        Compute per-slot usage entropy over a batch of attention weights.
        Low entropy (< 0.1) -> dictionary collapse.
        Uniform attention -> high entropy (ln(K)).

        Args:
            a_s: Tensor[batch, K] or [K] -- attention weights across multiple steps.
        Returns:
            Scalar float entropy.
        """
        if a_s.dim() == 1:
            p = a_s
        else:
            p = a_s.mean(dim=0)  # average usage per slot across batch/time

        p = p / (p.sum() + eps)
        p = torch.clamp(p, min=eps)
        entropy = -torch.sum(p * torch.log(p))
        return float(entropy.item())

    def reconstruct(self, window: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the full window (not last-step-only) from motif summary z_s.
        Used for L_pattern loss: MSE(recon, window).

        Args:
            window: Tensor[k, N, dim] or [k, dim]
        Returns:
            recon: Tensor of same shape as window.
        """
        orig_shape = window.shape
        pool_s = self._pool_window(window)  # [dim]
        z_s, _, _ = self.forward(pool_s)    # [dim]
        decoded = self.decoder(z_s)         # [dim]

        if window.dim() == 3:
            # window is [k, N, dim]
            recon = decoded.unsqueeze(0).unsqueeze(0).expand(orig_shape).clone()
        elif window.dim() == 2:
            # window is [k, dim]
            recon = decoded.unsqueeze(0).expand(orig_shape).clone()
        else:
            recon = decoded.reshape(orig_shape).clone()

        return recon
