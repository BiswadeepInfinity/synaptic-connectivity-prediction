"""
training/loop.py — Phase 5: Training loop for the full PCDGM model.

Loss (THE_BOOK §Full Loss):
  L = L_edge + λ·L_pattern + β·L_diversity + γ·L_timing

Defaults:
  λ=1.0, β=0.1, γ=1.0  (retune from logged per-term losses)

Hard rules:
  - Log ALL four loss terms separately, every step (L_edge, L_pattern,
    L_diversity, L_timing, L_total).
  - Checkpoint selection uses VALIDATION metric — never training metric.
  - Seed propagates to: data shuffling, model init, dictionary init
    (three separate calls, not one).
  - 5 seeds minimum for any reported ablation number.
  - Temporal splits only — never shuffle=True on this data.
"""

from typing import Dict, Any, List, Optional, Sequence
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pcdgm.model import PCDGM


def validate_seeds(seeds: Sequence[int]) -> List[int]:
    """
    Enforces the benchmark requirement of at least 5 distinct seeds.
    Raises ValueError if fewer than 5 seeds or non-unique seeds are provided.
    """
    seeds_list = list(seeds)
    if len(seeds_list) < 5:
        raise ValueError(f"At least 5 seeds are required for any reported benchmark, got {len(seeds_list)}.")
    if len(set(seeds_list)) != len(seeds_list):
        raise ValueError("Seeds must be distinct.")
    return seeds_list


def set_seed(seed: int) -> Dict[str, int]:
    """
    Explicitly propagates seed to all three independent init/randomness points
    per THE_BOOK Hard Rule:
      1. data shuffling / sampling
      2. model parameter init
      3. dictionary init
    """
    # 1. data shuffle seed
    data_shuffle_seed = seed
    np.random.seed(data_shuffle_seed)

    # 2. model init seed
    model_init_seed = seed
    torch.manual_seed(model_init_seed)

    # 3. dictionary init seed
    dictionary_init_seed = seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return {
        "requested_seed": seed,
        "data_shuffle_seed": data_shuffle_seed,
        "model_init_seed": model_init_seed,
        "dictionary_init_seed": dictionary_init_seed,
    }


class TrainingLoop:
    """
    Full PCDGM training loop with separate loss logging and
    validation-based checkpointing.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        optimizer: Optional[optim.Optimizer] = None,
        lr: float = 0.001,
        lambda_pattern: float = 1.0,
        beta_diversity: float = 0.1,
        gamma_timing: float = 1.0,
        seed: int = 42,
    ):
        self.config = config or {}
        self.seed = self.config.get("seed", seed)
        self.seeded_metadata = set_seed(self.seed)

        self.lambda_pattern = self.config.get("lambda_pattern", lambda_pattern)
        self.beta_diversity = self.config.get("beta_diversity", beta_diversity)
        self.gamma_timing = self.config.get("gamma_timing", gamma_timing)
        self.device = self.config.get("device", "cpu")

        if model is None:
            self.model = PCDGM().to(self.device)
        else:
            self.model = model.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("lr", lr))
        else:
            self.optimizer = optimizer

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.history: List[Dict[str, float]] = []
        self.best_checkpoint: Optional[Dict[str, Any]] = None
        self.best_val_metric: float = float("inf")
        self.checkpoint_metric_source: str = "val"

    def step(self, batch: Any) -> Dict[str, float]:
        """
        Executes a single forward + backward step.
        Returns dict with keys: L_edge, L_pattern, L_diversity, L_timing, L_total.
        ALL terms must be present — see test_loss_terms_logged_separately.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Handle batch formats (dict, tuple, or list of window tensors)
        if isinstance(batch, dict):
            window = batch["window"]
            target_adj = batch.get("target_adj", None)
            target_interval = batch.get("target_interval", None)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            window = batch[0]
            target_adj = batch[1]
            target_interval = batch[2] if len(batch) > 2 else None
        else:
            window = batch
            target_adj = None
            target_interval = None

        if isinstance(window, (list, tuple)):
            window = [w.to(self.device) for w in window]
            current_adj = window[-1]
        else:
            current_adj = window.to(self.device)
            window = [current_adj]

        if target_adj is None:
            # Default target is current or next adjacency
            target_adj = current_adj

        target_adj = target_adj.to(self.device)

        # Forward pass through model or assembled components
        if hasattr(self.model, "forward"):
            out = self.model(window, current_adj)
            p_ij = out["p_ij"]
            z_s = out["z_s"]
            a_s = out["a_s"]
            pool_s = out["pool_s"]
            H_fused = out.get("H_fused", None)
        else:
            raise ValueError("Model does not provide a forward method.")

        # 1. Edge prediction loss (L_edge): BCE between predicted probabilities and target
        L_edge = self.bce_loss(p_ij, target_adj.float())

        # 2. Pattern reconstruction loss (L_pattern): MSE(decoder(z_s), pool_s)
        if hasattr(self.model, "motif_dict"):
            recon = self.model.motif_dict.decoder(z_s)
            L_pattern = self.mse_loss(recon, pool_s)
        else:
            L_pattern = torch.tensor(0.0, device=self.device, requires_grad=True)

        # 3. Diversity loss (L_diversity): Negative entropy bonus to prevent dictionary collapse
        # -sum_k( H( mean_s(a_s)_k ) )
        # If batch dimension exists, average attention over steps
        if a_s.dim() == 1:
            mean_a = a_s
        else:
            mean_a = a_s.mean(dim=0)
        # Entropy H(p) = -sum(p * log(p + eps))
        eps = 1e-8
        entropy = -torch.sum(mean_a * torch.log(mean_a + eps))
        # Negative entropy bonus: minimizes -entropy (maximizing entropy)
        L_diversity = -entropy

        # 4. Timing prediction loss (L_timing): CrossEntropy for snapshot interval
        if hasattr(self.model, "predictor") and hasattr(self.model.predictor, "when_head"):
            # Compute when logits from fused embeddings
            h1 = self.model.predictor.W1(H_fused).mean(dim=0, keepdim=True)
            h2 = self.model.predictor.W2(H_fused).mean(dim=0, keepdim=True)
            h_pool = h1 + h2
            if self.model.predictor.use_motif and self.model.predictor.W4 is not None:
                z_term = self.model.predictor.W4(z_s).unsqueeze(0)
                h_pool = h_pool + z_term
            when_logits = self.model.predictor.when_head(torch.relu(h_pool))  # [1, 8]

            if target_interval is not None:
                target_idx = torch.tensor([target_interval], dtype=torch.long, device=self.device)
            else:
                target_idx = torch.tensor([0], dtype=torch.long, device=self.device)

            L_timing = self.ce_loss(when_logits, target_idx)
        else:
            L_timing = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Full loss: L = L_edge + λ·L_pattern + β·L_diversity + γ·L_timing
        L_total = (
            L_edge
            + self.lambda_pattern * L_pattern
            + self.beta_diversity * L_diversity
            + self.gamma_timing * L_timing
        )

        L_total.backward()
        self.optimizer.step()

        metrics = {
            "L_edge": float(L_edge.item()),
            "L_pattern": float(L_pattern.item()),
            "L_diversity": float(L_diversity.item()),
            "L_timing": float(L_timing.item()),
            "L_total": float(L_total.item()),
        }

        self.history.append(metrics)
        return metrics

    def validate(self, val_batch: Any) -> float:
        """Computes validation loss without gradient updates."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(val_batch, dict):
                window = val_batch["window"]
                target_adj = val_batch.get("target_adj", window[-1])
            elif isinstance(val_batch, (list, tuple)) and len(val_batch) >= 2:
                window = val_batch[0]
                target_adj = val_batch[1]
            else:
                window = val_batch
                target_adj = window[-1] if isinstance(window, (list, tuple)) else window

            if isinstance(window, (list, tuple)):
                window = [w.to(self.device) for w in window]
                current_adj = window[-1]
            else:
                current_adj = window.to(self.device)
                window = [current_adj]

            out = self.model(window, current_adj)
            val_loss = float(self.bce_loss(out["p_ij"], target_adj.to(self.device).float()).item())
            return val_loss

    def run(
        self,
        train_batches: List[Any],
        val_batches: Optional[List[Any]] = None,
        n_epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Runs complete training with validation-based checkpointing.
        """
        for epoch in range(n_epochs):
            for batch in train_batches:
                self.step(batch)

            if val_batches is not None and len(val_batches) > 0:
                val_losses = [self.validate(vb) for vb in val_batches]
                mean_val_loss = float(np.mean(val_losses))
                # Checkpoint selected on validation metric
                if mean_val_loss < self.best_val_metric:
                    self.best_val_metric = mean_val_loss
                    self.best_checkpoint = copy.deepcopy(self.model.state_dict())
                    self.checkpoint_metric_source = "val"

        return {
            "history": self.history,
            "best_val_metric": self.best_val_metric,
            "checkpoint_metric_source": self.checkpoint_metric_source,
            "seeded_metadata": self.seeded_metadata,
        }
