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

TODO (Phase 5): implement TrainingLoop and run_training().
"""


class TrainingLoop:
    """
    Full PCDGM training loop with separate loss logging and
    validation-based checkpointing.

    TODO (Phase 5): implement.
    """

    def __init__(self, model, config):
        raise NotImplementedError("Training loop (Phase 5) not yet implemented.")

    def step(self, batch) -> dict:
        """
        Returns dict with keys: L_edge, L_pattern, L_diversity, L_timing, L_total.
        ALL terms must be present — see test_loss_terms_logged_separately.
        """
        raise NotImplementedError("Training loop (Phase 5) not yet implemented.")

    def run(self, n_epochs: int):
        raise NotImplementedError("Training loop (Phase 5) not yet implemented.")
