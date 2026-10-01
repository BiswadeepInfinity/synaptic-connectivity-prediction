"""
training/test_loop.py — Phase 5 tests for the training loop.

Tests (THE_BOOK Part V §Training Loop):
  - All 5 loss keys present in step output
  - Checkpoint selected on validation, not train
  - Seed propagated to all 3 init points
  - Minimum 5 seeds
"""

import pytest


class TestTrainingLoop:

    def test_loss_terms_logged_separately(self):
        """
        All four loss terms + total must be in the step output dict.
        Logging only L_total is not acceptable (THE_BOOK Hard Rule 5).
        """
        pytest.skip("Phase 5 not yet implemented.")
        # from training.loop import TrainingLoop
        # step_output = TrainingLoop(...).step(batch)
        required_keys = {"L_edge", "L_pattern", "L_diversity", "L_timing", "L_total"}
        # assert required_keys.issubset(step_output.keys())

    def test_checkpoint_selection_uses_validation_not_train(self):
        """
        Best checkpoint must be selected by validation metric.
        Using training metric would be data leakage into model selection.
        """
        pytest.skip("Phase 5 not yet implemented.")
        # assert training_history["checkpoint_metric_source"] == "val"

    def test_seed_propagates_to_all_three_init_points(self):
        """
        Seed must be applied to: data shuffling, model init, dictionary init.
        Applying to only one or two of these gives non-reproducible results.
        """
        pytest.skip("Phase 5 not yet implemented.")
        # assert seeded_run_metadata["data_shuffle_seed"] == requested_seed
        # assert seeded_run_metadata["model_init_seed"] == requested_seed
        # assert seeded_run_metadata["dictionary_init_seed"] == requested_seed

    def test_five_seeds_minimum(self):
        """Any reported ablation result requires >= 5 seeds."""
        pytest.skip("Phase 5 not yet implemented.")
        # assert len(experiment_config["seeds"]) >= 5
