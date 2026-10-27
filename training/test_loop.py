"""
training/test_loop.py — Phase 5 tests for the training loop.

Tests (THE_BOOK Part V §Training Loop):
  - All 5 loss keys present in step output
  - Checkpoint selected on validation, not train
  - Seed propagated to all 3 init points
  - Minimum 5 seeds
"""

import pytest
import numpy as np
import torch
from pcdgm.model import PCDGM
from training.loop import TrainingLoop, set_seed, validate_seeds


class TestTrainingLoop:

    def test_loss_terms_logged_separately(self, synthetic_snapshots):
        """
        All four loss terms + total must be in the step output dict.
        Logging only L_total is not acceptable (THE_BOOK Hard Rule 5).
        """
        model = PCDGM(node_dim=64, encoder_hidden=100, fusion_out_dim=100, motif_K=10, window_k=3)
        loop = TrainingLoop(model=model)

        batch = {
            "window": synthetic_snapshots[:3],
            "target_adj": synthetic_snapshots[3],
            "target_interval": 3,
        }

        step_output = loop.step(batch)
        required_keys = {"L_edge", "L_pattern", "L_diversity", "L_timing", "L_total"}
        assert required_keys.issubset(step_output.keys()), (
            f"Step output missing required keys. Found: {step_output.keys()}"
        )

    def test_checkpoint_selection_uses_validation_not_train(self, synthetic_snapshots):
        """
        Best checkpoint must be selected by validation metric.
        Using training metric would be data leakage into model selection.
        """
        model = PCDGM(node_dim=64, encoder_hidden=100, fusion_out_dim=100, motif_K=10, window_k=3)
        loop = TrainingLoop(model=model)

        train_batches = [
            {"window": synthetic_snapshots[:3], "target_adj": synthetic_snapshots[3]}
        ]
        val_batches = [
            {"window": synthetic_snapshots[2:5], "target_adj": synthetic_snapshots[5]}
        ]

        training_history = loop.run(train_batches=train_batches, val_batches=val_batches, n_epochs=2)
        assert training_history["checkpoint_metric_source"] == "val", (
            f"Expected checkpoint metric source 'val', got {training_history.get('checkpoint_metric_source')}"
        )
        assert loop.best_checkpoint is not None, "Best checkpoint was not saved."

    def test_seed_propagates_to_all_three_init_points(self):
        """
        Verify behavioral reproducibility across all three initialization points:
          1. Data sampling / order
          2. Model parameter initialization
          3. Dictionary initialization
        Two runs with the SAME seed must produce identical outputs, while runs
        with DIFFERENT seeds must produce distinct outputs.
        """
        seed_a = 42
        seed_b = 12345

        # 1. Data order check (NumPy random state)
        set_seed(seed_a)
        data_order_a1 = np.random.permutation(50)
        set_seed(seed_a)
        data_order_a2 = np.random.permutation(50)
        set_seed(seed_b)
        data_order_b = np.random.permutation(50)

        assert np.array_equal(data_order_a1, data_order_a2), "Data order non-deterministic with same seed!"
        assert not np.array_equal(data_order_a1, data_order_b), "Data order identical across different seeds!"

        # 2. Model parameter initialization check (Torch weights)
        set_seed(seed_a)
        model_a1 = PCDGM(node_dim=64, encoder_hidden=100, fusion_out_dim=100, motif_K=10, window_k=3)
        set_seed(seed_a)
        model_a2 = PCDGM(node_dim=64, encoder_hidden=100, fusion_out_dim=100, motif_K=10, window_k=3)
        set_seed(seed_b)
        model_b = PCDGM(node_dim=64, encoder_hidden=100, fusion_out_dim=100, motif_K=10, window_k=3)

        w_a1 = next(model_a1.encoder.conv1.parameters())
        w_a2 = next(model_a2.encoder.conv1.parameters())
        w_b = next(model_b.encoder.conv1.parameters())

        assert torch.equal(w_a1, w_a2), "Model parameters non-deterministic with same seed!"
        assert not torch.equal(w_a1, w_b), "Model parameters identical across different seeds!"

        # 3. Dictionary initialization check (Motif dictionary D)
        dict_a1 = model_a1.motif_dict.D
        dict_a2 = model_a2.motif_dict.D
        dict_b = model_b.motif_dict.D

        assert torch.equal(dict_a1, dict_a2), "Dictionary D non-deterministic with same seed!"
        assert not torch.equal(dict_a1, dict_b), "Dictionary D identical across different seeds!"

    def test_five_seeds_minimum(self):
        """
        Verify behavioral requirement that at least 5 distinct seeds are
        enforced and executed for reported ablation runs.
        """
        seeds_valid = [42, 12345, 123, 456, 789]
        checked = validate_seeds(seeds_valid)
        assert len(checked) >= 5, "Fewer than 5 seeds provided!"
        assert len(set(checked)) == len(checked), "Seeds must be unique!"

        # Real seed validation must raise ValueError when given fewer than 5 seeds
        seeds_insufficient = [42, 12345, 123]
        with pytest.raises(ValueError, match="At least 5 seeds"):
            validate_seeds(seeds_insufficient)
