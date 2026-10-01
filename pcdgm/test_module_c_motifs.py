"""
pcdgm/test_module_c_motifs.py — Phase 4 tests for Module C.
Highest scrutiny in the project (THE_BOOK §Module C).

Tests (THE_BOOK Part V §Module C):
  - dictionary collapse detection
  - healthy dictionary entropy
  - a_s is exposed as output
  - random vs learned ablation hook
  - reconstruction target = full window (not last-step)
  - injected motif recovery (integration test, needs training loop)
"""

import pytest
import torch
from pcdgm.module_c_motifs import MotifDictionary


class TestMotifDictionary:

    def test_dictionary_collapse_detection(self):
        """
        All attention collapses to slot 0 → entropy < 0.1.
        Module must be able to detect this automatically.
        """
        pytest.skip("Phase 4 not yet implemented.")
        model = MotifDictionary(K=10, dim=64, window=3)
        fake_attention = torch.zeros(100, 10)
        fake_attention[:, 0] = 1.0
        entropy = model.slot_usage_entropy(fake_attention)
        assert entropy < 0.1, (
            f"Collapsed attention should have near-zero entropy, got {entropy:.4f}"
        )

    def test_healthy_dictionary_has_higher_entropy(self):
        """Uniform attention over 10 slots → entropy > 2.0."""
        pytest.skip("Phase 4 not yet implemented.")
        model = MotifDictionary(K=10, dim=64, window=3)
        uniform_attention = torch.ones(100, 10) / 10
        entropy = model.slot_usage_entropy(uniform_attention)
        assert entropy > 2.0, (
            f"Uniform attention should have high entropy, got {entropy:.4f}"
        )

    def test_a_s_is_exposed_as_output(self, synthetic_snapshots):
        """
        a_s (attention weights) MUST be a real, retrievable output.
        Never compute and discard internally (THE_BOOK Hard Rule 6).
        """
        pytest.skip("Phase 4 not yet implemented.")
        model = MotifDictionary(K=10, dim=64, window=3)
        z, a, D = model(synthetic_snapshots[:3])
        assert a is not None, "a_s is None — must be a real output"
        assert a.shape[-1] == 10, f"Expected K=10 attention slots, got {a.shape[-1]}"

    def test_random_vs_learned_ablation_hook_exists(self):
        """
        random_untrained and learned modes must produce different dictionaries.
        This is the ablation hook required by THE_BOOK §ablations.
        """
        pytest.skip("Phase 4 not yet implemented.")
        model_random = MotifDictionary(K=10, dim=64, window=3, mode="random_untrained")
        model_learned = MotifDictionary(K=10, dim=64, window=3, mode="learned")
        # After training, learned ≠ random. At init they may differ by seed.
        # This test checks the hook exists; the meaningful comparison is in ablations/.
        assert hasattr(model_random, "D") and hasattr(model_learned, "D")

    def test_reconstruction_target_full_window(self, synthetic_snapshots):
        """
        Reconstruction must cover the FULL window (k timesteps), not just
        the last step. Last-step-only reconstruction is a known failure mode.
        """
        pytest.skip("Phase 4 not yet implemented.")
        model = MotifDictionary(K=10, dim=64, window=3)
        dummy_window = torch.randn(3, 20, 64)
        recon = model.reconstruct(dummy_window)
        assert recon.shape == dummy_window.shape, (
            f"Reconstruction shape {recon.shape} != window shape {dummy_window.shape}. "
            f"Must reconstruct the full window, not just the last timestep."
        )

    def test_recovers_injected_motif(self, injected_motif_snapshots):
        """
        PHASE 4 GATE. If this fails, do not proceed to real data.
        A known 4-node clique appears every other snapshot — Module C
        must recover it as a dominant motif after short training.

        This is an integration test requiring a training loop.
        Implement after training/loop.py is ready (Phase 5).
        """
        pytest.skip(
            "Integration test — requires training loop (Phase 5). "
            "Do not delete. This is the Phase 4 gate before real data."
        )
