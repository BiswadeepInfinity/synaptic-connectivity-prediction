"""
pcdgm/test_module_c_motifs.py — Phase 4 tests for Module C (Structural Motif Discovery).
Highest scrutiny in the project (THE_BOOK §Module C).

Tests (THE_BOOK Part V §Module C):
  - dictionary collapse detection
  - healthy dictionary entropy
  - a_s is exposed as output
  - random vs learned ablation hook
  - reconstruction target = full window (not last-step)
  - injected motif recovery sanity check (Phase 4 gate)
"""

from collections import Counter
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pcdgm.module_c_motifs import MotifDictionary


class TestMotifDictionary:

    def test_dictionary_collapse_detection(self):
        """
        All attention collapses to slot 0 -> entropy < 0.1.
        Module must be able to detect this automatically.
        """
        model = MotifDictionary(K=10, dim=64, window=3)
        fake_attention = torch.zeros(100, 10)
        fake_attention[:, 0] = 1.0
        entropy = model.slot_usage_entropy(fake_attention)
        assert entropy < 0.1, (
            f"Collapsed attention should have near-zero entropy, got {entropy:.4f}"
        )

    def test_healthy_dictionary_has_higher_entropy(self):
        """Uniform attention over 10 slots -> entropy > 2.0 (ln(10) ≈ 2.302)."""
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
        model = MotifDictionary(K=10, dim=64, window=3)
        z, a, D = model(synthetic_snapshots[:3])
        assert a is not None, "a_s is None -- must be a real output"
        assert a.shape[-1] == 10, f"Expected K=10 attention slots, got {a.shape[-1]}"
        assert torch.isclose(a.sum(), torch.tensor(1.0), atol=1e-4), "a_s must sum to 1"

    def test_random_vs_learned_ablation_hook_exists(self):
        """
        random_untrained and learned modes must produce different dictionaries.
        This is the ablation hook required by THE_BOOK §ablations.
        """
        model_random = MotifDictionary(K=10, dim=64, window=3, mode="random_untrained")
        model_learned = MotifDictionary(K=10, dim=64, window=3, mode="learned")
        assert hasattr(model_random, "D") and hasattr(model_learned, "D")
        assert not torch.allclose(model_random.D, model_learned.D)
        assert not model_random.D.requires_grad, "random_untrained D must be frozen"
        assert model_learned.D.requires_grad, "learned D must require grad"

    def test_reconstruction_target_full_window(self, synthetic_snapshots):
        """
        Reconstruction must cover the FULL window (k timesteps), not just
        the last step. Last-step-only reconstruction is a known failure mode.
        """
        model = MotifDictionary(K=10, dim=64, window=3)
        dummy_window = torch.randn(3, 20, 64)
        recon = model.reconstruct(dummy_window)
        assert recon.shape == dummy_window.shape, (
            f"Reconstruction shape {recon.shape} != window shape {dummy_window.shape}. "
            f"Must reconstruct the full window, not just the last timestep."
        )

    def test_recovers_injected_motif(self):
        """
        PHASE 4 GATE: Synthetic Sanity Check.
        Inject a deliberate repeating structural pattern (4-node dense clique)
        into fake snapshot window sequence data, and verify:
          1. Model reconstructs pattern representations with decreasing MSE.
          2. Learned motif dictionary assigns a consistent dominant slot to the injected motif (>90% consistency).
          3. Dictionary does not collapse (usage entropy > 1.5).
        """
        torch.manual_seed(42)
        N = 20
        dim = 64
        k = 3
        num_windows = 40

        # Construct synthetic dataset with injected motif in every alternate window
        data = []
        labels = []
        for i in range(num_windows):
            is_motif = (i % 2 == 0)
            window = []
            for _ in range(k):
                adj = (torch.rand(N, N) < 0.05).float()
                adj.fill_diagonal_(0)
                if is_motif:
                    # Injected repeating structural pattern: 4-node dense clique
                    adj[0:4, 0:4] = 1.0
                    adj[0:4, 0:4].fill_diagonal_(0)
                window.append(adj)
            data.append(window)
            labels.append(1 if is_motif else 0)

        model = MotifDictionary(K=8, dim=dim, window=k, mode="learned")
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Measure pre-training reconstruction MSE
        initial_mse = 0.0
        with torch.no_grad():
            for w in data:
                pool_target = model._pool_window(w)
                z, _, _ = model(w)
                initial_mse += nn.functional.mse_loss(model.decoder(z), pool_target).item()
        initial_mse /= len(data)

        # Train model on synthetic sequences
        for _ in range(50):
            optimizer.zero_grad()
            all_a = []
            total_recon = 0.0
            for w in data:
                pool_target = model._pool_window(w)
                z, a, _ = model(w)
                recon = model.decoder(z)
                total_recon = total_recon + nn.functional.mse_loss(recon, pool_target)
                all_a.append(a)

            a_batch = torch.stack(all_a, dim=0)  # [B, K]
            p_mean = a_batch.mean(dim=0)
            l_diversity = -torch.sum(p_mean * torch.log(p_mean + 1e-9))

            loss = (total_recon / len(data)) - 0.05 * l_diversity
            loss.backward()
            optimizer.step()

        # Measure post-training reconstruction MSE
        final_mse = 0.0
        with torch.no_grad():
            for w in data:
                pool_target = model._pool_window(w)
                z, _, _ = model(w)
                final_mse += nn.functional.mse_loss(model.decoder(z), pool_target).item()
        final_mse /= len(data)

        # Verify loss reduction
        assert final_mse < initial_mse * 0.3, (
            f"Pattern reconstruction did not decrease sufficiently: {initial_mse:.4f} -> {final_mse:.4f}"
        )

        # Check slot usage entropy (collapse prevention)
        with torch.no_grad():
            a_batch = torch.stack([model(w)[1] for w in data], dim=0)
            entropy = model.slot_usage_entropy(a_batch)

        assert entropy > 1.5, f"Dictionary collapsed to low entropy: {entropy:.4f}"

        # Check recovery of injected repeating structural pattern
        motif_slots = [model(w)[1].argmax().item() for w, m in zip(data, labels) if m == 1]
        motif_counts = Counter(motif_slots)
        dominant_slot, dominant_count = motif_counts.most_common(1)[0]
        recovery_rate = dominant_count / len(motif_slots)

        assert recovery_rate >= 0.90, (
            f"Injected motif not consistently recovered! Dominant slot {dominant_slot} "
            f"selected only {dominant_count}/{len(motif_slots)} ({recovery_rate*100:.1f}%)"
        )
