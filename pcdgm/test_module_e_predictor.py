"""
pcdgm/test_module_e_predictor.py — Phase 5 tests for Module E.

Tests (THE_BOOK Part V §Module E):
  - predicted positive rate not trivially zero (model collapse check)
  - timing output uses interval (not fake precision)
"""

import pytest
import torch
from pcdgm.module_e_predictor import ChangePredictor


class TestChangePredictor:

    def test_predicted_positive_rate_not_trivially_zero(self):
        """
        Class imbalance (many more non-edges than edges) can cause the model
        to collapse and predict zero everywhere. Mean prediction must be > 0.001.
        THE_BOOK anomaly: model collapses under class imbalance.
        """
        pytest.skip("Phase 5 not yet implemented.")
        model = ChangePredictor(dim=100)
        H = torch.randn(20, 100)
        z = torch.randn(20, 64)
        p = model.which(H, H, z)
        assert p.mean() > 0.001, (
            f"Model collapsed: mean prediction {p.mean():.6f} ≈ 0. "
            f"Check class-imbalance handling / loss weighting."
        )

    def test_timing_output_uses_interval_not_fake_precision(self):
        """
        Timing must be reported as a snapshot interval (int in range(8)),
        not a raw float hour. Claiming hour-level precision is not supported
        by the data (8 discrete points). THE_BOOK §Resolution honesty rule.
        """
        pytest.skip("Phase 5 not yet implemented.")
        model = ChangePredictor(dim=100)
        H = torch.randn(20, 100)
        z = torch.randn(20, 64)
        interval = model.when_interval(H, H, z, real_hours=[0, 5, 8, 16, 23, 27, 45, 45])
        assert isinstance(interval, int), (
            f"when_interval returned {type(interval)}, expected int. "
            f"Do not return a float hour — resolution honesty rule."
        )
        assert interval in range(8), (
            f"Interval {interval} out of range(8)."
        )
