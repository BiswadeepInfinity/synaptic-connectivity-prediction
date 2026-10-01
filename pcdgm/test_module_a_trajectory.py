"""
pcdgm/test_module_a_trajectory.py — Phase 3 tests for Module A.

Tests (THE_BOOK Part V §Module A):
  H2 validation: infer_trajectory_A2 recovers real order above chance.
  A2 blindness:  infer_trajectory_A2 takes no timestamps argument.
  A1 shape:      interpolated adjacency has correct shape.
"""

import pytest
from scipy.stats import spearmanr
from pcdgm.module_a_trajectory import interpolate_A1, infer_trajectory_A2


class TestModuleA1:

    def test_a1_interpolation_shape(self, synthetic_snapshots, real_hours):
        pytest.skip("Phase 3 not yet implemented.")
        interp_fn = interpolate_A1(synthetic_snapshots, real_hours)
        mid_point = interp_fn(s=2.5)
        assert mid_point.shape == synthetic_snapshots[0].shape

    def test_a1_values_bounded_zero_one(self, synthetic_snapshots, real_hours):
        pytest.skip("Phase 3 not yet implemented.")
        interp_fn = interpolate_A1(synthetic_snapshots, real_hours)
        out = interp_fn(s=10.0)
        assert (out >= 0).all() and (out <= 1).all(), (
            "Interpolated adjacency values outside [0, 1] — PCHIP overshoot?"
        )


class TestModuleA2:

    def test_a2_is_blind_to_timestamps(self, synthetic_snapshots):
        """
        A2 must NOT accept real_hours as an argument.
        Passing timestamps would defeat the entire purpose of H2.
        """
        pytest.skip("Phase 3 not yet implemented.")
        import inspect
        sig = inspect.signature(infer_trajectory_A2)
        assert "real_hours" not in sig.parameters, (
            "infer_trajectory_A2 must not accept real_hours — it is blind by design."
        )
        order = infer_trajectory_A2(synthetic_snapshots)
        assert len(order) == len(synthetic_snapshots)

    def test_a2_recovers_real_order_above_chance(self, synthetic_snapshots, real_hours):
        """
        H2. A failure here is a valid, reportable result.
        Do NOT weaken the threshold just to force a pass (THE_BOOK §H2).
        """
        pytest.skip("Phase 3 not yet implemented.")
        inferred_order = infer_trajectory_A2(synthetic_snapshots)
        rho, pval = spearmanr(inferred_order, real_hours)
        assert rho > 0.3, (
            f"H2 fails to clear even a weak bar: rho={rho:.3f}. "
            f"This is a valid negative result — report it, do not weaken the threshold."
        )
