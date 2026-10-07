"""
pcdgm/test_module_a_trajectory.py — Phase 3 tests for Module A.

Tests (THE_BOOK Part V §Module A):
  H2 validation: infer_trajectory_A2 recovers real order above chance.
  A2 blindness:  infer_trajectory_A2 takes no timestamps argument.
  A1 shape:      interpolated adjacency has correct shape.
  A1 bounds:     interpolated probabilities stay bounded in [0, 1].
"""

import inspect
import pytest
from scipy.stats import spearmanr
import torch
from pcdgm.module_a_trajectory import interpolate_A1, infer_trajectory_A2, validate_h2_trajectory


class TestModuleA1:

    def test_a1_interpolation_shape(self, synthetic_snapshots, real_hours):
        interp_fn = interpolate_A1(synthetic_snapshots, real_hours)
        mid_point = interp_fn(s=2.5)
        assert mid_point.shape == synthetic_snapshots[0].shape

    def test_a1_values_bounded_zero_one(self, synthetic_snapshots, real_hours):
        interp_fn = interpolate_A1(synthetic_snapshots, real_hours)
        out = interp_fn(s=10.0)
        assert (out >= 0).all() and (out <= 1).all(), (
            "Interpolated adjacency values outside [0, 1] — PCHIP overshoot?"
        )

    def test_a1_no_self_loops(self, synthetic_snapshots, real_hours):
        interp_fn = interpolate_A1(synthetic_snapshots, real_hours)
        out = interp_fn(s=16.0)
        diag = torch.diag(out)
        assert torch.all(diag == 0.0), "Interpolated connectome has non-zero self-loops on diagonal."

    def test_a1_replicates_properly_averaged(self, synthetic_snapshots, real_hours):
        # Test evaluation at s=45h matches average of the two 45h snapshots
        interp_fn = interpolate_A1(synthetic_snapshots, real_hours)
        out_45 = interp_fn(s=45.0)
        expected_45 = (synthetic_snapshots[-2] + synthetic_snapshots[-1]) / 2.0
        assert torch.allclose(out_45, expected_45, atol=1e-5), (
            "A1 at t=45h must equal the average of the two 45h replicates."
        )


class TestModuleA2:

    def test_a2_is_blind_to_timestamps(self, synthetic_snapshots):
        """
        A2 must NOT accept real_hours as an argument.
        Passing timestamps would defeat the entire purpose of H2.
        """
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
        inferred_order = infer_trajectory_A2(synthetic_snapshots)
        rho, pval = spearmanr(inferred_order, real_hours)
        assert rho > 0.3, (
            f"H2 fails to clear even a weak bar: rho={rho:.3f}. "
            f"This is a valid negative result — report it, do not weaken the threshold."
        )

    def test_a2_recovers_real_witvliet_order(self):
        """
        H2 validation on real biological connectome data.
        Verifies that blind structural spectral seriation strongly correlates
        with developmental hours on the 8 real Witvliet snapshots.
        """
        from data.loader import load_all_snapshots
        try:
            snaps, hours, _ = load_all_snapshots(use_cect=True)
        except Exception as e:
            pytest.skip(f"Witvliet connectome not available: {e}")

        inferred = infer_trajectory_A2(snaps)
        rho, pval = validate_h2_trajectory(inferred, hours)
        assert rho > 0.8, (
            f"H2 on real Witvliet data expected rho > 0.8, got rho={rho:.4f} (p={pval:.4e})"
        )

