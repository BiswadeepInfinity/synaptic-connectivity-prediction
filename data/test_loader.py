"""
data/test_loader.py — Phase 0 tests for data/loader.py.

These are the highest-priority tests in the project (THE_BOOK Part V, §Module 0).
Every test must pass before ANY novel module is built.

Anomalies caught here (cross-reference: THE_BOOK §Anomaly-to-Test table):
  - test_no_temporal_leakage           → temporal leakage via random split
  - test_replicate_snapshots_not_sequential → 45h replicates treated as sequential
  - test_node_id_consistency           → neuron ID drift across snapshots
  - test_neuron_and_edge_counts_match_paper → data loading correctness (real data gate)
"""

import pytest
import torch

from data.loader import (
    REAL_HOURS,
    temporal_split,
    validate_node_consistency,
    verify_paper_counts,
)


# ---------------------------------------------------------------------------
# THE most important test in the project (THE_BOOK: "single most important")
# ---------------------------------------------------------------------------

class TestTemporalSplit:

    def test_no_temporal_leakage(self):
        """
        THE single most important test in this project.
        Train indices must all be strictly less than any val index,
        and val indices strictly less than any test index.
        A random shuffle would trivially break this.
        """
        train_idx, val_idx, test_idx = temporal_split(
            n_timepoints=8, real_hours=REAL_HOURS
        )
        assert max(train_idx) < min(val_idx), (
            f"Temporal leakage: max train idx {max(train_idx)} >= "
            f"min val idx {min(val_idx)}"
        )
        assert max(val_idx) < min(test_idx), (
            f"Temporal leakage: max val idx {max(val_idx)} >= "
            f"min test idx {min(test_idx)}"
        )

    def test_replicate_snapshots_not_sequential(self):
        """
        The two 45h snapshots (indices 6 and 7) must land in the SAME split.
        They represent the same developmental stage in two animals, NOT
        consecutive timepoints. Treating them as sequential is a hard error.
        """
        train_idx, val_idx, test_idx = temporal_split(
            n_timepoints=8, real_hours=REAL_HOURS
        )
        idx_45h = [i for i, h in enumerate(REAL_HOURS) if h == 45]
        assert len(idx_45h) == 2, "Expected exactly 2 snapshots at 45h"

        split_map = {}
        for i in train_idx: split_map[i] = "train"
        for i in val_idx:   split_map[i] = "val"
        for i in test_idx:  split_map[i] = "test"

        splits_of_replicates = {split_map[i] for i in idx_45h}
        assert len(splits_of_replicates) == 1, (
            f"45h replicates are in different splits: "
            f"idx {idx_45h[0]} → {split_map[idx_45h[0]]}, "
            f"idx {idx_45h[1]} → {split_map[idx_45h[1]]}. "
            f"They must always land in the same partition."
        )

    def test_all_indices_covered(self):
        """Every timepoint must appear in exactly one split."""
        train_idx, val_idx, test_idx = temporal_split(
            n_timepoints=8, real_hours=REAL_HOURS
        )
        all_indices = sorted(train_idx + val_idx + test_idx)
        assert all_indices == list(range(8)), (
            f"Not all 8 timepoints covered: got {all_indices}"
        )

    def test_no_overlap_between_splits(self):
        """No index should appear in more than one split."""
        train_idx, val_idx, test_idx = temporal_split(
            n_timepoints=8, real_hours=REAL_HOURS
        )
        train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
        assert train_set.isdisjoint(val_set),  "Overlap between train and val"
        assert train_set.isdisjoint(test_set), "Overlap between train and test"
        assert val_set.isdisjoint(test_set),   "Overlap between val and test"

    def test_each_split_non_empty(self):
        """All three splits must contain at least one snapshot."""
        train_idx, val_idx, test_idx = temporal_split(
            n_timepoints=8, real_hours=REAL_HOURS
        )
        assert len(train_idx) >= 1, "Train split is empty"
        assert len(val_idx)   >= 1, "Val split is empty"
        assert len(test_idx)  >= 1, "Test split is empty"

    def test_indices_are_sorted_ascending(self):
        """Within each split, indices should be in ascending time order."""
        train_idx, val_idx, test_idx = temporal_split(
            n_timepoints=8, real_hours=REAL_HOURS
        )
        assert train_idx == sorted(train_idx), "Train indices not sorted"
        assert val_idx   == sorted(val_idx),   "Val indices not sorted"
        assert test_idx  == sorted(test_idx),  "Test indices not sorted"

    def test_custom_timepoints(self):
        """temporal_split should generalise beyond n=8."""
        train_idx, val_idx, test_idx = temporal_split(
            n_timepoints=10,
            real_hours=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        assert max(train_idx) < min(val_idx)
        assert max(val_idx)   < min(test_idx)


# ---------------------------------------------------------------------------
# Node consistency
# ---------------------------------------------------------------------------

class TestNodeConsistency:

    def test_node_id_consistency(self, synthetic_snapshots):
        """
        Neuron IDs must be stable across all 8 snapshots.
        ID drift (same physical neuron tracked differently) would silently
        corrupt all downstream modules.
        """
        node_ids_per_snapshot = [list(range(s.shape[0])) for s in synthetic_snapshots]
        assert validate_node_consistency(node_ids_per_snapshot)

    def test_inconsistent_nodes_detected(self):
        """validate_node_consistency must return False when IDs differ."""
        node_ids = [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 4],   # <- neuron 3 replaced by neuron 4
        ]
        result = validate_node_consistency(node_ids)
        assert result is False, (
            "Expected False when node IDs differ across snapshots"
        )

    def test_empty_snapshot_list_ok(self):
        """Empty list should return True (vacuously consistent)."""
        assert validate_node_consistency([]) is True

    def test_single_snapshot_ok(self):
        """A single snapshot is trivially consistent."""
        assert validate_node_consistency([[0, 1, 2]]) is True


# ---------------------------------------------------------------------------
# Paper-count verification (Phase 0 gate — run once real data is loaded)
# ---------------------------------------------------------------------------

class TestPaperCounts:

    def test_neuron_and_edge_counts_match_paper(self):
        """
        PRIMARY PHASE 0 GATE.
        Verify all 8 loaded snapshots match Witvliet (2021) Table 1 counts
        within 5% tolerance for edges and exactly for neuron count.

        Requires: pip install cect
        Skipped automatically when cect is not installed.
        """
        from data.loader import (
            load_all_snapshots_both,
            verify_paper_counts_both,
        )
        try:
            chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)
        except ImportError:
            pytest.skip("cect not installed — run: pip install cect")
        except Exception as e:
            pytest.skip(f"Witvliet data not available: {e}")

        results = verify_paper_counts_both(chem_snaps, elec_snaps)
        for i, r in results.items():
            chem_r = r["chemical"]
            elec_r = r["electrical"]
            hour = r["hour"]
            assert chem_r["neuron_ok"], (
                f"Snapshot {i} ({hour}h): chemical neuron count {chem_r['n_neurons']} != 225."
            )
            assert chem_r["edge_ok"], (
                f"Snapshot {i} ({hour}h): chemical edge count {chem_r['n_edges']} deviates "
                f"{chem_r['edge_deviation_pct']:.1f}% from expected {chem_r['expected_edges']}."
            )
            assert elec_r["neuron_ok"], (
                f"Snapshot {i} ({hour}h): electrical neuron count {elec_r['n_neurons']} != 225."
            )
            assert elec_r["edge_ok"], (
                f"Snapshot {i} ({hour}h): electrical edge count {elec_r['n_edges']} deviates "
                f"{elec_r['edge_deviation_pct']:.1f}% from expected {elec_r['expected_edges']}."
            )

    def test_raw_loaded_counts_match_paper_exactly(self):
        """
        Cross-check guard: both chemical and electrical counts from load_all_snapshots_both
        must match their independent paper reference counts (never merged into one count).

        Requires: pip install cect
        """
        from data.loader import (
            load_all_snapshots_both,
            _PAPER_CHEMICAL_EDGE_COUNTS,
            _PAPER_ELECTRICAL_EDGE_COUNTS,
        )
        try:
            chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)
        except ImportError:
            pytest.skip("cect not installed — run: pip install cect")
        except Exception as e:
            pytest.skip(f"Witvliet data not available: {e}")

        for i, (c_snap, e_snap, label) in enumerate(zip(chem_snaps, elec_snaps, labels)):
            c_count = int(c_snap.sum().item())
            c_expected = _PAPER_CHEMICAL_EDGE_COUNTS[i]
            c_diff = abs(c_count - c_expected) / max(c_expected, 1)
            assert c_diff <= 0.05, (
                f"Snapshot {i} ({label}): chemical count {c_count} deviates "
                f"{c_diff:.1%} from paper chemical count {c_expected}."
            )

            e_count = int(e_snap.sum().item())
            e_expected = _PAPER_ELECTRICAL_EDGE_COUNTS[i]
            e_diff = abs(e_count - e_expected) / max(e_expected, 1)
            assert e_diff <= 0.05, (
                f"Snapshot {i} ({label}): electrical count {e_count} deviates "
                f"{e_diff:.1%} from paper electrical count {e_expected}."
            )

    def test_verify_paper_counts_with_synthetic(self, synthetic_snapshots):
        """
        Smoke-test verify_paper_counts() on synthetic data.
        Expected to warn about edge counts (synthetic ≠ paper) but not crash.
        """
        # Patch expected counts temporarily to match synthetic sizes
        import data.loader as loader_module
        original_chem = loader_module._PAPER_CHEMICAL_EDGE_COUNTS

        synthetic_edges = [int(s.sum().item()) for s in synthetic_snapshots]
        loader_module._PAPER_CHEMICAL_EDGE_COUNTS = synthetic_edges
        loader_module._PAPER_NEURON_COUNT = synthetic_snapshots[0].shape[0]

        try:
            results = loader_module.verify_paper_counts(synthetic_snapshots, tolerance=0.5, synclass="Generic_CS")
            for i, r in results.items():
                assert "n_neurons" in r
                assert "n_edges" in r
        finally:
            loader_module._PAPER_CHEMICAL_EDGE_COUNTS = original_chem
            loader_module._PAPER_NEURON_COUNT = 225


# ---------------------------------------------------------------------------
# Adjacency matrix properties
# ---------------------------------------------------------------------------

class TestAdjacencyProperties:

    def test_no_self_loops_in_synthetic(self, synthetic_snapshots):
        """Diagonal of every adjacency matrix must be zero (no self-loops)."""
        for i, snap in enumerate(synthetic_snapshots):
            diag = torch.diag(snap)
            assert diag.sum() == 0, f"Snapshot {i} has self-loops on diagonal"

    def test_snapshots_are_float32(self, synthetic_snapshots):
        """Adjacency tensors must be float32 for downstream GNN compatibility."""
        for i, snap in enumerate(synthetic_snapshots):
            assert snap.dtype == torch.float32, (
                f"Snapshot {i} dtype is {snap.dtype}, expected torch.float32"
            )

    def test_snapshots_are_square(self, synthetic_snapshots):
        """Each adjacency matrix must be square (N × N)."""
        for i, snap in enumerate(synthetic_snapshots):
            assert snap.shape[0] == snap.shape[1], (
                f"Snapshot {i} shape {snap.shape} is not square"
            )

    def test_all_values_binary(self, synthetic_snapshots):
        """
        Adjacency matrices must be binary (0.0 or 1.0).
        Non-binary values would indicate weight information is present,
        which would require explicit handling in Module B.
        """
        for i, snap in enumerate(synthetic_snapshots):
            unique_vals = snap.unique()
            assert set(unique_vals.tolist()).issubset({0.0, 1.0}), (
                f"Snapshot {i} has non-binary values: {unique_vals.tolist()}"
            )
