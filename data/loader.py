"""
data/loader.py — Phase 0: Data loading and temporal splitting.

Implements:
  - temporal_split()           strict chronological splits, no shuffling ever
  - load_snapshot()            load a single Witvliet snapshot as adjacency tensor
  - load_all_snapshots()       load all 8 snapshots with metadata
  - validate_node_consistency() ensure neuron IDs are stable across snapshots
  - verify_paper_counts()      check edge / neuron counts against Witvliet 2021

Hard rules from THE_BOOK Part II:
  1. Temporal splits only. Never shuffle across time.
  2. The two 45h replicates are NOT sequential — same developmental stage,
     two animals. They must always land in the same split.
  3. No activity data anywhere.

Design choices (not fully specified in THE_BOOK, documented here per Rule 6):
  - PCHIP is used for trajectory interpolation backbone (Module A). This file
    only does loading + splitting; PCHIP lives in module_a_trajectory.py.
  - The 45h replicates are identified by position (indices 6 and 7) rather
    than by value so that future data updates don't silently break the logic.
  - We use a 5/1/2 split (snapshots 0-4 train, 5 val, 6-7 test) giving the
    test set the two 45h replicates together — this keeps the replicates in
    the same partition as required.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants — Witvliet et al. 2021, Table 1
# ---------------------------------------------------------------------------
REAL_HOURS: List[int] = [0, 5, 8, 16, 23, 27, 45, 45]

# Paper-reported edge counts per snapshot (Witvliet et al. Nature 2021 Table 1)
# Chemical synapses (directed connections):
_PAPER_CHEMICAL_EDGE_COUNTS: List[int] = [775, 986, 1012, 1136, 1515, 1525, 2191, 2186]
# Electrical synapses / gap junctions (symmetric directed edges, undirected pairs * 2):
# Directed edges in connectome graphs: [162, 244, 184, 412, 572, 424, 570, 604]
# Undirected gap junction pairs: [81, 122, 92, 206, 286, 212, 285, 302]
_PAPER_ELECTRICAL_EDGE_COUNTS: List[int] = [162, 244, 184, 412, 572, 424, 570, 604]
_PAPER_ELECTRICAL_PAIR_COUNTS: List[int] = [81, 122, 92, 206, 286, 212, 285, 302]

# Backwards compatibility alias for chemical synapses
_PAPER_EDGE_COUNTS: List[int] = _PAPER_CHEMICAL_EDGE_COUNTS
# Paper-reported neuron count (all snapshots)
_PAPER_NEURON_COUNT: int = 225
# Tolerance: allow ±5 % deviation from paper counts before raising
_COUNT_TOLERANCE: float = 0.05


# ---------------------------------------------------------------------------
# Temporal splitting
# ---------------------------------------------------------------------------

def temporal_split(
    n_timepoints: int = 8,
    real_hours: List[int] | None = None,
    train_frac: float = 0.625,   # 5/8 snapshots
    val_frac:   float = 0.125,   # 1/8 snapshot
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (train_indices, val_indices, test_indices) as strict temporal
    partitions — no data from a later timepoint ever appears in an earlier
    split.

    Special handling for the two 45h replicates (indices 6 and 7 when
    n_timepoints=8 and real_hours=[0,5,8,16,23,27,45,45]):
      - Both replicates are forced into the same split (test by default).
      - Their equal timestamp means neither is "later" than the other; we
        must not treat them as a forward prediction pair.

    Args:
        n_timepoints: total number of snapshots (default 8).
        real_hours:   timestamps in hours post-hatching; used to detect
                      tied timestamps (replicates). Defaults to REAL_HOURS.
        train_frac:   fraction of snapshots for training.
        val_frac:     fraction of snapshots for validation.

    Returns:
        Three lists of integer indices, sorted ascending within each split.

    Raises:
        ValueError: if resulting splits overlap or any split is empty.
    """
    if real_hours is None:
        real_hours = REAL_HOURS

    hours = list(real_hours)
    assert len(hours) == n_timepoints, (
        f"real_hours length {len(hours)} != n_timepoints {n_timepoints}"
    )

    # Identify tied-hour groups (e.g. the two 45h replicates).
    # All members of a tied group must land in the same split.
    seen: dict[int, int] = {}
    tied_group: dict[int, int] = {}  # index → representative index of its group
    for i, h in enumerate(hours):
        if h in seen:
            tied_group[i] = seen[h]
            tied_group[seen[h]] = seen[h]
        else:
            seen[h] = i
            tied_group[i] = i

    # Build split boundaries (strict: train < val < test in time)
    n_train = max(1, round(n_timepoints * train_frac))
    n_val   = max(1, round(n_timepoints * val_frac))
    n_test  = n_timepoints - n_train - n_val

    if n_test < 1:
        raise ValueError(
            f"Split leaves no test snapshots: train={n_train}, val={n_val}, "
            f"total={n_timepoints}."
        )

    train_idx = list(range(n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, n_timepoints))

    # Verify replicate integrity: every tied group must be in one partition
    split_map = {i: "train" for i in train_idx}
    split_map.update({i: "val" for i in val_idx})
    split_map.update({i: "test" for i in test_idx})

    for i, rep in tied_group.items():
        if i != rep and split_map[i] != split_map[rep]:
            raise ValueError(
                f"Tied-hour replicates {i} and {rep} (both {hours[i]}h) "
                f"landed in different splits ({split_map[i]}, {split_map[rep]}). "
                f"Adjust train_frac/val_frac so both fall in the same partition."
            )

    # Sanity: no overlap, strictly ordered
    _assert_no_overlap(train_idx, val_idx, test_idx)
    assert max(train_idx) < min(val_idx), "Temporal ordering violated: train/val"
    assert max(val_idx)   < min(test_idx), "Temporal ordering violated: val/test"

    return train_idx, val_idx, test_idx


def _assert_no_overlap(
    train_idx: List[int],
    val_idx:   List[int],
    test_idx:  List[int],
) -> None:
    """Raise if any index appears in more than one split."""
    all_sets = [set(train_idx), set(val_idx), set(test_idx)]
    for i, s1 in enumerate(all_sets):
        for j, s2 in enumerate(all_sets):
            if i < j:
                overlap = s1 & s2
                if overlap:
                    raise ValueError(
                        f"Split overlap detected between split {i} and split {j}: "
                        f"indices {overlap}."
                    )


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

def load_snapshot(
    reader,                  # cect WitvlietDataReader instance (from get_instance())
    synclass: str = "Generic_CS",
    all_nodes: List[str] | None = None,
    as_tensor: bool = True,
) -> torch.Tensor | np.ndarray:
    """
    Load a single Witvliet snapshot from a cect reader instance.

    Uses reader.to_networkx_graph(synclass) → networkx DiGraph → dense adjacency
    matrix. Returns an (N, N) float32 tensor (or numpy array).
    Diagonal is zeroed — no self-loops.
    """
    import networkx as nx

    G = reader.to_networkx_graph(synclass)
    if all_nodes is not None:
        nodes = all_nodes
    else:
        nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    adj_np = np.zeros((n, n), dtype=np.float32)
    for src, dst, data in G.edges(data=True):
        if src in node_idx and dst in node_idx:
            i, j = node_idx[src], node_idx[dst]
            adj_np[i, j] = 1.0   # binarise

    np.fill_diagonal(adj_np, 0)   # no self-loops

    if as_tensor:
        return torch.from_numpy(adj_np)
    return adj_np


def load_all_snapshots(
    use_cect:    bool = True,
    devotg_path: Path | None = None,
    synclass:    str = "Generic_CS",
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Load all 8 Witvliet snapshots.

    Args:
        use_cect:    if True, load via `cect` (default, independent cross-check).
        devotg_path: if provided, also load from DevoTG's pre-parsed format and
                     cross-check edge counts. Used in Phase 1.
        synclass:    "Generic_CS" for chemical synapses, "Generic_GJ" for electrical gap junctions.

    Returns:
        snapshots : list of 8 (N, N) float32 tensors, index 0 = 0h.
        hours     : list of 8 ints [0, 5, 8, 16, 23, 27, 45, 45].
        labels    : human-readable stage labels.

    Raises:
        ImportError: if cect is not installed.
        RuntimeError: if neuron counts differ across snapshots.
    """
    labels = [
        "L1_0h", "L1_5h", "L1_8h", "L1_16h",
        "L2_23h", "L3_27h", "YA_45h_A", "YA_45h_B",
    ]

    if use_cect:
        snapshots = _load_via_cect(synclass=synclass)
    else:
        if devotg_path is None:
            raise ValueError("devotg_path required when use_cect=False.")
        snapshots = _load_via_devotg(devotg_path)

    if not validate_node_consistency([list(range(s.shape[0])) for s in snapshots]):
        raise RuntimeError(
            "Node count is not consistent across snapshots. "
            "Check whether all 8 Witvliet datasets loaded correctly."
        )

    return snapshots, list(REAL_HOURS), labels


def load_all_snapshots_both(
    use_cect: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[str]]:
    """
    Load all 8 Witvliet snapshots for both chemical synapses AND electrical gap junctions separately.

    Chemical and electrical adjacency matrices are returned as separate lists of tensors,
    both strictly aligned to the exact same global union of 225 neuron nodes. They are NEVER merged.

    Returns:
        chem_snapshots : list of 8 (N, N) float32 tensors for chemical synapses.
        elec_snapshots : list of 8 (N, N) float32 tensors for electrical synapses (gap junctions).
        hours          : list of 8 ints [0, 5, 8, 16, 23, 27, 45, 45].
        labels         : human-readable stage labels.
    """
    labels = [
        "L1_0h", "L1_5h", "L1_8h", "L1_16h",
        "L2_23h", "L3_27h", "YA_45h_A", "YA_45h_B",
    ]

    try:
        from cect.readers.WitvlietDataReader1 import get_instance as get1
        from cect.readers.WitvlietDataReader2 import get_instance as get2
        from cect.readers.WitvlietDataReader3 import get_instance as get3
        from cect.readers.WitvlietDataReader4 import get_instance as get4
        from cect.readers.WitvlietDataReader5 import get_instance as get5
        from cect.readers.WitvlietDataReader6 import get_instance as get6
        from cect.readers.WitvlietDataReader7 import get_instance as get7
        from cect.readers.WitvlietDataReader8 import get_instance as get8
    except ImportError as exc:
        raise ImportError(
            "cect not installed. Run: pip install cect\n"
            "See THE_BOOK §6 for data access options."
        ) from exc

    readers = [get1(), get2(), get3(), get4(), get5(), get6(), get7(), get8()]
    all_nodes_set = set()
    for r in readers:
        all_nodes_set.update(r.to_networkx_graph("Generic_CS").nodes())
        all_nodes_set.update(r.to_networkx_graph("Generic_GJ").nodes())
    all_nodes = sorted(list(all_nodes_set))

    chem_snapshots = []
    elec_snapshots = []
    for reader in readers:
        c_snap = load_snapshot(reader, synclass="Generic_CS", all_nodes=all_nodes, as_tensor=True)
        e_snap = load_snapshot(reader, synclass="Generic_GJ", all_nodes=all_nodes, as_tensor=True)
        chem_snapshots.append(c_snap)
        elec_snapshots.append(e_snap)

    return chem_snapshots, elec_snapshots, list(REAL_HOURS), labels


def _load_via_cect(synclass: str = "Generic_CS") -> List[torch.Tensor]:
    """Load the 8 snapshots using the cect WitvlietDataReader classes aligned to global union nodes."""
    try:
        from cect.readers.WitvlietDataReader1 import get_instance as get1
        from cect.readers.WitvlietDataReader2 import get_instance as get2
        from cect.readers.WitvlietDataReader3 import get_instance as get3
        from cect.readers.WitvlietDataReader4 import get_instance as get4
        from cect.readers.WitvlietDataReader5 import get_instance as get5
        from cect.readers.WitvlietDataReader6 import get_instance as get6
        from cect.readers.WitvlietDataReader7 import get_instance as get7
        from cect.readers.WitvlietDataReader8 import get_instance as get8
    except ImportError as exc:
        raise ImportError(
            "cect not installed. Run: pip install cect\n"
            "See THE_BOOK §6 for data access options."
        ) from exc

    # get_instance() is the stable factory function in cect 0.3.5
    readers = [get1(), get2(), get3(), get4(), get5(), get6(), get7(), get8()]
    all_nodes_set = set()
    for r in readers:
        all_nodes_set.update(r.to_networkx_graph("Generic_CS").nodes())
        all_nodes_set.update(r.to_networkx_graph("Generic_GJ").nodes())
    all_nodes = sorted(list(all_nodes_set))

    snapshots = []
    for reader in readers:
        snap = load_snapshot(reader, synclass=synclass, all_nodes=all_nodes, as_tensor=True)
        snapshots.append(snap)

    return snapshots


def _load_via_devotg(devotg_path: Path) -> List[torch.Tensor]:
    """
    Load pre-parsed snapshots from DevoTG's repository.
    Used ONLY in Phase 1 (reproduction gate). Not the canonical data path.

    DESIGN CHOICE: DevoTG stores adjacency matrices as .npy files named
    by timepoint index. If their format differs, update this function only.
    """
    snapshots = []
    for i in range(8):
        candidate = devotg_path / f"data" / f"connectome_{i}.npy"
        if not candidate.exists():
            raise FileNotFoundError(
                f"DevoTG snapshot not found: {candidate}\n"
                f"Clone the repo into external/devotg/ first."
            )
        arr = np.load(str(candidate)).astype(np.float32)
        np.fill_diagonal(arr, 0)
        snapshots.append(torch.from_numpy(arr))
    return snapshots


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_node_consistency(
    node_ids_per_snapshot: List[List[int]],
) -> bool:
    """
    Return True iff every snapshot has the same set of node IDs.
    Neuron ID drift (same neuron tracked differently in different
    reconstructions) is an anomaly that would silently corrupt all
    downstream modules — catch it early (THE_BOOK anomaly table).
    """
    if not node_ids_per_snapshot:
        return True
    reference = set(node_ids_per_snapshot[0])
    for i, ids in enumerate(node_ids_per_snapshot[1:], start=1):
        if set(ids) != reference:
            warnings.warn(
                f"Node ID mismatch at snapshot {i}: "
                f"{len(set(ids))} nodes vs {len(reference)} in snapshot 0.",
                stacklevel=2,
            )
            return False
    return True


def verify_paper_counts(
    snapshots: List[torch.Tensor],
    tolerance: float = _COUNT_TOLERANCE,
    synclass: str = "Generic_CS",
) -> dict:
    """
    Compare loaded neuron and edge counts against the Witvliet (2021) paper.
    Returns a summary dict. Raises RuntimeError if neuron count is wrong;
    warns (does not raise) if edge counts deviate beyond tolerance.

    Args:
        snapshots: list of (N, N) adjacency tensors.
        tolerance: fraction allowed deviation (default 5%).
        synclass: "Generic_CS" (chemical, default) or "Generic_GJ" (electrical).

    This is the primary Phase 0 verification step.
    """
    results = {}
    n_neurons_expected = _PAPER_NEURON_COUNT
    expected_list = (
        _PAPER_CHEMICAL_EDGE_COUNTS if synclass == "Generic_CS"
        else _PAPER_ELECTRICAL_EDGE_COUNTS
    )

    for i, snap in enumerate(snapshots):
        n_neurons = snap.shape[0]
        n_edges   = int(snap.sum().item())
        expected_edges = expected_list[i]
        hour      = REAL_HOURS[i]

        neuron_ok = (n_neurons == n_neurons_expected)
        edge_dev  = abs(n_edges - expected_edges) / max(expected_edges, 1)
        edge_ok   = (edge_dev <= tolerance)

        if not neuron_ok:
            raise RuntimeError(
                f"Snapshot {i} ({hour}h): neuron count {n_neurons} != "
                f"expected {n_neurons_expected}. Data may have loaded incorrectly."
            )
        if not edge_ok:
            warnings.warn(
                f"Snapshot {i} ({hour}h): {synclass} edge count {n_edges} deviates "
                f"{edge_dev:.1%} from paper ({expected_edges}). "
                f"Exceeds {tolerance:.0%} tolerance — check data source.",
                stacklevel=2,
            )

        results[i] = {
            "hour": hour,
            "synclass": synclass,
            "n_neurons": n_neurons,
            "n_edges": n_edges,
            "expected_edges": expected_edges,
            "edge_deviation_pct": round(edge_dev * 100, 2),
            "neuron_ok": neuron_ok,
            "edge_ok": edge_ok,
        }

    return results


def verify_paper_counts_both(
    chem_snapshots: List[torch.Tensor],
    elec_snapshots: List[torch.Tensor],
    tolerance: float = _COUNT_TOLERANCE,
) -> dict:
    """
    Verify chemical synapses and electrical gap junctions independently against their paper values.

    Never merges the two counts.
    """
    chem_results = verify_paper_counts(chem_snapshots, tolerance=tolerance, synclass="Generic_CS")
    elec_results = verify_paper_counts(elec_snapshots, tolerance=tolerance, synclass="Generic_GJ")

    combined = {}
    for i in range(len(chem_snapshots)):
        hour = REAL_HOURS[i]
        combined[i] = {
            "hour": hour,
            "chemical": chem_results[i],
            "electrical": elec_results[i],
            "both_ok": chem_results[i]["edge_ok"] and elec_results[i]["edge_ok"] and chem_results[i]["neuron_ok"],
        }
    return combined
