"""
conftest.py — Shared pytest fixtures for all modules.
Defined once here; imported automatically by pytest in every test file.

Design notes (THE_BOOK Part V):
- N_NEURONS=20 keeps tests fast without being trivially small.
- REAL_HOURS matches the 8 Witvliet snapshots exactly; two 45h entries are
  intentional (biological replicates, NOT sequential timepoints).
- injected_motif_snapshots embeds a known 4-node clique every other snapshot
  so Module C's recovery can be verified on synthetic data before real data.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_NEURONS    = 20
K_TIMEPOINTS = 8
REAL_HOURS   = [0, 5, 8, 16, 23, 27, 45, 45]


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_snapshots():
    """
    8 adjacency matrices (N×N float) with slow temporal drift.
    Probability of each edge increases slightly over time.
    No temporal leakage possible — snapshots are independent draws.
    """
    torch.manual_seed(0)
    snapshots = []
    base_prob = torch.rand(N_NEURONS, N_NEURONS) * 0.1
    for t in range(K_TIMEPOINTS):
        drift = 0.02 * t
        prob  = torch.clamp(base_prob + drift * torch.rand(N_NEURONS, N_NEURONS), 0, 1)
        adj   = (torch.rand(N_NEURONS, N_NEURONS) < prob).float()
        adj.fill_diagonal_(0)          # no self-loops
        snapshots.append(adj)
    return snapshots


@pytest.fixture
def real_hours():
    """Real collection timestamps from Witvliet et al. (2021)."""
    return list(REAL_HOURS)  # return a copy so tests can't mutate the global


@pytest.fixture
def injected_motif_snapshots():
    """
    A fixed 4-node clique (neurons 0-3) appears in every EVEN snapshot.
    Used to test that Module C can recover a known, injected motif before
    touching real data (Phase 4 gate — THE_BOOK §Phase 4).
    If Module C fails this, do not proceed to real data.
    """
    torch.manual_seed(1)
    snapshots = []
    for t in range(K_TIMEPOINTS):
        adj = (torch.rand(N_NEURONS, N_NEURONS) < 0.05).float()
        adj.fill_diagonal_(0)
        if t % 2 == 0:                          # inject clique at even steps
            adj[0:4, 0:4] = 1.0
            adj[0:4, 0:4].fill_diagonal_(0)     # still no self-loops
        snapshots.append(adj)
    return snapshots
