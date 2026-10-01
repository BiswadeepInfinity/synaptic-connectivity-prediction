"""
pcdgm/module_a_trajectory.py — Phase 3: Trajectory Reconstruction.

A1 (primary):
  Direct interpolation using real known timestamps (PCHIP — monotonic,
  no overshoot). The two 45h replicates are averaged for the interpolation
  target; both are kept as separate eval samples at test time.

A2 (secondary, novelty-relevant):
  Blind structural trajectory inference — infer ordering from graph
  structure alone, validated against real hours (H2).
  Steps:
    1. Pairwise graph-to-graph distance (WL kernel or GNN-embedding L2).
    2. Build 8-node meta-graph from distances.
    3. Infer ordering via diffusion/random-walk.
    4. Spearman correlation vs. real hours (H2 validation).

TODO (Phase 3): implement interpolate_A1() and infer_trajectory_A2().

Hard rules:
  - infer_trajectory_A2() must NOT accept real_hours as an argument.
    It is blind to timestamps by design.
  - A1 and A2 are both reported — A1 is the working pipeline;
    A2's H2 validation is a standalone finding.
"""

# Phase 3 — NOT YET IMPLEMENTED


def interpolate_A1(snapshots, real_hours):
    """
    Return a callable f(s) → adjacency tensor at continuous time s.
    Uses PCHIP monotonic interpolation over the 8 real snapshots.

    TODO (Phase 3): implement.
    """
    raise NotImplementedError("Module A1 (Phase 3) not yet implemented.")


def infer_trajectory_A2(snapshots):
    """
    Infer developmental ordering from graph structure only.
    NOTE: does NOT accept real_hours — blind by design (H2 test).

    Returns: list of inferred ordering indices (length == len(snapshots)).

    TODO (Phase 3): implement WL-kernel distance + diffusion ordering.
    """
    raise NotImplementedError("Module A2 (Phase 3) not yet implemented.")
