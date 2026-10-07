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
    1. Pairwise graph-to-graph structural distance between all snapshots.
    2. Build an 8-node meta-graph from distances / affinities.
    3. Infer ordering via diffusion/spectral seriation (Fiedler vector of L_sym).
    4. Spearman correlation vs. real hours (H2 validation).

Hard rules (THE_BOOK §Part II & §Part III):
  - infer_trajectory_A2() must NOT accept real_hours as an argument.
    It is blind to timestamps by design.
  - A1 and A2 are both reported — A1 is the working pipeline;
    A2's H2 validation is a standalone finding.
"""

from typing import Callable, Sequence
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr


def interpolate_A1(
    snapshots: Sequence[torch.Tensor | np.ndarray],
    real_hours: Sequence[int | float],
) -> Callable[[float | torch.Tensor], torch.Tensor]:
    """
    Module A1: Direct monotonic trajectory interpolation.

    Constructs a continuous-time developmental axis s in [min(real_hours), max(real_hours)]
    via Piecewise Cubic Hermite Interpolating Polynomials (PCHIP).

    Design specifications (THE_BOOK §Module A):
      - Replicate snapshots sharing identical timestamps (e.g. 45h Replicates A & B)
        are averaged to produce a unique, well-defined target at that developmental stage.
      - PCHIP guarantees monotonicity between monotonic data points and eliminates
        spline overshoot.
      - Output adjacency values are clamped to [0.0, 1.0] and diagonal elements zeroed.

    Returns:
      A callable f(s) -> torch.Tensor of shape (N, N) representing the continuous
      interpolated connectome adjacency matrix at continuous developmental time s.
    """
    if len(snapshots) != len(real_hours):
        raise ValueError(
            f"Snapshot count ({len(snapshots)}) must match real_hours count ({len(real_hours)})."
        )

    # Convert all snapshots to numpy arrays
    np_snaps = [
        s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s, dtype=np.float32)
        for s in snapshots
    ]

    # Group by unique developmental timestamps (e.g. 0, 5, 8, 16, 23, 27, 45)
    unique_hours = sorted(list(set(real_hours)))
    h_to_snaps: dict[float, list[np.ndarray]] = {h: [] for h in unique_hours}
    for snap, h in zip(np_snaps, real_hours):
        h_to_snaps[h].append(snap)

    # Average duplicates (e.g. YA_45h_A and YA_45h_B) for the interpolation target
    averaged_snapshots = []
    for h in unique_hours:
        mats = h_to_snaps[h]
        avg_mat = np.mean(mats, axis=0).astype(np.float32)
        averaged_snapshots.append(avg_mat)

    # Shape: (T_unique, N, N)
    traj_tensor = np.stack(averaged_snapshots, axis=0)

    # Fit PCHIP interpolator along axis 0 (developmental time)
    pchip = PchipInterpolator(x=unique_hours, y=traj_tensor, axis=0)

    t_min = float(unique_hours[0])
    t_max = float(unique_hours[-1])

    def trajectory_fn(s: float | torch.Tensor) -> torch.Tensor:
        if isinstance(s, torch.Tensor):
            s_val = float(s.detach().cpu().item())
        else:
            s_val = float(s)

        # Clamp s to observed range
        s_clamped = np.clip(s_val, t_min, t_max)
        interp_adj = pchip(s_clamped)

        # Monotonicity & valid probability bounds: clamp to [0.0, 1.0]
        interp_adj = np.clip(interp_adj, 0.0, 1.0)

        # Ensure no self-loops
        np.fill_diagonal(interp_adj, 0.0)

        return torch.from_numpy(interp_adj).float()

    return trajectory_fn


def infer_trajectory_A2(
    snapshots: Sequence[torch.Tensor | np.ndarray],
) -> np.ndarray:
    """
    Module A2: Blind structural trajectory inference.

    CRITICAL RULE (THE_BOOK §Hard Rules & §Module A):
      This function must NOT accept timestamps, hours, or stage metadata.
      It infers developmental progression purely from graph structural topology.

    Methodology:
      1. Extract a topological feature vector per snapshot that captures the
         *shape* of the wiring, not just how many connections exist:
           - Normalized in/out degree distributions (per-node share of total weight)
           - Reciprocity ratio (fraction of edges that are bidirectional)
           - Degree entropy (Shannon entropy of in/out degree distributions)
           - Clustering tendency (triangle density relative to 2-paths)
         NOTE: raw total edge count is intentionally NOT included as a standalone
         feature. All degree-based features are normalized by the snapshot's own
         total, so the feature space is scale-invariant. This prevents the method
         from trivially sorting by "more edges = later" (that is not topology).
      2. Construct a KxK meta-graph of pairwise Euclidean distances between
         these topological feature vectors.
      3. Compute a Gaussian affinity matrix W from those distances.
      4. Perform spectral seriation via the Fiedler vector (2nd smallest eigenvector
         of the normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}).
      5. Orient the 1D trajectory using network density (normalized edge count)
         as a biological anchor for synaptogenesis direction — denser networks
         are later developmental stages.

    Returns:
      inferred_order: 1D np.ndarray of shape (K,) with inferred pseudotime
      coordinates for each snapshot.
    """
    K = len(snapshots)
    if K < 2:
        raise ValueError("At least 2 snapshots required for trajectory inference.")

    mats = [
        s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s, dtype=np.float32)
        for s in snapshots
    ]

    # ------------------------------------------------------------------
    # Topological feature extraction — scale-invariant, NOT raw edge count
    # ------------------------------------------------------------------
    feature_vectors = []
    for m in mats:
        total = float(m.sum()) + 1e-8
        n = m.shape[0]

        # Normalized degree distributions (each neuron's share of total weight)
        # Scale-invariant: dividing by total removes the "more edges = later" signal
        norm_in  = m.sum(axis=0) / total   # shape (N,)
        norm_out = m.sum(axis=1) / total   # shape (N,)

        # Reciprocity: fraction of edges that are bidirectional (mutual)
        # Captures the directed structure of the wiring, not its volume
        reciprocity = float((m * m.T).sum()) / (2.0 * total)

        # Degree entropy: Shannon entropy of the normalized degree distribution
        # High entropy = many neurons share connections; low = hub-dominated
        # This captures the *shape* of the degree distribution
        in_prob  = norm_in  / (norm_in.sum()  + 1e-12)
        out_prob = norm_out / (norm_out.sum() + 1e-12)
        in_entropy  = float(-np.sum(in_prob  * np.log(in_prob  + 1e-12)))
        out_entropy = float(-np.sum(out_prob * np.log(out_prob + 1e-12)))

        # Clustering tendency: triangles relative to 2-paths
        # A^2 diagonal counts paths of length 2 from i back to i (via a common neighbour)
        # A^3 diagonal counts closed triangles rooted at each node
        A2 = m @ m                                       # shape (N, N)
        triangles  = float(np.trace(A2 @ m))             # sum of diagonal of A^3
        two_paths  = float(A2.sum() - np.trace(A2))     # off-diagonal sum of A^2
        clustering = triangles / (two_paths + 1e-8)

        vec = np.concatenate([
            norm_in,                                      # N dims
            norm_out,                                     # N dims
            [reciprocity, in_entropy, out_entropy, clustering],   # 4 structural scalars
        ])
        feature_vectors.append(vec)

    feat_matrix = np.array(feature_vectors, dtype=np.float32)

    # Compute pairwise Euclidean distance matrix D
    D = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            D[i, j] = float(np.linalg.norm(feat_matrix[i] - feat_matrix[j]))

    # Compute similarity affinity matrix W via adaptive Gaussian bandwidth
    non_zero_dists = D[D > 0]
    sigma = float(np.median(non_zero_dists)) if len(non_zero_dists) > 0 else 1.0
    W = np.exp(-(D ** 2) / (2.0 * (sigma ** 2) + 1e-8))
    np.fill_diagonal(W, 0.0)

    # Normalized symmetric graph Laplacian: L_sym = I - D_deg^{-1/2} W D_deg^{-1/2}
    degrees = np.sum(W, axis=1)
    deg_inv_sqrt = np.power(degrees + 1e-8, -0.5)
    L_sym = np.eye(K, dtype=np.float32) - np.outer(deg_inv_sqrt, deg_inv_sqrt) * W

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(L_sym)

    # The Fiedler vector is eigenvector 1 (corresponding to 2nd smallest eigenvalue)
    fiedler_vec = eigvecs[:, 1].copy()

    # Biological orientation: developmental synaptogenesis increases network density.
    # Orient the Fiedler vector so that denser networks score higher.
    # Using density (edges / possible_edges) rather than raw total_edges so that
    # the anchor is a structural property, not absolute size.
    n_nodes = mats[0].shape[0]
    max_edges = float(n_nodes * (n_nodes - 1))
    densities = np.array([m.sum() / max_edges for m in mats])
    correlation = np.corrcoef(fiedler_vec, densities)[0, 1]
    if correlation < 0:
        fiedler_vec = -fiedler_vec

    return fiedler_vec


def validate_h2_trajectory(
    inferred_order: np.ndarray | Sequence[float],
    real_hours: Sequence[int | float],
) -> tuple[float, float]:
    """
    Validate Hypothesis H2: compute Spearman rank correlation between blind inferred
    structural pseudotime and real biological hours.

    Returns:
      (spearman_rho, p_value)
    """
    res = spearmanr(inferred_order, real_hours)
    return float(res.statistic), float(res.pvalue)

