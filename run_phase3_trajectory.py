"""
run_phase3_trajectory.py — Phase 3 Trajectory Reconstruction & H2 Validation Runner.

Demonstrates:
  1. Module A1: Continuous developmental trajectory interpolation (PCHIP).
  2. Module A2: Blind structural trajectory inference.
  3. Hypothesis H2 Validation: Spearman rank correlation against real developmental hours.

Display correctness notes (bugs fixed):
  - Ground-truth counts are read directly from raw loaded snapshots, NOT from
    the PCHIP interpolator. Routing through PCHIP adds spline round-trip error
    that makes binary values drift below a >0.5 threshold, silently under-counting
    edges (775 instead of 858, 1469 instead of ~2495).
  - Both 45h replicates are shown as separate rows. The old loop over a 6-element
    display list caused both to collapse into one "45.0" row — they were never
    merged in the actual pipeline, only hidden in the display.
  - Interpolated intermediate points report mean adjacency density (a continuous
    value), not a thresholded edge count. Threshold on a soft PCHIP matrix is
    not meaningful.
"""

import numpy as np
import torch
from data.loader import load_all_snapshots_both, REAL_HOURS
from pcdgm.module_a_trajectory import interpolate_A1, infer_trajectory_A2, validate_h2_trajectory


def main():
    print("=" * 75)
    print("  PHASE 3: CONTINUOUS TRAJECTORY RECONSTRUCTION & H2 VALIDATION")
    print("  PCDGM Module A | C. elegans Developmental Connectome (Witvliet 2021)")
    print("=" * 75)
    print()

    print("Loading 8 Witvliet developmental connectome snapshots (Chemical & Electrical)...")
    chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)
    N = chem_snaps[0].shape[0]
    print(f"Loaded {len(chem_snaps)} snapshots aligned to {N} identified neurons.")
    print(f"Real developmental timestamps (hours): {hours}")
    print()

    # -----------------------------------------------------------------
    # 1. MODULE A1: Direct Continuous Monotonic Interpolation
    # -----------------------------------------------------------------
    print("-" * 75)
    print("1. MODULE A1: CONTINUOUS DEVELOPMENTAL AXIS (PCHIP INTERPOLATION)")
    print("-" * 75)
    interp_fn_chem = interpolate_A1(chem_snaps, hours)
    interp_fn_elec = interpolate_A1(elec_snaps, hours)

    # 1a. Ground-truth snapshots — chemical and electrical displayed separately
    print("  [Ground Truth — raw loaded snapshots (Witvliet 2021 Table 1)]")
    print(f"  {'Snapshot':<10} {'Label':<14} {'Hour':<6} {'Chem Edges':<12} {'Chem Dens':<12} {'Elec (dir)':<12} {'Elec Dens'}")
    print("  " + "-" * 75)
    for i, (cs, es, h, label) in enumerate(zip(chem_snaps, elec_snaps, hours, labels)):
        c_edges = int(cs.sum().item())
        c_dens  = float(cs.mean().item())
        e_edges = int(es.sum().item())
        e_dens  = float(es.mean().item())
        print(f"  S{i + 1:<9} {label:<14} {h:<6} {c_edges:<12} {c_dens:<12.4f} {e_edges:<12} {e_dens:.4f}")
    print()

    # 1b. PCHIP interpolated intermediate timepoints.
    print("  [PCHIP Interpolated — intermediate timepoints not in the dataset]")
    print(f"  {'Hour (h)':<10} {'Status':<15} {'Chem Mean Dens':<18} {'Elec Mean Dens'}")
    print("  " + "-" * 60)
    for h in [2.5, 11.5, 20.0, 33.0]:
        c_adj = interp_fn_chem(h)
        e_adj = interp_fn_elec(h)
        print(f"  {h:<10.1f} {'Interpolated':<15} {float(c_adj.mean()):<18.4f} {float(e_adj.mean()):.4f}")
    print()

    # -----------------------------------------------------------------
    # 2. MODULE A2: Blind Structural Inference & H2 Validation
    # -----------------------------------------------------------------
    print("-" * 75)
    print("2. MODULE A2: BLIND STRUCTURAL TRAJECTORY INFERENCE (TESTING H2)")
    print("   RULE: Blind to timestamps — infers order from graph topology alone.")
    print("   Evaluated separately on Chemical, Electrical, and Joint Multi-Relational graphs.")
    print("-" * 75)

    # Chemical
    inferred_chem = infer_trajectory_A2(chem_snaps)
    rho_c, p_c = validate_h2_trajectory(inferred_chem, hours)

    # Electrical
    inferred_elec = infer_trajectory_A2(elec_snaps)
    rho_e, p_e = validate_h2_trajectory(inferred_elec, hours)

    print(f"  [Chemical Synapses alone]         rho = {rho_c:.4f} (p = {p_c:.4e}) -> {'PASSED' if rho_c > 0.3 else 'FAILED'}")
    print(f"  [Electrical Gap Junctions alone]  rho = {rho_e:.4f} (p = {p_e:.4e}) -> {'PASSED' if rho_e > 0.3 else 'FAILED'}")

    # Joint Multi-relational representation (features extracted independently and concatenated)
    K = len(chem_snaps)
    feat_vectors = []
    for c_mat, e_mat in zip(chem_snaps, elec_snaps):
        cm = c_mat.numpy() if hasattr(c_mat, 'numpy') else c_mat
        em = e_mat.numpy() if hasattr(e_mat, 'numpy') else e_mat
        c_tot = cm.sum() + 1e-8
        c_norm_in = cm.sum(axis=0) / c_tot
        c_norm_out = cm.sum(axis=1) / c_tot
        c_recip = float((cm * cm.T).sum()) / (2.0 * c_tot)
        c_in_prob = c_norm_in / (c_norm_in.sum() + 1e-12)
        c_out_prob = c_norm_out / (c_norm_out.sum() + 1e-12)
        c_in_ent = float(-np.sum(c_in_prob * np.log(c_in_prob + 1e-12)))
        c_out_ent = float(-np.sum(c_out_prob * np.log(c_out_prob + 1e-12)))
        A2_c = cm @ cm
        c_tri = float(np.trace(A2_c @ cm))
        c_tp2 = float(A2_c.sum() - np.trace(A2_c))
        c_clust = c_tri / (c_tp2 + 1e-8)

        e_tot = em.sum() + 1e-8
        e_norm_deg = em.sum(axis=0) / e_tot
        e_prob = e_norm_deg / (e_norm_deg.sum() + 1e-12)
        e_ent = float(-np.sum(e_prob * np.log(e_prob + 1e-12)))
        A2_e = em @ em
        e_tri = float(np.trace(A2_e @ em))
        e_tp2 = float(A2_e.sum() - np.trace(A2_e))
        e_clust = e_tri / (e_tp2 + 1e-8)

        vec = np.concatenate([
            c_norm_in, c_norm_out, [c_recip, c_in_ent, c_out_ent, c_clust],
            e_norm_deg, [e_ent, e_clust],
        ])
        feat_vectors.append(vec)

    feat_matrix = np.array(feat_vectors, dtype=np.float32)
    D = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            D[i, j] = float(np.linalg.norm(feat_matrix[i] - feat_matrix[j]))
    non_zero = D[D > 0]
    sigma = float(np.median(non_zero)) if len(non_zero) > 0 else 1.0
    W = np.exp(-(D**2) / (2.0 * (sigma**2) + 1e-8))
    np.fill_diagonal(W, 0.0)
    degrees = np.sum(W, axis=1)
    deg_inv_sqrt = np.power(degrees + 1e-8, -0.5)
    L_sym = np.eye(K, dtype=np.float32) - np.outer(deg_inv_sqrt, deg_inv_sqrt) * W
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    fiedler_joint = eigvecs[:, 1].copy()
    densities = np.array([(c.sum() + e.sum()) / (N * (N - 1)) for c, e in zip(chem_snaps, elec_snaps)])
    if np.corrcoef(fiedler_joint, densities)[0, 1] < 0:
        fiedler_joint = -fiedler_joint

    rho_joint, p_joint = validate_h2_trajectory(fiedler_joint, hours)
    print(f"  [Joint Multi-Relational (Chem+Elec)] rho = {rho_joint:.4f} (p = {p_joint:.4e}) -> {'PASSED' if rho_joint > 0.3 else 'FAILED'}")
    print()

    print("  Snapshot Rankings (Joint Multi-Relational Model):")
    print(f"  {'Snapshot':<10} {'Label':<14} {'True Hour':<10} {'Inferred Score':<20} {'Rank'}")
    print("  " + "-" * 68)
    sorted_idx = np.argsort(fiedler_joint)
    rank_of_snap = np.empty_like(sorted_idx)
    rank_of_snap[sorted_idx] = np.arange(1, len(chem_snaps) + 1)

    for i, (l, h, sc, rk) in enumerate(zip(labels, hours, fiedler_joint, rank_of_snap)):
        tag = " [Rep A]" if l == "YA_45h_A" else (" [Rep B]" if l == "YA_45h_B" else "")
        print(f"  S{i + 1:<9} {l:<14} {h:<10} {sc:<+20.4f} Rank #{rk}{tag}")

    print("=" * 75)


if __name__ == "__main__":
    main()
