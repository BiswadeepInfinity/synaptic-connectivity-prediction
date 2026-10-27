"""
scripts/validate_module_c_stability.py — Rigorous 5-Seed Stability & Degree Confound Analysis
Using Canonical Module B Embeddings.

Addresses:
  (1) Save/load canonical trained Module C model on Module B embeddings.
  (2) Train with 5 seeds, match slots across seeds by cosine similarity of slot dictionary vectors,
      and compute stability of neuron-to-slot assignments.
  (3) Re-run the degree-confound test on the canonical model (comparing slot membership vs degree in each window).
  (4) Show full neuron lists (and cell classes) for the AVA/AVB/AVE slots without alphabetical truncation.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.loader import load_all_snapshots_both
from cect.readers.WitvlietDataReader1 import get_instance as get1
from cect.readers.WitvlietDataReader8 import get_instance as get8
from pcdgm.module_b_encoder import StructuralEncoder
from pcdgm.module_c_motifs import MotifDictionary

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CANONICAL_MODEL_PATH = RESULTS_DIR / "canonical_module_c_model.pt"

SEEDS = [42, 12345, 123, 456, 789]
WINDOW_LABELS = [
    "W0 (0-8h)", "W1 (5-16h)", "W2 (8-23h)", "W3 (16-27h)", "W4 (23-45hA)", "W5 (27-45hB)"
]

def get_node_metadata():
    from cect.readers.WitvlietDataReader2 import get_instance as get2
    from cect.readers.WitvlietDataReader3 import get_instance as get3
    from cect.readers.WitvlietDataReader4 import get_instance as get4
    from cect.readers.WitvlietDataReader5 import get_instance as get5
    from cect.readers.WitvlietDataReader6 import get_instance as get6
    from cect.readers.WitvlietDataReader7 import get_instance as get7
    readers = [get1(), get2(), get3(), get4(), get5(), get6(), get7(), get8()]
    all_nodes_set = set()
    for r in readers:
        all_nodes_set.update(r.to_networkx_graph("Generic_CS").nodes())
        all_nodes_set.update(r.to_networkx_graph("Generic_GJ").nodes())
    all_nodes = sorted(list(all_nodes_set))
    return all_nodes

def train_module_c(windows_H, seed: int, K: int = 10, dim: int = 64, k: int = 3, epochs: int = 100):
    torch.manual_seed(seed)
    model = MotifDictionary(K=K, dim=dim, window=k, mode="learned")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        optimizer.zero_grad()
        all_a = []
        tot = 0.0
        for w in windows_H:
            p = model._pool_window(w)
            z, a, _ = model(w)
            recon = model.decoder(z)
            tot = tot + F.mse_loss(recon, p)
            all_a.append(a)
        a_b = torch.stack(all_a, dim=0)
        p_m = a_b.mean(dim=0)
        l_div = -torch.sum(p_m * torch.log(p_m + 1e-9))
        loss = (tot / len(windows_H)) - 0.05 * l_div
        loss.backward()
        optimizer.step()
    return model

def compute_node_slot_assignments(model, windows_H):
    """
    Computes per-node slot assignment for each window.
    windows_H is list of 6 tensors [k=3, N=225, dim=64].
    Returns: Tensor [num_windows, N] of assigned slot indices (0..K-1).
    """
    model.eval()
    with torch.no_grad():
        window_assignments = []
        D_norm = F.normalize(model.D, dim=-1) # [K, dim]
        for w in windows_H:
            # Average node embedding across the 3 snapshots in window
            h_node_win = w.mean(dim=0) # [N, dim]
            h_norm = F.normalize(h_node_win, dim=-1)
            sims = torch.matmul(h_norm, D_norm.t()) # [N, K] cosine similarity
            assigned = sims.argmax(dim=-1) # [N]
            window_assignments.append(assigned)
        return torch.stack(window_assignments, dim=0) # [num_windows, N]

def match_slots_hungarian(D_ref, D_target):
    """
    Matches slots of D_target to D_ref using cosine similarity and Hungarian algorithm.
    Returns: permutation mapping target slot -> ref slot
    """
    ref_norm = F.normalize(D_ref, dim=-1)
    target_norm = F.normalize(D_target, dim=-1)
    cos_sim = torch.matmul(ref_norm, target_norm.t()).detach().cpu().numpy() # [K, K]
    cost_matrix = -cos_sim
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # col_ind[i] in target maps to row_ind[i] in ref
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return mapping

def main():
    print("=" * 85)
    print("  RIGOROUS STABILITY & DEGREE CONFOUND AUDIT FOR MODULE C (H3 FOUNDATION)")
    print("=" * 85)

    all_nodes = get_node_metadata()
    N = len(all_nodes)
    n2i = {n: i for i, n in enumerate(all_nodes)}
    print(f"Total aligned neurons across 8 snapshots: {N}")

    # Load canonical chemical snapshots
    chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)
    k = 3

    # Generate Module B structural representations
    print("\n[Step 1] Generating Module B structural representations (normalized topological features)...")
    torch.manual_seed(42)
    encoder = StructuralEncoder()
    encoder.eval()
    with torch.no_grad():
        H_snaps = [encoder(s) for s in chem_snaps] # 8 tensors of [N, 64]

    windows_H = [torch.stack(H_snaps[i:i+k], dim=0) for i in range(len(H_snaps) - k + 1)] # 6 windows of [3, N, 64]
    windows_adj = [chem_snaps[i:i+k] for i in range(len(chem_snaps) - k + 1)]

    # Train canonical model (Seed 42) and save
    print(f"\n[Step 2] Training canonical model (Seed 42) and saving to {CANONICAL_MODEL_PATH}...")
    canonical_model = train_module_c(windows_H, seed=42)
    torch.save(canonical_model.state_dict(), CANONICAL_MODEL_PATH)
    print("Canonical model saved successfully.")

    # Multi-seed training and slot alignment
    print(f"\n[Step 3] Training across {len(SEEDS)} seeds: {SEEDS} and matching slots...")
    models = {}
    aligned_assignments = {} # seed -> [num_windows, N]

    canonical_assign = compute_node_slot_assignments(canonical_model, windows_H).numpy()

    for seed in SEEDS:
        if seed == 42:
            models[seed] = canonical_model
            aligned_assignments[seed] = canonical_assign
        else:
            m = train_module_c(windows_H, seed=seed)
            models[seed] = m
            # Align slots to canonical model (Seed 42)
            slot_map = match_slots_hungarian(canonical_model.D, m.D)
            raw_assign = compute_node_slot_assignments(m, windows_H).numpy() # [W, N]
            aligned_assign = np.vectorize(slot_map.get)(raw_assign)
            aligned_assignments[seed] = aligned_assign

    # Compute neuron stability across seeds
    # For each window and neuron, what fraction of seeds agree on the assigned slot?
    print("\n--- NEURON-TO-SLOT ASSIGNMENT STABILITY ACROSS 5 SEEDS ---")
    
    # Measure overall stability
    stability_matrix = np.zeros((len(windows_H), N)) # [W, N]
    majority_slot_matrix = np.zeros((len(windows_H), N), dtype=int)

    for w_idx in range(len(windows_H)):
        for i in range(N):
            slots_across_seeds = [aligned_assignments[s][w_idx, i] for s in SEEDS]
            counts = Counter(slots_across_seeds)
            most_common_slot, top_count = counts.most_common(1)[0]
            stability_matrix[w_idx, i] = top_count / len(SEEDS)
            majority_slot_matrix[w_idx, i] = most_common_slot

    mean_stability_per_win = stability_matrix.mean(axis=1)
    for w_idx, (wl, stab) in enumerate(zip(WINDOW_LABELS, mean_stability_per_win)):
        pct_perfect = (stability_matrix[w_idx] == 1.0).mean() * 100
        pct_majority = (stability_matrix[w_idx] >= 0.8).mean() * 100
        print(f"  {wl:<14}: Mean agreement = {stab*100:5.1f}% | >=80% stable: {pct_majority:5.1f}% | 100% stable: {pct_perfect:5.1f}%")

    # Command circuit neurons tracking
    command_circuit = ['AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVEL', 'AVER', 'PVCL', 'PVCR']
    print("\nStability of Key Command Neurons across Seeds (Window 0):")
    for n in command_circuit:
        idx = n2i[n]
        slots = [aligned_assignments[s][0, idx] for s in SEEDS]
        canon_slot = canonical_assign[0, idx]
        print(f"  {n:<6}: Canonical Slot = {canon_slot:>2d} | Across Seeds = {slots} (Agreement: {stability_matrix[0, idx]*100:.0f}%)")

    # -------------------------------------------------------------------------
    # Re-run Degree-Confound Test on Canonical Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("--- DEGREE-CONFOUND TEST ON CANONICAL MODULE B/C MODEL ---")
    print("=" * 85)
    print(f"{'Window':<14} | {'Target Slots':<14} | {'Tgt Mean Deg':<12} | {'Other Mean Deg':<15} | {'Tgt Hubs/Top20':<16} | {'Mann-Whitney p'}")
    print("-" * 90)

    for w_idx, (w_adj, wl) in enumerate(zip(windows_adj, WINDOW_LABELS)):
        # Compute exact node degrees within this window
        w_degs = np.zeros(N)
        for s in w_adj:
            w_degs += s.numpy().sum(axis=-1) # out-degree

        assign_w = canonical_assign[w_idx]
        
        # Where do AVA/AVB/AVE sit in this window?
        key_slots = sorted(list(set(assign_w[n2i[n]] for n in command_circuit)))
        is_target_slot = np.isin(assign_w, key_slots)

        deg_target = w_degs[is_target_slot]
        deg_other = w_degs[~is_target_slot]

        top20_idx = np.argsort(w_degs)[::-1][:20]
        top20_in_target = sum(1 for idx in top20_idx if is_target_slot[idx])

        stat, mwu_p = mannwhitneyu(deg_target, deg_other, alternative="greater")
        
        slots_str = ",".join(map(str, key_slots))
        print(
            f"{wl:<14} | "
            f"Slots {slots_str:<8} | "
            f"{np.mean(deg_target):<12.2f} | "
            f"{np.mean(deg_other):<15.2f} | "
            f"{top20_in_target:>2d} / 20 ({top20_in_target*5:>2d}%)    | "
            f"{mwu_p:.4e}"
        )

    # -------------------------------------------------------------------------
    # Full Neuron Lists for Slots Hosting AVA / AVB / AVE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("--- FULL NEURON LISTS FOR AVA/AVB/AVE SLOTS (WINDOW 0, NO TRUNCATION) ---")
    print("=" * 85)

    assign_w0 = canonical_assign[0]
    ava_slots = sorted(list(set(assign_w0[n2i[n]] for n in command_circuit)))

    for slot in ava_slots:
        members = [all_nodes[i] for i in range(N) if assign_w0[i] == slot]
        hubs_in_slot = [n for n in members if n in [all_nodes[idx] for idx in np.argsort(w_degs)[::-1][:20]]]
        print(f"\nSLOT {slot} (Total Members: {len(members)}):")
        print(f"  Command Circuit Present: {[n for n in command_circuit if n in members]}")
        print(f"  Top-20 Hubs in this Slot: {hubs_in_slot}")
        print(f"  Complete Neuron List ({len(members)}):")
        # Print wrapped in lines of 10
        for chunk in [members[i:i+10] for i in range(0, len(members), 10)]:
            print("   ", ", ".join(chunk))

if __name__ == "__main__":
    main()
