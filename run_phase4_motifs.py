"""
run_phase4_motifs.py — Phase 4 Structural Motif Discovery & K-Sweep Benchmark.

Executes and reports:
  1. Phase 4 Gate: Synthetic injected-motif sanity check (clique recovery).
  2. Real Witvliet connectome application (Chemical & Electrical separated).
  3. K-Sweep across K in {5, 10, 20, 50} reporting:
     - Initial MSE -> Final MSE (loss drop %)
     - Empirical slot usage entropy H(p) vs max theoretical ln(K)
     - Effective number of slots utilized (>5% attention mass)
  4. Temporal attention routing table across the 6 chronological developmental windows (k=3).
"""

from collections import Counter
import math
import torch
import torch.nn as nn
import torch.optim as optim
from data.loader import load_all_snapshots_both
from pcdgm.module_c_motifs import MotifDictionary


def run_synthetic_sanity_check():
    print("=" * 80)
    print("  PHASE 4 GATE: SYNTHETIC INJECTED MOTIF RECOVERY SANITY CHECK")
    print("=" * 80)
    print("  Protocol: Injecting 4-node dense clique into alternating sequence windows (k=3).")
    print("  Verifying model can detect and recover it before trusting real brain data.")
    print("-" * 80)

    torch.manual_seed(42)
    N = 20
    dim = 64
    k = 3
    num_windows = 40

    data = []
    labels = []
    for i in range(num_windows):
        is_motif = (i % 2 == 0)
        window = []
        for _ in range(k):
            adj = (torch.rand(N, N) < 0.05).float()
            adj.fill_diagonal_(0)
            if is_motif:
                adj[0:4, 0:4] = 1.0
                adj[0:4, 0:4].fill_diagonal_(0)
            window.append(adj)
        data.append(window)
        labels.append(1 if is_motif else 0)

    model = MotifDictionary(K=8, dim=dim, window=k, mode="learned")
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    initial_mse = 0.0
    with torch.no_grad():
        for w in data:
            p = model._pool_window(w)
            z, _, _ = model(w)
            initial_mse += nn.functional.mse_loss(model.decoder(z), p).item()
    initial_mse /= len(data)

    for _ in range(50):
        optimizer.zero_grad()
        all_a = []
        tot = 0.0
        for w in data:
            p = model._pool_window(w)
            z, a, _ = model(w)
            recon = model.decoder(z)
            tot = tot + nn.functional.mse_loss(recon, p)
            all_a.append(a)
        a_b = torch.stack(all_a, dim=0)
        p_m = a_b.mean(dim=0)
        l_div = -torch.sum(p_m * torch.log(p_m + 1e-9))
        loss = (tot / len(data)) - 0.05 * l_div
        loss.backward()
        optimizer.step()

    final_mse = 0.0
    with torch.no_grad():
        for w in data:
            p = model._pool_window(w)
            z, _, _ = model(w)
            final_mse += nn.functional.mse_loss(model.decoder(z), p).item()
    final_mse /= len(data)

    with torch.no_grad():
        a_b = torch.stack([model(w)[1] for w in data], dim=0)
        entropy = model.slot_usage_entropy(a_b)

    motif_slots = [model(w)[1].argmax().item() for w, m in zip(data, labels) if m == 1]
    motif_counts = Counter(motif_slots)
    dominant_slot, dominant_count = motif_counts.most_common(1)[0]
    recovery_rate = dominant_count / len(motif_slots)

    print(f"  Initial Reconstruction MSE : {initial_mse:.5f}")
    print(f"  Final Reconstruction MSE   : {final_mse:.5f}  (Drop: {(1 - final_mse/initial_mse)*100:.1f}%)")
    print(f"  Slot Usage Entropy H(p)    : {entropy:.4f}  (Theoretical max: {math.log(8):.4f})")
    print(f"  Injected Motif Dominant Slot: Slot #{dominant_slot} ({dominant_count}/{len(motif_slots)} = {recovery_rate*100:.1f}%)")
    print(f"  GATE STATUS                : {'PASSED (Ready for Real Data)' if recovery_rate >= 0.90 else 'FAILED'}")
    print()


def run_real_witvliet_k_sweep():
    print("=" * 80)
    print("  PHASE 4: REAL WITVLIET CONNECTOME MOTIF DISCOVERY (K-SWEEP)")
    print("=" * 80)
    print("  Loading 8 Witvliet snapshots (Chemical & Electrical separated)...")
    chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)

    k = 3
    chem_windows = [chem_snaps[i:i+k] for i in range(len(chem_snaps) - k + 1)]
    elec_windows = [elec_snaps[i:i+k] for i in range(len(elec_snaps) - k + 1)]

    window_labels = [
        f"W0 (0-8h: S1-S3)",
        f"W1 (5-16h: S2-S4)",
        f"W2 (8-23h: S3-S5)",
        f"W3 (16-27h: S4-S6)",
        f"W4 (23-45hA: S5-S7)",
        f"W5 (27-45hB: S6-S8)",
    ]

    print(f"  Evaluated over {len(chem_windows)} chronological windows (window size k={k}):")
    for wl in window_labels:
        print(f"    - {wl}")
    print()

    # Table 1: Chemical Synapses K-Sweep
    _evaluate_synapse_channel("CHEMICAL SYNAPSES (Generic_CS)", chem_windows, k, window_labels)

    # Table 2: Electrical Synapses K-Sweep
    _evaluate_synapse_channel("ELECTRICAL SYNAPSES / GAP JUNCTIONS (Generic_GJ)", elec_windows, k, window_labels)


def _evaluate_synapse_channel(title: str, windows, k: int, window_labels):
    print("-" * 80)
    print(f"  TABLE: {title} — DICTIONARY CAPACITY (K) SWEEP")
    print("-" * 80)
    print(f"  {'K':<5} {'Init MSE':<11} {'Final MSE':<11} {'Loss Drop':<12} {'Entropy H':<12} {'Max ln(K)':<11} {'Entropy %':<11} {'Used Slots'}")
    print("  " + "-" * 78)

    models = {}
    for K in [5, 10, 20, 50]:
        torch.manual_seed(42)
        model = MotifDictionary(K=K, dim=64, window=k, mode="learned")
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        init_mse = 0.0
        with torch.no_grad():
            for w in windows:
                p = model._pool_window(w)
                z, _, _ = model(w)
                init_mse += nn.functional.mse_loss(model.decoder(z), p).item()
        init_mse /= len(windows)

        for _ in range(100):
            optimizer.zero_grad()
            all_a = []
            tot = 0.0
            for w in windows:
                p = model._pool_window(w)
                z, a, _ = model(w)
                recon = model.decoder(z)
                tot = tot + nn.functional.mse_loss(recon, p)
                all_a.append(a)
            a_b = torch.stack(all_a, dim=0)
            p_m = a_b.mean(dim=0)
            l_div = -torch.sum(p_m * torch.log(p_m + 1e-9))
            loss = (tot / len(windows)) - 0.05 * l_div
            loss.backward()
            optimizer.step()

        final_mse = 0.0
        with torch.no_grad():
            for w in windows:
                p = model._pool_window(w)
                z, _, _ = model(w)
                final_mse += nn.functional.mse_loss(model.decoder(z), p).item()
        final_mse /= len(windows)

        with torch.no_grad():
            a_b = torch.stack([model(w)[1] for w in windows], dim=0)
            ent = model.slot_usage_entropy(a_b)
            p_mean = a_b.mean(dim=0)
            used_slots = int((p_mean > 0.05).sum().item())

        max_ent = math.log(K)
        ent_ratio = (ent / max_ent) * 100
        drop_pct = (1.0 - final_mse / init_mse) * 100.0

        print(f"  {K:<5} {init_mse:<11.4f} {final_mse:<11.4f} {drop_pct:<11.1f}% {ent:<12.4f} {max_ent:<11.4f} {ent_ratio:<10.1f}% {used_slots}/{K}")
        models[K] = (model, a_b)
    print()

    # Show Developmental Attention Routing for recommended K=10
    print(f"  [DEVELOPMENTAL ATTENTION ROUTING — Primary Config K=10]")
    print(f"  {'Window':<22} {'Dominant Slot':<16} {'Top Slot Attention Weights'}")
    print("  " + "-" * 78)
    _, a_batch_10 = models[10]
    for i, wl in enumerate(window_labels):
        top_vals, top_inds = torch.topk(a_batch_10[i], 3)
        top_str = ", ".join([f"Slot {int(idx)}: {float(val):.3f}" for idx, val in zip(top_inds, top_vals)])
        dom = f"Slot #{int(top_inds[0])} ({float(top_vals[0])*100:.1f}%)"
        print(f"  {wl:<22} {dom:<16} {top_str}")
    print()


def main():
    run_synthetic_sanity_check()
    run_real_witvliet_k_sweep()


if __name__ == "__main__":
    main()
