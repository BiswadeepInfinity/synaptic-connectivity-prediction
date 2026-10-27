"""
scripts/run_attempts_gate.py — Evaluates Attempt A (tau=0.1) and Attempt B (tau=0.05) on ConnectomePCDGM
with neuron mean-centering and degree regression.
"""

import sys
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import load_all_snapshots_both
from baseline.tgn import build_connectome_temporal_data
from run_phase5_training import ConnectomePCDGM

def evaluate_attempts():
    chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)
    data, num_nodes = build_connectome_temporal_data()
    N = num_nodes
    k = 3
    K = 10
    dim = 64

    # Compute node degree per window (sum of degrees over the 3 snapshots in the window)
    windows_deg = []
    for i in range(len(chem_snaps) - k + 1):
        w_chem = chem_snaps[i:i+k]
        deg_w = np.zeros(N)
        for s in w_chem:
            deg_w += s.numpy().sum(axis=-1) + s.numpy().sum(axis=0) # total degree
        windows_deg.append(deg_w)

    WINDOW_LABELS = ['W0 (0-8h)', 'W1 (5-16h)', 'W2 (8-23h)', 'W3 (16-27h)', 'W4 (23-45hA)', 'W5 (27-45hB)']
    window_hour_sets = [
        [0, 5, 8],
        [5, 8, 16],
        [8, 16, 23],
        [16, 23, 27],
        [23, 27, 45],
        [27, 45]
    ]

    torch.manual_seed(42)
    model = ConnectomePCDGM(num_nodes=num_nodes, msg_dim=2, variant='learned', K=K, node_dim=dim, motif_dim=dim)
    model.eval()

    # Extract node embeddings h_proj [N, 64] for each window from ConnectomePCDGM
    windows_h_proj = []
    for w_idx, wh in enumerate(window_hour_sets):
        model.memory.reset_state()
        model.neighbor_loader.reset_state()
        mask = torch.zeros(data.num_events, dtype=torch.bool)
        for h in wh:
            mask = mask | (data.t == h)
        events_idx = torch.nonzero(mask).squeeze(-1)

        # Batch update memory & neighbor loader
        model.neighbor_loader.insert(data.src[events_idx], data.dst[events_idx])
        model.memory.update_state(data.src[events_idx], data.dst[events_idx], data.t[events_idx], data.msg[events_idx])

        all_n_ids = torch.arange(num_nodes, dtype=torch.long)
        n_id, edge_index, e_id = model.neighbor_loader(all_n_ids)
        assoc = torch.arange(n_id.size(0))
        z, last_update = model.memory(n_id)
        z = model.gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
        z_all = z[assoc[all_n_ids]]
        h_proj = model.proj_h(z_all) # [N, 64]
        windows_h_proj.append(h_proj.detach())

    max_ln_K = math.log(K)

    for attempt_name, tau in [("Attempt A", 0.1), ("Attempt B", 0.05)]:
        print("\n" + "=" * 95)
        print(f"EVALUATING {attempt_name.upper()}: tau = {tau} (Centered Embeddings + L2 Norm)")
        print(f"Model used: ConnectomePCDGM (TGN + Dictionary layer, num_nodes={num_nodes}, K={K}, dim={dim})")
        print("=" * 95)
        print(f"{'Window':<14} | {'Mean Cos Sim':<12} | {'Std':<8} | {'Entropy/ln(K)':<14} | {'Deg Corr r':<12} | {'Deg R^2':<10}")
        print("-" * 95)

        cos_sims_all = []
        stds_all = []
        entropy_fracs_all = []
        r_all = []
        r2_all = []

        D_norm = F.normalize(model.D.detach(), dim=-1)

        for w_idx, (h_proj, wl, deg_w) in enumerate(zip(windows_h_proj, WINDOW_LABELS, windows_deg)):
            # Subtract mean neuron embedding (mean over all 225 neurons)
            mean_h = h_proj.mean(dim=0, keepdim=True) # [1, 64]
            h_centered = h_proj - mean_h # [225, 64]

            # L2-normalize centered embeddings
            h_norm = F.normalize(h_centered, dim=-1) # [225, 64]

            # Attention logits: cos_sim / tau
            cos_logits = torch.matmul(h_norm, D_norm.t()) / tau # [225, K]
            a = torch.softmax(cos_logits, dim=-1) # [225, K]

            # Final motif vectors z_i (untransformed, direct output)
            z_m = torch.matmul(a, D_norm) # [225, 64]

            # 1. Mean pairwise cosine similarity across neurons
            z_m_norm = F.normalize(z_m, dim=-1)
            cos_mat = torch.mm(z_m_norm, z_m_norm.t())
            mean_cos = (cos_mat.sum() - torch.trace(cos_mat)) / (N * (N - 1))
            cos_sims_all.append(mean_cos.item())

            # 2. Std across neurons
            mean_std = z_m.std(dim=0).mean().item()
            stds_all.append(mean_std)

            # 3. Mean attention entropy / ln(K)
            entropy_per_node = -torch.sum(a * torch.log(a + 1e-12), dim=-1)
            mean_entropy_frac = (entropy_per_node.mean().item()) / max_ln_K
            entropy_fracs_all.append(mean_entropy_frac)

            # 4. Degree check: R^2 and Pearson r of first PC of z_i vs node degree
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(z_m.numpy()).squeeze(-1) # [225]
            r, _ = pearsonr(pc1, deg_w)
            r2 = r ** 2
            r_all.append(r)
            r2_all.append(r2)

            print(f"{wl:<14} | {mean_cos.item():<12.4f} | {mean_std:<8.4f} | {mean_entropy_frac:<14.4f} | {r:<12.4f} | {r2:<10.4f}")

        overall_mean_cos = float(np.mean(cos_sims_all))
        overall_mean_std = float(np.mean(stds_all))
        overall_mean_entropy_frac = float(np.mean(entropy_fracs_all))
        overall_r = float(np.mean(r_all))
        overall_r2 = float(np.mean(r2_all))

        print("-" * 95)
        print(f"Overall Across-Neuron Cosine Sim: {overall_mean_cos:.4f}")
        print(f"Overall Feature Std:             {overall_mean_std:.4f}")
        print(f"Overall Entropy / ln(K):         {overall_mean_entropy_frac:.4f}")
        print(f"Overall Mean Degree Corr r:      {overall_r:.4f}")
        print(f"Overall Mean Degree R^2:         {overall_r2:.4f}")
        print("=" * 95)

        if overall_mean_cos <= 0.95:
            print(f">>> {attempt_name.upper()} PASSED THE SPREAD GATE ({overall_mean_cos:.4f} <= 0.95)! <<<")
            return attempt_name, tau, overall_mean_cos
        else:
            print(f">>> {attempt_name.upper()} FAILED THE SPREAD GATE ({overall_mean_cos:.4f} > 0.95). <<<")

    return "Failed", None, None

if __name__ == "__main__":
    evaluate_attempts()
