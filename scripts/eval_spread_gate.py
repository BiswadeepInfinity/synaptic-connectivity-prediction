import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import load_all_snapshots_both
from pcdgm.module_b_encoder import StructuralEncoder
from pcdgm.module_c_motifs import MotifDictionary
from scripts.validate_module_c_stability import CANONICAL_MODEL_PATH

def compute_spread_gate():
    chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)
    k = 3
    K = 10
    dim = 64
    tau = 0.1

    encoder = StructuralEncoder()
    encoder.eval()
    with torch.no_grad():
        H_snaps = [encoder(s) for s in chem_snaps]

    windows_H = [torch.stack(H_snaps[i:i+k], dim=0) for i in range(len(H_snaps) - k + 1)] # 6 windows

    model = MotifDictionary(K=K, dim=dim, window=k)
    model.load_state_dict(torch.load(CANONICAL_MODEL_PATH, weights_only=True))
    model.eval()

    WINDOW_LABELS = ['W0 (0-8h)', 'W1 (5-16h)', 'W2 (8-23h)', 'W3 (16-27h)', 'W4 (23-45hA)', 'W5 (27-45hB)']

    print("=" * 80)
    print("STEP 3: SPREAD GATE EVALUATION (tau = 0.1, L2-normalized embeddings & dictionary)")
    print("=" * 80)
    print(f"{'Window':<14} | {'Mean Cosine Sim':<16} | {'Mean Std':<12} | {'Mean Entropy / ln(K)':<22}")
    print("-" * 72)

    max_ln_K = math.log(K)
    cos_sims_all = []
    stds_all = []
    entropy_fracs_all = []

    for w_idx, (w, w_label) in enumerate(zip(windows_H, WINDOW_LABELS)):
        h_node = w.mean(dim=0) # [N=225, dim=64]
        
        # Step 2: L2-normalize both neuron embeddings and dictionary vectors
        h_norm = F.normalize(h_node, dim=-1) # [N, 64]
        D_norm = F.normalize(model.D, dim=-1) # [K, 64]
        
        # Cosine similarity / tau
        cos_logits = torch.matmul(h_norm, D_norm.t()) / tau # [N, K]
        a = torch.softmax(cos_logits, dim=-1) # [N, K]
        z_m = torch.matmul(a, D_norm) # [N, 64]
        
        # 1. Mean pairwise cosine similarity across neurons
        z_m_norm = F.normalize(z_m, dim=-1)
        cos_mat = torch.mm(z_m_norm, z_m_norm.t())
        N = cos_mat.shape[0]
        mean_cos = (cos_mat.sum() - torch.trace(cos_mat)) / (N * (N - 1))
        cos_sims_all.append(mean_cos.item())
        
        # 2. Std across neurons
        mean_std = z_m.std(dim=0).mean().item()
        stds_all.append(mean_std)
        
        # 3. Mean attention entropy per neuron as fraction of ln(K)
        entropy_per_node = -torch.sum(a * torch.log(a + 1e-12), dim=-1) # [N]
        mean_entropy_frac = (entropy_per_node.mean().item()) / max_ln_K
        entropy_fracs_all.append(mean_entropy_frac)
        
        print(f"{w_label:<14} | {mean_cos.item():<16.4f} | {mean_std:<12.4f} | {mean_entropy_frac:<22.4f}")

    overall_mean_cos = float(np.mean(cos_sims_all))
    overall_mean_std = float(np.mean(stds_all))
    overall_mean_entropy_frac = float(np.mean(entropy_fracs_all))
    print("-" * 72)
    print(f"Overall Mean Across-Neuron Cosine Similarity: {overall_mean_cos:.4f}")
    print(f"Overall Mean Across-Neuron Feature Std:       {overall_mean_std:.4f}")
    print(f"Overall Mean Attention Entropy / ln(K):       {overall_mean_entropy_frac:.4f}")
    print("=" * 80)
    
    passed = overall_mean_cos <= 0.95
    if not passed:
        print(f"SPREAD GATE RESULT: FAILED (Mean Cosine Sim = {overall_mean_cos:.4f} > 0.95). STOPPING.")
    else:
        print(f"SPREAD GATE RESULT: PASSED (Mean Cosine Sim = {overall_mean_cos:.4f} <= 0.95). PROCEEDING.")
    return passed, overall_mean_cos, cos_sims_all, stds_all, entropy_fracs_all

if __name__ == "__main__":
    compute_spread_gate()
