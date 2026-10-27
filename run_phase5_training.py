"""
run_phase5_training.py — Phase 5 Retest Benchmark across 3 Pattern Variants (tau = 0.1).

Evaluates 3 variants on the Witvliet connectome using:
  - Exact Phase 2 temporal split (5 train / 1 val / 2 test with 45h replicates together)
  - Negative edge sampling (1:1 ratio matching Phase 2 baseline)
  - Standard link prediction metrics (AUROC and AUPRC)
  - Paired 5 seeds: [42, 12345, 123, 456, 789]
  - Target reference: 0.6527

Variants:
  1. No Motif Module (pure temporal graph encoder)
  2. Random Untrained Patterns (frozen random motif dictionary D with tau=0.1)
  3. Learned Patterns (trainable dictionary D with tau=0.1)

The ONLY difference between arms is the motif input (centered + L2-normalized with tau=0.1).
"""

import os
import random
import sys
from pathlib import Path
from typing import Sequence, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, average_precision_score

# Set deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEVOTG_DIR = REPO_ROOT / "external" / "devotg"
if str(DEVOTG_DIR) not in sys.path:
    sys.path.insert(0, str(DEVOTG_DIR))

from devotg.models.tgn_model import GraphAttentionEmbedding, LinkPredictor
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader

from baseline.tgn import build_connectome_temporal_data, split_connectome_temporal_data


class ConnectomePCDGM(nn.Module):
    """
    Connectome temporal graph model supporting all 3 motif variants with tau=0.1 attention.
    """

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        variant: str = "learned",  # "no_motif" | "random_untrained" | "learned"
        K: int = 10,
        node_dim: int = 64,
        motif_dim: int = 64,
        tau: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.variant = variant
        self.node_dim = node_dim
        self.motif_dim = motif_dim if variant != "no_motif" else 0
        self.tau = tau
        self.device = torch.device(device)

        # Base TGN Memory & GNN (Identical across all 3 arms)
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=msg_dim,
            memory_dim=node_dim,
            time_dim=node_dim,
            message_module=IdentityMessage(msg_dim, node_dim, node_dim),
            aggregator_module=LastAggregator(),
        ).to(self.device)

        self.gnn = GraphAttentionEmbedding(
            in_channels=node_dim,
            out_channels=node_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        ).to(self.device)

        if self.variant == "learned":
            self.D = nn.Parameter(torch.randn(K, motif_dim, device=self.device) / (motif_dim**0.5))
            self.proj_h = nn.Linear(node_dim, motif_dim).to(self.device)
            self.fusion_layer = nn.Linear(node_dim + motif_dim, node_dim).to(self.device)
        elif self.variant == "random_untrained":
            self.D = nn.Parameter(torch.randn(K, motif_dim, device=self.device) / (motif_dim**0.5), requires_grad=False)
            self.proj_h = nn.Linear(node_dim, motif_dim).to(self.device)
            self.fusion_layer = nn.Linear(node_dim + motif_dim, node_dim).to(self.device)
        else:
            self.D = None
            self.proj_h = None
            self.fusion_layer = None

        self.link_pred = LinkPredictor(in_channels=node_dim).to(self.device)
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=self.device)
        self.neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        trainable_params = list(self.memory.parameters()) + list(self.gnn.parameters()) + list(self.link_pred.parameters())
        if self.variant in ("learned", "random_untrained"):
            trainable_params += list(self.proj_h.parameters()) + list(self.fusion_layer.parameters())
            if self.variant == "learned":
                trainable_params += [self.D]

        self.optimizer = torch.optim.Adam(trainable_params, lr=0.001)

    def forward_nodes(self, n_id, edge_index, t, msg):
        z, last_update = self.memory(n_id)
        z = self.gnn(z, last_update, edge_index, t, msg)

        if self.variant in ("learned", "random_untrained"):
            # Per-neuron motif vector derived from its own node embedding
            h_proj = self.proj_h(z)  # [B, motif_dim]
            # Mean-centering over batch neurons
            mean_h = h_proj.mean(dim=0, keepdim=True)
            h_centered = h_proj - mean_h
            # L2-normalize centered embeddings and dictionary vectors
            h_norm = F.normalize(h_centered, dim=-1)
            D_norm = F.normalize(self.D, dim=-1)
            # Scaled cosine similarity logits with tau = 0.1
            logits = torch.matmul(h_norm, D_norm.t()) / self.tau  # [B, K]
            a = torch.softmax(logits, dim=-1)  # [B, K]
            z_m = torch.matmul(a, D_norm)      # [B, motif_dim]
            z = torch.relu(self.fusion_layer(torch.cat([z, z_m], dim=-1)))

        return z

    def step(self, batch, temporal_data) -> float:
        self.train()
        self.optimizer.zero_grad()
        batch = batch.to(self.device)

        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        z = self.forward_nodes(
            n_id,
            edge_index,
            temporal_data.t[e_id].to(self.device),
            temporal_data.msg[e_id].to(self.device),
        )

        pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
        neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

        loss = self.criterion(pos_out, torch.ones_like(pos_out)) + self.criterion(neg_out, torch.zeros_like(neg_out))

        self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        self.optimizer.step()
        self.memory.detach()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, loader, temporal_data):
        self.eval()
        aps, aucs = [], []
        for batch in loader:
            batch = batch.to(self.device)
            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z = self.forward_nodes(
                n_id,
                edge_index,
                temporal_data.t[e_id].to(self.device),
                temporal_data.msg[e_id].to(self.device),
            )

            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
            )

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

        mean_ap = float(np.mean(aps)) if aps else 0.0
        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        return mean_ap, mean_auc


def run_paired_experiment(seeds: Sequence[int] = (42, 12345, 123, 456, 789), epochs: int = 5, batch_size: int = 200):
    data, num_nodes = build_connectome_temporal_data()
    train_data, val_data, test_data = split_connectome_temporal_data(data)

    variants = ["no_motif", "random_untrained", "learned"]
    results_by_var = {v: [] for v in variants}

    print("\n" + "=" * 95)
    print("CONFIG DIFF BETWEEN THE THREE ARMS:")
    print("=" * 95)
    print("  [Shared by all 3 arms]")
    print(f"    - Data Split: Train (0h, 5h, 8h, 16h, 23h), Val (27h), Test (45h Replicates A & B)")
    print(f"    - Negative Sampling: 1:1 ratio (neg_sampling_ratio=1.0)")
    print(f"    - Base Architecture: TGN Memory (64-dim) + GraphAttentionEmbedding (64-dim) + LinkPredictor")
    print(f"    - Optimizer: Adam, lr=0.001, epochs={epochs}, batch_size={batch_size}")
    print(f"    - Seeding: Paired seeds {list(seeds)} applied identically before data loader & model init")
    print("  [Differences]")
    print("    - no_motif:         motif_dim=0, no motif dictionary or projection layer.")
    print("    - random_untrained: motif_dim=64, frozen D (requires_grad=False), centered h + L2 norm, tau=0.1.")
    print("    - learned:          motif_dim=64, trainable D (requires_grad=True), centered h + L2 norm, tau=0.1.")
    print("=" * 95 + "\n")

    for s in seeds:
        print(f"---> Running Paired Seed: {s}")
        for var in variants:
            # Paired seed ensures exact identical data batching order and initial state
            random.seed(s)
            torch.manual_seed(s)
            np.random.seed(s)

            train_loader = TemporalDataLoader(train_data, batch_size=batch_size, neg_sampling_ratio=1.0)
            val_loader = TemporalDataLoader(val_data, batch_size=batch_size, neg_sampling_ratio=1.0)
            test_loader = TemporalDataLoader(test_data, batch_size=batch_size, neg_sampling_ratio=1.0)

            # Re-seed for model init
            random.seed(s)
            torch.manual_seed(s)
            np.random.seed(s)
            model = ConnectomePCDGM(
                num_nodes=num_nodes,
                msg_dim=data.msg.size(-1),
                variant=var,
                K=10,
                node_dim=64,
                motif_dim=64,
                tau=0.1,
                device="cpu",
            )

            for epoch in range(epochs):
                model.memory.reset_state()
                model.neighbor_loader.reset_state()
                for batch in train_loader:
                    model.step(batch, train_data)

            val_ap, val_auc = model.evaluate(val_loader, data)
            test_ap, test_auc = model.evaluate(test_loader, data)

            results_by_var[var].append({
                "seed": s,
                "val_auc": val_auc,
                "val_ap": val_ap,
                "test_auc": test_auc,
                "test_ap": test_ap,
            })

    return results_by_var


def main():
    print("=" * 95)
    print("  PHASE 5 RETEST: MULTI-SEED 3-VARIANT BENCHMARK (tau = 0.1)")
    print("  Paired 5 Seeds across No-Motif, Random-Untrained, and Learned PCDGM")
    print("=" * 95)

    seeds = (42, 12345, 123, 456, 789)
    results = run_paired_experiment(seeds=seeds, epochs=5)

    print("\n" + "=" * 95)
    print("  TABLE 1: PER-SEED TEST PERFORMANCE")
    print("=" * 95)
    print(f"{'Seed':<8} | {'No-Motif AUC':<14} | {'Random AUC':<14} | {'Learned AUC':<14} | {'No-Motif AP':<14} | {'Random AP':<14} | {'Learned AP':<14}")
    print("-" * 95)
    for i, s in enumerate(seeds):
        r_no = results["no_motif"][i]
        r_rand = results["random_untrained"][i]
        r_learn = results["learned"][i]
        print(f"{s:<8} | {r_no['test_auc']:<14.4f} | {r_rand['test_auc']:<14.4f} | {r_learn['test_auc']:<14.4f} | {r_no['test_ap']:<14.4f} | {r_rand['test_ap']:<14.4f} | {r_learn['test_ap']:<14.4f}")

    print("\n" + "=" * 95)
    print("  TABLE 2: AGGREGATE TEST PERFORMANCE & REFERENCE COMPARISON (0.6527)")
    print("=" * 95)
    print(f"{'Variant':<28} | {'Test AUROC':<20} | {'Test AUPRC':<20} | {'Val AUROC':<16}")
    print("-" * 95)
    labels = {
        "no_motif": "1. No Motif Module",
        "random_untrained": "2. Random Untrained (tau=0.1)",
        "learned": "3. Learned Patterns (tau=0.1)",
    }
    for k in ["no_motif", "random_untrained", "learned"]:
        aucs = [r["test_auc"] for r in results[k]]
        aps = [r["test_ap"] for r in results[k]]
        v_aucs = [r["val_auc"] for r in results[k]]
        print(f"{labels[k]:<28} | {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}     | {np.mean(aps):.4f} +/- {np.std(aps):.4f}     | {np.mean(v_aucs):.4f} +/- {np.std(v_aucs):.4f}")
    print("=" * 95)

    # Paired t-tests on test AUROC
    learn_aucs = np.array([r["test_auc"] for r in results["learned"]])
    nomotif_aucs = np.array([r["test_auc"] for r in results["no_motif"]])
    rand_aucs = np.array([r["test_auc"] for r in results["random_untrained"]])

    diff_nomotif = learn_aucs - nomotif_aucs
    t_no, p_no = ttest_rel(learn_aucs, nomotif_aucs)

    diff_rand = learn_aucs - rand_aucs
    t_rand, p_rand = ttest_rel(learn_aucs, rand_aucs)

    print("\n" + "=" * 95)
    print("  TABLE 3: PAIRED T-TESTS (TEST AUROC)")
    print("=" * 95)
    print(f"  Learned vs. No-Motif: Mean Paired Diff = {diff_nomotif.mean():+.4f} | t = {t_no:.4f} | p = {p_no:.4f}")
    print(f"  Learned vs. Random:   Mean Paired Diff = {diff_rand.mean():+.4f} | t = {t_rand:.4f} | p = {p_rand:.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
