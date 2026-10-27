"""
run_phase5_training.py — Phase 5 Training Benchmark across 3 Pattern Variants.

Evaluates 3 variants on the Witvliet connectome using:
  - Exact Phase 2 temporal split (5 train / 1 val / 2 test with 45h replicates together)
  - Negative edge sampling (1:1 ratio matching Phase 2 baseline)
  - Standard link prediction metrics (AUROC and AUPRC)
  - 5 seeds: [42, 12345, 123, 456, 789]
  - Target reference: 0.6527

Variants:
  1. No Motif Module (pure temporal graph encoder)
  2. Random Untrained Patterns (frozen random motif dictionary D)
  3. Learned Patterns (trainable dictionary D with soft slot attention)
"""

import sys
from pathlib import Path
from typing import Sequence, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

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
    Connectome temporal graph model supporting all 3 motif variants.
    """

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        variant: str = "learned",  # "no_motif" | "random_untrained" | "learned"
        K: int = 10,
        node_dim: int = 64,
        motif_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.variant = variant
        self.node_dim = node_dim
        self.motif_dim = motif_dim if variant != "no_motif" else 0
        self.device = torch.device(device)

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
            logits = torch.matmul(h_proj, self.D.t()) / (self.motif_dim**0.5)
            a = torch.softmax(logits, dim=-1)  # [B, K]
            z_m = torch.matmul(a, self.D)  # [B, motif_dim]
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


def run_single_variant(variant: str, seeds: Sequence[int], epochs: int = 5, batch_size: int = 200) -> Dict[str, Any]:
    data, num_nodes = build_connectome_temporal_data()
    train_data, val_data, test_data = split_connectome_temporal_data(data)

    results = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)

        model = ConnectomePCDGM(
            num_nodes=num_nodes,
            msg_dim=data.msg.size(-1),
            variant=variant,
            node_dim=64,
            motif_dim=64,
            device="cpu",
        )

        train_loader = TemporalDataLoader(train_data, batch_size=batch_size, neg_sampling_ratio=1.0)
        val_loader = TemporalDataLoader(val_data, batch_size=batch_size, neg_sampling_ratio=1.0)
        test_loader = TemporalDataLoader(test_data, batch_size=batch_size, neg_sampling_ratio=1.0)

        for epoch in range(epochs):
            model.memory.reset_state()
            model.neighbor_loader.reset_state()
            for batch in train_loader:
                model.step(batch, train_data)

        val_ap, val_auc = model.evaluate(val_loader, data)
        test_ap, test_auc = model.evaluate(test_loader, data)

        results.append({
            "seed": s,
            "val_auc": val_auc,
            "val_ap": val_ap,
            "test_auc": test_auc,
            "test_ap": test_ap,
        })

    test_aucs = [r["test_auc"] for r in results]
    test_aps = [r["test_ap"] for r in results]
    return {
        "variant": variant,
        "results": results,
        "mean_auc": float(np.mean(test_aucs)),
        "std_auc": float(np.std(test_aucs)),
        "mean_ap": float(np.mean(test_aps)),
        "std_ap": float(np.std(test_aps)),
    }


def main():
    print("=" * 85)
    print("  PHASE 5: MULTI-SEED CONNECTOME TRAINING BENCHMARK")
    print("  Comparison of 3 Variants: No Motif Module vs. Random Patterns vs. Learned Patterns")
    print("  Evaluation on 45h Connectome Test Snapshots (Replicates Together)")
    print("  Target Reference Baseline: 0.6527")
    print("=" * 85)

    seeds = (42, 12345, 123, 456, 789)
    variants = [
        ("no_motif", "1. No Motif Module (Plain Temporal GNN)"),
        ("random_untrained", "2. Random Untrained Patterns (Frozen D)"),
        ("learned", "3. Learned Patterns (Trainable Dictionary D)"),
    ]

    all_summaries = {}
    for var_key, var_title in variants:
        print(f"\n---> Training Variant: {var_title} across {len(seeds)} seeds...")
        res = run_single_variant(var_key, seeds=seeds, epochs=5)
        all_summaries[var_key] = res
        print(f"     Mean Test AUROC: {res['mean_auc']:.4f} +/- {res['std_auc']:.4f}")
        print(f"     Mean Test AUPRC: {res['mean_ap']:.4f} +/- {res['std_ap']:.4f}")

    print("\n" + "=" * 85)
    print("  PHASE 5 RESULTS TABLE (REPORTED NEXT TO 0.6527 REFERENCE)")
    print("=" * 85)
    print(f"{'Variant':<35} | {'Test AUROC':<18} | {'Test AUPRC':<18} | {'Diff vs 0.6527':<15}")
    print("-" * 92)
    ref = 0.6527
    labels = {
        "no_motif": "No Motif Module",
        "random_untrained": "Random Untrained Patterns",
        "learned": "Learned Patterns (PCDGM)",
    }
    for k in ["no_motif", "random_untrained", "learned"]:
        s = all_summaries[k]
        diff = s["mean_auc"] - ref
        diff_str = f"{diff:+.4f}"
        print(f"{labels[k]:<35} | {s['mean_auc']:.4f} +/- {s['std_auc']:.4f} | {s['mean_ap']:.4f} +/- {s['std_ap']:.4f} | {diff_str:<15}")
    print("=" * 85)


if __name__ == "__main__":
    main()
