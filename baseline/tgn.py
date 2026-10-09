"""
baseline/tgn.py — Phase 1 & 2: Temporal Graph Network baseline.

PHASE 1: Reproduce DevoTG's lineage-prediction baseline.
  Gate: target AUC 0.839 ± 0.007 before any novel module is trusted.
  Uses DevoTG's model and data from external/devotg/.

PHASE 2: Plain TGN trained on the 8 Witvliet connectome snapshots,
  predicting E_{t+1} from E_{<=t}.

Hard rules (THE_BOOK Part II):
  - Temporal splits only — never shuffle=True.
  - Two 45h replicates are NOT sequential.
  - No activity data.
  - Report L_edge separately from total loss.
"""

from pathlib import Path
import sys
from typing import Sequence
import numpy as np
import torch

DEVOTG_DIR = Path(__file__).resolve().parent.parent / "external" / "devotg"


def run_devotg_single_seed(
    seed: int = 42,
    epochs: int = 8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 200,
    device: str = "cpu",
    csv_path: str | Path | None = None,
) -> float:
    """Run DevoTG training for a single seed and return the final test AUC."""
    if str(DEVOTG_DIR) not in sys.path:
        sys.path.insert(0, str(DEVOTG_DIR))

    from devotg.data import build_cell_ctdg
    from devotg.models import TGNModel

    if csv_path is None:
        csv_path = DEVOTG_DIR / "data" / "cell_lineage_datasets" / "cells_birth_and_pos.csv"

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    temporal_data = build_cell_ctdg(str(csv_path), feature_dim=172)

    model = TGNModel(
        num_nodes=temporal_data.num_nodes,
        msg_dim=temporal_data.msg.size(-1),
        memory_dim=100,
        time_dim=100,
        embedding_dim=100,
        device=device,
    )

    history = model.train_model(
        temporal_data,
        epochs=epochs,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        verbose=False,
    )

    return float(history["test_auc"][-1])


def run_devotg_reproduction(
    seeds: Sequence[int] | None = None,
    epochs: int = 8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 200,
    device: str = "cpu",
    csv_path: str | Path | None = None,
) -> float:
    """
    PHASE 1 GATE: Runs DevoTG lineage-prediction baseline reproduction.

    Evaluates mean test AUC across 5 random seeds to reproduce the published
    result of 0.839 ± 0.007 (Gayen & Alicea, arXiv:2606.21940).
    """
    if seeds is None:
        seeds = (42, 12345, 123, 456, 789)

    aucs = []
    for idx, s in enumerate(seeds):
        print(f"  [DevoTG Reproduction] Running Seed {s} ({idx+1}/{len(seeds)})...", flush=True)
        auc = run_devotg_single_seed(
            seed=s,
            epochs=epochs,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            batch_size=batch_size,
            device=device,
            csv_path=csv_path,
        )
        print(f"  --> Seed {s} Test AUC: {auc:.4f}", flush=True)
        aucs.append(auc)

    mean_auc = float(np.mean(aucs))
    return mean_auc


# ---------------------------------------------------------------------------
# PHASE 2: Connectome-level Plain TGN Baseline
# ---------------------------------------------------------------------------

def build_connectome_temporal_data(
    snapshots: Sequence[torch.Tensor] | None = None,
    real_hours: Sequence[int] | None = None,
    excel_dir: str | Path | None = None,
) -> tuple[torch.Tensor, any]:
    """
    Build a continuous-time dynamic graph (TemporalData) from the 8 Witvliet snapshots
    or provided adjacency tensors.
    
    Hard rules:
      - Temporal ordering preserved.
      - 45h replicates share identical timestamp.
      - Directed synapses.
      - No activity features.
    """
    from torch_geometric.data import TemporalData

    if snapshots is not None:
        # Use synthetic or loaded adjacency tensors
        if real_hours is None:
            real_hours = [0, 5, 8, 16, 23, 27, 45, 45][:len(snapshots)]
        src_l, dst_l, t_l, msg_l = [], [], [], []
        num_nodes = snapshots[0].shape[0]
        for snap_idx, (adj, h) in enumerate(zip(snapshots, real_hours)):
            indices = torch.nonzero(adj)
            for edge in indices:
                u, v = int(edge[0]), int(edge[1])
                if u != v:
                    src_l.append(u)
                    dst_l.append(v)
                    t_l.append(float(h))
                    msg_l.append([1.0])
    else:
        # Load from canonical cect loader with chemical and electrical synapses properly separated
        try:
            from data.loader import load_all_snapshots_both, REAL_HOURS
            chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)
            num_nodes = chem_snaps[0].shape[0]
            if real_hours is None:
                real_hours = hours

            src_l, dst_l, t_l, msg_l = [], [], [], []
            for snap_idx, (c_adj, e_adj, h) in enumerate(zip(chem_snaps, elec_snaps, real_hours)):
                # Chemical synapse events (channel 0 = chemical indicator)
                c_indices = torch.nonzero(c_adj)
                for edge in c_indices:
                    u, v = int(edge[0]), int(edge[1])
                    if u != v:
                        src_l.append(u)
                        dst_l.append(v)
                        t_l.append(float(h))
                        msg_l.append([1.0, 0.0])  # [is_chem, is_elec]

                # Electrical gap junction events (channel 1 = electrical indicator)
                e_indices = torch.nonzero(e_adj)
                for edge in e_indices:
                    u, v = int(edge[0]), int(edge[1])
                    if u != v:
                        src_l.append(u)
                        dst_l.append(v)
                        t_l.append(float(h))
                        msg_l.append([0.0, 1.0])  # [is_chem, is_elec]
        except Exception:
            # Fallback to excel files if cect is unavailable
            import pandas as pd
            if excel_dir is None:
                excel_dir = DEVOTG_DIR / "data" / "connectome_datasets"
            else:
                excel_dir = Path(excel_dir)

            files = sorted(excel_dir.glob("witvliet_2020_*.xlsx"))
            if not files:
                raise FileNotFoundError(f"No Witvliet excel datasets found in {excel_dir}")

            dfs = [pd.read_excel(f) for f in files]
            all_neurons = sorted(list(set.union(*[set(df["pre"]).union(set(df["post"])) for df in dfs])))
            num_nodes = len(all_neurons)
            neuron_to_idx = {n: i for i, n in enumerate(all_neurons)}
            if real_hours is None:
                real_hours = [0, 5, 8, 16, 23, 27, 45, 45]

            src_l, dst_l, t_l, msg_l = [], [], [], []
            for snap_idx, (df, h) in enumerate(zip(dfs, real_hours)):
                for _, row in df.iterrows():
                    u = neuron_to_idx[row["pre"]]
                    v = neuron_to_idx[row["post"]]
                    if u != v:
                        src_l.append(u)
                        dst_l.append(v)
                        t_l.append(float(h))
                        is_chem = 1.0 if row.get("type") == "chemical" else 0.0
                        is_elec = 1.0 if row.get("type") == "electrical" else 0.0
                        msg_l.append([is_chem, is_elec])

    src = torch.tensor(src_l, dtype=torch.long)
    dst = torch.tensor(dst_l, dtype=torch.long)
    t = torch.tensor(t_l, dtype=torch.long)
    msg = torch.tensor(msg_l, dtype=torch.float)

    perm = torch.argsort(t)
    data = TemporalData(src=src[perm], dst=dst[perm], t=t[perm], msg=msg[perm])
    return data, num_nodes



def split_connectome_temporal_data(
    temporal_data,
    real_hours: Sequence[int] | None = None,
    train_frac: float = 0.625,  # 5/8 snapshots -> [0, 5, 8, 16, 23]
    val_frac: float = 0.125,    # 1/8 snapshot  -> [27]
):
    """
    Split connectome TemporalData strictly chronologically.
    Guarantees:
      - Train snapshots: [0, 5, 8, 16, 23] (first 5)
      - Val snapshot: [27] (6th)
      - Test snapshots: [45, 45] (both replicates together in test)
      - No temporal leakage, no empty splits.
    """
    from data.loader import temporal_split, REAL_HOURS

    if real_hours is None:
        real_hours = REAL_HOURS

    unique_hours_in_order = []
    for h in real_hours:
        unique_hours_in_order.append(h)

    train_idx, val_idx, test_idx = temporal_split(
        n_timepoints=len(unique_hours_in_order),
        real_hours=list(unique_hours_in_order),
        train_frac=train_frac,
        val_frac=val_frac,
    )

    train_hours = set(unique_hours_in_order[i] for i in train_idx)
    val_hours = set(unique_hours_in_order[i] for i in val_idx)
    test_hours = set(unique_hours_in_order[i] for i in test_idx)

    # Compute boolean index masks based on event timestamps
    t_cpu = temporal_data.t.cpu().numpy()
    train_mask = np.isin(t_cpu, list(train_hours))
    val_mask = np.isin(t_cpu, list(val_hours))
    test_mask = np.isin(t_cpu, list(test_hours))

    # Convert masks to torch long tensors (PyG TemporalData requires list, tuple, or torch long/bool tensor)
    train_indices = torch.from_numpy(np.where(train_mask)[0]).long()
    val_indices = torch.from_numpy(np.where(val_mask)[0]).long()
    test_indices = torch.from_numpy(np.where(test_mask)[0]).long()

    return (
        temporal_data[train_indices],
        temporal_data[val_indices],
        temporal_data[test_indices],
    )




class ConnectomeTGNBaseline:
    """
    Phase 2 Plain TGN Baseline on connectome temporal graphs.
    Predicts future synaptic connectivity E_{t+1} from E_{<=t}.
    """

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int = 1,
        memory_dim: int = 64,
        time_dim: int = 64,
        embedding_dim: int = 64,
        device: str = "cpu",
    ):
        if str(DEVOTG_DIR) not in sys.path:
            sys.path.insert(0, str(DEVOTG_DIR))
        from devotg.models.tgn_model import GraphAttentionEmbedding, LinkPredictor
        from torch_geometric.nn import TGNMemory
        from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

        self.device = torch.device(device)
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim

        self.memory = TGNMemory(
            num_nodes=self.num_nodes,
            raw_msg_dim=self.msg_dim,
            memory_dim=self.memory_dim,
            time_dim=self.time_dim,
            message_module=IdentityMessage(self.msg_dim, self.memory_dim, self.time_dim),
            aggregator_module=LastAggregator(),
        ).to(self.device)

        self.gnn = GraphAttentionEmbedding(
            in_channels=self.memory_dim,
            out_channels=self.embedding_dim,
            msg_dim=self.msg_dim,
            time_enc=self.memory.time_enc,
        ).to(self.device)

        self.link_pred = LinkPredictor(in_channels=self.embedding_dim).to(self.device)
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long, device=self.device)
        self.neighbor_loader = LastNeighborLoader(self.num_nodes, size=10, device=self.device)

        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters()) | set(self.gnn.parameters()) | set(self.link_pred.parameters()),
            lr=0.001,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def step(self, batch, temporal_data):
        """
        Execute a single forward-backward optimization step.
        Returns dictionary with individual loss terms:
          - L_edge: edge binary classification loss
          - L_total: total loss (identical to L_edge in plain baseline)
        """
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.optimizer.zero_grad()
        batch = batch.to(self.device)

        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        z, last_update = self.memory(n_id)
        z = self.gnn(
            z,
            last_update,
            edge_index,
            temporal_data.t[e_id].to(self.device),
            temporal_data.msg[e_id].to(self.device),
        )

        pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
        neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

        pos_loss = self.criterion(pos_out, torch.ones_like(pos_out))
        neg_loss = self.criterion(neg_out, torch.zeros_like(neg_out))
        l_edge = pos_loss + neg_loss
        l_total = l_edge

        # Update memory state sequentially
        self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        self.neighbor_loader.insert(batch.src, batch.dst)

        l_total.backward()
        self.optimizer.step()
        self.memory.detach()

        return {
            "L_edge": float(l_edge.item()),
            "L_total": float(l_total.item()),
        }

    @torch.no_grad()
    def evaluate(self, loader, temporal_data):
        """Evaluate AP and AUC over a temporal data loader."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        aps, aucs = [], []
        for batch in loader:
            batch = batch.to(self.device)
            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z, last_update = self.memory(n_id)
            z = self.gnn(
                z,
                last_update,
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


def run_connectome_tgn_experiment(
    seeds: Sequence[int] = (42, 12345, 123, 456, 789),
    epochs: int = 5,
    batch_size: int = 200,
    device: str = "cpu",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """
    Run Phase 2 connectome-level TGN experiment across >= 5 seeds.
    Returns summary statistics including mean test AUC and AP.
    """
    from torch_geometric.loader import TemporalDataLoader

    data, num_nodes = build_connectome_temporal_data()

    results = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)

        model = ConnectomeTGNBaseline(num_nodes=num_nodes, msg_dim=data.msg.size(-1), device=device)
        model.memory.reset_state()
        model.neighbor_loader.reset_state()

        train_data, val_data, test_data = split_connectome_temporal_data(data)

        train_loader = TemporalDataLoader(train_data, batch_size=batch_size, neg_sampling_ratio=1.0)
        val_loader = TemporalDataLoader(val_data, batch_size=batch_size, neg_sampling_ratio=1.0)
        test_loader = TemporalDataLoader(test_data, batch_size=batch_size, neg_sampling_ratio=1.0)


        print(f"  [TGN Benchmark] Training Seed {s} ({len(results)+1}/{len(seeds)})...", flush=True)
        for epoch in range(epochs):
            model.memory.reset_state()
            model.neighbor_loader.reset_state()
            for batch in train_loader:
                model.step(batch, train_data)
            print(f"    Seed {s} | Epoch {epoch+1}/{epochs} finished", flush=True)

        val_ap, val_auc = model.evaluate(val_loader, data)
        test_ap, test_auc = model.evaluate(test_loader, data)
        print(f"    --> Seed {s} Done: Val AUC={val_auc:.4f}, Test AUC={test_auc:.4f}, Test AP={test_ap:.4f}\n", flush=True)

        results.append({
            "seed": s,
            "val_ap": val_ap,
            "val_auc": val_auc,
            "test_ap": test_ap,
            "test_auc": test_auc,
        })

    test_aucs = [r["test_auc"] for r in results]
    test_aps = [r["test_ap"] for r in results]
    return {
        "seeds": list(seeds),
        "results": results,
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)),
        "mean_test_ap": float(np.mean(test_aps)),
        "std_test_ap": float(np.std(test_aps)),
    }

