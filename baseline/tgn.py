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

    aucs = [
        run_devotg_single_seed(
            seed=s,
            epochs=epochs,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            batch_size=batch_size,
            device=device,
            csv_path=csv_path,
        )
        for s in seeds
    ]

    mean_auc = float(np.mean(aucs))
    return mean_auc
