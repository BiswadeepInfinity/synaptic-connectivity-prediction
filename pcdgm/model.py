"""
pcdgm/model.py — Phase 5: Assembled PCDGM Model (Modules B + C + D + E).

Architecture (THE_BOOK §Full Architecture & Roadmap Phase 5):
  1. Module B (StructuralEncoder): Adjacency -> Node Embeddings H_s [N, 64]
  2. Module C (MotifDictionary): Window of snapshots / encodings ->
     Motif representation z_s [64], soft attention a_s [K], dictionary D [K, 64]
  3. Module D (GraphFusion): Concatenate H_s and broadcast(z_s) [N, 64+64]
     and pass through GATConv + LayerNorm -> Fused Node Embeddings H_s' [N, 100]
  4. Module E (ChangePredictor):
     which: pairwise change probabilities p_ij [N, N]
     when: predicted snapshot interval in range(8)
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from pcdgm.module_b_encoder import StructuralEncoder
from pcdgm.module_c_motifs import MotifDictionary
from pcdgm.module_d_fusion import GraphFusion
from pcdgm.module_e_predictor import ChangePredictor


class PCDGM(nn.Module):
    """
    Assembled PCDGM (Pattern-Conditioned Developmental Graph Model).

    Combines:
      - Module B: StructuralEncoder
      - Module C: MotifDictionary
      - Module D: GraphFusion
      - Module E: ChangePredictor
    """

    def __init__(
        self,
        node_dim: int = 64,
        encoder_hidden: int = 100,
        fusion_out_dim: int = 100,
        motif_K: int = 50,
        window_k: int = 3,
        motif_mode: str = "learned",
        use_motif: bool = True,
        heads: int = 2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.fusion_out_dim = fusion_out_dim
        self.window_k = window_k
        self.use_motif = use_motif

        # Module B: Structural Encoder
        self.encoder = StructuralEncoder(
            in_dim=4,
            hidden_dim=encoder_hidden,
            out_dim=node_dim,
            heads=heads,
        )

        # Module C: Motif Discovery
        self.motif_dict = MotifDictionary(
            K=motif_K,
            dim=node_dim,
            window=window_k,
            mode=motif_mode,
        )

        # Module D: Graph Fusion
        self.fusion = GraphFusion(
            node_dim=node_dim,
            motif_dim=node_dim,
            out_dim=fusion_out_dim,
            heads=heads,
            use_motif=use_motif,
        )

        # Module E: Change Predictor
        self.predictor = ChangePredictor(
            dim=fusion_out_dim,
            motif_dim=node_dim,
            hidden_dim=node_dim,
            use_motif=use_motif,
        )

    def forward(
        self,
        window_snapshots: List[torch.Tensor],
        current_snapshot: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for full PCDGM model.

        Args:
            window_snapshots: list of k adjacency tensors [N, N] representing the temporal window H_{s-k:s}
            current_snapshot: [N, N] adjacency of current time step (defaults to window_snapshots[-1])

        Returns:
            Dictionary containing:
              - p_ij: [N, N] change probability matrix
              - z_s: [64] motif representation
              - a_s: [K] slot attention distribution
              - D: [K, 64] motif dictionary
              - H_s: [N, 64] structural node embeddings
              - H_fused: [N, 100] fused node embeddings
              - pool_s: [64] mean pooled window representation
        """
        if current_snapshot is None:
            current_snapshot = window_snapshots[-1]

        N = current_snapshot.shape[0]

        # 1. Module B: Node embeddings for current snapshot
        H_s = self.encoder(current_snapshot)  # [N, 64]

        # 2. Module C: Motif representation from window
        # Either pass window of snapshots or pre-encoded representations
        z_s, a_s, D = self.motif_dict(window_snapshots)  # [64], [K], [K, 64]

        # 3. Module D: Broadcast z_s to [N, 64] and fuse
        if self.use_motif:
            z_per_node = z_s.unsqueeze(0).expand(N, -1)  # [N, 64]
        else:
            z_per_node = None

        H_fused = self.fusion(H=H_s, z_per_node=z_per_node, adjacency=current_snapshot)  # [N, 100]

        # 4. Module E: Which change predictor
        # Predict pairwise edge probabilities
        p_ij = self.predictor.which(H_i=H_fused, H_j=H_fused, z_s=z_per_node if self.use_motif else None)

        pool_s = self.motif_dict._pool_window(window_snapshots)

        return {
            "p_ij": p_ij,
            "z_s": z_s,
            "a_s": a_s,
            "D": D,
            "H_s": H_s,
            "H_fused": H_fused,
            "pool_s": pool_s,
        }

    def predict_interval(
        self,
        H_fused: torch.Tensor,
        z_s: Optional[torch.Tensor] = None,
        real_hours: Optional[List[int]] = None,
    ) -> int:
        """Predicts snapshot interval index in range(len(real_hours))."""
        return self.predictor.when_interval(
            H_i=H_fused,
            H_j=H_fused,
            z_s=z_s,
            real_hours=real_hours,
        )
