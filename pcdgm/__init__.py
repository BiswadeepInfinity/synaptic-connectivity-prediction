"""
pcdgm package — Pattern-Conditioned Developmental Graph Model.
"""

from pcdgm.module_a_trajectory import interpolate_A1, infer_trajectory_A2
from pcdgm.module_b_encoder import StructuralEncoder
from pcdgm.module_c_motifs import MotifDictionary
from pcdgm.module_d_fusion import GraphFusion
from pcdgm.module_e_predictor import ChangePredictor
from pcdgm.model import PCDGM

__all__ = [
    "interpolate_A1",
    "infer_trajectory_A2",
    "StructuralEncoder",
    "MotifDictionary",
    "GraphFusion",
    "ChangePredictor",
    "PCDGM",
]
