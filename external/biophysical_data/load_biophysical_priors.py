"""
external/biophysical_data/load_biophysical_priors.py

Unified loader for biophysical priors:
  1. 3D spatial soma coordinates (x, y, z) and Euclidean distance matrix.
  2. CeNGEN single-cell RNA-seq expression profiles for neuronal classes.

Aligns seamlessly with Witvliet developmental connectome neurons.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
POS_FILE = DATA_DIR / "3D_coordinates_celegans.csv"
CEN_FILE = DATA_DIR / "cengen_threshold4.csv"


def load_3d_coordinates():
    """
    Load 3D soma coordinates for C. elegans neurons.
    Returns:
        dict: {neuron_name: np.array([x, y, z], dtype=np.float32)}
    """
    if not POS_FILE.exists():
        raise FileNotFoundError(f"Missing 3D coordinates file: {POS_FILE}")
    
    df = pd.read_csv(POS_FILE)
    df.columns = [c.strip() for c in df.columns]
    coords = {}
    for _, row in df.iterrows():
        name = str(row['Name']).strip()
        coords[name] = np.array([float(row['x']), float(row['y']), float(row['z'])], dtype=np.float32)
    return coords


def get_spatial_distance_matrix(node_names, sigma=50.0):
    """
    Compute pairwise 3D Euclidean distance matrix and Gaussian proximity kernel.
    For missing neurons (e.g. non-neuronal cells), imputes median distance.
    
    Args:
        node_names: list of neuron names (length N)
        sigma: scale parameter for Gaussian proximity kernel exp(-d^2 / 2*sigma^2)
        
    Returns:
        dist_matrix: [N, N] pairwise Euclidean distances (in microns)
        proximity_matrix: [N, N] spatial contact probability in [0, 1]
    """
    coords = load_3d_coordinates()
    N = len(node_names)
    pos = np.full((N, 3), np.nan, dtype=np.float32)
    
    for i, name in enumerate(node_names):
        if name in coords:
            pos[i] = coords[name]
    
    # Pairwise differences
    diff = pos[:, None, :] - pos[None, :, :]  # [N, N, 3]
    dist = np.sqrt(np.nansum(diff**2, axis=-1))  # [N, N]
    
    # Impute missing pairs with 90th percentile distance (far apart assumption)
    valid_mask = ~np.isnan(pos).any(axis=1)
    if valid_mask.sum() > 1:
        default_dist = np.nanpercentile(dist, 90)
    else:
        default_dist = 200.0
        
    # Apply default where either neuron had no 3D coordinate
    pair_valid = valid_mask[:, None] & valid_mask[None, :]
    dist[~pair_valid] = default_dist
    np.fill_diagonal(dist, 0.0)
    
    proximity = np.exp(- (dist**2) / (2.0 * sigma**2)).astype(np.float32)
    return dist.astype(np.float32), proximity


def load_cengen_profiles(node_names, top_n_genes=256):
    """
    Load CeNGEN single-cell RNA-seq expression profiles for given neurons.
    
    Args:
        node_names: list of neuron names (length N)
        top_n_genes: number of top variable genes to retain for compact embedding
        
    Returns:
        profiles: [N, top_n_genes] float32 array of normalized log1p expression
        gene_names: list of gene names corresponding to columns
    """
    if not CEN_FILE.exists():
        raise FileNotFoundError(f"Missing CeNGEN file: {CEN_FILE}")
        
    df = pd.read_csv(CEN_FILE)
    cengen_classes = [c for c in df.columns[3:]]
    
    # Select top variable genes across neuron classes
    expr_matrix = df[cengen_classes].values.astype(np.float32)  # [G, C]
    variances = np.var(expr_matrix, axis=1)
    top_indices = np.argsort(variances)[::-1][:top_n_genes]
    
    selected_expr = expr_matrix[top_indices, :]  # [top_n, C]
    selected_genes = df['gene_name'].iloc[top_indices].tolist()
    
    # Map node name to CeNGEN class (e.g. AVAL -> AVA, AVAR -> AVA)
    def map_to_class(n):
        if n in cengen_classes:
            return n
        for length in [4, 3, 2]:
            prefix = n[:length]
            if prefix in cengen_classes:
                return prefix
        return None
        
    N = len(node_names)
    profiles = np.zeros((N, top_n_genes), dtype=np.float32)
    
    for i, name in enumerate(node_names):
        cls = map_to_class(name)
        if cls is not None:
            col_idx = cengen_classes.index(cls)
            val = selected_expr[:, col_idx]
            # log1p normalization
            profiles[i] = np.log1p(val)
        else:
            # Missing neurons get zero vector
            pass
            
    return profiles, selected_genes


if __name__ == "__main__":
    print("Testing biophysical loaders...")
    sample_nodes = ["AVAL", "AVAR", "AVBL", "AVBR", "AVER", "AVJL", "RMHR", "DA1"]
    dist, prox = get_spatial_distance_matrix(sample_nodes)
    print(f"Computed distance matrix for {len(sample_nodes)} nodes. Min={dist[0,1]:.1f} um, Max={dist.max():.1f} um")
    
    profiles, genes = load_cengen_profiles(sample_nodes, top_n_genes=64)
    print(f"Extracted CeNGEN profiles: shape {profiles.shape}, sample gene: {genes[0]}")
    print("Success!")
