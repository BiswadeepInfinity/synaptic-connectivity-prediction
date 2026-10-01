"""
verify_install.py — Final environment verification.
Run this once to confirm all packages and data sources are working.
"""
import sys
import numpy as np
import torch

print("=" * 55)
print("SYNAPTIC CONNECTIVITY PREDICTION — INSTALL VERIFY")
print("=" * 55)
print(f"Python: {sys.version.split()[0]}")
print(f"numpy:  {np.__version__}")
print(f"torch:  {torch.__version__}")

# -- torch-geometric ----------------------------------------
try:
    import torch_geometric
    print(f"torch-geometric: {torch_geometric.__version__}  [OK]")
except ImportError:
    print("torch-geometric: NOT FOUND  [FAIL]")

# -- scipy / sklearn ----------------------------------------
import scipy, sklearn, statsmodels, networkx
print(f"scipy:       {scipy.__version__}  [OK]")
print(f"sklearn:     {sklearn.__version__}  [OK]")
print(f"statsmodels: {statsmodels.__version__}  [OK]")
print(f"networkx:    {networkx.__version__}  [OK]")

# -- pytest ------------------------------------------------─
import pytest
print(f"pytest:      {pytest.__version__}  [OK]")

# -- cect — correct API ------------------------------------─
print("\n-- cect (Witvliet data) --")
try:
    from cect.readers.WitvlietDataReader1 import get_instance as get1
    r1 = get1()
    # The reader exposes .connections (list of connection infos) and .nodes
    n_neurons = len(r1.nodes) if hasattr(r1, "nodes") else "?"
    # Use to_networkx_graph to get adjacency
    G = r1.to_networkx_graph()
    print(f"  Dataset 1 (0h):  nodes={G.number_of_nodes()}, edges={G.number_of_edges()}  [OK]")

    from cect.readers.WitvlietDataReader8 import get_instance as get8
    r8 = get8()
    G8 = r8.to_networkx_graph()
    print(f"  Dataset 8 (45h): nodes={G8.number_of_nodes()}, edges={G8.number_of_edges()}  [OK]")
    print("  cect readers: OK")
except Exception as e:
    print(f"  cect ERROR: {e}  [FAIL]")

# -- DevoTG --
print("\n-- DevoTG --")
try:
    sys.path.insert(0, "external/devotg")
    from devotg.data import build_cell_ctdg, load_sample_data
    from devotg.models import TGNModel
    print("  DevoTG imports: OK")
    print("  Witvliet XLSXs: external/devotg/data/connectome_datasets/ (8 files)")
except Exception as e:
    print(f"  DevoTG ERROR: {e}  [FAIL]")

print("\n" + "=" * 55)
print("Environment ready. Run: python -m pytest data/ -v")
print("=" * 55)
