import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (
    load_all_snapshots_both,
    _PAPER_CHEMICAL_EDGE_COUNTS,
    _PAPER_ELECTRICAL_EDGE_COUNTS,
)

print('Snapshot count crosscheck vs Witvliet 2021 (Chemical & Electrical):')
chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)

print(f'  {"Label":<10} {"ChemLoad":<10} {"ChemPaper":<10} {"Status":<8} {"ElecLoad":<10} {"ElecPaper":<10} Status')
print('  ' + '-' * 70)
for i, label in enumerate(labels):
    c_loaded = int(chem_snaps[i].sum().item())
    c_paper  = _PAPER_CHEMICAL_EDGE_COUNTS[i]
    c_ok = abs(c_loaded - c_paper) / max(c_paper, 1) <= 0.05
    c_status = 'OK' if c_ok else 'WARN'

    e_loaded = int(elec_snaps[i].sum().item())
    e_paper  = _PAPER_ELECTRICAL_EDGE_COUNTS[i]
    e_ok = abs(e_loaded - e_paper) / max(e_paper, 1) <= 0.05
    e_status = 'OK' if e_ok else 'WARN'

    print(f'  {label:<10} {c_loaded:<10} {c_paper:<10} {c_status:<8} {e_loaded:<10} {e_paper:<10} {e_status}')

