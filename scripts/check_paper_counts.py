import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (
    load_all_snapshots_both,
    _PAPER_CHEMICAL_EDGE_COUNTS,
    _PAPER_ELECTRICAL_EDGE_COUNTS,
)

print('Loading snapshots via cect (chemical and electrical separately)...')
chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both(use_cect=True)

print()
print('=== CHEMICAL SYNAPSES (Generic_CS) ===')
print(f'  {"Snapshot":<10} {"Label":<14} {"Hour":<6} {"Loaded":<10} {"Paper":<10} {"Diff":<8} {"Status"}')
print('  ' + '-' * 68)
all_chem_ok = True
for i, (snap, h, label) in enumerate(zip(chem_snaps, hours, labels)):
    loaded = int(snap.sum().item())
    paper  = _PAPER_CHEMICAL_EDGE_COUNTS[i]
    diff   = loaded - paper
    pct    = abs(diff) / max(paper, 1) * 100
    ok     = pct <= 5.0
    status = 'OK' if ok else f'WARN ({pct:.1f}% off)'
    if not ok:
        all_chem_ok = False
    print(f'  S{i+1:<9} {label:<14} {h:<6} {loaded:<10} {paper:<10} {diff:<+8} {status}')

print()
print('=== ELECTRICAL SYNAPSES / GAP JUNCTIONS (Generic_GJ, directed edges) ===')
print(f'  {"Snapshot":<10} {"Label":<14} {"Hour":<6} {"Loaded":<10} {"Paper":<10} {"Diff":<8} {"Status"}')
print('  ' + '-' * 68)
all_elec_ok = True
for i, (snap, h, label) in enumerate(zip(elec_snaps, hours, labels)):
    loaded = int(snap.sum().item())
    paper  = _PAPER_ELECTRICAL_EDGE_COUNTS[i]
    diff   = loaded - paper
    pct    = abs(diff) / max(paper, 1) * 100
    ok     = pct <= 5.0
    status = 'OK' if ok else f'WARN ({pct:.1f}% off)'
    if not ok:
        all_elec_ok = False
    print(f'  S{i+1:<9} {label:<14} {h:<6} {loaded:<10} {paper:<10} {diff:<+8} {status}')

print()
if all_chem_ok and all_elec_ok:
    print('  RESULT: All chemical and electrical counts independently within 5% of paper values.')
else:
    print('  RESULT: Some snapshot counts deviate from paper values.')

