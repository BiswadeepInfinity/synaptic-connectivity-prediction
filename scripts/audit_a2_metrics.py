import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import load_all_snapshots

print('Loading snapshots...')
snaps, hours, labels = load_all_snapshots(use_cect=True)
print()
print(f'  {"Label":<14} {"Hour":<6} {"Edges":<8} {"Density":<10} {"Reciprocity":<14} {"InEntropy":<12} {"OutEntropy":<12} {"Clustering"}')
print('  ' + '-' * 92)
for snap, h, label in zip(snaps, hours, labels):
    m = snap.numpy()
    total = m.sum() + 1e-8
    n = m.shape[0]
    reciprocity = float((m * m.T).sum()) / (2.0 * total)
    norm_in  = m.sum(axis=0) / total
    norm_out = m.sum(axis=1) / total
    in_prob  = norm_in  / (norm_in.sum()  + 1e-12)
    out_prob = norm_out / (norm_out.sum() + 1e-12)
    in_ent   = float(-np.sum(in_prob  * np.log(in_prob  + 1e-12)))
    out_ent  = float(-np.sum(out_prob * np.log(out_prob + 1e-12)))
    A2 = m @ m
    tri = float(np.trace(A2 @ m))
    tp2 = float(A2.sum() - np.trace(A2))
    clust = tri / (tp2 + 1e-8)
    edges = int(m.sum())
    dens  = float(m.mean())
    print(f'  {label:<14} {h:<6} {edges:<8} {dens:<10.4f} {reciprocity:<14.4f} {in_ent:<12.3f} {out_ent:<12.3f} {clust:.4f}')
print()
print('  Reciprocity: fraction of connections that are bidirectional (mutual)')
print('  InEntropy:   Shannon entropy of incoming-connection distribution across neurons')
print('  OutEntropy:  Shannon entropy of outgoing-connection distribution across neurons')
print('  Clustering:  triangles / 2-paths (local wiring clique density)')
