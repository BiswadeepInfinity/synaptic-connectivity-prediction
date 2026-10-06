"""
research_audit.py — Deep-dive research diagnostic tool for C. elegans connectome.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

DEVOTG_DIR = Path(__file__).resolve().parent / "external" / "devotg"

def main():
    print("=" * 85)
    print("           DEEP-DIVE CONNECTOME RESEARCH DIAGNOSTICS & ANATOMICAL AUDIT")
    print("=" * 85)
    
    excel_dir = DEVOTG_DIR / "data" / "connectome_datasets"
    files = sorted(excel_dir.glob("witvliet_2020_*.xlsx"))
    
    if not files:
        print("Error: Connectome excel files not found in", excel_dir)
        return

    dfs = [pd.read_excel(f) for f in files]
    snapshot_hours = [0, 5, 8, 16, 23, 27, 45, 45]
    stages = ["L1 (Birth)", "L1 (5h)", "L1 (8h)", "L1 (16h)", "L2 (23h)", "L3 (27h)", "Adult (Rep A)", "Adult (Rep B)"]

    print("\n1. TEMPORAL TRAJECTORY PROGRESSION ACROSS DEVELOPMENT:")
    print("-" * 85)
    print(f"{'Snapshot':<18} | {'Age (h)':<8} | {'Raw Synapses':<14} | {'Unique Pre':<12} | {'Unique Post':<12}")
    print("-" * 85)

    all_pre, all_post = set(), set()
    for i, (df, h, stage) in enumerate(zip(dfs, snapshot_hours, stages)):
        all_pre.update(df["pre"].unique())
        all_post.update(df["post"].unique())
        print(f"{stage:<18} | {h:<8} | {len(df):<14} | {df['pre'].nunique():<12} | {df['post'].nunique():<12}")
    
    all_neurons = sorted(list(all_pre.union(all_post)))
    print("-" * 85)
    print(f"Total Unique Identified Neurons across entire lifespan: {len(all_neurons)} (Expected: 225)")

    print("\n2. SYNAPSE STABILITY CLASSIFICATION (WITVLIET 2021 PRINCIPLES):")
    print("-" * 85)
    # Check edge recurrence across all snapshots
    edge_counts = {}
    for df in dfs:
        for _, row in df.iterrows():
            edge = (row["pre"], row["post"])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            
    stable = sum(1 for c in edge_counts.values() if c >= 7)
    developmental = sum(1 for c in edge_counts.values() if 3 <= c <= 6)
    variable = sum(1 for c in edge_counts.values() if c <= 2)
    total_unique_edges = len(edge_counts)

    print(f"  - Stable Connections (present in >= 7 snapshots):        {stable:5d} ({stable/total_unique_edges*100:.1f}%)")
    print(f"  - Developmental Connections (formed mid-life):          {developmental:5d} ({developmental/total_unique_edges*100:.1f}%)")
    print(f"  - Variable / Transient Connections (in <= 2 snapshots): {variable:5d} ({variable/total_unique_edges*100:.1f}%)")
    print(f"  - Total Unique Physical Synaptic Pathways Observed:     {total_unique_edges:5d}")

    print("\n3. TOP 10 COMMAND HUB NEURONS BY ADULT SYNAPSE DENSITY (H3 FOCUS):")
    print("-" * 85)
    adult_df = dfs[6] # 45h adult A
    birth_df = dfs[0] # 0h birth
    
    hub_data = []
    for n in all_neurons:
        birth_deg = int(((birth_df["pre"] == n) | (birth_df["post"] == n)).sum())
        adult_deg = int(((adult_df["pre"] == n) | (adult_df["post"] == n)).sum())
        hub_data.append((n, birth_deg, adult_deg, adult_deg - birth_deg))
        
    hub_data.sort(key=lambda x: x[2], reverse=True)
    print(f"{'Neuron Name':<15} | {'Birth Degree (0h)':<18} | {'Adult Degree (45h)':<18} | {'Net Synapses Added':<20}")
    print("-" * 85)
    for n, b, a, g in hub_data[:10]:
        star = " ⭐ (AVA/AVB Hub)" if n.startswith("AV") else ""
        print(f"{n:<15} | {b:<18} | {a:<18} | {g:+<20} {star}")
    print("=" * 85)

if __name__ == "__main__":
    main()
