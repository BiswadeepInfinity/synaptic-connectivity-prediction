"""
scripts/run_h3_full_brain.py — Full-Brain H3: Which neurons across all 225 gain
connections faster than degree-matched chance?

Exact same null protocol as run_h3_direct.py:
  - Matched null: 1,000 draws, 0h chemical in-degree ±20% window
  - Criterion: exceed 95th percentile in BOTH 45h replicates
  - Separate analysis for chemical in-degree and gap-junction degree
  - Skip neurons with matched pool < 5 (too few peers to draw a null)

Output: results/h3_full_brain.md
"""

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import load_all_snapshots_both
from cect.readers.WitvlietDataReader1 import get_instance as get1
from cect.readers.WitvlietDataReader2 import get_instance as get2
from cect.readers.WitvlietDataReader3 import get_instance as get3
from cect.readers.WitvlietDataReader4 import get_instance as get4
from cect.readers.WitvlietDataReader5 import get_instance as get5
from cect.readers.WitvlietDataReader6 import get_instance as get6
from cect.readers.WitvlietDataReader7 import get_instance as get7
from cect.readers.WitvlietDataReader8 import get_instance as get8


MIN_POOL = 5      # minimum matched-null pool size to include a neuron
B        = 1000   # bootstrap draws
SEED     = 42


def main():
    print("Loading 8 Witvliet snapshots (chemical + electrical)...")
    chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both()

    # Build node list — same as H3 direct
    readers = [get1(), get2(), get3(), get4(), get5(), get6(), get7(), get8()]
    all_nodes = sorted(list(set.union(*[
        set(r.to_networkx_graph("Generic_CS").nodes()).union(
            set(r.to_networkx_graph("Generic_GJ").nodes()))
        for r in readers
    ])))
    node2idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)
    print(f"Total neurons in union node set: {N}")

    # In-degree matrices: [N, 8]  (in-degree = column sum of adj)
    chem_in  = np.zeros((N, 8), dtype=float)
    elec_deg = np.zeros((N, 8), dtype=float)
    for t in range(8):
        chem_in[:, t]  = chem_snaps[t].numpy().sum(axis=0)   # col sum = in-degree
        elec_deg[:, t] = elec_snaps[t].numpy().sum(axis=0)

    rng = np.random.default_rng(SEED)

    print("Running matched-null analysis for all neurons...")
    results = []
    skipped = 0

    for idx, neuron in enumerate(all_nodes):
        d0_c = chem_in[idx, 0]

        # Matched pool: same 0h chemical in-degree ±20%, excluding self
        low, high = d0_c * 0.8, d0_c * 1.2
        matched = [j for j in range(N) if j != idx and low <= chem_in[j, 0] <= high]

        if len(matched) < MIN_POOL:
            skipped += 1
            continue

        # Actual gains
        g_c_a = chem_in[idx, 6]  - d0_c
        g_c_b = chem_in[idx, 7]  - d0_c
        d0_e  = elec_deg[idx, 0]
        g_e_a = elec_deg[idx, 6] - d0_e
        g_e_b = elec_deg[idx, 7] - d0_e

        # Null distribution
        sample_idx = rng.choice(matched, size=B, replace=True)
        null_c_a = chem_in[sample_idx, 6]  - chem_in[sample_idx, 0]
        null_c_b = chem_in[sample_idx, 7]  - chem_in[sample_idx, 0]
        null_e_a = elec_deg[sample_idx, 6] - elec_deg[sample_idx, 0]
        null_e_b = elec_deg[sample_idx, 7] - elec_deg[sample_idx, 0]

        pct_c_a = np.mean(null_c_a <= g_c_a) * 100
        pct_c_b = np.mean(null_c_b <= g_c_b) * 100
        pct_e_a = np.mean(null_e_a <= g_e_a) * 100
        pct_e_b = np.mean(null_e_b <= g_e_b) * 100

        both_c_95 = (pct_c_a >= 95.0) and (pct_c_b >= 95.0)
        both_e_95 = (pct_e_a >= 95.0) and (pct_e_b >= 95.0)
        mean_chem_pct = (pct_c_a + pct_c_b) / 2.0

        results.append({
            "neuron":         neuron,
            "0h_c":           d0_c,
            "45hA_c":         chem_in[idx, 6],
            "45hB_c":         chem_in[idx, 7],
            "g_c_a":          g_c_a,
            "g_c_b":          g_c_b,
            "pct_c_a":        pct_c_a,
            "pct_c_b":        pct_c_b,
            "mean_chem_pct":  mean_chem_pct,
            "0h_e":           d0_e,
            "45hA_e":         elec_deg[idx, 6],
            "45hB_e":         elec_deg[idx, 7],
            "g_e_a":          g_e_a,
            "g_e_b":          g_e_b,
            "pct_e_a":        pct_e_a,
            "pct_e_b":        pct_e_b,
            "matched_pool":   len(matched),
            "both_c_95":      both_c_95,
            "both_e_95":      both_e_95,
        })

    print(f"Analysed: {len(results)} neurons, skipped (pool < {MIN_POOL}): {skipped}")

    # Sort by mean chemical percentile descending
    results.sort(key=lambda r: r["mean_chem_pct"], reverse=True)

    # Subsets
    hits_c  = [r for r in results if r["both_c_95"]]
    hits_e  = [r for r in results if r["both_e_95"]]
    hits_both = [r for r in results if r["both_c_95"] and r["both_e_95"]]

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    out_file = REPO_ROOT / "results" / "h3_full_brain.md"
    with open(out_file, "w", encoding="utf-8") as f:

        f.write("# H3 Full-Brain Screen: Which Neurons Gain Synapses Faster Than Matched Chance?\n\n")
        f.write("> Same null protocol as `h3_direct.md`: 1,000 draws, 0h chemical "
                "in-degree ±20% window, criterion = exceed 95th percentile in **both** "
                "45h replicates (A and B independently). Seed 42. "
                f"Neurons with fewer than {MIN_POOL} matched peers excluded.\n\n")
        f.write(f"**{N} total neurons · {len(results)} analysed · {skipped} excluded (small pool)**\n\n---\n\n")

        # --- Summary counts ---
        f.write("## 1. Summary Counts\n\n")
        f.write(f"| Criterion | Count |\n|---|---|\n")
        f.write(f"| Exceed 95th pct chemical in-degree, BOTH replicates | **{len(hits_c)}** |\n")
        f.write(f"| Exceed 95th pct gap-junction degree, BOTH replicates | **{len(hits_e)}** |\n")
        f.write(f"| Exceed 95th pct for BOTH synapse types, BOTH replicates | **{len(hits_both)}** |\n\n")
        f.write("---\n\n")

        # --- Top 20 by chemical ---
        f.write("## 2. Top 20 Neurons by Chemical In-Degree Growth Percentile\n\n")
        f.write("Sorted by mean of Replicate A and B percentile.\n\n")
        f.write("| Rank | Neuron | 0h In | Gain RepA | Pct RepA | Gain RepB | Pct RepB | Mean Pct | Pool N | Chem ≥95% Both? |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(results[:20], 1):
            flag = "✅" if r["both_c_95"] else ""
            f.write(f"| {i} | **{r['neuron']}** | {r['0h_c']:.0f} | "
                    f"+{r['g_c_a']:.1f} | {r['pct_c_a']:.1f}% | "
                    f"+{r['g_c_b']:.1f} | {r['pct_c_b']:.1f}% | "
                    f"**{r['mean_chem_pct']:.1f}%** | {r['matched_pool']} | {flag} |\n")
        f.write("\n---\n\n")

        # --- All neurons that pass the 95th pct criterion (chemical) ---
        f.write("## 3. All Neurons Exceeding 95th Percentile Chemical Growth in Both Replicates\n\n")
        if hits_c:
            f.write("| Neuron | 0h In | Gain RepA | Pct RepA | Gain RepB | Pct RepB | Also ≥95% Gap Both? |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for r in sorted(hits_c, key=lambda x: x["mean_chem_pct"], reverse=True):
                gap_flag = "✅" if r["both_e_95"] else ""
                f.write(f"| **{r['neuron']}** | {r['0h_c']:.0f} | "
                        f"+{r['g_c_a']:.1f} | {r['pct_c_a']:.1f}% | "
                        f"+{r['g_c_b']:.1f} | {r['pct_c_b']:.1f}% | {gap_flag} |\n")
        else:
            f.write("_No neurons met this criterion._\n")
        f.write("\n---\n\n")

        # --- All neurons that pass for gap junctions ---
        f.write("## 4. All Neurons Exceeding 95th Percentile Gap-Junction Growth in Both Replicates\n\n")
        if hits_e:
            f.write("| Neuron | 0h Gap | Gain RepA | Pct RepA | Gain RepB | Pct RepB | Also ≥95% Chem Both? |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for r in sorted(hits_e, key=lambda x: (x["pct_e_a"] + x["pct_e_b"]) / 2, reverse=True):
                chem_flag = "✅" if r["both_c_95"] else ""
                f.write(f"| **{r['neuron']}** | {r['0h_e']:.0f} | "
                        f"+{r['g_e_a']:.1f} | {r['pct_e_a']:.1f}% | "
                        f"+{r['g_e_b']:.1f} | {r['pct_e_b']:.1f}% | {chem_flag} |\n")
        else:
            f.write("_No neurons met this criterion._\n")
        f.write("\n---\n\n")

        # --- Full ranked table ---
        f.write("## 5. Full Ranked Table — All Analysed Neurons\n\n")
        f.write("| Rank | Neuron | 0h Chem In | Gain A | Pct A | Gain B | Pct B | Mean Pct | Chem✅ | Gap✅ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(results, 1):
            c = "✅" if r["both_c_95"] else ""
            e = "✅" if r["both_e_95"] else ""
            f.write(f"| {i} | {r['neuron']} | {r['0h_c']:.0f} | "
                    f"+{r['g_c_a']:.1f} | {r['pct_c_a']:.1f}% | "
                    f"+{r['g_c_b']:.1f} | {r['pct_c_b']:.1f}% | "
                    f"{r['mean_chem_pct']:.1f}% | {c} | {e} |\n")

    print(f"\nSaved: {out_file}")
    print(f"\nKey results:")
    print(f"  Neurons with preferential chemical growth (both reps): {len(hits_c)}")
    print(f"  Neurons with preferential gap-junction growth (both reps): {len(hits_e)}")
    print(f"  Neurons with both: {len(hits_both)}")
    if hits_c:
        print(f"\n  Chemical hits:")
        for r in sorted(hits_c, key=lambda x: x["mean_chem_pct"], reverse=True):
            print(f"    {r['neuron']:12s}  chem pct A={r['pct_c_a']:.1f}%  B={r['pct_c_b']:.1f}%")
    if hits_e:
        print(f"\n  Gap-junction hits:")
        for r in sorted(hits_e, key=lambda x: (x["pct_e_a"]+x["pct_e_b"])/2, reverse=True):
            print(f"    {r['neuron']:12s}  gap pct A={r['pct_e_a']:.1f}%  B={r['pct_e_b']:.1f}%")


if __name__ == "__main__":
    main()
