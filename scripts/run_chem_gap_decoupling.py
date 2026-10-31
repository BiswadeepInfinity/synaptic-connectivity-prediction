"""
scripts/run_chem_gap_decoupling.py — Do chemical and gap-junction synapses
grow together or independently across the 225-neuron population?

Analysis:
  1. For each analysed neuron: compute total chemical gain and gap-junction gain
     (average of Rep A and Rep B to reduce replicate noise).
  2. Pearson and Spearman correlation of chem_gain vs gap_gain.
  3. Scatter statistics: how many neurons grow a lot in chem but not gap, vice versa?
  4. Check where the H3 hits fall in the 2D space.
  5. Save results/chem_gap_decoupling.md
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats

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

MIN_POOL = 5
SEED = 42


def main():
    print("Loading data...")
    chem_snaps, elec_snaps, hours, labels = load_all_snapshots_both()
    readers = [get1(),get2(),get3(),get4(),get5(),get6(),get7(),get8()]
    all_nodes = sorted(list(set.union(*[
        set(r.to_networkx_graph("Generic_CS").nodes()).union(
            set(r.to_networkx_graph("Generic_GJ").nodes()))
        for r in readers
    ])))
    node2idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    chem_in  = np.stack([chem_snaps[t].numpy().sum(axis=0) for t in range(8)], axis=1)  # [N,8]
    elec_deg = np.stack([elec_snaps[t].numpy().sum(axis=0) for t in range(8)], axis=1)

    rng = np.random.default_rng(SEED)
    B   = 1000

    # Known hits from full-brain screen
    chem_hits = {"AVJL","AVJR","RMHR","RMFR","RMDR","RMHL"}
    gap_hits  = {"AQR","AVKL","AVEL","AVKR","RIGL","RIBR","OLQDR","SIBVR","AVM","RIGR","RMHR"}
    both_hits = chem_hits & gap_hits  # {"RMHR"}

    # H3-direct neurons (confirmed in bilateral analysis)
    h3_direct = {"AVAL","AVAR","AVBL","AVBR","AVEL","AVER"}

    # ------------------------------------------------------------------ #
    # Build per-neuron gain vectors (mean of Rep A and Rep B)
    # ------------------------------------------------------------------ #
    neurons, chem_gains, gap_gains = [], [], []
    chem_pcts_a, chem_pcts_b = [], []
    gap_pcts_a,  gap_pcts_b  = [], []

    for idx, neuron in enumerate(all_nodes):
        d0_c = chem_in[idx, 0]
        low, high = d0_c * 0.8, d0_c * 1.2
        matched = [j for j in range(N) if j != idx and low <= chem_in[j,0] <= high]
        if len(matched) < MIN_POOL:
            continue

        g_c = (chem_in[idx,6] + chem_in[idx,7]) / 2.0 - d0_c
        g_e = (elec_deg[idx,6] + elec_deg[idx,7]) / 2.0 - elec_deg[idx,0]

        samp = rng.choice(matched, size=B, replace=True)
        null_c_a = chem_in[samp,6] - chem_in[samp,0]
        null_c_b = chem_in[samp,7] - chem_in[samp,0]
        null_e_a = elec_deg[samp,6] - elec_deg[samp,0]
        null_e_b = elec_deg[samp,7] - elec_deg[samp,0]

        pct_ca = np.mean(null_c_a <= (chem_in[idx,6] - d0_c)) * 100
        pct_cb = np.mean(null_c_b <= (chem_in[idx,7] - d0_c)) * 100
        d0_e   = elec_deg[idx,0]
        pct_ea = np.mean(null_e_a <= (elec_deg[idx,6] - d0_e)) * 100
        pct_eb = np.mean(null_e_b <= (elec_deg[idx,7] - d0_e)) * 100

        neurons.append(neuron)
        chem_gains.append(g_c)
        gap_gains.append(g_e)
        chem_pcts_a.append(pct_ca)
        chem_pcts_b.append(pct_cb)
        gap_pcts_a.append(pct_ea)
        gap_pcts_b.append(pct_eb)

    chem_gains = np.array(chem_gains)
    gap_gains  = np.array(gap_gains)
    chem_pcts_a = np.array(chem_pcts_a)
    chem_pcts_b = np.array(chem_pcts_b)
    gap_pcts_a  = np.array(gap_pcts_a)
    gap_pcts_b  = np.array(gap_pcts_b)
    n_analysed  = len(neurons)

    # ------------------------------------------------------------------ #
    # Correlations
    # ------------------------------------------------------------------ #
    pearson_r,  pearson_p  = stats.pearsonr(chem_gains, gap_gains)
    spearman_r, spearman_p = stats.spearmanr(chem_gains, gap_gains)

    # Also correlate percentiles
    mean_chem_pct = (chem_pcts_a + chem_pcts_b) / 2
    mean_gap_pct  = (gap_pcts_a  + gap_pcts_b)  / 2
    pr_pct, pp_pct = stats.pearsonr(mean_chem_pct, mean_gap_pct)
    sr_pct, sp_pct = stats.spearmanr(mean_chem_pct, mean_gap_pct)

    # ------------------------------------------------------------------ #
    # Quadrant analysis
    # ------------------------------------------------------------------ #
    chem_median = np.median(chem_gains)
    gap_median  = np.median(gap_gains)

    q_high_both  = [(neurons[i], chem_gains[i], gap_gains[i])
                    for i in range(n_analysed)
                    if chem_gains[i] > chem_median and gap_gains[i] > gap_median]
    q_chem_only  = [(neurons[i], chem_gains[i], gap_gains[i])
                    for i in range(n_analysed)
                    if chem_gains[i] > chem_median and gap_gains[i] <= gap_median]
    q_gap_only   = [(neurons[i], chem_gains[i], gap_gains[i])
                    for i in range(n_analysed)
                    if chem_gains[i] <= chem_median and gap_gains[i] > gap_median]
    q_low_both   = [(neurons[i], chem_gains[i], gap_gains[i])
                    for i in range(n_analysed)
                    if chem_gains[i] <= chem_median and gap_gains[i] <= gap_median]

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print(f"  CHEMICAL vs GAP-JUNCTION GROWTH CORRELATION ({n_analysed} neurons)")
    print(f"{'='*65}")
    print(f"\n  Raw gain correlation:")
    print(f"    Pearson  r = {pearson_r:+.4f}  p = {pearson_p:.4f}")
    print(f"    Spearman r = {spearman_r:+.4f}  p = {spearman_p:.4f}")
    print(f"\n  Null-percentile correlation:")
    print(f"    Pearson  r = {pr_pct:+.4f}  p = {pp_pct:.4f}")
    print(f"    Spearman r = {sr_pct:+.4f}  p = {sp_pct:.4f}")
    print(f"\n  Medians:  chem gain = {chem_median:.1f},  gap gain = {gap_median:.1f}")
    print(f"\n  Quadrant counts (split at medians):")
    print(f"    High chem & high gap  : {len(q_high_both):3d} neurons")
    print(f"    High chem only        : {len(q_chem_only):3d} neurons")
    print(f"    High gap only         : {len(q_gap_only):3d} neurons")
    print(f"    Low both              : {len(q_low_both):3d} neurons")

    print(f"\n  Where do H3 hits land?")
    for label_set, tag in [
        (chem_hits, "Chem hit"),
        (gap_hits,  "Gap hit"),
        (h3_direct, "H3-direct"),
    ]:
        for n in sorted(label_set):
            if n in neurons:
                i = neurons.index(n)
                cg, gg = chem_gains[i], gap_gains[i]
                quad = ("High-both" if cg > chem_median and gg > gap_median
                        else "Chem-only" if cg > chem_median
                        else "Gap-only"  if gg > gap_median
                        else "Low-both")
                print(f"    {tag:<12} {n:<8}  chem={cg:+6.1f}  gap={gg:+5.1f}  [{quad}]")

    # ------------------------------------------------------------------ #
    # Write markdown
    # ------------------------------------------------------------------ #
    out = REPO_ROOT / "results" / "chem_gap_decoupling.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Chemical vs Gap-Junction Growth Decoupling Analysis\n\n")
        f.write(f"> {n_analysed} neurons analysed (0h chemical in-degree matched pool >= {MIN_POOL}). "
                f"Gain = mean of 45h-A and 45h-B minus 0h. Seed {SEED}.\n\n---\n\n")

        f.write("## 1. Correlation of Chemical and Gap-Junction Growth\n\n")
        f.write("| Metric | Pearson r | p-value | Spearman r | p-value |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Raw gain (synapses) | {pearson_r:+.4f} | {pearson_p:.4f} | "
                f"{spearman_r:+.4f} | {spearman_p:.4f} |\n")
        f.write(f"| Null percentile | {pr_pct:+.4f} | {pp_pct:.4f} | "
                f"{sr_pct:+.4f} | {sp_pct:.4f} |\n\n")

        if abs(pearson_r) < 0.3:
            interp = ("**Weak/no correlation.** Chemical and gap-junction growth are largely "
                      "independent — different wiring programmes act on different neuron populations.")
        elif abs(pearson_r) < 0.6:
            interp = ("**Moderate correlation.** There is partial overlap between chemical "
                      "and gap-junction growth, but the two synapse types are not co-regulated.")
        else:
            interp = ("**Strong correlation.** Chemical and gap-junction growth co-occur in "
                      "the same neurons — a single shared wiring programme dominates.")

        f.write(f"**Interpretation:** {interp}\n\n---\n\n")

        f.write("## 2. Quadrant Analysis (split at population medians)\n\n")
        f.write(f"Chemical gain median: {chem_median:.1f} synapses | "
                f"Gap-junction gain median: {gap_median:.1f}\n\n")
        f.write("| Quadrant | Count | % of total |\n|---|---|---|\n")
        for label, q in [
            ("High chemical & high gap-junction", q_high_both),
            ("High chemical only", q_chem_only),
            ("High gap-junction only", q_gap_only),
            ("Low both", q_low_both),
        ]:
            pct = len(q) / n_analysed * 100
            f.write(f"| {label} | {len(q)} | {pct:.1f}% |\n")
        f.write("\n---\n\n")

        f.write("## 3. Position of H3 Hits in 2D Growth Space\n\n")
        f.write("| Neuron | Source | Chem Gain | Gap Gain | Quadrant |\n")
        f.write("|---|---|---|---|---|\n")
        all_hits = {n: "Chem+Gap hit" for n in both_hits}
        all_hits.update({n: "Chem hit" for n in chem_hits - both_hits})
        all_hits.update({n: "Gap hit" for n in gap_hits - both_hits})
        all_hits.update({n: "H3-direct" for n in h3_direct if n not in all_hits})
        for n, tag in sorted(all_hits.items()):
            if n in neurons:
                i = neurons.index(n)
                cg, gg = chem_gains[i], gap_gains[i]
                quad = ("High-both" if cg > chem_median and gg > gap_median
                        else "Chem-only" if cg > chem_median
                        else "Gap-only"  if gg > gap_median
                        else "Low-both")
                f.write(f"| **{n}** | {tag} | {cg:+.1f} | {gg:+.1f} | {quad} |\n")
        f.write("\n---\n\n")

        f.write("## 4. Top 15 Neurons by Chemical Gain\n\n")
        order = np.argsort(chem_gains)[::-1]
        f.write("| Rank | Neuron | Chem Gain | Gap Gain |\n|---|---|---|---|\n")
        for rank, i in enumerate(order[:15], 1):
            f.write(f"| {rank} | {neurons[i]} | {chem_gains[i]:+.1f} | {gap_gains[i]:+.1f} |\n")
        f.write("\n## 5. Top 15 Neurons by Gap-Junction Gain\n\n")
        order_g = np.argsort(gap_gains)[::-1]
        f.write("| Rank | Neuron | Gap Gain | Chem Gain |\n|---|---|---|---|\n")
        for rank, i in enumerate(order_g[:15], 1):
            f.write(f"| {rank} | {neurons[i]} | {gap_gains[i]:+.1f} | {chem_gains[i]:+.1f} |\n")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
