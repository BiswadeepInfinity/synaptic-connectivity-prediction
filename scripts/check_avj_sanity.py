"""
scripts/check_avj_sanity.py — Sanity check for AVJ full-brain hit.

Questions answered:
  1. What is AVJ's in-degree at each snapshot? Is the pool size (N=18) a concern?
  2. Who sends connections TO AVJ (presynaptic partners) at 0h vs 45h?
     i.e., which neurons are responsible for the gain?
  3. Do AVJ's presynaptic partners overlap with those of AVE/AVB?
     (if the locomotor circuit wires as a unit, they should share senders)
  4. Is AVJ's low 0h baseline (4 connections) causing a small-pool artefact?
     Re-run null with wider window (±40%) to check stability.
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

HOURS = [0, 5, 8, 16, 23, 27, 45, 45]
REP_LABELS = ["0h","5h","8h","16h","23h","27h","45h-A","45h-B"]

def sep(t=""): print("="*65); print(f"  {t}") if t else None; print("="*65) if t else None

def null_percentile(chem_in, idx, t_adult, B, window, rng, N):
    d0 = chem_in[idx, 0]
    g  = chem_in[idx, t_adult] - d0
    low, high = d0 * (1 - window), d0 * (1 + window)
    matched = [j for j in range(N) if j != idx and low <= chem_in[j, 0] <= high]
    if len(matched) < 3:
        return None, None, len(matched)
    samp = rng.choice(matched, size=B, replace=True)
    null_g = chem_in[samp, t_adult] - chem_in[samp, 0]
    pct = np.mean(null_g <= g) * 100
    return pct, g, len(matched)

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

    # Raw adjacency arrays [N, N, 8]
    chem = np.stack([chem_snaps[t].numpy() for t in range(8)], axis=2)  # [N,N,8]
    elec = np.stack([elec_snaps[t].numpy() for t in range(8)], axis=2)

    chem_in = chem.sum(axis=0)   # [N, 8] in-degree (col sums)
    elec_deg = elec.sum(axis=0)

    rng = np.random.default_rng(42)
    B = 1000

    targets = {
        "AVJL": node2idx["AVJL"],
        "AVJR": node2idx["AVJR"],
        "AVEL": node2idx["AVEL"],
        "AVER": node2idx["AVER"],
        "AVBL": node2idx["AVBL"],
        "AVBR": node2idx["AVBR"],
    }

    # ------------------------------------------------------------------ #
    # CHECK 1: Full 8-snapshot in-degree profile for AVJ
    # ------------------------------------------------------------------ #
    sep("CHECK 1: AVJ longitudinal chemical in-degree profile")
    print(f"  {'Neuron':<8} " + "  ".join(f"{l:>6}" for l in REP_LABELS))
    print("  " + "-"*60)
    for name in ["AVJL","AVJR"]:
        idx = targets[name]
        row = [int(chem_in[idx, t]) for t in range(8)]
        print(f"  {name:<8} " + "  ".join(f"{v:>6}" for v in row))
    print()

    # ------------------------------------------------------------------ #
    # CHECK 2: Who sends connections to AVJL at 0h vs 45h-A?
    # New connections = present in 45h-A but not in 0h
    # ------------------------------------------------------------------ #
    sep("CHECK 2: Presynaptic partners of AVJL — new connections by 45h-A")
    avjl = targets["AVJL"]
    pre_0h  = set(np.where(chem[:, avjl, 0] > 0)[0])   # neurons sending to AVJL at 0h
    pre_45a = set(np.where(chem[:, avjl, 6] > 0)[0])   # at 45h-A
    pre_45b = set(np.where(chem[:, avjl, 7] > 0)[0])   # at 45h-B

    new_A = pre_45a - pre_0h
    new_B = pre_45b - pre_0h
    new_both = new_A & new_B

    print(f"  In-degree 0h:   {len(pre_0h)}")
    print(f"  In-degree 45h-A: {len(pre_45a)}  (new: {len(new_A)})")
    print(f"  In-degree 45h-B: {len(pre_45b)}  (new: {len(new_B)})")
    print(f"  New in BOTH replicates: {len(new_both)}")
    print(f"  New senders (both reps): {sorted([all_nodes[j] for j in new_both])}")
    print(f"  New senders (A only):    {sorted([all_nodes[j] for j in new_A - new_B])}")
    print(f"  New senders (B only):    {sorted([all_nodes[j] for j in new_B - new_A])}")
    print()

    # ------------------------------------------------------------------ #
    # CHECK 3: Overlap — do AVJL's new senders overlap with AVE/AVB new senders?
    # ------------------------------------------------------------------ #
    sep("CHECK 3: Overlap of new presynaptic partners with AVE and AVB")
    locomotor = {
        "AVEL": targets["AVEL"],
        "AVER": targets["AVER"],
        "AVBL": targets["AVBL"],
        "AVBR": targets["AVBR"],
    }
    avj_new_both = new_both  # new senders to AVJL confirmed in both reps

    print(f"  AVJL new senders (both reps): {sorted([all_nodes[j] for j in avj_new_both])}")
    print()
    for name, idx in locomotor.items():
        pre0  = set(np.where(chem[:, idx, 0] > 0)[0])
        p45a  = set(np.where(chem[:, idx, 6] > 0)[0])
        p45b  = set(np.where(chem[:, idx, 7] > 0)[0])
        new_n = (p45a - pre0) & (p45b - pre0)
        overlap = avj_new_both & new_n
        print(f"  {name}: new senders (both reps) = {len(new_n)}, "
              f"overlap with AVJL = {len(overlap)}")
        if overlap:
            print(f"    Shared: {sorted([all_nodes[j] for j in overlap])}")
    print()

    # ------------------------------------------------------------------ #
    # CHECK 4: Pool-size stability — does the percentile hold at ±40%?
    # ------------------------------------------------------------------ #
    sep("CHECK 4: Null stability — does result hold with wider ±40% window?")
    print(f"  {'Neuron':<8} {'Window':>8} {'Pool N':>8} {'Pct Rep-A':>11} {'Pct Rep-B':>11}")
    print("  " + "-"*50)
    for name in ["AVJL","AVJR"]:
        idx = targets[name]
        for window, label in [(0.20, "±20%"), (0.40, "±40%"), (0.60, "±60%")]:
            pct_a, g_a, pool = null_percentile(chem_in, idx, 6, B, window, rng, N)
            pct_b, g_b, _    = null_percentile(chem_in, idx, 7, B, window, rng, N)
            if pct_a is None:
                print(f"  {name:<8} {label:>8} {'<3 (skip)':>8}")
            else:
                print(f"  {name:<8} {label:>8} {pool:>8} {pct_a:>10.1f}% {pct_b:>10.1f}%")
    print()

    # ------------------------------------------------------------------ #
    # CHECK 5: Is the gain driven by one big snapshot jump or gradual?
    # ------------------------------------------------------------------ #
    sep("CHECK 5: Is AVJL's growth gradual or a single-step jump?")
    print(f"  Snapshot-to-snapshot gains for AVJL (chemical in-degree):")
    avjl_profile = [int(chem_in[avjl, t]) for t in range(8)]
    for t in range(1, 8):
        delta = avjl_profile[t] - avjl_profile[t-1]
        label = f"{REP_LABELS[t-1]}->{REP_LABELS[t]}"
        bar = "+" * max(0, delta) if delta >= 0 else "-" * abs(delta)
        print(f"    {label:<14} {delta:+3d}  {bar}")

    sep("SANITY CHECK COMPLETE")


if __name__ == "__main__":
    main()
