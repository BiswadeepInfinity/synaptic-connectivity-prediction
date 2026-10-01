"""
ablations/run_ablations.py — Phase 6: Ablation study runner.

Full ablation set (THE_BOOK §Phase 6):
  1. With / without pattern module (Module C)
  2. Learned vs. random (untrained) motif dictionary
  3. K sweep: {5, 10, 20, 50} dictionary slots
  4. Window sweep: k ∈ {2, 3, 4}
  5. A1 vs. A2 trajectory (interpolation vs. blind structural inference)

Full baseline ladder (THE_BOOK §Phase 6):
  persistence → static GNN → plain TGN → motif-representation baseline
  → proposed PCDGM

Rules:
  - 5 seeds minimum on every reported number.
  - Seed propagated to data, model, and dictionary init.
  - Do NOT change evaluation protocol between baseline and proposed-model runs.
  - Report ΔAUROC(learned) − ΔAUROC(random) with paired significance test.
  - Report slot usage entropy alongside K sweep results.

TODO (Phase 6): implement run_all_ablations().
"""

ABLATION_K_VALUES      = [5, 10, 20, 50]
ABLATION_WINDOW_VALUES = [2, 3, 4]
MIN_SEEDS              = 5


def run_all_ablations(config: dict) -> dict:
    """
    Run the full ablation set and return a results dict.
    TODO (Phase 6): implement.
    """
    raise NotImplementedError("Ablations (Phase 6) not yet implemented.")


if __name__ == "__main__":
    raise NotImplementedError(
        "Phase 6 not yet implemented. Complete Phases 0–5 first."
    )
