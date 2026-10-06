"""
run_benchmarks.py — Helper runner script for Phase 2 connectome benchmark.
"""

from baseline.tgn import run_connectome_tgn_experiment

if __name__ == "__main__":
    print("=" * 65)
    print("Running Phase 2 Connectome 5-Seed Benchmark (Real Witvliet Data)...")
    print("=" * 65)
    res = run_connectome_tgn_experiment(epochs=5, seeds=(42, 12345, 123, 456, 789))
    print("-" * 65)
    print(f"Mean Test AUC (AUROC): {res['mean_test_auc']:.4f} +/- {res['std_test_auc']:.4f}")
    print(f"Mean Test AP  (AUPRC): {res['mean_test_ap']:.4f} +/- {res['std_test_ap']:.4f}")
    print("-" * 65)
    for r in res["results"]:
        print(f"  Seed {r['seed']}: Val AUC = {r['val_auc']:.4f} | Test AUC = {r['test_auc']:.4f}, Test AP = {r['test_ap']:.4f}")
    print("=" * 65)
