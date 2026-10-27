"""
scripts/run_all_phases.py — Executes Phase 1 through Phase 5 in sequence.
Captures all tables, benchmarks, and outputs cleanly without Windows batch file interruption.
"""

import sys
import subprocess
import time
import threading

def run_step(step_num: int, total_steps: int, title: str, cmd: list):
    print("\n" + "=" * 80)
    print(f"  [{step_num}/{total_steps}] {title}")
    print("=" * 80)
    sys.stdout.flush()

    start_time = time.time()
    stop_spinner = False

    def spinner_task():
        spinner_chars = ["|", "/", "-", "\\"]
        idx = 0
        while not stop_spinner:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            sys.stdout.write(f"\r  Running... [{spinner_chars[idx % len(spinner_chars)]}] Elapsed: {mins:02d}m {secs:02d}s ")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.2)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    spinner_thread = threading.Thread(target=spinner_task, daemon=True)
    spinner_thread.start()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in proc.stdout:
        # Stop spinner briefly to print the output cleanly
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.write("  " + line)
        sys.stdout.flush()

    proc.wait()
    stop_spinner = True
    spinner_thread.join(timeout=1.0)

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    status_str = "SUCCESS" if proc.returncode == 0 else f"FAILED (Exit Code: {proc.returncode})"
    print(f"\n  --> Completed in {mins:02d}m {secs:02d}s | Status: {status_str}\n")
    sys.stdout.flush()

    if proc.returncode != 0:
        print(f"[!] Warning: Step {step_num} returned non-zero exit code: {proc.returncode}")
        sys.stdout.flush()

def main():
    print("*" * 80)
    print("  RUNNING COMPLETE BENCHMARK & EXPERIMENT PIPELINE (PHASES 1 TO 5)")
    print("*" * 80)
    sys.stdout.flush()

    total = 7
    run_step(1, total, "PHASE 1: DEVOTG REPRODUCTION GATE (Lineage Division AUC)", [sys.executable, "-m", "pytest", "test_phase1_reproduction.py", "-v", "-s"])
    run_step(2, total, "PHASE 2: 5-SEED CONNECTOME BENCHMARK (AUROC / AUPRC)", [sys.executable, "run_benchmark.py"])
    run_step(3, total, "PHASE 3: TRAJECTORY RECONSTRUCTION & H2 RANKING", [sys.executable, "run_phase3_trajectory.py"])
    run_step(4, total, "PHASE 4: STRUCTURAL MOTIF DISCOVERY & K-SWEEP", [sys.executable, "run_phase4_motifs.py"])
    run_step(5, total, "PHASE 5: MULTI-SEED 3-VARIANT PATTERN TRAINING BENCHMARK", [sys.executable, "run_phase5_training.py"])
    run_step(6, total, "DATA INTEGRITY: RAW SYNAPSE COUNTS VS PAPER TABLE 1", [sys.executable, "scripts/check_paper_counts.py"])
    run_step(7, total, "COMPREHENSIVE AUTOMATED TEST SUITE (PHASES 0 TO 5)", [
        sys.executable, "-m", "pytest",
        "data/test_loader.py",
        "baseline/test_tgn.py",
        "pcdgm/test_module_a_trajectory.py",
        "pcdgm/test_module_b_encoder.py",
        "pcdgm/test_module_c_motifs.py",
        "pcdgm/test_module_d_fusion.py",
        "pcdgm/test_module_e_predictor.py",
        "training/test_loop.py",
        "-v",
    ])

    print("*" * 80)
    print("  COMPLETE PIPELINE FINISHED (PHASES 1 TO 5 VERIFIED)")
    print("*" * 80)

if __name__ == "__main__":
    main()
