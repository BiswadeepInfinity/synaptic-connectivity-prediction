@echo off
setlocal EnableDelayedExpansion

:menu
cls
echo =========================================================================================
echo       PREDICTING SYNAPTIC CONNECTIVITY CHANGE IN DEVELOPMENTAL CONNECTOMES
echo       Research Workbench ^& Experiment Runner ^| C. elegans Connectome (Witvliet 2021)
echo =========================================================================================
echo.
echo  SCIENTIFIC OBJECTIVE:
echo   Predict WHEN and WHERE synapses form, strengthen, or disappear across development
echo   using continuous developmental trajectories and discovered structural motifs.
echo.
echo  DATASET SUMMARY:
echo   - Organism: Caenorhabditis elegans (Witvliet et al., Nature 2021)
echo   - 8 EM reconstructions: Birth (0h) to Adult (45h Replicates A ^& B)
echo   - Scale: 225 identified neurons ^| 12,912 cumulative synaptic interactions
echo   - Train: 0h, 5h, 8h, 16h, 23h (6,211 events) ^| Val: 27h (1,737) ^| Test: 45h (4,964)
echo.
echo -----------------------------------------------------------------------------------------
echo  SELECT EXPERIMENT OR DIAGNOSTIC:
echo -----------------------------------------------------------------------------------------
echo   [1] Run Phase 1 Gate: DevoTG Lineage Reproduction (Target: AUC 0.839 +/- 0.007)
echo       - Verifies that our environment reproduces Gayen ^& Alicea (arXiv:2606.21940)
echo.
echo   [2] Run Phase 2 Connectome Unit Tests (PyTest)
echo       - Verifies zero temporal leakage and separate L_edge / L_total loss logging
echo.
echo   [3] Run Phase 2 Connectome 5-Seed Baseline Benchmark (Real Witvliet Data)
echo       - Trains 5 independent models across seeds [42, 12345, 123, 456, 789]
echo       - Prints per-seed and aggregate AUROC (mean ~ 64.08%%) and AUPRC (mean ~ 62.11%%)
echo.
echo   [4] Run Comprehensive Diagnostics ^& Data Integrity Verification
echo       - Neuron ID consistency across 8 snapshots, degree distributions, edge checks
echo.
echo   [5] Launch Interactive Synaptic Visualizer (Web Dashboard)
echo       - Explore node-by-node predictions, true positives, false positives, hub neurons
echo.
echo   [6] Run ALL Automated Tests (Phase 0 + Phase 1 + Phase 2)
echo.
echo   -----------------------------------------------------------------------------------------
echo   PHASE 3 -- TRAJECTORY RECONSTRUCTION ^& H2 VALIDATION  [NEW]
echo   -----------------------------------------------------------------------------------------
echo   [7]  Phase 3 -- Full Output: Ground-Truth Counts + Interpolation + H2 Ranking
echo        - Raw synapse counts for all 8 snapshots (both 45h replicates shown separately)
echo        - PCHIP interpolated density at intermediate timepoints
echo        - Module A2 blind structural ordering + Spearman rho vs real hours
echo.
echo   [8]  Phase 3 -- Data Integrity Check: Raw Counts vs Witvliet 2021 Paper
echo        - Compares int(snap.sum()) for every snapshot against paper Table 1 values
echo        - Catches silent loader discrepancies before they propagate downstream
echo.
echo   [9]  Phase 3 -- A2 Metric Audit: Topology vs Edge Count
echo        - Shows feature vector components per snapshot (reciprocity, entropy, clustering)
echo        - Confirms ordering is structural, not trivially edge-count-based
echo.
echo   [10] Phase 3 -- Full Test Suite (Module A unit tests + paper-count gate)
echo        - All A1/A2 unit tests + live paper-count verification + raw count cross-check
echo.
echo   -----------------------------------------------------------------------------------------
echo   PHASE 4 -- STRUCTURAL MOTIF DISCOVERY ^& K-SWEEP  [NEW]
echo   -----------------------------------------------------------------------------------------
echo   [11] Phase 4 -- Full Output: Synthetic Sanity Check + Real K-Sweep Tables
echo        - Verifies 4-node clique recovery on synthetic data (Phase 4 gate)
echo        - K-sweep across {5, 10, 20, 50} for Chemical and Electrical synapses
echo        - Full reconstruction loss, slot usage entropy, and developmental routing
echo.
echo   [12] Phase 4 -- Unit ^& Integration Test Suite (Module C tests)
echo        - Dictionary collapse detection, entropy checks, and synthetic recovery gate
echo.
echo   -----------------------------------------------------------------------------------------
echo   PHASE 5 -- PATTERN-CONDITIONED TRAINING LOOP ^& BENCHMARK  [NEW]
echo   -----------------------------------------------------------------------------------------
echo   [13] Phase 5 -- Full Benchmark: 3 Motif Variants across 5 Seeds
echo        - Evaluates: No Motif vs. Random Untrained vs. Learned Patterns (PCDGM)
echo        - Phase 2 split (replicates together), negative sampling, AUROC / AUPRC vs 0.6527
echo.
echo   [14] Phase 5 -- Comprehensive Test Suite (Modules B, D, E ^& Training Loop)
echo        - Tests: Module B encoder, Module D fusion, Module E predictor, TrainingLoop
echo.
echo   -----------------------------------------------------------------------------------------
echo   RUN EVERYTHING ^& EXIT
echo   -----------------------------------------------------------------------------------------
echo   [15] Run EVERYTHING: Phase 1 (Reproduction) + Phase 2 (5-Seed Benchmark) + Phase 3 (Trajectory) + Phase 4 (Motifs) + Phase 5 (PCDGM Benchmark ^& Tests)
echo        - Runs all benchmark models, outputs all tables, and executes all test suites (Phases 0 to 5)
echo.
echo   [16] Exit
echo -----------------------------------------------------------------------------------------
echo.

set /p OPTION="Select option [1-16]: "

if "%OPTION%"=="" goto menu
goto opt_%OPTION% 2>nul
echo.
echo [!] Invalid selection '%OPTION%'. Please enter a number between 1 and 16.
goto pause_step

:opt_1
    echo.
    echo =====================================================================================
    echo [RUNNING PHASE 1 GATE]
    echo Target: AUC 0.839 +/- 0.007 across 5 seeds on cell-lineage division events.
    echo Rule: Must pass before any novel PCDGM module is trusted.
    echo =====================================================================================
    python -m pytest test_phase1_reproduction.py -v -s
    goto pause_step

:opt_2
    echo.
    echo =====================================================================================
    echo [RUNNING PHASE 2 UNIT TESTS]
    echo Checking: Strict chronological order, no future leakage, replicate integrity.
    echo =====================================================================================
    python -m pytest baseline/test_tgn.py -k "TestPhase2TGNBaseline" -v -s
    goto pause_step

:opt_3
    echo.
    echo =====================================================================================
    echo [RUNNING PHASE 2 5-SEED CONNECTOME BENCHMARK]
    echo Evaluating Plain TGN link prediction on real 45h adult connectomes.
    echo =====================================================================================
    python run_benchmark.py
    goto pause_step

:opt_4
    python research_audit.py
    goto pause_step

:opt_5
    echo.
    echo =====================================================================================
    echo [STARTING LOCAL VISUALIZER SERVER]
    echo URL: http://localhost:8000
    echo =====================================================================================
    start http://localhost:8000
    python -m http.server 8000 --directory visualizer
    goto pause_step

:opt_6
    echo.
    echo =====================================================================================
    echo [RUNNING COMPLETE TEST SUITE - Phase 0 + Phase 1 + Phase 2]
    echo =====================================================================================
    python -m pytest data/test_loader.py baseline/test_tgn.py test_phase1_reproduction.py -v -s
    goto pause_step

:opt_7
    echo.
    echo =====================================================================================
    echo [PHASE 3] FULL OUTPUT: Ground-Truth Counts + Interpolation + H2 Ranking
    echo.
    echo  What you will see:
    echo   TABLE 1 - Ground Truth: raw synapse counts for all 8 snapshots
    echo             Both 45h replicates ^(YA_45h_A and YA_45h_B^) shown as separate rows
    echo             These numbers come directly from the loaded tensors - no processing
    echo.
    echo   TABLE 2 - PCHIP Interpolated: mean adjacency density at intermediate hours
    echo             ^(2.5h, 11.5h, 20.0h, 33.0h^) - timepoints not in the real dataset
    echo.
    echo   TABLE 3 - Module A2 H2 Ranking: blind structural ordering of all 8 snapshots
    echo             Spearman rho vs real hours - target gate is rho ^> 0.3
    echo             Feature: normalized degrees + reciprocity + entropy + clustering
    echo             ^(NOT raw edge count^)
    echo =====================================================================================
    echo.
    python run_phase3_trajectory.py
    goto pause_step

:opt_8
    echo.
    echo =====================================================================================
    echo [PHASE 3] DATA INTEGRITY: Raw Loaded Counts vs Witvliet 2021 Paper Values
    echo.
    echo  This compares int^(snap.sum^(^)^) for every snapshot against the paper's
    echo  reported edge counts. A mismatch here means the loader is filtering or
    echo  missing connections before any downstream module even runs.
    echo.
    echo  Paper reference counts ^(Witvliet 2021 Table 1 - Chemical / Electrical directed^):
    echo    Chem: S1: 775  S2: 986  S3: 1012 S4: 1136 S5: 1515 S6: 1525 S7: 2191 S8: 2186
    echo    Elec: S1: 162  S2: 244  S3: 184  S4: 412  S5: 572  S6: 424  S7: 570  S8: 604
    echo =====================================================================================
    echo.
    python scripts\check_paper_counts.py
    goto pause_step

:opt_9
    echo.
    echo =====================================================================================
    echo [PHASE 3] A2 METRIC AUDIT: What Is the Structural Feature Actually Measuring?
    echo.
    echo  Shows the 4 topological scalars per snapshot ^(reciprocity, in-entropy,
    echo  out-entropy, clustering^) alongside raw edge count for comparison.
    echo  If the structural ordering were just edge-counting, reciprocity/entropy/
    echo  clustering would carry no signal. This table lets you verify they vary.
    echo =====================================================================================
    echo.
    python scripts\audit_a2_metrics.py
    goto pause_step

:opt_10
    echo.
    echo =====================================================================================
    echo [PHASE 3] FULL TEST SUITE: Module A Unit Tests + Paper-Count Gate
    echo.
    echo  Tests run:
    echo   data/test_loader.py           - Phase 0 gate incl. live paper-count check
    echo   pcdgm/test_module_a_trajectory.py - A1/A2 unit tests + H2 on real data
    echo =====================================================================================
    echo.
    python -m pytest data/test_loader.py pcdgm/test_module_a_trajectory.py -v -s
    goto pause_step

:opt_11
    echo.
    echo =====================================================================================
    echo [PHASE 4] FULL OUTPUT: Synthetic Sanity Check + Real K-Sweep Tables
    echo.
    echo  What you will see:
    echo   1. Synthetic Sanity Check: 4-node clique recovery on alternating windows
    echo   2. Table 1: Chemical Synapses K-Sweep across K in {5, 10, 20, 50}
    echo   3. Table 2: Electrical Synapses K-Sweep across K in {5, 10, 20, 50}
    echo   4. Developmental Attention Routing across 6 developmental windows (k=3)
    echo =====================================================================================
    echo.
    python run_phase4_motifs.py
    goto pause_step

:opt_12
    echo.
    echo =====================================================================================
    echo [PHASE 4] MODULE C TEST SUITE: Dictionary Unit Tests + Injected Recovery Gate
    echo.
    echo  Tests run:
    echo   pcdgm/test_module_c_motifs.py - collapse detection, entropy, full window recon,
    echo                                  random vs learned ablation hook, synthetic gate
    echo =====================================================================================
    echo.
    python -m pytest pcdgm/test_module_c_motifs.py -v -s
    goto pause_step

:opt_13
    echo.
    echo =====================================================================================
    echo [PHASE 5] STEPS 3 ^& 4: SPREAD GATE EVALUATION ^& PAIRED 3-VARIANT BENCHMARK
    echo  Step 3: Spread Gate (Cosine Sim, Std, Entropy, Degree R^2 across 6 Windows)
    echo  Step 4: Paired 5-Seed Retest (No-Motif vs. Random vs. Learned PCDGM with tau=0.1)
    echo =====================================================================================
    echo.
    python scripts\run_attempts_gate.py
    echo.
    echo -------------------------------------------------------------------------------------
    echo  Proceeding to Step 4 Paired Multi-Seed Retest Benchmark...
    echo -------------------------------------------------------------------------------------
    python run_phase5_training.py
    goto pause_step

:opt_14
    echo.
    echo =====================================================================================
    echo [PHASE 5] TEST SUITE: Modules B, D, E and Training Loop Tests
    echo  Tests run:
    echo   pcdgm/test_module_b_encoder.py
    echo   pcdgm/test_module_d_fusion.py
    echo   pcdgm/test_module_e_predictor.py
    echo   training/test_loop.py
    echo =====================================================================================
    echo.
    python -m pytest pcdgm/test_module_b_encoder.py pcdgm/test_module_d_fusion.py pcdgm/test_module_e_predictor.py training/test_loop.py -v -s
    goto pause_step

:opt_15
    echo.
    echo =====================================================================================
    echo [RUNNING FULL PIPELINE: Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5]
    echo  All benchmarks, tables, and test suites (Phases 0 to 5) executed in sequence.
    echo =====================================================================================
    echo.
    python scripts\run_all_phases.py
    goto pause_step

:opt_16
    echo Exiting workbench.
    exit /b 0

:pause_step
echo.
echo =====================================================================================
echo  [M] Return to Main Menu     [E] Exit Workbench
echo =====================================================================================
set /p NEXT_ACTION="Enter your choice (M/E) [Default: M]: "
if /i "%NEXT_ACTION%"=="E" (
    echo Exiting workbench.
    exit /b 0
)
goto menu
