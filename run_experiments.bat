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
echo   [11] Run EVERYTHING: Phase 1 + Phase 2 + Phase 3 (full pipeline, all outputs)
echo.
echo   [12] Exit
echo -----------------------------------------------------------------------------------------
echo.

set /p OPTION="Select option [1-12]: "

if "%OPTION%"=="1" (
    echo.
    echo =====================================================================================
    echo [RUNNING PHASE 1 GATE]
    echo Target: AUC 0.839 +/- 0.007 across 5 seeds on cell-lineage division events.
    echo Rule: Must pass before any novel PCDGM module is trusted.
    echo =====================================================================================
    python -m pytest test_phase1_reproduction.py -v -s
    goto pause_step
)

if "%OPTION%"=="2" (
    echo.
    echo =====================================================================================
    echo [RUNNING PHASE 2 UNIT TESTS]
    echo Checking: Strict chronological order, no future leakage, replicate integrity.
    echo =====================================================================================
    python -m pytest baseline/test_tgn.py -k "TestPhase2TGNBaseline" -v -s
    goto pause_step
)

if "%OPTION%"=="3" (
    echo.
    echo =====================================================================================
    echo [RUNNING PHASE 2 5-SEED CONNECTOME BENCHMARK]
    echo Evaluating Plain TGN link prediction on real 45h adult connectomes.
    echo =====================================================================================
    python run_benchmark.py
    goto pause_step
)

if "%OPTION%"=="4" (
    python research_audit.py
    goto pause_step
)

if "%OPTION%"=="5" (
    echo.
    echo =====================================================================================
    echo [STARTING LOCAL VISUALIZER SERVER]
    echo URL: http://localhost:8000
    echo =====================================================================================
    start http://localhost:8000
    python -m http.server 8000 --directory visualizer
    goto pause_step
)

if "%OPTION%"=="6" (
    echo.
    echo =====================================================================================
    echo [RUNNING COMPLETE TEST SUITE - Phase 0 + Phase 1 + Phase 2]
    echo =====================================================================================
    python -m pytest data/test_loader.py baseline/test_tgn.py test_phase1_reproduction.py -v -s
    goto pause_step
)

if "%OPTION%"=="7" (
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
)

if "%OPTION%"=="8" (
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
)

if "%OPTION%"=="9" (
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
)

if "%OPTION%"=="10" (
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
)

if "%OPTION%"=="11" (
    echo.
    echo =====================================================================================
    echo [RUNNING FULL PIPELINE: Phase 1 + Phase 2 + Phase 3]
    echo  This will take several minutes. All tables and test results printed in sequence.
    echo =====================================================================================
    echo.

    echo ---- [1/4] PHASE 3 OUTPUT -----------------------------------------------------------
    python run_phase3_trajectory.py
    echo.

    echo ---- [2/4] PHASE 3 DATA INTEGRITY CHECK --------------------------------------------
    python scripts\crosscheck_counts.py
    echo.

    echo ---- [3/4] PHASE 3 TEST SUITE -------------------------------------------------------
    python -m pytest data/test_loader.py pcdgm/test_module_a_trajectory.py -v
    echo.

    echo ---- [4/4] PHASE 1 + PHASE 2 TEST SUITE --------------------------------------------
    python -m pytest data/test_loader.py baseline/test_tgn.py test_phase1_reproduction.py -v
    echo.

    echo =====================================================================================
    echo  FULL PIPELINE COMPLETE
    echo =====================================================================================
    goto pause_step
)

if "%OPTION%"=="12" (
    echo Exiting workbench.
    exit /b 0
)

echo.
echo [!] Invalid selection. Please enter a number between 1 and 12.

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
