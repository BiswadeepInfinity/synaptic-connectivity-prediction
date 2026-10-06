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
echo   [7] Exit
echo -----------------------------------------------------------------------------------------
echo.

set /p OPTION="Select option [1-7]: "

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
    echo [RUNNING COMPLETE TEST SUITE]
    echo =====================================================================================
    python -m pytest data/test_loader.py baseline/test_tgn.py test_phase1_reproduction.py -v -s
    goto pause_step
)

if "%OPTION%"=="7" (
    echo Exiting workbench.
    exit /b 0
)

echo.
echo [!] Invalid selection. Please enter a number between 1 and 7.

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

