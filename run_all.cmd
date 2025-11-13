@echo off
setlocal ENABLEDELAYEDEXECUTION

REM --- Run tag: use first arg or timestamp via PowerShell ---
if "%~1"=="" (
  for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyy-MM-dd_HH-mm-ss\")"') do set "RUN_TAG=%%i"
) else (
  set "RUN_TAG=%~1"
)
echo >>> Using RUN_TAG="%RUN_TAG%"

REM --- Make src a package so imports like src.common.* work ---
for %%D in (src src\common src\data src\energy src\eval src\features src\models src\perception src\scheduler) do (
  if not exist "%%D\__init__.py" type NUL > "%%D\__init__.py"
)

REM --- Ensure PY deps available (quiet) ---
pip show tqdm >NUL 2>&1 || pip install -q tqdm
python -c "import pyarrow" 2>NUL 1>NUL || pip install -q pyarrow fastparquet

REM --- Ensure Python can import from this repo root ---
set "PYTHONPATH=%CD%"

echo [1/8] Build frame index
python scripts\01_prepare_data.py --config configs\data.yaml || (echo Failed at step 1 & exit /b 1)

echo [2/8] Extract features
python scripts\02_extract_features.py --data-config configs\data.yaml --feat-config configs\features.yaml || (echo Failed at step 2 & exit /b 2)

echo [3/8] Baseline evaluation
python scripts\05_run_perception_baseline.py --data-config configs\data.yaml --eval-config configs\eval.yaml --run-tag "%RUN_TAG%" || (echo Failed at step 3 & exit /b 3)

echo [4/8] Train regression scheduler
python scripts\03_train_scheduler.py --model-config configs\model.yaml --run-tag "%RUN_TAG%" || (echo Failed at step 4 & exit /b 4)

echo [5/8] Calibrate scheduler
python scripts\04_calibrate_scheduler.py --sched-config configs\scheduler.yaml --eval-config configs\eval.yaml --run-tag "%RUN_TAG%" || (echo Failed at step 5 & exit /b 5)

echo [6/8] Scheduler evaluation
python scripts\06_run_scheduler_eval.py --data-config configs\data.yaml --feat-config configs\features.yaml --model-config configs\model.yaml --sched-config configs\scheduler.yaml --eval-config configs\eval.yaml --energy-config configs\energy.yaml --run-tag "%RUN_TAG%" || (echo Failed at step 6 & exit /b 6)

echo [7/8] Make figures
python scripts\07_make_figures.py --run-tag "%RUN_TAG%" || (echo Failed at step 7 & exit /b 7)

echo [8/8] Export tables
python scripts\08_export_report_tables.py --run-tag "%RUN_TAG%" || (echo Failed at step 8 & exit /b 8)

echo:
echo ✅ All steps completed successfully. Run tag: %RUN_TAG%
echo   results\runs\%RUN_TAG%\baseline
echo   results\runs\%RUN_TAG%\scheduler
echo   results\models\%RUN_TAG%
echo   results\policy\%RUN_TAG%
echo   report\figures
echo   results\tables

endlocal
