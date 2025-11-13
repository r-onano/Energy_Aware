#!/usr/bin/env bash
set -euo pipefail

# -------- run tag handling --------
RUN_TAG="${1:-}"
if [[ -z "${RUN_TAG}" ]]; then
  # timestamp tag: 2025-11-13_14-32-05
  RUN_TAG="$(date +"%Y-%m-%d_%H-%M-%S")"
fi
export RUN_TAG
echo ">>> Using RUN_TAG='${RUN_TAG}'"

# -------- 1. Create a tiny synthetic dataset --------
echo "[1/8] Create synthetic dataset"
python scripts/00_make_synth_demo.py --num-frames 60

# -------- 2. Extract features --------
echo "[2/8] Extract features"
python scripts/02_extract_features.py \
  --data-config configs/data.yaml \
  --feat-config configs/features.yaml

# -------- 3. Baseline eval (per-frame latency + energy) --------
echo "[3/8] Baseline eval"
python scripts/05_run_perception_baseline.py \
  --data-config configs/data.yaml \
  --eval-config  configs/eval.yaml \
  --run-tag      "${RUN_TAG}"

# -------- 4. Train regression scheduler --------
echo "[4/8] Train regression scheduler"
python scripts/03_train_scheduler.py \
  --model-config configs/model.yaml \
  --run-tag      "${RUN_TAG}"

# -------- 5. Calibrate scheduler thresholds --------
echo "[5/8] Calibrate scheduler"
python scripts/04_calibrate_scheduler.py \
  --sched-config configs/scheduler.yaml \
  --eval-config  configs/eval.yaml \
  --run-tag      "${RUN_TAG}"

# -------- 6. Run scheduler evaluation --------
echo "[6/8] Scheduler evaluation"
python scripts/06_run_scheduler_eval.py \
  --data-config  configs/data.yaml \
  --feat-config  configs/features.yaml \
  --model-config configs/model.yaml \
  --sched-config configs/scheduler.yaml \
  --eval-config  configs/eval.yaml \
  --energy-config configs/energy.yaml \
  --run-tag      "${RUN_TAG}"

# -------- 7. Make figures --------
echo "[7/8] Make figures"
python scripts/07_make_figures.py --run-tag "${RUN_TAG}"

# -------- 8. Export tables --------
echo "[8/8] Export tables"
python scripts/08_export_report_tables.py --run-tag "${RUN_TAG}"

echo "Done. Run tag: ${RUN_TAG}"
echo "See:"
echo "  results/runs/${RUN_TAG}/{baseline,scheduler}"
echo "  results/models/${RUN_TAG}"
echo "  results/policy/${RUN_TAG}"
echo "  report/figures/"
echo "  results/tables/"
