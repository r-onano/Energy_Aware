#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Create synthetic dataset"
python scripts/00_make_synth_demo.py --num-frames 60

echo "[2/6] Extract features"
python scripts/02_extract_features.py --data-config configs/data.yaml --feat-config configs/features.yaml

echo "[3/6] Baseline eval (per-frame latency + energy proxy)"
python scripts/05_run_perception_baseline.py --data-config configs/data.yaml --eval-config configs/eval.yaml

echo "[4/6] Train regression scheduler"
python scripts/03_train_scheduler.py --model-config configs/model.yaml

echo "[5/6] Calibrate scheduler thresholds"
python scripts/04_calibrate_scheduler.py --sched-config configs/scheduler.yaml --eval-config configs/eval.yaml

echo "[6/6] Run scheduler evaluation"
python scripts/06_run_scheduler_eval.py \
  --data-config configs/data.yaml \
  --feat-config configs/features.yaml \
  --model-config configs/model.yaml \
  --sched-config configs/scheduler.yaml \
  --eval-config configs/eval.yaml \
  --energy-config configs/energy.yaml

echo "Making figures and tables…"
python scripts/07_make_figures.py
python scripts/08_export_report_tables.py

echo "Done. See results/ and report/figures/"
