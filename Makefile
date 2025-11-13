.PHONY: setup prep features train calib baseline sched figs tables test

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Run: conda activate green-perception"

prep:
	python scripts/01_prepare_data.py --config configs/data.yaml

features:
	python scripts/02_extract_features.py --data-config configs/data.yaml --feat-config configs/features.yaml

train:
	python scripts/03_train_scheduler.py --model-config configs/model.yaml

calib:
	python scripts/04_calibrate_scheduler.py --sched-config configs/scheduler.yaml --eval-config configs/eval.yaml

baseline:
	python scripts/05_run_perception_baseline.py --data-config configs/data.yaml --eval-config configs/eval.yaml

sched:
	python scripts/06_run_scheduler_eval.py --data-config configs/data.yaml --feat-config configs/features.yaml --model-config configs/model.yaml --sched-config configs/scheduler.yaml --eval-config configs/eval.yaml --energy-config configs/energy.yaml

figs:
	python scripts/07_make_figures.py

tables:
	python scripts/08_export_report_tables.py

test:
	pytest -q
