# Energy-Aware Perception Scheduling

An end-to-end research codebase for reducing perception energy in autonomous-driving workloads by adapting inference frequency to scene complexity—while preserving safety/accuracy.

## Quickstart

```bash
conda env create -f environment.yml
conda activate green-perception

# 1) Prepare data index (edit configs/data.yaml to your paths)
make prep

# 2) Extract scene-complexity features
make features

# 3) Baseline perception eval (per-frame latency+energy)
make baseline

# 4) Train regression scheduler (predict cost from features)
make train

# 5) Calibrate thresholds → skip policy
make calib

# 6) Run scheduler evaluation
make sched

# 7) Figures & tables
make figs
make tables

# 8) Tests
make test
