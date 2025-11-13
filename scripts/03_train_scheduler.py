import argparse, json
from pathlib import Path
from src.common.io import read_yaml
from src.common.log import info, ok, warn
from src.models.regressors import train_regressor

def main(args):
    model_cfg = read_yaml(args.model_config)
    feat_path = Path("results/features.parquet")
    baseline_dir = Path("results/runs/baseline")
    out_dir = Path("results/models")
    if not feat_path.exists():
        raise SystemExit("Missing features.parquet. Run: make features")
    if not baseline_dir.exists():
        warn("Baseline results not found; will simulate target labels.")
    info("Training regression scheduler...")
    mtr = train_regressor(model_cfg, feat_path, baseline_dir, out_dir)
    ok(f"Saved model + scaler to {out_dir}. Metrics: {json.dumps(mtr[1], indent=2)} (val)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=str, required=True)
    main(ap.parse_args())
