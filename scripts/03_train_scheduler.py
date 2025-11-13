import argparse
from pathlib import Path
from typing import Tuple

from src.common.io import read_yaml
from src.common.log import info, ok, warn
from src.models.regressors import train_regressor
from src.common.runs import run_root, ensure_dir, write_manifest


def _resolve_baseline_dir(run_tag: str | None) -> Path:
    """
    Prefer results/runs/<tag>/baseline; fallback to results/runs/baseline.
    """
    tagged = Path("results/runs") / (run_tag or "") / "baseline"
    default = Path("results/runs") / "baseline"
    if run_tag and tagged.exists():
        return tagged
    if default.exists():
        if run_tag and not tagged.exists():
            warn(f"Tagged baseline not found at {tagged}; using {default}.")
        return default
    raise SystemExit("No baseline results found. Run scripts/05_run_perception_baseline.py first.")


def _model_out_dir(run_tag: str | None) -> Path:
    """
    results/models/<run-tag>/, or results/models/default/ if no tag provided.
    """
    tag = run_tag or "default"
    out = Path("results/models") / tag
    ensure_dir(out)
    return out


def main(args):
    model_cfg = read_yaml(args.model_config)
    feat_path = Path("results/features.parquet")
    if not feat_path.exists():
        raise SystemExit("Missing features.parquet. Run scripts/02_extract_features.py first.")

    # Resolve where to read baseline labels from (tagged preferred)
    baseline_dir = _resolve_baseline_dir(args.run_tag)

    # Where to write the trained model
    out_dir = _model_out_dir(args.run_tag)

    info("Training regression scheduler...")
    # train_regressor should read features from feat_path and (optionally)
    # merge in any label/latency targets from baseline_dir/per_frame.csv
    mtr_train, mtr_val = train_regressor(model_cfg, feat_path, baseline_dir, out_dir)

    write_manifest(
        out_dir,
        {
            "phase": "train",
            "baseline_dir": str(baseline_dir),
            "metrics_train": mtr_train,
            "metrics_val": mtr_val,
        },
    )
    ok(f"Saved model + scaler to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=str, required=True)
    ap.add_argument("--run-tag", type=str, default=None, help="Writes to results/models/<run-tag>/ and reads tagged baseline if present")
    main(ap.parse_args())
