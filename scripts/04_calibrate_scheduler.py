import argparse, json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import joblib

from src.common.io import read_yaml
from src.common.log import info, ok, warn
from src.common.runs import run_root, ensure_dir, write_manifest


def _resolve_model_paths(run_tag: str | None) -> Tuple[Path, Path | None]:
    """
    Prefer results/models/<tag> if available; otherwise fall back to results/models/.
    Returns (model_dir, scaler_path_or_None).
    """
    base = Path("results/models")
    tagged = base / (run_tag or "")
    if run_tag and (tagged / "scheduler_model.pkl").exists():
        model_dir = tagged
    elif (base / "scheduler_model.pkl").exists():
        model_dir = base
        if run_tag:
            warn(f"Tagged model not found in {tagged}; using {base}.")
    else:
        raise SystemExit("No trained model found. Run scripts/03_train_scheduler.py first.")

    scaler = model_dir / "feature_scaler.pkl"
    return model_dir, (scaler if scaler.exists() else None)


def main(args):
    sched_cfg = read_yaml(args.sched_config)["scheduler"]
    _ = read_yaml(args.eval_config)  # kept for future use

    # Load features (use all; you can switch to train/val if preferred)
    feats_path = Path("results/features.parquet")
    if not feats_path.exists():
        raise SystemExit(f"Missing features at {feats_path}. Run scripts/02_extract_features.py first.")
    feats = pd.read_parquet(feats_path)

    # Resolve model & scaler (tagged preferred)
    model_dir, scaler_path = _resolve_model_paths(args.run_tag)
    model = joblib.load(model_dir / "scheduler_model.pkl")
    scaler = joblib.load(scaler_path) if scaler_path is not None else None

    # Feature columns (read from config to avoid drift)
    feat_cols = read_yaml("configs/model.yaml")["features"]["cols"]
    X = feats[feat_cols].values
    if scaler is not None:
        X = scaler.transform(X)

    # Predict cost distribution on your dataset
    y_cost = model.predict(X)

    # Quantile-based thresholds for k-bins
    k_values = sched_cfg["k_values"]
    if len(k_values) < 2:
        raise SystemExit("scheduler.k_values must have at least 2 entries.")
    # e.g., for k_values=[1,2,3,5], we need 3 thresholds
    q = np.linspace(0.25, 0.85, num=len(k_values) - 1)
    thresholds = list(np.quantile(y_cost, q).astype(float))

    # Prepare policy payload
    out = {
        "k_values": k_values,
        "cost_thresholds": thresholds,
        "max_skip": sched_cfg["max_skip"],
        "safety": sched_cfg["safety_floors"],
        "change_reset": sched_cfg.get("change_reset", {}),
    }

    # Save policy under results/policy/<tag>/calibrated_policy.json
    policy_root = Path("results/policy") / (args.run_tag or "")
    ensure_dir(policy_root)
    policy_path = policy_root / "calibrated_policy.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    write_manifest(policy_root, {"phase": "calibrate", "thresholds": thresholds, "model_dir": str(model_dir)})

    ok(f"Calibrated policy saved → {policy_path}")
    info(f"Thresholds: {thresholds}")
    info(f"k_values:  {k_values}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched-config", type=str, required=True)
    ap.add_argument("--eval-config", type=str, required=True)
    ap.add_argument("--run-tag", type=str, default=None, help="results/policy/<run-tag>/calibrated_policy.json")
    main(ap.parse_args())
