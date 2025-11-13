import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.common.io import read_yaml
from src.common.log import info, ok
from src.scheduler.policy import SkipPolicy
from src.scheduler.safety import in_no_skip_zone

def main(args):
    sched_cfg = read_yaml(args.sched_config)["scheduler"]
    eval_cfg  = read_yaml(args.eval_config)

    feats = pd.read_parquet("results/features.parquet")
    model_dir = Path("results/models")
    model = joblib.load(model_dir / "scheduler_model.pkl")
    try:
        scaler = joblib.load(model_dir / "feature_scaler.pkl")
    except Exception:
        scaler = None

    feat_cols = read_yaml("configs/model.yaml")["features"]["cols"]
    X = feats[feat_cols].values
    if scaler is not None:
        X = scaler.transform(X)
    y_cost = model.predict(X)

    # Quantile-based thresholds
    k_values = sched_cfg["k_values"]
    q = np.linspace(0.25, 0.85, num=len(k_values)-1)  # flexible spread
    thresholds = list(np.quantile(y_cost, q).astype(float))

    # Save policy + safety
    out = {
        "k_values": k_values,
        "cost_thresholds": thresholds,
        "max_skip": sched_cfg["max_skip"],
        "safety": sched_cfg["safety_floors"],
        "change_reset": sched_cfg.get("change_reset", {}),
    }
    Path("results/policy").mkdir(parents=True, exist_ok=True)
    with open("results/policy/calibrated_policy.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    ok(f"Calibrated policy saved with thresholds={thresholds}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched-config", type=str, required=True)
    ap.add_argument("--eval-config", type=str, required=True)
    main(ap.parse_args())
