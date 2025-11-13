import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.common.io import read_yaml
from src.common.log import info, ok, warn
from src.perception.yolo import Detector
from src.perception.metrics import map_at_iou  # (optional; kept for future GT AP)
from src.scheduler.safety import in_no_skip_zone
from src.scheduler.policy import SkipPolicy
from src.energy.nvml import PowerLogger
from src.energy.simulator import SimEnergy
from src.common.runs import run_root, ensure_dir, write_manifest


def load_policy(policy_json: Path) -> tuple[SkipPolicy, dict, dict]:
    with open(policy_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    pol = SkipPolicy(d["k_values"], d["cost_thresholds"], d["max_skip"])
    saf = d.get("safety", {})
    change = d.get("change_reset", {})
    return pol, saf, change


def _resolve_model_paths(run_tag: str | None) -> tuple[Path, Path | None]:
    """
    Prefer results/models/<tag> if available; otherwise fall back to results/models root.
    Returns: (model_dir, scaler_path or None)
    """
    tagged = Path("results/models") / (run_tag or "")
    root = Path("results/models")
    if run_tag and (tagged / "scheduler_model.pkl").exists():
        model_dir = tagged
    elif (root / "scheduler_model.pkl").exists():
        model_dir = root
        if run_tag:
            warn(f"Tagged model not found under {tagged}; falling back to {root}.")
    else:
        raise SystemExit("No trained model found. Run: 03_train_scheduler.py first.")
    scaler_path = model_dir / "feature_scaler.pkl"
    if not scaler_path.exists():
        scaler_path = None
    return model_dir, scaler_path


def _resolve_policy_path(run_tag: str | None) -> Path:
    """
    Prefer results/policy/<tag>/calibrated_policy.json; fallback to results/policy/calibrated_policy.json.
    """
    tagged = Path("results/policy") / (run_tag or "") / "calibrated_policy.json"
    root = Path("results/policy") / "calibrated_policy.json"
    if run_tag and tagged.exists():
        return tagged
    if root.exists():
        if run_tag and not tagged.exists():
            warn(f"Tagged policy not found under {tagged.parent}; using {root}.")
        return root
    raise SystemExit("No calibrated policy found. Run: 04_calibrate_scheduler.py first.")


def main(args):
    data_cfg = read_yaml(args.data_config)
    feat_cfg = read_yaml(args.feat_config)
    model_cfg = read_yaml(args.model_config)
    sched_cfg = read_yaml(args.sched_config)
    eval_cfg  = read_yaml(args.eval_config)
    energy_cfg= read_yaml(args.energy_config)

    # Per-run output directory
    run_dir = run_root(args.run_tag)
    out_dir = ensure_dir(run_dir / "scheduler")
    write_manifest(out_dir, {"phase": "scheduler"})

    # Load features (test split)
    feats = pd.read_parquet("results/features.parquet")
    feats = feats[feats["split"] == "test"].copy()
    feat_cols = model_cfg["features"]["cols"]

    # Load model & scaler (prefer tagged)
    model_dir, scaler_path = _resolve_model_paths(args.run_tag)
    model = joblib.load(model_dir / "scheduler_model.pkl")
    scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None

    # Load calibrated policy (prefer tagged)
    policy_path = _resolve_policy_path(args.run_tag)
    pol, safety_floors, change_reset = load_policy(policy_path)

    det = Detector(weights=eval_cfg["baseline"]["detector_weights"],
                   conf=eval_cfg["baseline"]["conf"],
                   device=eval_cfg["baseline"]["device"])

    # Energy source
    mode = energy_cfg["energy"]["mode"]
    logger = None
    sim = None
    if mode == "nvml":
        logger = PowerLogger(gpu=energy_cfg["energy"]["nvml_gpu_index"], hz=10).__enter__()
    else:
        sim = SimEnergy(**energy_cfg["energy"]["sim_params"], calib_path=energy_cfg["energy"]["calibration_path"])

    per_rows = []
    k_counter = 0
    last_det = None
    t0 = time.time()
    n_frames = 0
    prev_motion_p90 = None

    for _, row in feats.iterrows():
        n_frames += 1
        # Predict cost
        x = row[feat_cols].values.reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        y_cost = float(model.predict(x)[0])

        # Safety guard
        safety_guard = lambda f=row: in_no_skip_zone(
            f, safety_floors["density_hi"], safety_floors["motion_hi"], safety_floors["brightness_lo"]
        )
        k = pol.decide(y_cost, row.to_dict(), safety_guard=safety_guard)

        # Reset on large motion change if configured
        if change_reset.get("enabled", False) and prev_motion_p90 is not None:
            ratio = (row["motion_mag_p90"] + 1e-6) / (prev_motion_p90 + 1e-6)
            if ratio > change_reset.get("flow_delta_p90", 1.5):
                k_counter = 0  # force inference
        prev_motion_p90 = row["motion_mag_p90"]

        run_infer = (k_counter == 0)  # run now, then skip next k-1 frames
        if run_infer:
            r = det.infer(row["img_path"])
            latency_ms = r["latency_ms"]
            last_det = r
            k_counter = max(k - 1, 0)
        else:
            # carry forward last detections and assume tiny latency
            r = last_det if last_det is not None else {"boxes": np.zeros((0, 4)),
                                                       "scores": np.zeros((0,)),
                                                       "cls": np.zeros((0,)),
                                                       "latency_ms": 1.0}
            latency_ms = 1.0
            k_counter -= 1

        # Energy
        fps = n_frames / max(time.time() - t0, 1e-3)
        if logger:
            logger.tick()
            w = logger.mean()
        else:
            sim.observe(fps=fps, util=0.55 if run_infer else 0.15)
            w = sim.mean()

        # Collect row (add AP later if GT available)
        per_rows.append({
            "frame_id": row["frame_id"],
            "img_path": row["img_path"],
            "k_used": k,
            "ran_infer": int(run_infer),
            "latency_ms": float(latency_ms),
            "power_w": float(w),
        })

    if logger:
        logger.__exit__(None, None, None)

    per = pd.DataFrame(per_rows)
    per.to_csv(out_dir / "per_frame.csv", index=False)

    agg = {
        "latency_ms_mean": float(per["latency_ms"].mean()),
        "latency_ms_p95": float(per["latency_ms"].quantile(0.95)),
        "power_w_mean": float(per["power_w"].mean()),
        "n_frames": int(len(per)),
        "skip_rate": float(1.0 - per["ran_infer"].mean()),
        "model_dir": str(model_dir),
        "policy_path": str(policy_path),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    ok(f"Scheduler run complete → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=str, required=True)
    ap.add_argument("--feat-config", type=str, required=True)
    ap.add_argument("--model-config", type=str, required=True)
    ap.add_argument("--sched-config", type=str, required=True)
    ap.add_argument("--eval-config", type=str, required=True)
    ap.add_argument("--energy-config", type=str, required=True)
    ap.add_argument("--run-tag", type=str, default=None, help="results/runs/<run-tag> and matching models/policy")
    main(ap.parse_args())
