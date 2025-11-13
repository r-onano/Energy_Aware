import argparse, json, time
from pathlib import Path
import pandas as pd

from src.common.io import read_yaml
from src.common.log import info, ok, warn
from src.perception.yolo import Detector
from src.energy.simulator import SimEnergy
from src.common.runs import run_root, ensure_dir, write_manifest


def main(args):
    data_cfg = read_yaml(args.data_config)
    eval_cfg = read_yaml(args.eval_config)

    # Per-run output directory (results/runs/<tag>/baseline)
    run_dir = run_root(args.run_tag)
    out_dir = ensure_dir(run_dir / "baseline")
    write_manifest(out_dir, {"phase": "baseline"})

    # Load test split from the frame index
    index_path = Path(data_cfg["index"]["out_path"])
    if not index_path.exists():
        raise SystemExit(f"Frame index not found: {index_path}. Run prepare_data first.")
    df = pd.read_parquet(index_path)
    df = df[df["split"] == "test"].copy()

    # Detector
    det = Detector(
        weights=eval_cfg["baseline"]["detector_weights"],
        conf=eval_cfg["baseline"]["conf"],
        device=eval_cfg["baseline"]["device"],
    )

    # Energy mode: try NVML; fallback to simulator
    energy_mode = "nvml"
    try:
        from src.energy.nvml import PowerLogger  # import here to allow fallback cleanly
        logger = PowerLogger(gpu=0, hz=10).__enter__()
    except Exception as e:
        warn(f"NVML unavailable ({e}); using simulator energy model.")
        energy_mode = "sim"
        logger = None
        sim = SimEnergy()

    per_rows = []
    t_start = time.time()
    n_frames = 0

    try:
        for _, row in df.iterrows():
            r = det.infer(row["img_path"])
            n_frames += 1
            fps = n_frames / max(time.time() - t_start, 1e-3)

            if energy_mode == "nvml":
                logger.tick()
                w = logger.mean()
            else:
                # simple util heuristic for baseline (always infer)
                sim.observe(fps=fps, util=0.6)
                w = sim.mean()

            per_rows.append({
                "frame_id": row["frame_id"],
                "img_path": row["img_path"],
                "latency_ms": float(r["latency_ms"]),
                "power_w": float(w),
            })
    finally:
        if energy_mode == "nvml" and logger is not None:
            logger.__exit__(None, None, None)

    # Save per-frame + aggregate
    per = pd.DataFrame(per_rows)
    per.to_csv(out_dir / "per_frame.csv", index=False)

    agg = {
        "latency_ms_mean": float(per["latency_ms"].mean()),
        "latency_ms_p95": float(per["latency_ms"].quantile(0.95)),
        "power_w_mean": float(per["power_w"].mean()),
        "n_frames": int(len(per)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    ok(f"Baseline per-frame + metrics saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=str, required=True)
    ap.add_argument("--eval-config", type=str, required=True)
    ap.add_argument("--run-tag", type=str, default=None, help="results/runs/<run-tag>/baseline")
    main(ap.parse_args())
