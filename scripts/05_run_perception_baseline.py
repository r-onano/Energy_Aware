import argparse, json, time
from pathlib import Path
import pandas as pd

from src.common.io import read_yaml
from src.common.log import info, ok
from src.perception.yolo import Detector
from src.energy.nvml import PowerLogger
from src.energy.smi import SMIPoller
from src.energy.simulator import SimEnergy

def main(args):
    data_cfg = read_yaml(args.data_config)
    eval_cfg = read_yaml(args.eval_config)
    out_dir = Path(eval_cfg["eval"]["output_dir"]) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_cfg["index"]["out_path"])
    df = df[df["split"] == "test"].copy()  # measure on test split

    det = Detector(weights=eval_cfg["baseline"]["detector_weights"],
                   conf=eval_cfg["baseline"]["conf"],
                   device=eval_cfg["baseline"]["device"])

    # Energy mode
    energy_mode = "sim"
    try:
        from src.energy.nvml import PowerLogger  # noqa
        energy_mode = "nvml"
    except Exception:
        energy_mode = "sim"  # fallback

    logger = None
    sim = SimEnergy()
    per_rows = []
    t_start = time.time()
    n_frames = 0

    if energy_mode == "nvml":
        logger = PowerLogger(gpu=0, hz=10)
        logger.__enter__()

    try:
        for _, row in df.iterrows():
            r = det.infer(row["img_path"])
            n_frames += 1
            fps = n_frames / max(time.time() - t_start, 1e-3)
            if energy_mode == "nvml":
                logger.tick()
                w = logger.mean()
            else:
                sim.observe(fps=fps, util=0.6)
                w = sim.mean()

            per_rows.append({
                "frame_id": row["frame_id"],
                "img_path": row["img_path"],
                "latency_ms": r["latency_ms"],
                "power_w": w,
            })
    finally:
        if logger:
            logger.__exit__(None, None, None)

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
    main(ap.parse_args())
