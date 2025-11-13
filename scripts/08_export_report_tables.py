import json
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd

OUT_DIR = Path("results/tables")
RUNS_ROOT = Path("results/runs")


def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_run_dir() -> Optional[Path]:
    if not RUNS_ROOT.exists():
        return None
    runs: List[Path] = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _resolve_run_dir(run_tag: Optional[str]) -> Optional[Path]:
    if run_tag:
        rd = RUNS_ROOT / run_tag
        return rd if rd.exists() else None
    return _latest_run_dir()


def export_summary_table(run_dir: Path):
    base_m = _read_json(run_dir / "baseline" / "metrics.json")
    sched_m = _read_json(run_dir / "scheduler" / "metrics.json")
    if not base_m or not sched_m:
        print(f"Missing metrics for run '{run_dir.name}'; cannot export summary.")
        return

    df = pd.DataFrame(
        [
            {
                "Run": "Baseline",
                "Mean Latency (ms)": base_m.get("latency_ms_mean", ""),
                "P95 Latency (ms)": base_m.get("latency_ms_p95", ""),
                "Mean Power (W)": base_m.get("power_w_mean", ""),
                "Frames": base_m.get("n_frames", ""),
            },
            {
                "Run": "Scheduler",
                "Mean Latency (ms)": sched_m.get("latency_ms_mean", ""),
                "P95 Latency (ms)": sched_m.get("latency_ms_p95", ""),
                "Mean Power (W)": sched_m.get("power_w_mean", ""),
                "Frames": sched_m.get("n_frames", ""),
                "Skip Rate": sched_m.get("skip_rate", ""),
            },
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"table_summary_{run_dir.name}.csv", index=False)
    # also write a generic name for convenience (overwrites each time)
    df.to_csv(OUT_DIR / "table_summary.csv", index=False)


def export_ablation_placeholder():
    cols = ["Setting", "Mean Power (W)", "ΔPower (%)", "Mean Lat (ms)", "ΔLat (%)", "Rel Acc"]
    pd.DataFrame(columns=cols).to_csv(OUT_DIR / "table_ablation.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Name of the run under results/runs/<run-tag>. If omitted, uses the most recent run."
    )
    args = parser.parse_args()

    _ensure_out()
    run_dir = _resolve_run_dir(args.run_tag)
    if run_dir is None:
        print("No run directory found under results/runs/. Run the pipeline first.")
        return

    export_summary_table(run_dir)
    export_ablation_placeholder()
    print(f"Tables saved to {OUT_DIR} (run: {run_dir.name})")


if __name__ == "__main__":
    main()
