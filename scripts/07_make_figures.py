import os, json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

OUT_DIR = Path("report/figures")
RUNS_ROOT = Path("results/runs")


def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_run_dir() -> Optional[Path]:
    if not RUNS_ROOT.exists():
        return None
    # list only subdirectories
    runs: List[Path] = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    if not runs:
        return None
    # sort by mtime desc
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _resolve_run_dir(run_tag: Optional[str]) -> Optional[Path]:
    if run_tag:
        rd = RUNS_ROOT / run_tag
        return rd if rd.exists() else None
    # auto-pick most recent
    return _latest_run_dir()


def plot_energy_accuracy(baseline_dir: Path, scheduler_dir: Path):
    base_m = _load_metrics(baseline_dir)
    sched_m = _load_metrics(scheduler_dir)
    if not base_m or not sched_m:
        print("Missing metrics.json for baseline or scheduler.")
        return

    # Single-point “curve” for now; extend if you sweep different policies
    x = [base_m.get("power_w_mean", 0.0), sched_m.get("power_w_mean", 0.0)]
    y = [1.0, 1.0]  # placeholder until GT AP is enabled
    labels = ["Baseline", "Scheduler"]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi))
    plt.xlabel("Mean Power (W)")
    plt.ylabel("Relative Accuracy (arb.)")
    plt.title("Energy–Accuracy Trade-off (proxy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_energy_accuracy.png", dpi=200)
    plt.close()


def plot_latency_hist(run_dir: Path, title: str, fname: str):
    per_path = run_dir / "per_frame.csv"
    if not per_path.exists():
        print(f"Missing per_frame.csv for {run_dir}")
        return
    df = pd.read_csv(per_path)
    if "latency_ms" not in df.columns:
        print(f"Column 'latency_ms' missing in {per_path}")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(df["latency_ms"], bins=40)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()


def plot_skip_rate(scheduler_dir: Path):
    per_path = scheduler_dir / "per_frame.csv"
    if not per_path.exists():
        print("Missing per_frame.csv for scheduler.")
        return
    df = pd.read_csv(per_path)
    if "ran_infer" not in df.columns:
        print(f"Column 'ran_infer' missing in {per_path}")
        return
    skip_rate = 1.0 - df["ran_infer"].mean()
    plt.figure(figsize=(4, 4))
    plt.bar(["Skip rate"], [skip_rate])
    plt.ylim(0, 1)
    plt.title("Scheduler Skip Rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_skip_rate.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Name of the run under results/runs/<run-tag>. "
             "If omitted, the most recent run directory is used."
    )
    args = parser.parse_args()

    _ensure_out()

    run_dir = _resolve_run_dir(args.run_tag)
    if run_dir is None:
        print("No run directory found. Create one by running baseline/scheduler first.")
        return

    baseline_dir = run_dir / "baseline"
    scheduler_dir = run_dir / "scheduler"

    if not baseline_dir.exists():
        print(f"Baseline directory not found: {baseline_dir}")
    if not scheduler_dir.exists():
        print(f"Scheduler directory not found: {scheduler_dir}")

    plot_energy_accuracy(baseline_dir, scheduler_dir)
    plot_latency_hist(baseline_dir, "Latency Distribution — Baseline", "fig_latency_baseline.png")
    plot_latency_hist(scheduler_dir, "Latency Distribution — Scheduler", "fig_latency_scheduler.png")
    plot_skip_rate(scheduler_dir)

    print(f"Saved figures to {OUT_DIR} (run: {run_dir.name})")


if __name__ == "__main__":
    main()
