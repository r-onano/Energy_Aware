import os, json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("report/figures")
RESULTS_DIR = Path("results/runs")

def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def _load_metrics(run_dir):
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_energy_accuracy(baseline_dir, scheduler_dir):
    base_m = _load_metrics(baseline_dir)
    sched_m = _load_metrics(scheduler_dir)
    if not base_m or not sched_m:
        print("Missing metrics.json for baseline or scheduler.")
        return

    # Single-point “curve” for now; extend if you sweep different policies
    x = [base_m["power_w_mean"], sched_m["power_w_mean"]]
    y = [1.0, 1.0]  # placeholder for mAP if you add GT; keep as 1.0 to show energy-only for now
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

def plot_latency_hist(run_dir, title, fname):
    per_path = run_dir / "per_frame.csv"
    if not per_path.exists():
        print(f"Missing per_frame.csv for {run_dir}")
        return
    df = pd.read_csv(per_path)
    plt.figure(figsize=(6,4))
    plt.hist(df["latency_ms"], bins=40)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()

def plot_skip_rate(scheduler_dir):
    per_path = scheduler_dir / "per_frame.csv"
    if not per_path.exists():
        print("Missing per_frame.csv for scheduler.")
        return
    df = pd.read_csv(per_path)
    skip_rate = 1.0 - df["ran_infer"].mean()
    plt.figure(figsize=(4,4))
    plt.bar(["Skip rate"], [skip_rate])
    plt.ylim(0,1)
    plt.title("Scheduler Skip Rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_skip_rate.png", dpi=200)
    plt.close()

def main():
    _ensure_out()
    baseline_dir  = RESULTS_DIR / "baseline"
    scheduler_dir = RESULTS_DIR / "scheduler"

    plot_energy_accuracy(baseline_dir, scheduler_dir)
    plot_latency_hist(baseline_dir,  "Latency Distribution — Baseline",  "fig_latency_baseline.png")
    plot_latency_hist(scheduler_dir, "Latency Distribution — Scheduler", "fig_latency_scheduler.png")
    plot_skip_rate(scheduler_dir)

    print(f"Saved figures to {OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()
