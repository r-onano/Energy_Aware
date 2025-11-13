import json, argparse
from pathlib import Path
import pandas as pd

OUT_DIR = Path("results/tables")
RUNS_DIR = Path("results/runs")

def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def _read_json(p):
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def export_summary_table():
    base_m = _read_json(RUNS_DIR / "baseline/metrics.json")
    sched_m = _read_json(RUNS_DIR / "scheduler/metrics.json")
    if not base_m or not sched_m:
        print("Missing baseline/scheduler metrics; cannot export tables.")
        return

    df = pd.DataFrame([
        {
            "Run": "Baseline",
            "Mean Latency (ms)": base_m["latency_ms_mean"],
            "P95 Latency (ms)": base_m["latency_ms_p95"],
            "Mean Power (W)": base_m["power_w_mean"],
            "Frames": base_m["n_frames"],
        },
        {
            "Run": "Scheduler",
            "Mean Latency (ms)": sched_m["latency_ms_mean"],
            "P95 Latency (ms)": sched_m["latency_ms_p95"],
            "Mean Power (W)": sched_m["power_w_mean"],
            "Frames": sched_m["n_frames"],
        },
    ])

    # If skip_rate exists:
    if "skip_rate" in sched_m:
        df.loc[df["Run"] == "Scheduler", "Skip Rate"] = sched_m["skip_rate"]
    else:
        df["Skip Rate"] = ""

    df.to_csv(OUT_DIR / "table_summary.csv", index=False)

def export_ablation_placeholder():
    # Drop an empty ablation table for later sweeps (extend as needed)
    cols = ["Setting", "Mean Power (W)", "ΔPower (%)", "Mean Lat (ms)", "ΔLat (%)", "Rel Acc"]
    pd.DataFrame(columns=cols).to_csv(OUT_DIR / "table_ablation.csv", index=False)

def main():
    _ensure_out()
    export_summary_table()
    export_ablation_placeholder()
    print(f"Tables saved to {OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()
