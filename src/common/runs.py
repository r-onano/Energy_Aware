from pathlib import Path
import json, hashlib, os, time

def _ts():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def run_root(run_tag: str | None) -> Path:
    tag = run_tag or os.environ.get("RUN_TAG")
    if not tag:
        tag = _ts()
    root = Path("results/runs") / tag
    root.mkdir(parents=True, exist_ok=True)
    return root

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_manifest(run_dir: Path, extra: dict | None = None):
    m = {"run_dir": str(run_dir), "time": _ts()}
    if extra:
        m.update(extra)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
