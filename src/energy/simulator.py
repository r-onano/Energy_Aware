import json, time
from pathlib import Path

class SimEnergy:
    def __init__(self, alpha=0.8, beta=0.2, gamma=5.0, calib_path: str | None = None):
        if calib_path and Path(calib_path).exists():
            with open(calib_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            alpha, beta, gamma = d.get("alpha", alpha), d.get("beta", beta), d.get("gamma", gamma)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.samples = []

    def observe(self, fps: float, util: float = 0.5):
        # crude model: higher fps -> lower per-frame energy time; combine with util
        w = self.alpha * (1.0 / max(fps, 1e-3)) * 10.0 + self.beta * util * 100.0 + self.gamma
        self.samples.append(w)

    def mean(self):
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)
