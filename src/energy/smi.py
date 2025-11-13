import subprocess, time

class SMIPoller:
    def __init__(self, interval_ms=100):
        self.interval = interval_ms / 1000.0
        self.samples = []

    def tick(self):
        cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        try:
            out = subprocess.check_output(cmd).decode("utf-8").strip().splitlines()
            # first GPU
            w = float(out[0])
            self.samples.append(w)
        except Exception:
            pass

    def mean(self):
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)
