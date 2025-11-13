import time
from contextlib import contextmanager
import pynvml

class PowerLogger:
    def __init__(self, gpu=0, hz=10):
        self.gpu = gpu
        self.hz = hz
        self.samples = []
        self._running = False

    def __enter__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu)
        self._running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        pynvml.nvmlShutdown()

    def tick(self):
        mwatts = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
        self.samples.append(mwatts / 1000.0)

    def mean(self):
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)
