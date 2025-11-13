import time
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, weights="yolov8n.pt", conf=0.25, device=0):
        self.model = YOLO(weights)
        self.model.fuse()
        self.device = device
        self.conf = conf

    def infer(self, img_path: str):
        t0 = time.time()
        res = self.model.predict(img_path, conf=self.conf, device=self.device, verbose=False)[0]
        latency = (time.time() - t0) * 1000.0
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        cls = res.boxes.cls.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        return {"boxes": boxes, "scores": scores, "cls": cls, "latency_ms": float(latency)}
