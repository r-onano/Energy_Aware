import cv2
import numpy as np


class FlowTracker:
    def __init__(self, resize_w: int, resize_h: int, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2):
        self.prev_gray = None
        self.size = (resize_w, resize_h)
        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )

    def compute(self, img_bgr):
        img = cv2.resize(img_bgr, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return {
                "motion_mag_mean": 0.0,
                "motion_mag_p90": 0.0,
                "motion_blur_proxy": 0.0,
            }
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self.params)
        self.prev_gray = gray
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_mean = float(np.mean(mag))
        p90 = float(np.percentile(mag, 90))
        # blur proxy via Laplacian variance (lower variance => more blur)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_proxy = float(1.0 / (lap + 1e-6))
        return {
            "motion_mag_mean": mag_mean,
            "motion_mag_p90": p90,
            "motion_blur_proxy": blur_proxy,
        }
