import cv2
import numpy as np


def _calc_entropy(gray: np.ndarray, kernel: int = 5) -> float:
    # histogram-based entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-8)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def illumination_features(img_bgr, resize_w: int, resize_h: int, glare_sat_thresh: int):
    img = cv2.resize(img_bgr, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mean = float(v.mean())
    std = float(v.std())
    entropy = _calc_entropy(v)
    glare_ratio = float((v >= glare_sat_thresh).mean())  # saturated pixels proxy
    return {
        "brightness_mean": mean,
        "brightness_std": std,
        "entropy": entropy,
        "glare_ratio": glare_ratio,
    }
