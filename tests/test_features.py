import numpy as np
import cv2
from src.features.illum import illumination_features
from src.features.motion import FlowTracker

def test_illum_features_shapes():
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    feats = illumination_features(img, 160, 90, 240)
    assert "brightness_mean" in feats and "entropy" in feats

def test_motion_tracker():
    ft = FlowTracker(160, 90)
    img1 = np.zeros((240,320,3), dtype=np.uint8)
    img2 = np.zeros((240,320,3), dtype=np.uint8); img2[:, 10:] = 5
    f1 = ft.compute(img1)
    f2 = ft.compute(img2)
    assert f2["motion_mag_p90"] >= f1["motion_mag_p90"]
