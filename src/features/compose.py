import cv2
from src.features.illum import illumination_features
from src.features.motion import FlowTracker
from src.features.objects import object_features


class FeaturePipeline:
    def __init__(self, cfg):
        self.use_objects = cfg["features"].get("objects", True)
        self.use_illum = cfg["features"].get("illumination", True)
        self.use_motion = cfg["features"].get("motion", True)
        self.illum_cfg = cfg.get("illumination", {})
        self.motion_cfg = cfg.get("motion", {})
        self.flow = None
        if self.use_motion:
            self.flow = FlowTracker(
                resize_w=self.motion_cfg.get("resize_w", 640),
                resize_h=self.motion_cfg.get("resize_h", 360),
                pyr_scale=self.motion_cfg.get("pyr_scale", 0.5),
                levels=self.motion_cfg.get("levels", 3),
                winsize=self.motion_cfg.get("winsize", 15),
                iterations=self.motion_cfg.get("iterations", 3),
                poly_n=self.motion_cfg.get("poly_n", 5),
                poly_sigma=self.motion_cfg.get("poly_sigma", 1.2),
            )

    def compute(self, img_bgr, row):
        feats = {}
        if self.use_objects:
            feats.update(object_features(row))
        if self.use_illum:
            feats.update(
                illumination_features(
                    img_bgr,
                    resize_w=self.illum_cfg.get("resize_w", 640),
                    resize_h=self.illum_cfg.get("resize_h", 360),
                    glare_sat_thresh=self.illum_cfg.get("glare_saturation_threshold", 240),
                )
            )
        if self.use_motion:
            feats.update(self.flow.compute(img_bgr))
        # Add texture/edges (cheap extras)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        feats["texture_var"] = float(gray.var())
        edges = cv2.Canny(gray, 100, 200)
        feats["edge_density"] = float(edges.mean() / 255.0)
        return feats


def build_feature_pipeline(cfg: dict) -> FeaturePipeline:
    return FeaturePipeline(cfg)
