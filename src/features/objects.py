# We already computed object_density/small_obj_ratio/occlusion_proxy during indexing.
# This module is mainly a placeholder if you later want to recompute from model detections.
def object_features(precomputed_row: dict):
    return {
        "object_density": float(precomputed_row.get("object_density", 0.0)),
        "small_obj_ratio": float(precomputed_row.get("small_obj_ratio", 0.0)),
        "occlusion_proxy": float(precomputed_row.get("occlusion_proxy", 0.0)),
    }
