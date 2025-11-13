def in_no_skip_zone(feats: dict, density_hi: float, motion_hi: float, brightness_lo: float) -> bool:
    if feats.get("object_density", 0.0) >= density_hi:
        return True
    if feats.get("motion_mag_p90", 0.0) >= motion_hi:
        return True
    if feats.get("brightness_mean", 1e9) <= brightness_lo:
        return True
    return False
