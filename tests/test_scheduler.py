from src.scheduler.policy import SkipPolicy
from src.scheduler.safety import in_no_skip_zone

def test_policy_thresholds():
    pol = SkipPolicy([1,2,3,5], [5,10,20], max_skip=5)
    assert pol.map_cost_to_k(1.0) == 1
    assert pol.map_cost_to_k(7.0) == 2
    assert pol.map_cost_to_k(25.0) == 5

def test_safety_zone():
    f = {"object_density": 12, "motion_mag_p90": 0.4, "brightness_mean": 50}
    assert in_no_skip_zone(f, density_hi=10, motion_hi=2.0, brightness_lo=25) is True
