import numpy as np
from typing import List

class SkipPolicy:
    def __init__(self, k_values: List[int], cost_thresholds: List[float], max_skip: int):
        assert len(cost_thresholds) == len(k_values) - 1, "cost thresholds should be len(k)-1"
        self.k_values = k_values
        self.cost_thresholds = cost_thresholds
        self.max_skip = max_skip

    def map_cost_to_k(self, y_cost: float) -> int:
        # Piecewise: [ -inf, t1 ) -> k1, [t1, t2) -> k2, ...
        for t, k in zip(self.cost_thresholds, self.k_values):
            if y_cost < t:
                return min(k, self.max_skip)
        return min(self.k_values[-1], self.max_skip)

    def decide(self, pred_cost: float, feats: dict, safety_guard=None) -> int:
        k = self.map_cost_to_k(pred_cost)
        if safety_guard and safety_guard(feats):
            return 1
        return k
