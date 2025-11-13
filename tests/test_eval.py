import numpy as np
from src.perception.metrics import iou_xyxy, map_at_iou

def test_iou_basic():
    a = np.array([[0,0,10,10]])
    b = np.array([[5,5,15,15]])
    iou = iou_xyxy(a,b)[0,0]
    assert 0.0 <= iou <= 1.0

def test_map_proxy_empty():
    assert map_at_iou(np.zeros((0,4)), np.zeros((0,4))) == 1.0
