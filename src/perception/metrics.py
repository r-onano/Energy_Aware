import numpy as np

def iou_xyxy(a, b):
    # a: (N,4), b: (M,4)
    N, M = len(a), len(b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=float)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, a_min=0, a_max=None) * np.clip(yy2 - yy1, a_min=0, a_max=None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.clip(union, 1e-6, None)
    return iou

def map_at_iou(pred_boxes, gt_boxes, iou_thr=0.5):
    # Simple one-threshold AP proxy: precision at best matching
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0:
        return 0.0
    iou = iou_xyxy(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    for i in range(iou.shape[0]):
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= iou_thr and j not in matched_gt:
            tp += 1
            matched_gt.add(j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    # F1-like proxy → rough AP surrogate
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)
