"""
YOLO-based perception models with energy tracking.
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_device


class YOLOPerception:
    """YOLO-based perception with multiple model variants."""
    
    # Energy consumption values (in Joules)
    ENERGY_COSTS = {
        'skip': 0.1,    # Frame reuse
        'light': 2.5,   # YOLOv5s
        'medium': 4.2,  # YOLOv5m
        'heavy': 6.8    # YOLOv5l
    }
    
    def __init__(self, config: Dict, device=None):
        """
        Initialize YOLO perception models.
        
        Args:
            config: Configuration dictionary
            device: Torch device (cuda/cpu/mps)
        """
        self.config = config
        self.device = get_device(device)
        self.models = {}
        self.last_detection = None
        
        # Load YOLO models
        self._load_models()
    
    def _load_models(self):
        """Load YOLO model variants."""
        print("Loading YOLO models...")
        
        try:
            from ultralytics import YOLO
            
            # Load different model sizes
            for variant in ['light', 'medium', 'heavy']:
                model_name = self.config['models']['perception'][variant]['model']
                print(f"Loading {variant}: {model_name}")
                
                try:
                    self.models[variant] = YOLO(f'{model_name}.pt')
                    self.models[variant].to(self.device)
                except Exception as e:
                    print(f"Warning: Could not load {model_name}: {e}")
                    # Use smaller model as fallback
                    if variant == 'heavy':
                        model_name = 'yolov5m'
                    elif variant == 'medium':
                        model_name = 'yolov5s'
                    else:
                        model_name = 'yolov5n'
                    print(f"Using fallback model: {model_name}")
                    self.models[variant] = YOLO(f'{model_name}.pt')
                    self.models[variant].to(self.device)
            
            print(f"Loaded {len(self.models)} YOLO models")
            
        except ImportError:
            print("Warning: ultralytics not installed. Using dummy models.")
            self.models = {
                'light': None,
                'medium': None,
                'heavy': None
            }
    
    def detect(self, image: np.ndarray, model_level: str = 'medium') -> Tuple[List[Dict], float, float]:
        """
        Run object detection.
        
        Args:
            image: Input image (RGB)
            model_level: Model complexity level ('skip', 'light', 'medium', 'heavy')
            
        Returns:
            Tuple of (detections, energy_consumed, latency)
        """
        start_time = time.time()
        
        # Handle skip case (reuse last detection)
        if model_level == 'skip':
            detections = self.last_detection if self.last_detection is not None else []
            latency = time.time() - start_time
            return detections, self.ENERGY_COSTS['skip'], latency
        
        # Get model
        model = self.models.get(model_level)
        
        if model is None:
            # Fallback to dummy detection
            detections = self._dummy_detection(image)
        else:
            # Run YOLO detection
            conf = self.config['models']['perception'][model_level]['confidence']
            iou = self.config['models']['perception'][model_level]['iou']
            
            results = model(image, conf=conf, iou=iou, verbose=False)
            detections = self._parse_yolo_results(results[0])
        
        # Store for skip
        self.last_detection = detections
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get energy cost
        energy = self.ENERGY_COSTS[model_level]
        
        return detections, energy, latency
    
    def _parse_yolo_results(self, result) -> List[Dict]:
        """
        Parse YOLO detection results.
        
        Args:
            result: YOLO result object
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # Get box coordinates
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            # Get class name
            class_name = result.names[cls]
            
            # Determine if critical
            is_critical = class_name.lower() in ['person', 'bicycle', 'motorcycle']
            
            detection = {
                'bbox': xyxy.tolist(),  # [x1, y1, x2, y2]
                'confidence': conf,
                'class_id': cls,
                'class_name': class_name,
                'is_critical': is_critical
            }
            
            detections.append(detection)
        
        return detections
    
    def _dummy_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Dummy detection for testing without YOLO.
        
        Args:
            image: Input image
            
        Returns:
            List of dummy detections
        """
        h, w = image.shape[:2]
        
        # Generate random detections
        n_detections = np.random.randint(0, 5)
        detections = []
        
        for _ in range(n_detections):
            x1 = np.random.randint(0, w - 100)
            y1 = np.random.randint(0, h - 100)
            x2 = x1 + np.random.randint(50, min(200, w - x1))
            y2 = y1 + np.random.randint(50, min(200, h - y1))
            
            class_id = np.random.randint(0, 3)
            class_names = ['car', 'person', 'bicycle']
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': np.random.uniform(0.5, 0.95),
                'class_id': class_id,
                'class_name': class_names[class_id],
                'is_critical': class_id in [1, 2]
            }
            
            detections.append(detection)
        
        return detections
    
    def evaluate_detection(self, pred_detections: List[Dict], 
                          gt_annotations: List[Dict],
                          iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate detection performance.
        
        Args:
            pred_detections: Predicted detections
            gt_annotations: Ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary of metrics
        """
        if not gt_annotations:
            # No ground truth
            return {
                'precision': 1.0 if not pred_detections else 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'critical_detection_rate': 1.0
            }
        
        # Match predictions to ground truth
        matched_pred = set()
        matched_gt = set()
        
        for i, pred in enumerate(pred_detections):
            for j, gt in enumerate(gt_annotations):
                if j in matched_gt:
                    continue
                
                iou = self._compute_iou(pred['bbox'], gt['bbox_2d'])
                
                if iou >= iou_threshold:
                    matched_pred.add(i)
                    matched_gt.add(j)
                    break
        
        # Calculate metrics
        tp = len(matched_pred)
        fp = len(pred_detections) - tp
        fn = len(gt_annotations) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Critical object detection rate
        critical_gt = [ann for ann in gt_annotations if ann.get('is_critical', False)]
        critical_detected = 0
        
        for gt_idx in matched_gt:
            if gt_annotations[gt_idx].get('is_critical', False):
                critical_detected += 1
        
        critical_rate = critical_detected / len(critical_gt) if critical_gt else 1.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'critical_detection_rate': critical_rate
        }
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    # Test YOLO perception
    from utils import load_config
    
    config = load_config('configs/config.yaml')
    perception = YOLOPerception(config)
    
    # Create dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test different model levels
    for level in ['skip', 'light', 'medium', 'heavy']:
        detections, energy, latency = perception.detect(image, level)
        print(f"\n{level.upper()} Model:")
        print(f"  Detections: {len(detections)}")
        print(f"  Energy: {energy:.2f} J")
        print(f"  Latency: {latency:.2f} ms")
