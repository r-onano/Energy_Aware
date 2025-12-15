"""
Energy-aware perception scheduler for autonomous vehicles.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import ComplexityPredictor
from perception.yolo_detector import YOLOPerception
from data_processing.feature_extraction import FeatureExtractor


class EnergyAwareScheduler:
    """
    Energy-aware perception scheduler that dynamically selects
    appropriate perception model based on scene complexity.
    """
    
    def __init__(self, config: Dict, complexity_model_path: str = None):
        """
        Initialize scheduler.
        
        Args:
            config: Configuration dictionary
            complexity_model_path: Path to trained complexity prediction model
        """
        self.config = config
        
        # Load complexity predictor
        if complexity_model_path and Path(complexity_model_path).exists():
            self.complexity_predictor = ComplexityPredictor.load(complexity_model_path)
            print(f"Loaded complexity predictor from {complexity_model_path}")
        else:
            print("Warning: No complexity predictor loaded. Using heuristic complexity estimation.")
            self.complexity_predictor = None
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Initialize perception models
        self.perception = YOLOPerception(config)
        
        # Get thresholds from config
        self.thresholds = config['scheduling']['thresholds']
        self.safety_rules = config['scheduling']['safety']
        
        # Tracking
        self.total_energy = 0.0
        self.frame_count = 0
        self.model_usage = {'skip': 0, 'light': 0, 'medium': 0, 'heavy': 0}
        self.prev_image = None
    
    def select_model(self, image: np.ndarray, annotations: List[Dict] = None) -> str:
        """
        Select appropriate perception model based on scene complexity.
        
        Args:
            image: Input image (RGB)
            annotations: Optional ground truth annotations for feature extraction
            
        Returns:
            Selected model level ('skip', 'light', 'medium', 'heavy')
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(
            image, annotations if annotations else [], self.prev_image
        )
        
        # Predict or compute complexity
        if self.complexity_predictor is not None:
            feature_vec = self.feature_extractor.get_feature_vector(features)
            complexity = float(self.complexity_predictor.predict(feature_vec.reshape(1, -1))[0])
        else:
            complexity = self.feature_extractor.compute_complexity_score(features)
        
        # Base selection on complexity thresholds
        if complexity < self.thresholds['skip']:
            model_level = 'skip'
        elif complexity < self.thresholds['light']:
            model_level = 'light'
        elif complexity < self.thresholds['medium']:
            model_level = 'medium'
        else:
            model_level = 'heavy'
        
        # Apply safety overrides
        model_level = self._apply_safety_rules(features, model_level)
        
        return model_level
    
    def _apply_safety_rules(self, features: Dict[str, float], base_level: str) -> str:
        """
        Apply safety rules to override model selection.
        
        Args:
            features: Extracted features
            base_level: Base model selection
            
        Returns:
            Final model level after safety rules
        """
        # Pedestrian/cyclist present -> use at least specified level
        if features['pedestrian_count'] > 0 or features['cyclist_count'] > 0:
            safety_level = self.safety_rules['pedestrian_present']
            base_level = self._max_model_level(base_level, safety_level)
        
        # Low visibility -> use at least specified level
        if features['brightness_mean'] < 0.3:
            safety_level = self.safety_rules['low_visibility']
            base_level = self._max_model_level(base_level, safety_level)
        
        # High motion -> use at least specified level
        if features.get('motion_score', 0) > 0.5:
            safety_level = self.safety_rules['high_motion']
            base_level = self._max_model_level(base_level, safety_level)
        
        return base_level
    
    def _max_model_level(self, level1: str, level2: str) -> str:
        """
        Return the higher complexity model level.
        
        Args:
            level1: First model level
            level2: Second model level
            
        Returns:
            Higher complexity level
        """
        levels = ['skip', 'light', 'medium', 'heavy']
        idx1 = levels.index(level1)
        idx2 = levels.index(level2)
        return levels[max(idx1, idx2)]
    
    def process_frame(self, image: np.ndarray, 
                     annotations: List[Dict] = None) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Process a single frame with energy-aware scheduling.
        
        Args:
            image: Input image
            annotations: Optional ground truth annotations
            
        Returns:
            Tuple of (detections, metrics)
        """
        # Select model
        model_level = self.select_model(image, annotations)
        
        # Run detection
        detections, energy, latency = self.perception.detect(image, model_level)
        
        # Update tracking
        self.total_energy += energy
        self.frame_count += 1
        self.model_usage[model_level] += 1
        self.prev_image = image
        
        # Compute metrics
        metrics = {
            'model_level': model_level,
            'energy': energy,
            'latency': latency,
            'n_detections': len(detections),
            'avg_energy': self.total_energy / self.frame_count
        }
        
        # Evaluate if ground truth available
        if annotations:
            eval_metrics = self.perception.evaluate_detection(detections, annotations)
            metrics.update(eval_metrics)
        
        return detections, metrics
    
    def get_statistics(self) -> Dict:
        """
        Get scheduling statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_frames = sum(self.model_usage.values())
        
        stats = {
            'total_frames': total_frames,
            'total_energy': self.total_energy,
            'avg_energy_per_frame': self.total_energy / total_frames if total_frames > 0 else 0,
            'model_usage': self.model_usage.copy(),
            'model_usage_pct': {
                level: count / total_frames * 100 if total_frames > 0 else 0
                for level, count in self.model_usage.items()
            }
        }
        
        # Calculate baseline energy (always using heavy model)
        baseline_energy = total_frames * self.perception.ENERGY_COSTS['heavy']
        energy_savings = (baseline_energy - self.total_energy) / baseline_energy * 100 if baseline_energy > 0 else 0
        
        stats['baseline_energy'] = baseline_energy
        stats['energy_savings_pct'] = energy_savings
        
        return stats
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.total_energy = 0.0
        self.frame_count = 0
        self.model_usage = {'skip': 0, 'light': 0, 'medium': 0, 'heavy': 0}
        self.prev_image = None


if __name__ == "__main__":
    # Test scheduler
    from utils import load_config
    
    config = load_config('configs/config.yaml')
    
    # Initialize scheduler (without trained model for testing)
    scheduler = EnergyAwareScheduler(config)
    
    # Create dummy image and annotations
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    annotations = [
        {
            'class': 'Car',
            'bbox_2d': [100, 100, 200, 200],
            'is_critical': False
        }
    ]
    
    # Process frame
    detections, metrics = scheduler.process_frame(image, annotations)
    
    print("Frame Processing Results:")
    print(f"  Model selected: {metrics['model_level']}")
    print(f"  Energy consumed: {metrics['energy']:.2f} J")
    print(f"  Latency: {metrics['latency']:.2f} ms")
    print(f"  Detections: {metrics['n_detections']}")
    
    # Process multiple frames
    print("\nProcessing 100 frames...")
    for i in range(100):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections, metrics = scheduler.process_frame(image)
    
    # Get statistics
    stats = scheduler.get_statistics()
    print("\nScheduling Statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Total energy: {stats['total_energy']:.2f} J")
    print(f"  Average energy: {stats['avg_energy_per_frame']:.2f} J/frame")
    print(f"  Energy savings: {stats['energy_savings_pct']:.1f}%")
    print("\nModel usage:")
    for level, pct in stats['model_usage_pct'].items():
        print(f"  {level}: {pct:.1f}%")
