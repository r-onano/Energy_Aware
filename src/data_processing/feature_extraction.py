"""
Feature extraction for scene complexity analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import entropy


class FeatureExtractor:
    """Extract visual features for complexity assessment."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
            'object_count',
            'object_density',
            'pedestrian_count',
            'vehicle_count',
            'cyclist_count',
            'brightness_mean',
            'brightness_std',
            'contrast',
            'motion_score',
            'edge_density',
            'spatial_entropy',
            'color_variance'
        ]
    
    def extract_all_features(self, image: np.ndarray, 
                            annotations: List[Dict],
                            prev_image: np.ndarray = None) -> Dict[str, float]:
        """
        Extract all features from an image.
        
        Args:
            image: Input image (RGB)
            annotations: List of object annotations
            prev_image: Previous frame for motion estimation (optional)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Object-based features
        obj_features = self._extract_object_features(image, annotations)
        features.update(obj_features)
        
        # Lighting features
        light_features = self._extract_lighting_features(image)
        features.update(light_features)
        
        # Motion features
        if prev_image is not None:
            motion_features = self._extract_motion_features(image, prev_image)
        else:
            motion_features = {'motion_score': 0.0}
        features.update(motion_features)
        
        # Structural features
        struct_features = self._extract_structural_features(image)
        features.update(struct_features)
        
        return features
    
    def _extract_object_features(self, image: np.ndarray, 
                                 annotations: List[Dict]) -> Dict[str, float]:
        """
        Extract object-based features.
        
        Args:
            image: Input image
            annotations: List of annotations
            
        Returns:
            Dictionary of object features
        """
        h, w = image.shape[:2]
        image_area = h * w
        
        features = {
            'object_count': len(annotations),
            'object_density': 0.0,
            'pedestrian_count': 0,
            'vehicle_count': 0,
            'cyclist_count': 0,
        }
        
        if not annotations:
            return features
        
        total_bbox_area = 0
        
        for ann in annotations:
            bbox = ann['bbox_2d']
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_bbox_area += bbox_area
            
            # Count by class
            obj_class = ann['class']
            if obj_class in ['Pedestrian', 'Person_sitting']:
                features['pedestrian_count'] += 1
            elif obj_class in ['Car', 'Van', 'Truck']:
                features['vehicle_count'] += 1
            elif obj_class == 'Cyclist':
                features['cyclist_count'] += 1
        
        # Object density as ratio of bbox area to image area
        features['object_density'] = total_bbox_area / image_area
        
        return features
    
    def _extract_lighting_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract lighting-related features.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary of lighting features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Brightness statistics
        brightness_mean = np.mean(gray) / 255.0
        brightness_std = np.std(gray) / 255.0
        
        # Contrast (using Michelson contrast) - FIXED: explicit float conversion
        gray_min = float(np.min(gray))
        gray_max = float(np.max(gray))
        if (gray_max + gray_min) > 0:
            contrast = (gray_max - gray_min) / (gray_max + gray_min)
        else:
            contrast = 0.0
        
        return {
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'contrast': contrast
        }
    
    def _extract_motion_features(self, current_image: np.ndarray,
                                 prev_image: np.ndarray) -> Dict[str, float]:
        """
        Extract motion-related features.
        
        Args:
            current_image: Current frame
            prev_image: Previous frame
            
        Returns:
            Dictionary of motion features
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        
        # FIXED: Resize prev frame to match current frame if needed
        if gray1.shape != gray2.shape:
            gray1 = cv2.resize(gray1, (gray2.shape[1], gray2.shape[0]))
        
        # Compute frame difference
        frame_diff = cv2.absdiff(gray2, gray1)
        
        # Motion score as mean absolute difference
        motion_score = np.mean(frame_diff) / 255.0
        
        return {
            'motion_score': motion_score
        }
    
    def _extract_structural_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract structural features.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of structural features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge density using Canny edge detector
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Spatial entropy
        # Divide image into grid and compute entropy of intensity distribution
        h, w = gray.shape
        grid_size = 16
        grid_h = h // grid_size
        grid_w = w // grid_size
        
        grid_means = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                grid_means.append(np.mean(cell))
        
        # Normalize and compute entropy
        grid_means = np.array(grid_means)
        hist, _ = np.histogram(grid_means, bins=32, density=True)
        hist = hist[hist > 0]  # Remove zeros for entropy calculation
        spatial_entropy_val = entropy(hist)
        
        # Color variance
        color_variance = np.mean([np.var(image[:, :, i]) for i in range(3)]) / (255.0 ** 2)
        
        return {
            'edge_density': edge_density,
            'spatial_entropy': spatial_entropy_val,
            'color_variance': color_variance
        }
    
    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to vector.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Feature vector
        """
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def compute_complexity_score(self, features: Dict[str, float]) -> float:
        """
        Compute complexity score from features (heuristic).
        
        Args:
            features: Dictionary of features
            
        Returns:
            Complexity score (0-1)
        """
        # Weighted combination of features
        weights = {
            'object_count': 0.15,
            'object_density': 0.15,
            'pedestrian_count': 0.20,  # Higher weight for safety-critical
            'vehicle_count': 0.05,
            'cyclist_count': 0.15,     # Higher weight for safety-critical
            'brightness_mean': -0.05,  # Lower brightness = more complex
            'brightness_std': 0.05,
            'contrast': 0.05,
            'motion_score': 0.10,
            'edge_density': 0.05,
            'spatial_entropy': 0.05,
            'color_variance': 0.00
        }
        
        score = 0.0
        
        # Normalize and weight each feature
        # Object count (normalize to 0-1, assuming max 20 objects)
        score += weights['object_count'] * min(features.get('object_count', 0) / 20.0, 1.0)
        
        # Object density (already 0-1)
        score += weights['object_density'] * features.get('object_density', 0)
        
        # Pedestrian count (normalize to 0-1, assuming max 10)
        score += weights['pedestrian_count'] * min(features.get('pedestrian_count', 0) / 10.0, 1.0)
        
        # Vehicle count (normalize to 0-1, assuming max 15)
        score += weights['vehicle_count'] * min(features.get('vehicle_count', 0) / 15.0, 1.0)
        
        # Cyclist count (normalize to 0-1, assuming max 5)
        score += weights['cyclist_count'] * min(features.get('cyclist_count', 0) / 5.0, 1.0)
        
        # Brightness (inverted - darker is more complex)
        score += weights['brightness_mean'] * (1.0 - features.get('brightness_mean', 0.5))
        
        # Other features (already normalized)
        score += weights['brightness_std'] * features.get('brightness_std', 0)
        score += weights['contrast'] * features.get('contrast', 0)
        score += weights['motion_score'] * features.get('motion_score', 0)
        score += weights['edge_density'] * features.get('edge_density', 0)
        score += weights['spatial_entropy'] * min(features.get('spatial_entropy', 0) / 3.5, 1.0)
        
        # Clip to [0, 1]
        return np.clip(score, 0.0, 1.0)


if __name__ == "__main__":
    # Test feature extractor
    extractor = FeatureExtractor()
    
    # Create dummy data
    image = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    annotations = [
        {
            'class': 'Car',
            'bbox_2d': [100, 200, 300, 400],
            'is_critical': False
        },
        {
            'class': 'Pedestrian',
            'bbox_2d': [500, 150, 550, 350],
            'is_critical': True
        }
    ]
    
    # Extract features
    features = extractor.extract_all_features(image, annotations)
    print("Extracted features:")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")
    
    # Get feature vector
    feature_vec = extractor.get_feature_vector(features)
    print(f"\nFeature vector shape: {feature_vec.shape}")
    print(f"Feature vector: {feature_vec}")
    
    # Compute complexity score
    complexity = extractor.compute_complexity_score(features)
    print(f"\nComplexity score: {complexity:.4f}")