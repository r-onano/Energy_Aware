"""
KITTI dataset loader and processor.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


class KITTIDataset:
    """KITTI dataset loader for object detection."""
    
    # KITTI class mappings
    CLASSES = {
        'Car': 0,
        'Van': 0,
        'Truck': 0,
        'Pedestrian': 1,
        'Person_sitting': 1,
        'Cyclist': 2,
        'Tram': 3,
        'Misc': 4,
        'DontCare': -1
    }
    
    CRITICAL_CLASSES = ['Pedestrian', 'Person_sitting', 'Cyclist']
    
    def __init__(self, data_root: str, split: str = 'training'):
        """
        Initialize KITTI dataset.
        
        Args:
            data_root: Path to KITTI dataset root
            split: 'training' or 'testing'
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_dir = self.data_root / split / 'image_2'
        self.label_dir = self.data_root / split / 'label_2'
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"Loaded {len(self.image_files)} images from KITTI {split} set")
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get image and annotations.
        
        Args:
            idx: Index of image
            
        Returns:
            Tuple of (image, annotations)
        """
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations (if available)
        annotations = []
        if self.split == 'training':
            label_path = self.label_dir / (image_path.stem + '.txt')
            if label_path.exists():
                annotations = self._parse_label(label_path)
        
        return image, annotations
    
    def _parse_label(self, label_path: Path) -> List[Dict]:
        """
        Parse KITTI label file.
        
        Args:
            label_path: Path to label file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                obj_class = parts[0]
                if obj_class == 'DontCare':
                    continue
                
                truncated = float(parts[1])
                occluded = int(parts[2])
                alpha = float(parts[3])
                
                # 2D bounding box
                bbox = [
                    float(parts[4]),  # left
                    float(parts[5]),  # top
                    float(parts[6]),  # right
                    float(parts[7])   # bottom
                ]
                
                # 3D dimensions
                dimensions = [
                    float(parts[8]),  # height
                    float(parts[9]),  # width
                    float(parts[10])  # length
                ]
                
                # 3D location
                location = [
                    float(parts[11]),  # x
                    float(parts[12]),  # y
                    float(parts[13])   # z
                ]
                
                rotation_y = float(parts[14])
                
                annotations.append({
                    'class': obj_class,
                    'class_id': self.CLASSES.get(obj_class, 4),
                    'truncated': truncated,
                    'occluded': occluded,
                    'alpha': alpha,
                    'bbox_2d': bbox,
                    'dimensions_3d': dimensions,
                    'location_3d': location,
                    'rotation_y': rotation_y,
                    'is_critical': obj_class in self.CRITICAL_CLASSES
                })
        
        return annotations
    
    def get_image_path(self, idx: int) -> str:
        """Get path to image file."""
        return str(self.image_files[idx])
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_images': len(self),
            'class_counts': {cls: 0 for cls in self.CLASSES.keys()},
            'critical_objects': 0,
            'avg_objects_per_image': 0,
            'images_with_pedestrians': 0,
        }
        
        total_objects = 0
        
        for idx in tqdm(range(len(self)), desc="Computing statistics"):
            _, annotations = self[idx]
            total_objects += len(annotations)
            
            has_pedestrian = False
            for ann in annotations:
                obj_class = ann['class']
                stats['class_counts'][obj_class] += 1
                
                if ann['is_critical']:
                    stats['critical_objects'] += 1
                    has_pedestrian = True
            
            if has_pedestrian:
                stats['images_with_pedestrians'] += 1
        
        stats['avg_objects_per_image'] = total_objects / len(self) if len(self) > 0 else 0
        
        return stats
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                     random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset into train/val/test indices.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        np.random.seed(random_seed)
        
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        
        train_end = int(len(self) * train_ratio)
        val_end = train_end + int(len(self) * val_ratio)
        
        train_indices = indices[:train_end].tolist()
        val_indices = indices[train_end:val_end].tolist()
        test_indices = indices[val_end:].tolist()
        
        return train_indices, val_indices, test_indices


def visualize_annotations(image: np.ndarray, annotations: List[Dict],
                         save_path: Optional[str] = None):
    """
    Visualize image with bounding boxes.
    
    Args:
        image: Input image
        annotations: List of annotations
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = {
        0: 'blue',    # Vehicle
        1: 'red',     # Pedestrian
        2: 'green',   # Cyclist
        3: 'yellow',  # Tram
        4: 'gray'     # Misc
    }
    
    for ann in annotations:
        bbox = ann['bbox_2d']
        class_id = ann['class_id']
        obj_class = ann['class']
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=colors.get(class_id, 'white'),
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(bbox[0], bbox[1] - 5, obj_class,
               color=colors.get(class_id, 'white'),
               fontsize=10, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test KITTI dataset loader
    dataset = KITTIDataset("data/raw/kitti", split='training')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first image
    if len(dataset) > 0:
        image, annotations = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Number of annotations: {len(annotations)}")
        
        if annotations:
            print("\nFirst annotation:")
            print(annotations[0])
        
        # Get statistics
        stats = dataset.get_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Split dataset
        train_idx, val_idx, test_idx = dataset.split_dataset()
        print(f"\nDataset split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
