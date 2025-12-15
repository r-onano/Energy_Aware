"""
BDD100K dataset loader and processor.
Handles BDD100K format with individual JSON label files per image.
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


class BDD100KDataset:
    """BDD100K dataset loader for object detection."""
    
    # BDD100K class mappings (map to same classes as KITTI for consistency)
    CLASSES = {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'trailer': 0,
        'pedestrian': 1,
        'rider': 2,  # Similar to cyclist
        'bike': 2,
        'motor': 2,
        'motorcycle': 2,
        'train': 3,
        'traffic light': 4,
        'traffic sign': 4,
        'other vehicle': 4,
        'other person': 1
    }
    
    CRITICAL_CLASSES = ['pedestrian', 'rider', 'bike', 'motor', 'motorcycle', 'other person']
    
    def __init__(self, data_root: str, split: str = 'train', max_images: Optional[int] = None):
        """
        Initialize BDD100K dataset.
        
        Args:
            data_root: Path to BDD100K dataset root
            split: 'train', 'val', or 'test'
            max_images: Maximum number of images to load (for quick testing)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_images = max_images
        
        # BDD100K structure: bdd100k/images/100k/{train,val,test}
        self.image_dir = self.data_root / 'images' / '100k' / split
        
        # Labels: bdd100k/labels/100k/{train,val,test}/*.json (one file per image)
        self.label_dir = self.data_root / 'labels' / '100k' / split
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')))
        
        # Limit if max_images specified
        if self.max_images and len(self.image_files) > self.max_images:
            self.image_files = self.image_files[:self.max_images]
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"Loaded {len(self.image_files)} images from BDD100K {split} set")
        if self.max_images:
            print(f"  (Limited to {self.max_images} images for testing)")
        
        # Check if labels exist
        if self.label_dir.exists():
            print(f"Labels directory found: {self.label_dir}")
        else:
            print(f"WARNING: Labels directory not found: {self.label_dir}")
            print(f"  Running in image-only mode (no annotations)")
    
    def _load_annotation_file(self, image_path: Path) -> List[Dict]:
        """
        Load annotation from individual JSON file for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of annotations
        """
        # Label file has same name as image but .json extension
        # e.g., cabc30fc-e7726578.jpg -> cabc30fc-e7726578.json
        label_file = self.label_dir / (image_path.stem + '.json')
        
        if not label_file.exists():
            return []
        
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
            
            annotations = []
            
            # BDD100K JSON structure can vary:
            # Format 1: {"frames": [{"objects": [...]}]}
            # Format 2: {"labels": [...]}
            # Format 3: Direct array of labels
            labels = []
            
            if 'frames' in data:
                # New format with frames - check for both 'objects' and 'labels'
                for frame in data['frames']:
                    # BDD100K uses 'objects' key in frames
                    labels.extend(frame.get('objects', frame.get('labels', [])))
            elif 'labels' in data:
                # Direct labels format
                labels = data['labels']
            elif isinstance(data, list):
                # Direct array
                labels = data
            
            for label in labels:
                category = label.get('category', '').lower()
                
                # Skip if no box2d or ignored categories
                if 'box2d' not in label:
                    continue
                    
                if category in ['drivable area', 'lane']:
                    continue
                
                box = label['box2d']
                
                # Check valid bbox
                if box['x1'] >= box['x2'] or box['y1'] >= box['y2']:
                    continue
                
                bbox = [
                    box['x1'],
                    box['y1'],
                    box['x2'],
                    box['y2']
                ]
                
                annotations.append({
                    'class': category,
                    'class_id': self.CLASSES.get(category, 4),
                    'bbox_2d': bbox,
                    'is_critical': category in self.CRITICAL_CLASSES,
                    'truncated': 0.0,  # Not available in BDD100K
                    'occluded': 0,     # Not available in BDD100K
                })
            
            return annotations
            
        except Exception as e:
            # Silently skip failed files (common for test set without labels)
            return []
    
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
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations from individual JSON file
        annotations = self._load_annotation_file(image_path)
        
        return image, annotations
    
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
            'class_counts': {cls: 0 for cls in set(self.CLASSES.values())},
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
                class_id = ann['class_id']
                stats['class_counts'][class_id] = stats['class_counts'].get(class_id, 0) + 1
                
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
        2: 'green',   # Cyclist/Rider
        3: 'yellow',  # Train
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
    # Test BDD100K dataset loader
    dataset = BDD100KDataset("data/raw/bdd100k", split='train', max_images=100)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get first image
    if len(dataset) > 0:
        image, annotations = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Number of annotations: {len(annotations)}")
        
        if annotations:
            print("\nFirst annotation:")
            print(annotations[0])
        
        # Test a few more images
        print("\nTesting first 10 images:")
        for i in range(min(10, len(dataset))):
            img, anns = dataset[i]
            print(f"  Image {i}: {img.shape}, {len(anns)} annotations")
        
        # Get statistics
        print("\nComputing statistics (this may take a while)...")
        stats = dataset.get_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Split dataset
        train_idx, val_idx, test_idx = dataset.split_dataset()
        print(f"\nDataset split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
