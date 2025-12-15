"""
Multi-dataset preprocessing for KITTI and BDD100K.
Extracts features and generates complexity labels for both datasets.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.kitti_loader import KITTIDataset
from data_processing.bdd100k_loader import BDD100KDataset
from data_processing.feature_extraction import FeatureExtractor
from utils import setup_logging, load_config


def preprocess_dataset(dataset, dataset_name, output_dir, config, logger):
    """
    Preprocess a single dataset.
    
    Args:
        dataset: Dataset object (KITTI or BDD100K)
        dataset_name: Name of dataset ('kitti' or 'bdd100k')
        output_dir: Output directory for processed data
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Dictionary with processing results
    """
    logger.info(f"="*70)
    logger.info(f"PREPROCESSING {dataset_name.upper()} DATASET")
    logger.info(f"="*70)
    
    feature_extractor = FeatureExtractor()
    
    # Storage for features and labels
    all_features = []
    all_complexity_scores = []
    all_labels = []  # Model selection labels
    image_paths = []
    
    # Process each image
    logger.info(f"Processing {len(dataset)} images...")
    
    for idx in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}"):
        try:
            # Load image and annotations
            image, annotations = dataset[idx]
            
            # Extract features
            features = feature_extractor.extract_all_features(image, annotations)
            
            # Compute complexity score
            complexity_score = feature_extractor.compute_complexity_score(features)
            
            # Determine model label based on complexity thresholds
            thresholds = config['scheduling']['thresholds']
            if complexity_score < thresholds['skip']:
                label = 0  # skip
            elif complexity_score < thresholds['light']:
                label = 1  # light
            elif complexity_score < thresholds['medium']:
                label = 2  # medium
            else:
                label = 3  # heavy
            
            # Store
            all_features.append(list(features.values()))
            all_complexity_scores.append(complexity_score)
            all_labels.append(label)
            image_paths.append(dataset.get_image_path(idx))
            
        except Exception as e:
            logger.warning(f"Error processing image {idx}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    complexity_array = np.array(all_complexity_scores)
    labels_array = np.array(all_labels)
    
    logger.info(f"\nProcessed {len(features_array)} images successfully")
    logger.info(f"Feature shape: {features_array.shape}")
    logger.info(f"Complexity score range: [{complexity_array.min():.4f}, {complexity_array.max():.4f}]")
    
    # Print label distribution
    unique, counts = np.unique(labels_array, return_counts=True)
    label_names = ['skip', 'light', 'medium', 'heavy']
    logger.info("\nLabel distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(labels_array) * 100
        logger.info(f"  {label_names[label]}: {count} ({pct:.1f}%)")
    
    # Split dataset
    logger.info("\nSplitting dataset...")
    train_idx, val_idx, test_idx = dataset.split_dataset()
    
    logger.info(f"Train: {len(train_idx)} samples")
    logger.info(f"Val: {len(val_idx)} samples")
    logger.info(f"Test: {len(test_idx)} samples")
    
    # Save processed data
    output_file = output_dir / f'{dataset_name}_processed.npz'
    
    np.savez(
        output_file,
        X=features_array,  # Changed from 'features' to 'X'
        y_complexity=complexity_array,  # Changed from 'complexity_scores'
        y_labels=labels_array,  # Changed from 'labels'
        indices=np.array(range(len(features_array))),  # Add indices array
        image_paths=np.array(image_paths),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_names=np.array(list(feature_extractor.feature_names))
    )
    
    logger.info(f"\nSaved processed data to {output_file}")
    
    return {
        'dataset': dataset_name,
        'total_samples': len(features_array),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'complexity_mean': float(complexity_array.mean()),
        'complexity_std': float(complexity_array.std()),
        'label_distribution': {label_names[i]: int(c) for i, c in zip(unique, counts)}
    }


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess datasets for energy-aware scheduling')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', type=str, nargs='+', default=['kitti', 'bdd100k'],
                       choices=['kitti', 'bdd100k'],
                       help='Datasets to process')
    parser.add_argument('--bdd-max-images', type=int, default=None,
                       help='Maximum BDD100K images to process (for testing)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['output']['logs'])
    
    try:
        logger.info("="*70)
        logger.info("MULTI-DATASET PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # Create output directory
        output_dir = Path(config['output']['processed_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process each requested dataset
        for dataset_name in args.datasets:
            logger.info(f"\n{'='*70}")
            logger.info(f"Loading {dataset_name.upper()} dataset...")
            logger.info(f"{'='*70}")
            
            # Load dataset
            if dataset_name == 'kitti':
                dataset_path = config['dataset']['kitti']['path']
                dataset = KITTIDataset(dataset_path, split='training')
                
            elif dataset_name == 'bdd100k':
                dataset_path = config['dataset']['bdd100k']['path']
                # Use training split for BDD100K
                max_images = args.bdd_max_images
                dataset = BDD100KDataset(dataset_path, split='train', max_images=max_images)
            
            # Preprocess
            result = preprocess_dataset(dataset, dataset_name, output_dir, config, logger)
            results.append(result)
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*70)
        
        for result in results:
            logger.info(f"\n{result['dataset'].upper()}:")
            logger.info(f"  Total samples: {result['total_samples']}")
            logger.info(f"  Train/Val/Test: {result['train_samples']}/{result['val_samples']}/{result['test_samples']}")
            logger.info(f"  Complexity: {result['complexity_mean']:.4f} ± {result['complexity_std']:.4f}")
            logger.info(f"  Label distribution:")
            for label, count in result['label_distribution'].items():
                logger.info(f"    {label}: {count}")
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
