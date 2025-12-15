"""
Data preprocessing pipeline for KITTI dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.kitti_loader import KITTIDataset
from data_processing.feature_extraction import FeatureExtractor
from utils import load_config, setup_logging, set_seed


def preprocess_dataset(config, logger, dataset_name='kitti'):
    """
    Preprocess dataset and extract features.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        dataset_name: Name of dataset ('kitti' or 'bdd100k')
    """
    logger.info(f"Starting preprocessing for {dataset_name} dataset...")
    
    # Set random seed
    set_seed(config['training']['random_seed'])
    
    # Load dataset
    if dataset_name == 'kitti':
        dataset_path = config['dataset']['kitti']['path']
        dataset = KITTIDataset(dataset_path, split='training')
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")
    
    logger.info(f"Loaded {len(dataset)} images")
    
    # Split dataset
    train_ratio = config['dataset'][dataset_name]['split']['train']
    val_ratio = config['dataset'][dataset_name]['split']['val']
    train_idx, val_idx, test_idx = dataset.split_dataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=config['training']['random_seed']
    )
    
    logger.info(f"Dataset split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features for all images
    all_features = []
    all_complexity = []
    all_labels = []
    all_indices = []
    
    prev_image = None
    
    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        image, annotations = dataset[idx]
        
        # Extract features
        features = extractor.extract_all_features(image, annotations, prev_image)
        
        # Compute complexity score
        complexity = extractor.compute_complexity_score(features)
        
        # Create label (for classification)
        # 0: skip, 1: light, 2: medium, 3: heavy
        if complexity < 0.1:
            label = 0  # skip
        elif complexity < 0.3:
            label = 1  # light
        elif complexity < 0.6:
            label = 2  # medium
        else:
            label = 3  # heavy
        
        # Override for safety
        if features['pedestrian_count'] > 0 or features['cyclist_count'] > 0:
            label = max(label, 2)  # At least medium for critical objects
        
        if features['brightness_mean'] < 0.3:  # Low visibility
            label = max(label, 2)  # At least medium
        
        # Store
        feature_vec = extractor.get_feature_vector(features)
        all_features.append(feature_vec)
        all_complexity.append(complexity)
        all_labels.append(label)
        all_indices.append(idx)
        
        prev_image = image
    
    # Convert to arrays
    X = np.array(all_features)
    y_complexity = np.array(all_complexity)
    y_labels = np.array(all_labels)
    indices = np.array(all_indices)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Complexity distribution: {np.histogram(y_complexity, bins=4)[0]}")
    logger.info(f"Label distribution: {np.bincount(y_labels)}")
    
    # Save processed data
    output_dir = Path(config['output']['processed_data'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame for easy analysis
    df = pd.DataFrame(X, columns=extractor.feature_names)
    df['complexity_score'] = y_complexity
    df['label'] = y_labels
    df['image_index'] = indices
    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'
    
    # Save
    csv_path = output_dir / f'{dataset_name}_features.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved features to {csv_path}")
    
    # Save numpy arrays
    npz_path = output_dir / f'{dataset_name}_processed.npz'
    np.savez(
        npz_path,
        X=X,
        y_complexity=y_complexity,
        y_labels=y_labels,
        indices=indices,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_names=extractor.feature_names
    )
    logger.info(f"Saved numpy arrays to {npz_path}")
    
    # Save split indices
    split_path = output_dir / f'{dataset_name}_splits.pkl'
    with open(split_path, 'wb') as f:
        pickle.dump({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }, f)
    logger.info(f"Saved splits to {split_path}")
    
    # Compute and save statistics
    stats = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'feature_means': X.mean(axis=0).tolist(),
        'feature_stds': X.std(axis=0).tolist(),
        'complexity_mean': float(y_complexity.mean()),
        'complexity_std': float(y_complexity.std()),
        'label_distribution': np.bincount(y_labels).tolist()
    }
    
    stats_path = output_dir / f'{dataset_name}_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    logger.info(f"Saved statistics to {stats_path}")
    
    logger.info("Preprocessing complete!")
    
    return df, stats


def main():
    """Main preprocessing function."""
    # Load config
    config = load_config('configs/config.yaml')
    
    # Setup logging
    logger = setup_logging(config['output']['logs'])
    
    try:
        # Preprocess KITTI dataset
        df, stats = preprocess_dataset(config, logger, dataset_name='kitti')
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total samples: {stats['n_samples']}")
        logger.info(f"Features: {stats['n_features']}")
        logger.info(f"Train/Val/Test: {stats['n_train']}/{stats['n_val']}/{stats['n_test']}")
        logger.info(f"Complexity mean±std: {stats['complexity_mean']:.3f}±{stats['complexity_std']:.3f}")
        logger.info(f"Label distribution: {stats['label_distribution']}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
