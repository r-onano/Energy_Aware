"""
Multi-dataset evaluation for energy-aware perception scheduling.
Evaluates performance on both KITTI and BDD100K datasets.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.kitti_loader import KITTIDataset
from data_processing.bdd100k_loader import BDD100KDataset
from scheduling.scheduler import EnergyAwareScheduler
from evaluation.evaluate import PerformanceEvaluator
from utils import setup_logging, load_config


def evaluate_dataset(dataset_name, config, logger, model_path, n_frames=None):
    """
    Evaluate on a single dataset.
    
    Args:
        dataset_name: 'kitti' or 'bdd100k'
        config: Configuration dictionary
        logger: Logger instance
        model_path: Path to complexity prediction model
        n_frames: Number of frames to evaluate (None for all)
    
    Returns:
        Results DataFrame
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATING ON {dataset_name.upper()}")
    logger.info(f"{'='*70}")
    
    # Load dataset
    if dataset_name == 'kitti':
        dataset_path = config['dataset']['kitti']['path']
        dataset = KITTIDataset(dataset_path, split='training')
    elif dataset_name == 'bdd100k':
        dataset_path = config['dataset']['bdd100k']['path']
        # For evaluation, use val split or limit train split
        max_eval = 500 if n_frames is None else n_frames
        dataset = BDD100KDataset(dataset_path, split='val', max_images=max_eval)
    
    # Load test indices
    processed_dir = Path(config['output']['processed_data'])
    splits_file = processed_dir / f'{dataset_name}_processed.npz'
    
    if splits_file.exists():
        data = np.load(splits_file, allow_pickle=True)
        test_idx = data['test_idx'].tolist()
    else:
        logger.warning(f"No processed splits found for {dataset_name}. Using last 15% as test set.")
        _, _, test_idx = dataset.split_dataset()
    
    # Limit frames if specified
    if n_frames and len(test_idx) > n_frames:
        test_idx = test_idx[:n_frames]
    
    logger.info(f"Evaluating on {len(test_idx)} test frames")
    
    # Create evaluator
    evaluator = PerformanceEvaluator(config, logger)
    
    # Compare with baselines
    results_df = evaluator.compare_baselines(dataset, test_idx, model_path)
    
    # Add dataset column
    results_df['dataset'] = dataset_name.upper()
    
    # Print results
    logger.info(f"\n{'='*70}")
    logger.info(f"{dataset_name.upper()} EVALUATION RESULTS")
    logger.info(f"{'='*70}")
    print(results_df.to_string(index=False))
    logger.info(f"{'='*70}")
    
    # Save results
    results_dir = Path(config['output']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / f'{dataset_name}_comparison_results.csv', index=False)
    logger.info(f"Results saved to {results_dir / f'{dataset_name}_comparison_results.csv'}")
    
    # Create visualizations
    logger.info(f"\nGenerating visualizations for {dataset_name}...")
    viz_dir = results_dir / dataset_name
    evaluator.visualize_results(results_df, viz_dir)
    
    return results_df


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate energy-aware scheduling on multiple datasets')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to complexity prediction model')
    parser.add_argument('--datasets', type=str, nargs='+', default=['kitti', 'bdd100k'],
                       choices=['kitti', 'bdd100k'],
                       help='Datasets to evaluate')
    parser.add_argument('--n-frames', type=int, default=None,
                       help='Number of frames to evaluate per dataset (None for all test set)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['output']['logs'])
    
    try:
        logger.info("="*70)
        logger.info("MULTI-DATASET EVALUATION PIPELINE")
        logger.info("="*70)
        
        # Find model if not specified
        if args.model is None:
            model_dir = Path(config['output']['models'])
            # Try random forest first, then other models
            for model_name in ['random_forest', 'xgboost', 'linear']:
                model_path = model_dir / f'{model_name}_complexity_predictor.pkl'
                if model_path.exists():
                    args.model = str(model_path)
                    logger.info(f"Using model: {args.model}")
                    break
            
            if args.model is None:
                raise ValueError("No trained model found. Please train models first.")
        
        # Evaluate each dataset
        all_results = []
        
        for dataset_name in args.datasets:
            results_df = evaluate_dataset(
                dataset_name,
                config,
                logger,
                args.model,
                args.n_frames
            )
            all_results.append(results_df)
        
        # Combine results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        results_dir = Path(config['output']['results'])
        combined_file = results_dir / 'all_datasets_comparison.csv'
        combined_results.to_csv(combined_file, index=False)
        
        # Print comparative summary
        logger.info("\n" + "="*70)
        logger.info("COMPARATIVE SUMMARY ACROSS DATASETS")
        logger.info("="*70)
        
        # Group by dataset and strategy
        summary = combined_results.groupby(['dataset', 'strategy']).agg({
            'energy_savings_pct': 'mean',
            'avg_f1': 'mean',
            'avg_critical_detection_rate': 'mean'
        }).round(3)
        
        print(summary)
        
        logger.info(f"\n{'='*70}")
        logger.info("MULTI-DATASET EVALUATION COMPLETED!")
        logger.info(f"{'='*70}")
        logger.info(f"Combined results saved to: {combined_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
