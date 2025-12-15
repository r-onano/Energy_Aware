"""
Evaluation module for energy-aware perception scheduling.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.kitti_loader import KITTIDataset
from scheduling.scheduler import EnergyAwareScheduler
from utils import load_config, setup_logging


class PerformanceEvaluator:
    """Evaluate energy-aware scheduling performance."""
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or print
        self.results = []
    
    def evaluate_scheduler(self, scheduler: EnergyAwareScheduler,
                          dataset: KITTIDataset,
                          indices: List[int]) -> Dict:
        """
        Evaluate scheduler on dataset.
        
        Args:
            scheduler: Energy-aware scheduler
            dataset: KITTI dataset
            indices: Frame indices to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Evaluating on {len(indices)} frames...")
        
        # Reset scheduler statistics
        scheduler.reset_statistics()
        
        # Process all frames
        all_metrics = []
        
        for idx in tqdm(indices, desc="Processing frames"):
            image, annotations = dataset[idx]
            detections, metrics = scheduler.process_frame(image, annotations)
            all_metrics.append(metrics)
        
        # Get scheduler statistics
        stats = scheduler.get_statistics()
        
        # Aggregate metrics
        results = {
            'total_frames': len(indices),
            'total_energy': stats['total_energy'],
            'baseline_energy': stats['baseline_energy'],
            'energy_savings_pct': stats['energy_savings_pct'],
            'avg_energy_per_frame': stats['avg_energy_per_frame'],
            'model_usage': stats['model_usage'],
            'model_usage_pct': stats['model_usage_pct'],
        }
        
        # Aggregate detection metrics (if available)
        metrics_with_gt = [m for m in all_metrics if 'precision' in m]
        
        if metrics_with_gt:
            results['avg_precision'] = np.mean([m['precision'] for m in metrics_with_gt])
            results['avg_recall'] = np.mean([m['recall'] for m in metrics_with_gt])
            results['avg_f1'] = np.mean([m['f1'] for m in metrics_with_gt])
            results['avg_critical_detection_rate'] = np.mean([m['critical_detection_rate'] for m in metrics_with_gt])
        
        # Store detailed results
        self.results = all_metrics
        
        return results
    
    def compare_baselines(self, dataset: KITTIDataset, indices: List[int],
                         model_path: str = None) -> pd.DataFrame:
        """
        Compare energy-aware scheduling with baseline strategies.
        
        Args:
            dataset: KITTI dataset
            indices: Frame indices
            model_path: Path to complexity prediction model
            
        Returns:
            Comparison DataFrame
        """
        self.logger.info("Comparing with baselines...")
        
        comparisons = []
        
        # 1. Always Heavy (baseline)
        self.logger.info("Evaluating: Always Heavy")
        results_heavy = self._evaluate_fixed_model(dataset, indices, 'heavy')
        results_heavy['strategy'] = 'Always Heavy'
        comparisons.append(results_heavy)
        
        # 2. Always Medium
        self.logger.info("Evaluating: Always Medium")
        results_medium = self._evaluate_fixed_model(dataset, indices, 'medium')
        results_medium['strategy'] = 'Always Medium'
        comparisons.append(results_medium)
        
        # 3. Always Light
        self.logger.info("Evaluating: Always Light")
        results_light = self._evaluate_fixed_model(dataset, indices, 'light')
        results_light['strategy'] = 'Always Light'
        comparisons.append(results_light)
        
        # 4. Energy-Aware (ours)
        self.logger.info("Evaluating: Energy-Aware")
        scheduler_adaptive = EnergyAwareScheduler(self.config, model_path)
        results_adaptive = self.evaluate_scheduler(scheduler_adaptive, dataset, indices)
        results_adaptive['strategy'] = 'Energy-Aware (Ours)'
        comparisons.append(results_adaptive)
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparisons)
        
        return df
    
    def _evaluate_fixed_model(self, dataset: KITTIDataset, indices: List[int],
                              model_level: str) -> Dict:
        """
        Evaluate using a fixed model (for baseline comparison).
        
        Args:
            dataset: KITTI dataset
            indices: Frame indices to evaluate
            model_level: Fixed model level ('light', 'medium', 'heavy')
            
        Returns:
            Evaluation results dictionary
        """
        from perception.yolo_detector import YOLOPerception
        
        # Initialize perception
        perception = YOLOPerception(self.config)
        
        # Track metrics
        total_energy = 0.0
        all_metrics = []
        model_usage = {'skip': 0, 'light': 0, 'medium': 0, 'heavy': 0}
        
        self.logger.info(f"Evaluating on {len(indices)} frames...")
        
        for idx in tqdm(indices, desc="Processing frames"):
            image, annotations = dataset[idx]
            
            # Always use the specified model
            detections, energy, latency = perception.detect(image, model_level)
            
            # Update tracking
            total_energy += energy
            model_usage[model_level] += 1
            
            # Compute metrics
            metrics = {
                'model_level': model_level,
                'energy': energy,
                'latency': latency,
                'n_detections': len(detections)
            }
            
            # Evaluate if ground truth available
            if annotations:
                eval_metrics = perception.evaluate_detection(detections, annotations)
                metrics.update(eval_metrics)
            
            all_metrics.append(metrics)
        
        # Aggregate results
        total_frames = len(indices)
        baseline_energy = total_frames * perception.ENERGY_COSTS['heavy']
        
        results = {
            'total_frames': total_frames,
            'total_energy': total_energy,
            'baseline_energy': baseline_energy,
            'energy_savings_pct': (baseline_energy - total_energy) / baseline_energy * 100 if baseline_energy > 0 else 0,
            'avg_energy_per_frame': total_energy / total_frames if total_frames > 0 else 0,
            'model_usage': model_usage.copy(),
            'model_usage_pct': {
                level: count / total_frames * 100 if total_frames > 0 else 0
                for level, count in model_usage.items()
            }
        }
        
        # Aggregate detection metrics (if available)
        metrics_with_gt = [m for m in all_metrics if 'precision' in m]
        
        if metrics_with_gt:
            results['avg_precision'] = np.mean([m['precision'] for m in metrics_with_gt])
            results['avg_recall'] = np.mean([m['recall'] for m in metrics_with_gt])
            results['avg_f1'] = np.mean([m['f1'] for m in metrics_with_gt])
            results['avg_critical_detection_rate'] = np.mean([m['critical_detection_rate'] for m in metrics_with_gt])
        
        return results
    
    def visualize_results(self, results_df: pd.DataFrame, save_dir: str = None):
        """
        Create visualizations of results.
        
        Args:
            results_df: Results DataFrame
            save_dir: Directory to save plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Energy vs Accuracy Trade-off
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if 'avg_f1' in results_df.columns:
            x = results_df['avg_energy_per_frame']
            y = results_df['avg_f1']
            
            for idx, row in results_df.iterrows():
                ax.scatter(row['avg_energy_per_frame'], row['avg_f1'], s=200)
                ax.annotate(row['strategy'], 
                          (row['avg_energy_per_frame'], row['avg_f1']),
                          xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Average Energy per Frame (J)', fontsize=12)
            ax.set_ylabel('Average F1 Score', fontsize=12)
            ax.set_title('Energy vs Detection Performance Trade-off', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if save_dir:
                plt.savefig(save_dir / 'energy_vs_accuracy.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # 2. Energy Savings Comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        strategies = results_df['strategy']
        energy_savings = results_df['energy_savings_pct']
        
        colors = ['#e74c3c' if 'Always' in s else '#27ae60' for s in strategies]
        bars = ax.bar(strategies, energy_savings, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Energy Savings (%)', fontsize=12)
        ax.set_title('Energy Savings Comparison vs Always Heavy Baseline', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'energy_savings.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # 3. Model Selection Frequency (for energy-aware)
        adaptive_row = results_df[results_df['strategy'] == 'Energy-Aware (Ours)']
        
        if not adaptive_row.empty:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            model_usage = adaptive_row.iloc[0]['model_usage_pct']
            labels = list(model_usage.keys())
            sizes = list(model_usage.values())
            colors = ['#95a5a6', '#3498db', '#f39c12', '#e74c3c']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                              colors=colors, startangle=90,
                                              textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            ax.set_title('Model Selection Frequency\n(Energy-Aware Strategy)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            if save_dir:
                plt.savefig(save_dir / 'model_selection_frequency.png', dpi=300, bbox_inches='tight')
            
            plt.close()
        
        # 4. Critical Object Detection Rate (if available)
        if 'avg_critical_detection_rate' in results_df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            strategies = results_df['strategy']
            detection_rates = results_df['avg_critical_detection_rate'] * 100
            
            colors = ['#e74c3c' if rate < 95 else '#27ae60' for rate in detection_rates]
            bars = ax.bar(strategies, detection_rates, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Critical Object Detection Rate (%)', fontsize=12)
            ax.set_title('Safety: Critical Object Detection Rate', fontsize=14, fontweight='bold')
            ax.axhline(y=95, color='red', linestyle='--', label='Target (95%)', linewidth=2)
            ax.set_ylim([0, 105])
            ax.legend()
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_dir / 'critical_detection_rate.png', dpi=300, bbox_inches='tight')
            
            plt.close()
        
        self.logger.info(f"Visualizations saved to {save_dir}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate energy-aware scheduling')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to complexity prediction model')
    parser.add_argument('--dataset', type=str, default='kitti',
                       help='Dataset name')
    parser.add_argument('--n_frames', type=int, default=None,
                       help='Number of frames to evaluate (None for all test set)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['output']['logs'])
    
    try:
        logger.info("Starting evaluation...")
        
        # Load dataset
        dataset_path = config['dataset'][args.dataset]['path']
        dataset = KITTIDataset(dataset_path, split='training')
        
        # Load splits
        processed_dir = Path(config['output']['processed_data'])
        splits_file = processed_dir / f'{args.dataset}_processed.npz'
        
        if splits_file.exists():
            data = np.load(splits_file, allow_pickle=True)
            test_idx = data['test_idx'].tolist()
        else:
            logger.warning("No processed splits found. Using last 15% as test set.")
            _, _, test_idx = dataset.split_dataset()
        
        # Limit frames if specified
        if args.n_frames:
            test_idx = test_idx[:args.n_frames]
        
        logger.info(f"Evaluating on {len(test_idx)} test frames")
        
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
        
        # Create evaluator
        evaluator = PerformanceEvaluator(config, logger)
        
        # Compare with baselines
        results_df = evaluator.compare_baselines(dataset, test_idx, args.model)
        
        # Print results
        logger.info("\n" + "="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        print(results_df.to_string(index=False))
        logger.info("="*70)
        
        # Save results
        results_dir = Path(config['output']['results'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(results_dir / 'comparison_results.csv', index=False)
        logger.info(f"Results saved to {results_dir / 'comparison_results.csv'}")
        
        # Create visualizations
        evaluator.visualize_results(results_df, results_dir)
        
        logger.info("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
