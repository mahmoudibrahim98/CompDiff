#!/usr/bin/env python
"""
Main script for training and evaluating downstream CheXpert classifiers.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import torch
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from downstream_eval_chest.train_classifier import train_model
from downstream_eval_chest.evaluate_classifier import (
    evaluate_and_save, 
    compute_fairness_metrics,
    compute_underdiagnosis_gap_only
)
from downstream_eval_chest.models import DenseNet121Classifier, load_checkpoint
from downstream_eval_chest.dataset import CheXpertClassifierDataset, CheXpertClassifierWebDataset
from downstream_eval_chest.data_utils import (
    compute_mimic_cxr_normalization,
    compute_synthetic_normalization,
    get_synthetic_training_paths,
    load_synthetic_test_set,
)
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate CheXpert classifier')
    
    # Config file (parse this first to allow loading config before requiring other args)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (overrides command-line args)')
    
    # Model strategy
    parser.add_argument('--strategy', type=str, required=False,
                       choices=['1a', '1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b', '6a'],
                       help='Training strategy')
    
    # Data paths
    parser.add_argument('--real_train_path', type=str, required=False,
                       help='Path to real training data (WebDataset directory with .tar files or CSV)')
    parser.add_argument('--real_val_path', type=str, required=False,
                       help='Path to real validation data (WebDataset directory with .tar files or CSV)')
    parser.add_argument('--real_test_path', type=str, default=None,
                       help='Path to real test data (WebDataset directory with .tar files or CSV)')
    parser.add_argument('--synthetic_base_path', type=str, default=None,
                       help='Base path to synthetic datasets')
    parser.add_argument('--synthetic_test_path', type=str, default=None,
                       help='Path to synthetic test data (for models 5a/5b)')
    
    # CheXpert evaluation
    parser.add_argument('--evaluate_on_chexpert', action='store_true',
                       help='Evaluate on CheXpert dataset instead of DTest')
    parser.add_argument('--chexpert_csv_path', type=str, default=None,
                       help='Path to CheXpert CSV file (e.g., chexpert_filtered.csv)')
    parser.add_argument('--chexpert_image_base_path', type=str, default=None,
                       help='Base path to CheXpert images directory')
    
    # Model configuration
    parser.add_argument('--model_version', type=str, default=None,
                       choices=['v0', 'v7'],
                       help='Synthetic model version (v0 or v7). Auto-set for 5a (v0) and 5b (v7)')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Synthetic dataset name (e.g., 0_train_baseline for v0, 6_train_hcn_age_from_promt for v7)')
    parser.add_argument('--num_generations', type=int, default=1,
                       help='Number of generations to combine (1, 2, or 3)')
    parser.add_argument('--subset_fraction', type=float, default=None,
                       help='Subset fraction for fine-tuning (Model 3a)')
    
    # Training hyperparameters
    parser.add_argument('--initial_lr', type=float, default=0.0001,
                       help='Initial learning rate')
    parser.add_argument('--fine_tune_lr', type=float, default=None,
                       help='Fine-tuning learning rate (default: initial_lr * 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--from_scratch', action='store_true',
                       help='Train from scratch (no ImageNet pretraining)')
    
    # Normalization
    parser.add_argument('--mimic_mean', type=float, default=None,
                       help='MIMIC-CXR normalization mean (computed if not provided)')
    parser.add_argument('--mimic_std', type=float, default=None,
                       help='MIMIC-CXR normalization std (computed if not provided)')
    parser.add_argument('--compute_normalization', action='store_true',
                       help='Compute MIMIC-CXR normalization from training data')
    parser.add_argument('--normalization_source', type=str, default='real',
                       choices=['real', 'synthetic'],
                       help='Source for normalization stats: "real" (default, MIMIC-CXR) or "synthetic" (for synthetic-only models)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/downstream_eval',
                       help='Output directory')
    
    # Evaluation
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only evaluate existing checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint for evaluation (if skip_training)')
    parser.add_argument('--use_old_order', action='store_true',
                       help='If set, model was trained with old CHEXPERT_CLASSES order. '
                            'Predictions will be remapped to new order automatically.')
    
    # Statistical analysis
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap resamples for confidence intervals (default: 1000)')
    parser.add_argument('--skip_statistics', action='store_true',
                       help='Skip statistical analysis (bootstrap CIs, etc.)')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for statistical analysis reproducibility')
    
    # Quick update option
    parser.add_argument('--only_compute_underdiagnosis_gap', action='store_true',
                       help='Only compute underdiagnosis gap and add to existing evaluation_results.json. '
                            'Requires existing predictions and metadata files. Skips full evaluation.')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse args first to get config path
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values (config takes precedence over defaults)
        for key, value in config.items():
            if hasattr(args, key):
                # Only override if value was not explicitly provided via command line
                # Check if the value matches the default (meaning it wasn't explicitly set)
                # For now, always override with config values when config is provided
                setattr(args, key, value)
            else:
                setattr(args, key, value)
    
    # Validate required arguments (either from config or command line)
    if not args.strategy:
        raise ValueError("--strategy is required (provide via --config or --strategy)")
    
    # Quick update: Only compute underdiagnosis gap if requested
    # This mode doesn't need training data or checkpoints, so check early
    if args.only_compute_underdiagnosis_gap:
        logger.info("="*60)
        logger.info("Quick update mode: Only computing underdiagnosis gap")
        logger.info("="*60)
        
        # Determine output directory and model name
        output_dir = Path(args.output_dir)
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b']
        
        if args.strategy in ['3a', '3b']:
            if args.subset_fraction is None:
                raise ValueError(f"subset_fraction is required for strategy {args.strategy}")
            # Format subset_fraction consistently (e.g., 1.0 instead of 1)
            subset_fraction_str = f"{args.subset_fraction:.1f}" if isinstance(args.subset_fraction, float) else str(args.subset_fraction)
            if args.strategy in strategies_with_synthetic and args.model_version:
                model_output_dir = output_dir / f'model_{args.strategy}_{args.model_version}_finetune_{subset_fraction_str}'
                model_name = f'model_{args.strategy}_{args.model_version}_finetune_{subset_fraction_str}'
            else:
                model_output_dir = output_dir / f'model_{args.strategy}_finetune_{subset_fraction_str}'
                model_name = f'model_{args.strategy}_finetune_{subset_fraction_str}'
        elif args.strategy in ['4a', '4b']:
            if args.strategy in strategies_with_synthetic and args.model_version:
                model_output_dir = output_dir / f'model_{args.strategy}_{args.model_version}_finetune'
                model_name = f'model_{args.strategy}_{args.model_version}_finetune'
            else:
                model_output_dir = output_dir / f'model_{args.strategy}_finetune'
                model_name = f'model_{args.strategy}_finetune'
        elif args.strategy in strategies_with_synthetic and args.model_version:
            model_output_dir = output_dir / f'model_{args.strategy}_{args.model_version}'
            model_name = f'model_{args.strategy}_{args.model_version}'
        else:
            model_output_dir = output_dir / f'model_{args.strategy}'
            model_name = f'model_{args.strategy}'
        
        # Determine evaluation dataset suffix
        from downstream_eval_chest.evaluate_classifier import sanitize_dataset_name_for_filename
        
        # For strategies 5a and 6a, we need to handle multiple datasets
        if args.strategy == '5a':
            # Default evaluation datasets for 5a (three synthetic + DTest)
            eval_datasets = ['synthetic_v0', 'synthetic_v7', 'synthetic_fairdiffusion', 'DTest']
        elif args.strategy == '6a':
            # Default evaluation datasets for 6a
            eval_datasets = ['test_data', 'GT_data', 'synthetic_v0', 'synthetic_v7']
        else:
            # For other strategies, determine from args
            if args.evaluate_on_chexpert:
                csv_filename = Path(args.chexpert_csv_path).name if args.chexpert_csv_path else 'chexpert.csv'
                eval_datasets = [f'CheXpert ({csv_filename})']
            elif args.strategy == '5b':
                eval_datasets = [f'synthetic_{args.model_version}']
            else:
                eval_datasets = ['DTest']
        
        # Process each evaluation dataset
        for eval_dataset_name in eval_datasets:
            dataset_suffix = sanitize_dataset_name_for_filename(eval_dataset_name)
            if dataset_suffix != 'dtest':
                file_suffix = f'_{dataset_suffix}'
            else:
                file_suffix = ''
            
            predictions_file = model_output_dir / f'predictions{file_suffix}.npz'
            metadata_file = model_output_dir / f'metadata{file_suffix}.json'
            results_file = model_output_dir / f'evaluation_results{file_suffix}.json'
            
            if not predictions_file.exists():
                logger.warning(f"Predictions file not found: {predictions_file}. Skipping {eval_dataset_name}.")
                continue
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found: {metadata_file}. Skipping {eval_dataset_name}.")
                continue
            if not results_file.exists():
                logger.warning(f"Results file not found: {results_file}. Skipping {eval_dataset_name}.")
                continue
            
                logger.info(f"\nProcessing {eval_dataset_name}...")
            logger.info(f"  Model directory: {model_output_dir}")
            logger.info(f"  Loading predictions: {predictions_file}")
            logger.info(f"  Loading metadata: {metadata_file}")
            logger.info(f"  Updating results: {results_file}")
            
            # Optional: Log where checkpoint would be (for reference, not used in this mode)
            checkpoint_path = model_output_dir / 'checkpoint_best.pth'
            if checkpoint_path.exists():
                logger.info(f"  (Checkpoint found at: {checkpoint_path} - not needed for this mode)")
            else:
                logger.debug(f"  (Checkpoint not found at: {checkpoint_path} - not needed for this mode)")
            
            # Load existing data
            import numpy as np
            predictions_data = np.load(predictions_file)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
            
            # Compute only underdiagnosis gap
            logger.info("Computing underdiagnosis gap...")
            gap_results = compute_underdiagnosis_gap_only(
                predictions_probs=predictions_data['predictions_probs'],
                ground_truth=predictions_data['ground_truth'],
                demographics=metadata['demographics'],
                thresholds=metadata['thresholds'],
                n_bootstrap=args.n_bootstrap,
                compute_statistics=not args.skip_statistics,
                random_seed=args.random_seed,
            )
            
            # Update existing results
            if 'fairness_metrics' not in existing_results:
                existing_results['fairness_metrics'] = {}
            
            existing_results['fairness_metrics']['underdiagnosis_gap'] = gap_results['underdiagnosis_gap']
            existing_results['fairness_metrics']['underdiagnosis_gap_ci'] = gap_results['underdiagnosis_gap_ci']
            
            # Save updated results
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            logger.info(f"✓ Updated underdiagnosis gap for {eval_dataset_name}")
            logger.info(f"  Gap: {gap_results['underdiagnosis_gap']}")
            if gap_results['underdiagnosis_gap_ci']['ci_lower'] is not None:
                logger.info(f"  95% CI: [{gap_results['underdiagnosis_gap_ci']['ci_lower']:.4f}, {gap_results['underdiagnosis_gap_ci']['ci_upper']:.4f}]")
        
        logger.info("\n" + "="*60)
        logger.info("Quick update completed!")
        logger.info("="*60)
        return
    
    # Normal mode: validate required arguments for training/evaluation
    if not args.real_train_path:
        raise ValueError("--real_train_path is required (provide via --config or --real_train_path)")
    if not args.real_val_path:
        raise ValueError("--real_val_path is required (provide via --config or --real_val_path)")
    
    # Auto-set model_version for strategies 5a and 5b
    if args.strategy == '5a':
        if args.model_version is None:
            args.model_version = 'v0'
            logger.info(f"Auto-set model_version to 'v0' for strategy 5a")
        elif args.model_version != 'v0':
            logger.warning(f"Strategy 5a typically uses v0, but model_version is set to {args.model_version}")
    elif args.strategy == '5b':
        if args.model_version is None:
            args.model_version = 'v7'
            logger.info(f"Auto-set model_version to 'v7' for strategy 5b")
        elif args.model_version != 'v7':
            logger.warning(f"Strategy 5b typically uses v7, but model_version is set to {args.model_version}")
    
    # Auto-set dataset_name if not provided for strategies that need it
    if args.dataset_name is None:
        if args.strategy in ['1b', '1c', '1d', '1e', '2b', '2c', '3a']:
            if args.model_version == 'v0':
                args.dataset_name = '0_train_baseline'
                logger.info(f"Auto-set dataset_name to '0_train_baseline' for v0")
            elif args.model_version == 'v7':
                args.dataset_name = '6_train_hcn_age_from_promt'
                logger.info(f"Auto-set dataset_name to '6_train_hcn_age_from_promt' for v7")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Compute or use normalization
    # Check if we should use synthetic normalization
    synthetic_only_strategies = ['1d', '1e', '2c']
    use_synthetic_normalization = (
        args.normalization_source == 'synthetic' and 
        args.strategy in synthetic_only_strategies
    )
    
    if args.compute_normalization or (args.mimic_mean is None or args.mimic_std is None):
        if use_synthetic_normalization:
            # Compute normalization from synthetic data
            if args.synthetic_base_path is None:
                raise ValueError(f"synthetic_base_path required for synthetic normalization with strategy {args.strategy}")
            if args.model_version is None or args.dataset_name is None:
                raise ValueError(f"model_version and dataset_name required for synthetic normalization with strategy {args.strategy}")
            
            logger.info("Computing synthetic normalization statistics...")
            synthetic_paths = get_synthetic_training_paths(
                args.synthetic_base_path,
                args.model_version,
                args.dataset_name,
                num_generations=args.num_generations,
            )
            
            # Use fewer samples for synthetic normalization (default 1000, faster than 10000)
            # Synthetic data loading is slower (individual files vs WebDataset streaming)
            num_samples = 1000  # Reduced from 10000 for faster computation
            logger.info(f"Computing normalization from {num_samples} synthetic samples (reduced for speed)")
            mimic_mean, mimic_std = compute_synthetic_normalization(
                synthetic_paths,
                num_samples=num_samples,
            )
            logger.info(f"Computed synthetic normalization: mean={mimic_mean:.6f}, std={mimic_std:.6f}")
        else:
            # Default: compute from real data (MIMIC-CXR)
            logger.info("Computing MIMIC-CXR normalization statistics...")
            # Determine data type for normalization computation
            train_path_obj = Path(args.real_train_path)
            if train_path_obj.is_dir() and any(train_path_obj.glob("*.tar")):
                norm_data_type = 'real_wds'
            elif train_path_obj.suffix == '.csv':
                norm_data_type = 'real_csv'
            else:
                norm_data_type = 'real_wds' if train_path_obj.is_dir() else 'real_csv'
            
            # FIX: Use same number of samples for both formats to ensure identical normalization
            # Previously: WebDataset used 10k samples, CSV used all samples → different stats!
            # Now: Use 10k samples for both (sufficient for statistics, avoids OOM)
            # Alternative: Use all samples for both if memory allows
            num_samples = 10000  # Use same for both formats to ensure identical normalization
            logger.info(f"Computing normalization from {num_samples} samples (same for both formats)")
            mimic_mean, mimic_std = compute_mimic_cxr_normalization(
                args.real_train_path,
                data_type=norm_data_type,
                split='train',
                num_samples=num_samples,
            )
            logger.info(f"Computed normalization: mean={mimic_mean:.6f}, std={mimic_std:.6f}")
    else:
        mimic_mean = args.mimic_mean
        mimic_std = args.mimic_std
        norm_source_str = "synthetic" if use_synthetic_normalization else "real"
        logger.info(f"Using provided normalization ({norm_source_str}): mean={mimic_mean:.4f}, std={mimic_std:.4f}")
    
    # Training
    output_dir = Path(args.output_dir)
    checkpoint_path = None
    
    if not args.skip_training:
        logger.info(f"Training model with strategy: {args.strategy}")
        checkpoint_path = train_model(
            strategy=args.strategy,
            real_train_path=Path(args.real_train_path),
            real_val_path=Path(args.real_val_path),
            synthetic_base_path=Path(args.synthetic_base_path) if args.synthetic_base_path else None,
            output_dir=output_dir,
            model_version=args.model_version,
            dataset_name=args.dataset_name,
            num_generations=args.num_generations,
            subset_fraction=args.subset_fraction,
            from_scratch=args.from_scratch,
            initial_lr=args.initial_lr,
            fine_tune_lr=args.fine_tune_lr,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mimic_mean=mimic_mean,  # Pass computed normalization values
            mimic_std=mimic_std,
            normalization_source=args.normalization_source,  # Pass normalization source
        )
        logger.info(f"Training completed. Best checkpoint: {checkpoint_path}")
    else:
        if args.checkpoint_path:
            checkpoint_path = Path(args.checkpoint_path)
        else:
            # Try to find best checkpoint in output directory
            # Check version-specific directory first for strategies that use synthetic data
            strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b']
            
            # Strategies 3a and 4a use two-phase training (pretrain + finetune)
            # The final checkpoint is in the finetune directory
            if args.strategy in ['3a', '3b']:
                # Strategy 3a/3b: finetune directory includes subset_fraction
                if args.subset_fraction is None:
                    raise ValueError(f"subset_fraction is required for strategy {args.strategy} when skipping training")
                # Format subset_fraction consistently (e.g., 1.0 instead of 1)
                subset_fraction_str = f"{args.subset_fraction:.1f}" if isinstance(args.subset_fraction, float) else str(args.subset_fraction)
                if args.strategy in strategies_with_synthetic and args.model_version:
                    checkpoint_path = output_dir / f'model_{args.strategy}_{args.model_version}_finetune_{subset_fraction_str}' / 'checkpoint_best.pth'
                    if not checkpoint_path.exists():
                        # Fallback to non-versioned directory
                        checkpoint_path = output_dir / f'model_{args.strategy}_finetune_{subset_fraction_str}' / 'checkpoint_best.pth'
                else:
                    checkpoint_path = output_dir / f'model_{args.strategy}_finetune_{subset_fraction_str}' / 'checkpoint_best.pth'
            elif args.strategy in ['4a', '4b']:
                # Strategy 4a/4b: finetune directory (no subset_fraction)
                if args.strategy in strategies_with_synthetic and args.model_version:
                    checkpoint_path = output_dir / f'model_{args.strategy}_{args.model_version}_finetune' / 'checkpoint_best.pth'
                    if not checkpoint_path.exists():
                        # Fallback to non-versioned directory
                        checkpoint_path = output_dir / f'model_{args.strategy}_finetune' / 'checkpoint_best.pth'
                else:
                    checkpoint_path = output_dir / f'model_{args.strategy}_finetune' / 'checkpoint_best.pth'
            elif args.strategy in strategies_with_synthetic and args.model_version:
                # Other strategies with synthetic data: standard directory
                checkpoint_path = output_dir / f'model_{args.strategy}_{args.model_version}' / 'checkpoint_best.pth'
                if not checkpoint_path.exists():
                    # Fallback to non-versioned directory
                    checkpoint_path = output_dir / f'model_{args.strategy}' / 'checkpoint_best.pth'
            else:
                checkpoint_path = output_dir / f'model_{args.strategy}' / 'checkpoint_best.pth'
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Using existing checkpoint: {checkpoint_path}")
    
    # Evaluation
    logger.info("Evaluating model...")
    
    # Determine evaluation dataset
    if args.evaluate_on_chexpert:
        # Evaluate on CheXpert dataset
        if args.chexpert_csv_path is None:
            raise ValueError("--chexpert_csv_path is required when --evaluate_on_chexpert is set")
        if args.chexpert_image_base_path is None:
            raise ValueError("--chexpert_image_base_path is required when --evaluate_on_chexpert is set")
        
        chexpert_csv_path = Path(args.chexpert_csv_path)
        chexpert_image_base_path = Path(args.chexpert_image_base_path)
        
        if not chexpert_csv_path.exists():
            raise FileNotFoundError(f"CheXpert CSV file not found: {chexpert_csv_path}")
        if not chexpert_image_base_path.exists():
            raise FileNotFoundError(f"CheXpert image base path not found: {chexpert_image_base_path}")
        
        # Use all data in CSV (no split filtering for CheXpert evaluation)
        test_dataset = CheXpertClassifierDataset(
            data_path=chexpert_csv_path,
            data_type='chexpert_csv',
            split=None,  # Don't filter by split - use all data
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            image_base_path=chexpert_image_base_path,
        )
        # Explicitly specify CheXpert dataset name with CSV filename for clarity
        csv_filename = chexpert_csv_path.name if hasattr(chexpert_csv_path, 'name') else Path(chexpert_csv_path).name
        evaluation_dataset = f'CheXpert ({csv_filename})'
        logger.info(f"Evaluating on CheXpert dataset: {evaluation_dataset}")
        logger.info(f"  CSV file: {chexpert_csv_path}")
        logger.info(f"  Image base path: {chexpert_image_base_path}")
        logger.info(f"  Using all data from CSV (no split filtering)")
    elif args.strategy == '5a':
        # Strategy 5a: Evaluate on multiple datasets (synthetic v0, synthetic v7, synthetic fairdiffusion, and DTest)
        # Prepare evaluation datasets list
        evaluation_datasets = []
        
        # 1. Synthetic v0 test set
        synthetic_v0_path = 'outputs_summarized/v0/0_train_baseline/test_images/step_10000'
        if args.synthetic_test_path:
            # Allow override for v0 path
            synthetic_v0_path = args.synthetic_test_path
        evaluation_datasets.append({
            'name': 'synthetic_v0',
            'path': synthetic_v0_path,
            'type': 'synthetic'
        })
        
        # 2. Synthetic v7 test set
        synthetic_v7_path = 'outputs_summarized/v7/6_train_hcn_age_from_promt/test_images/step_20000'
        evaluation_datasets.append({
            'name': 'synthetic_v7',
            'path': synthetic_v7_path,
            'type': 'synthetic'
        })
        
        # 3. Synthetic fairdiffusion test set
        synthetic_fairdiffusion_path = '/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/outputs_summarized/fairdiffusion/0_train_baseline_fairdiffusion/test_images/step_7500'
        evaluation_datasets.append({
            'name': 'synthetic_fairdiffusion',
            'path': synthetic_fairdiffusion_path,
            'type': 'synthetic'
        })
        
        # 4. DTest (real test set)
        if args.real_test_path is None:
            # Use validation set if test not provided
            test_path = args.real_val_path
            logger.warning("real_test_path not provided, using validation set for evaluation")
        else:
            test_path = args.real_test_path
        
        # Determine data type for DTest
        test_path_obj = Path(test_path)
        if test_path_obj.is_dir() and any(test_path_obj.glob("*.tar")):
            test_data_type = 'real_wds'
        elif test_path_obj.suffix == '.csv':
            test_data_type = 'real_csv'
        else:
            test_data_type = 'real_wds' if test_path_obj.is_dir() else 'real_csv'
        
        evaluation_datasets.append({
            'name': 'DTest',
            'path': test_path,
            'type': 'real',
            'data_type': test_data_type
        })
        
        # Set test_dataset and evaluation_dataset to None - will be handled in loop below
        test_dataset = None
        evaluation_dataset = None
        
    elif args.strategy == '5b':
        # Strategy 5b: Evaluate on synthetic test set (v7 only)
        if args.synthetic_test_path is None:
            # Auto-construct path for v7 test set
            args.synthetic_test_path = 'outputs_summarized/v7/6_train_hcn_age_from_promt/test_images/step_20000'
            logger.info(f"Auto-set synthetic_test_path to: {args.synthetic_test_path}")
        
        test_dataset = load_synthetic_test_set(
            args.synthetic_test_path,
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
        )
        evaluation_dataset = f'synthetic_{args.model_version}'
    elif args.strategy == '6a':
        # Strategy 6a: Evaluate on multiple datasets (test_data, GT_data, synthetic v0, synthetic v7)
        # Prepare evaluation datasets list
        evaluation_datasets = []
        
        # 1. test_data (real test set)
        test_data_path = 'real_data_summarized_four_splits/test_data'
        test_data_path_obj = Path(test_data_path)
        if test_data_path_obj.is_dir() and any(test_data_path_obj.glob("*.tar")):
            test_data_type = 'real_wds'
        elif test_data_path_obj.suffix == '.csv':
            test_data_type = 'real_csv'
        else:
            test_data_type = 'real_wds' if test_data_path_obj.is_dir() else 'real_csv'
        
        evaluation_datasets.append({
            'name': 'test_data',
            'path': test_data_path,
            'type': 'real',
            'data_type': test_data_type
        })
        
        # 2. GT_data (real ground truth set)
        gt_data_path = 'real_data_summarized_four_splits/GT_data'
        gt_data_path_obj = Path(gt_data_path)
        if gt_data_path_obj.is_dir() and any(gt_data_path_obj.glob("*.tar")):
            gt_data_type = 'real_wds'
        elif gt_data_path_obj.suffix == '.csv':
            gt_data_type = 'real_csv'
        else:
            gt_data_type = 'real_wds' if gt_data_path_obj.is_dir() else 'real_csv'
        
        evaluation_datasets.append({
            'name': 'GT_data',
            'path': gt_data_path,
            'type': 'real',
            'data_type': gt_data_type
        })
        
        # 3. Synthetic v0 test set
        synthetic_v0_path = 'synthetic_datasets/v0/0_train_baseline/GT-10000'
        evaluation_datasets.append({
            'name': 'synthetic_v0',
            'path': synthetic_v0_path,
            'type': 'synthetic'
        })
        
        # 4. Synthetic v7 test set
        synthetic_v7_path = 'synthetic_datasets/v7/6_train_hcn_age_from_promt/GT-20000'
        evaluation_datasets.append({
            'name': 'synthetic_v7',
            'path': synthetic_v7_path,
            'type': 'synthetic'
        })
        
        # Set test_dataset and evaluation_dataset to None - will be handled in loop below
        test_dataset = None
        evaluation_dataset = None
    else:
        # Evaluate on real test set (DTest)
        if args.real_test_path is None:
            # Use validation set if test not provided
            test_path = args.real_val_path
            logger.warning("real_test_path not provided, using validation set for evaluation")
        else:
            test_path = args.real_test_path
        
        # Determine data type: check if directory contains .tar files (WebDataset) or is a CSV file
        test_path_obj = Path(test_path)
        if test_path_obj.is_dir() and any(test_path_obj.glob("*.tar")):
            test_data_type = 'real_wds'
        elif test_path_obj.suffix == '.csv':
            test_data_type = 'real_csv'
        else:
            # Default to WebDataset if directory, CSV if file
            test_data_type = 'real_wds' if test_path_obj.is_dir() else 'real_csv'
        
        # Use streaming WebDataset for WebDataset data, regular Dataset for CSV
        if test_data_type == 'real_wds':
            import glob
            test_dataset = CheXpertClassifierWebDataset(
                url_list=sorted(glob.glob(str(test_path_obj / "*.tar"))),
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,  # IMPORTANT: Disable shuffle for evaluation to avoid duplicates/missing samples with multiple workers
            )
        else:
            test_dataset = CheXpertClassifierDataset(
                data_path=test_path_obj,
                data_type=test_data_type,
                split='test',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        evaluation_dataset = 'DTest'
    
    # Handle single dataset evaluation (non-5a and non-6a strategies)
    if args.strategy not in ['5a', '6a']:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    
    # Load validation loader for threshold optimization
    # Determine data type: check if directory contains .tar files (WebDataset) or is a CSV file
    val_path_obj = Path(args.real_val_path)
    if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
        val_data_type = 'real_wds'
    elif val_path_obj.suffix == '.csv':
        val_data_type = 'real_csv'
    else:
        # Default to WebDataset if directory, CSV if file
        val_data_type = 'real_wds' if val_path_obj.is_dir() else 'real_csv'
    
    # Use streaming WebDataset for WebDataset data, regular Dataset for CSV
    if val_data_type == 'real_wds':
        import glob
        val_dataset = CheXpertClassifierWebDataset(
            url_list=sorted(glob.glob(str(val_path_obj / "*.tar"))),
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            shuffle=False,  # IMPORTANT: Disable shuffle for validation to avoid duplicates/missing samples with multiple workers
        )
    else:
        val_dataset = CheXpertClassifierDataset(
            data_path=val_path_obj,
            data_type=val_data_type,
            split='val',
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Never shuffle for validation/IterableDataset
        num_workers=args.num_workers,
    )
    
    # Load model
    model = DenseNet121Classifier(num_classes=14)
    model = load_checkpoint(model, str(checkpoint_path), device)
    
    # Evaluate and save predictions
    # Include version in output directory and model name for all strategies that use synthetic data
    # Strategies that use synthetic data: 1b, 1c, 1d, 1e, 2b, 2c, 3a, 4a, 5a, 5b
    # Strategies 3a and 4a use two-phase training (pretrain + finetune), so output goes to finetune directory
    # Strategy 6a evaluates on synthetic but doesn't use it for training
    strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b']
    if args.strategy in ['3a', '3b']:
        # Strategy 3a/3b: finetune directory includes subset_fraction
        if args.subset_fraction is None:
            raise ValueError(f"subset_fraction is required for strategy {args.strategy}")
        # Format subset_fraction consistently (e.g., 1.0 instead of 1)
        subset_fraction_str = f"{args.subset_fraction:.1f}" if isinstance(args.subset_fraction, float) else str(args.subset_fraction)
        if args.strategy in strategies_with_synthetic and args.model_version:
            model_output_dir = output_dir / f'model_{args.strategy}_{args.model_version}_finetune_{subset_fraction_str}'
            model_name = f'model_{args.strategy}_{args.model_version}_finetune_{subset_fraction_str}'
        else:
            model_output_dir = output_dir / f'model_{args.strategy}_finetune_{subset_fraction_str}'
            model_name = f'model_{args.strategy}_finetune_{subset_fraction_str}'
    elif args.strategy in ['4a', '4b']:
        # Strategy 4a/4b: finetune directory (no subset_fraction)
        if args.strategy in strategies_with_synthetic and args.model_version:
            model_output_dir = output_dir / f'model_{args.strategy}_{args.model_version}_finetune'
            model_name = f'model_{args.strategy}_{args.model_version}_finetune'
        else:
            model_output_dir = output_dir / f'model_{args.strategy}_finetune'
            model_name = f'model_{args.strategy}_finetune'
    elif args.strategy in strategies_with_synthetic and args.model_version:
        # Other strategies with synthetic data: standard directory
        model_output_dir = output_dir / f'model_{args.strategy}_{args.model_version}'
        model_name = f'model_{args.strategy}_{args.model_version}'
    else:
        model_output_dir = output_dir / f'model_{args.strategy}'
        model_name = f'model_{args.strategy}'
    
    # Handle strategy 5a: Evaluate on multiple datasets
    if args.strategy == '5a':
        logger.info("Strategy 5a: Evaluating on multiple datasets (synthetic v0, synthetic v7, synthetic fairdiffusion, and DTest)")
        
        for eval_config in evaluation_datasets:
            eval_name = eval_config['name']
            eval_path = eval_config['path']
            eval_type = eval_config['type']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on: {eval_name}")
            logger.info(f"Path: {eval_path}")
            logger.info(f"{'='*60}")
            
            # Load test dataset based on type
            if eval_type == 'synthetic':
                test_dataset = load_synthetic_test_set(
                    eval_path,
                    mimic_mean=mimic_mean,
                    mimic_std=mimic_std,
                )
            else:  # real
                test_data_type = eval_config.get('data_type', 'real_csv')
                test_path_obj = Path(eval_path)
                
                if test_data_type == 'real_wds':
                    import glob
                    test_dataset = CheXpertClassifierWebDataset(
                        url_list=sorted(glob.glob(str(test_path_obj / "*.tar"))),
                        mimic_mean=mimic_mean,
                        mimic_std=mimic_std,
                        shuffle=False,
                    )
                else:
                    test_dataset = CheXpertClassifierDataset(
                        data_path=test_path_obj,
                        data_type=test_data_type,
                        split='test',
                        mimic_mean=mimic_mean,
                        mimic_std=mimic_std,
                    )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            
            # Evaluate on this dataset
            results = evaluate_and_save(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir=model_output_dir,
                model_name=model_name,
                checkpoint_path=str(checkpoint_path),
                evaluation_dataset=eval_name,
                val_loader=val_loader,
                use_old_order=args.use_old_order,
                n_bootstrap=args.n_bootstrap,
                compute_statistics=not args.skip_statistics,
                random_seed=args.random_seed,
            )
            
            # Compute fairness metrics for this dataset
            logger.info(f"Computing fairness metrics for {eval_name}...")
            from downstream_eval_chest.evaluate_classifier import sanitize_dataset_name_for_filename
            dataset_suffix = sanitize_dataset_name_for_filename(eval_name)
            if dataset_suffix != 'dtest':
                file_suffix = f'_{dataset_suffix}'
            else:
                file_suffix = ''
            
            predictions_file = model_output_dir / f'predictions{file_suffix}.npz'
            metadata_file = model_output_dir / f'metadata{file_suffix}.json'
            
            if predictions_file.exists() and metadata_file.exists():
                import numpy as np
                predictions_data = np.load(predictions_file)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                fairness_results = compute_fairness_metrics(
                    predictions_probs=predictions_data['predictions_probs'],
                    ground_truth=predictions_data['ground_truth'],
                    demographics=metadata['demographics'],
                    thresholds=metadata['thresholds'],
                    n_bootstrap=args.n_bootstrap,
                    compute_statistics=not args.skip_statistics,
                    random_seed=args.random_seed,
                )
                
                # Save fairness results
                results['fairness_metrics'] = fairness_results
                results_file = model_output_dir / f'evaluation_results{file_suffix}.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Fairness metrics computed and saved for {eval_name}")
        
        logger.info("\n" + "="*60)
        logger.info("Strategy 5a: All evaluations completed!")
        logger.info(f"Results saved to: {model_output_dir}")
        logger.info("="*60)
    
    elif args.strategy == '6a':
        logger.info("Strategy 6a: Evaluating on multiple datasets (test_data, GT_data, synthetic v0, synthetic v7)")
        
        for eval_config in evaluation_datasets:
            eval_name = eval_config['name']
            eval_path = eval_config['path']
            eval_type = eval_config['type']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on: {eval_name}")
            logger.info(f"Path: {eval_path}")
            logger.info(f"{'='*60}")
            
            # Load test dataset based on type
            if eval_type == 'synthetic':
                test_dataset = load_synthetic_test_set(
                    eval_path,
                    mimic_mean=mimic_mean,
                    mimic_std=mimic_std,
                )
            else:  # real
                test_data_type = eval_config.get('data_type', 'real_csv')
                test_path_obj = Path(eval_path)
                
                if test_data_type == 'real_wds':
                    import glob
                    test_dataset = CheXpertClassifierWebDataset(
                        url_list=sorted(glob.glob(str(test_path_obj / "*.tar"))),
                        mimic_mean=mimic_mean,
                        mimic_std=mimic_std,
                        shuffle=False,
                    )
                else:
                    test_dataset = CheXpertClassifierDataset(
                        data_path=test_path_obj,
                        data_type=test_data_type,
                        split='test',
                        mimic_mean=mimic_mean,
                        mimic_std=mimic_std,
                    )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            
            # Evaluate on this dataset
            results = evaluate_and_save(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dir=model_output_dir,
                model_name=model_name,
                checkpoint_path=str(checkpoint_path),
                evaluation_dataset=eval_name,
                val_loader=val_loader,
                use_old_order=args.use_old_order,
                n_bootstrap=args.n_bootstrap,
                compute_statistics=not args.skip_statistics,
                random_seed=args.random_seed,
            )
            
            # Compute fairness metrics for this dataset
            logger.info(f"Computing fairness metrics for {eval_name}...")
            from downstream_eval_chest.evaluate_classifier import sanitize_dataset_name_for_filename
            dataset_suffix = sanitize_dataset_name_for_filename(eval_name)
            if dataset_suffix != 'dtest':
                file_suffix = f'_{dataset_suffix}'
            else:
                file_suffix = ''
            
            predictions_file = model_output_dir / f'predictions{file_suffix}.npz'
            metadata_file = model_output_dir / f'metadata{file_suffix}.json'
            
            if predictions_file.exists() and metadata_file.exists():
                import numpy as np
                predictions_data = np.load(predictions_file)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                fairness_results = compute_fairness_metrics(
                    predictions_probs=predictions_data['predictions_probs'],
                    ground_truth=predictions_data['ground_truth'],
                    demographics=metadata['demographics'],
                    thresholds=metadata['thresholds'],
                    n_bootstrap=args.n_bootstrap,
                    compute_statistics=not args.skip_statistics,
                    random_seed=args.random_seed,
                )
                
                # Save fairness results
                results['fairness_metrics'] = fairness_results
                results_file = model_output_dir / f'evaluation_results{file_suffix}.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Fairness metrics computed and saved for {eval_name}")
        
        logger.info("\n" + "="*60)
        logger.info("Strategy 6a: All evaluations completed!")
        logger.info(f"Results saved to: {model_output_dir}")
        logger.info("="*60)
    
    else:
        # Single dataset evaluation (non-5a and non-6a strategies)
        results = evaluate_and_save(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=model_output_dir,
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            evaluation_dataset=evaluation_dataset,
            val_loader=val_loader,
            use_old_order=args.use_old_order,
            n_bootstrap=args.n_bootstrap,
            compute_statistics=not args.skip_statistics,
            random_seed=args.random_seed,
        )
        
        # Compute fairness metrics
        logger.info("Computing fairness metrics...")
        # Determine file suffix based on evaluation dataset
        from downstream_eval_chest.evaluate_classifier import sanitize_dataset_name_for_filename
        dataset_suffix = sanitize_dataset_name_for_filename(evaluation_dataset)
        if dataset_suffix != 'dtest':
            file_suffix = f'_{dataset_suffix}'
        else:
            file_suffix = ''
        
        predictions_file = model_output_dir / f'predictions{file_suffix}.npz'
        metadata_file = model_output_dir / f'metadata{file_suffix}.json'
        
        if predictions_file.exists() and metadata_file.exists():
            import numpy as np
            predictions_data = np.load(predictions_file)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            fairness_results = compute_fairness_metrics(
                predictions_probs=predictions_data['predictions_probs'],
                ground_truth=predictions_data['ground_truth'],
                demographics=metadata['demographics'],
                thresholds=metadata['thresholds'],
                n_bootstrap=args.n_bootstrap,
                compute_statistics=not args.skip_statistics,
                random_seed=args.random_seed,
            )
            
            # Save fairness results
            results['fairness_metrics'] = fairness_results
            results_file = model_output_dir / f'evaluation_results{file_suffix}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("Fairness metrics computed and saved")
        
        logger.info("Evaluation completed!")
        logger.info(f"Results saved to: {model_output_dir}")


if __name__ == '__main__':
    main()

