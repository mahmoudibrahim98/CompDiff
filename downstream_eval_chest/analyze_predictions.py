"""
Post-hoc analysis framework for computing new metrics from saved predictions.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import logging

from .dataset import CHEXPERT_CLASSES
from .utils import (
    compute_optimal_threshold,
    compute_auroc_per_label,
    compute_auprc_per_label,
    apply_thresholds,
)

logger = logging.getLogger(__name__)


def load_predictions(
    predictions_file: Union[str, Path],
    metadata_file: Union[str, Path],
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
    """
    Load saved predictions and metadata.
    
    Args:
        predictions_file: Path to .npz file with predictions
        metadata_file: Path to .json file with metadata
    
    Returns:
        Tuple of (predictions_dict, ground_truth, metadata_dict)
        predictions_dict contains 'predictions_logits' and 'predictions_probs'
    """
    predictions_file = Path(predictions_file)
    metadata_file = Path(metadata_file)
    
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # Load predictions
    predictions_data = np.load(predictions_file)
    predictions = {
        'predictions_logits': predictions_data['predictions_logits'],
        'predictions_probs': predictions_data['predictions_probs'],
        'sample_indices': predictions_data['sample_indices'],
    }
    ground_truth = predictions_data['ground_truth']
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded predictions: {len(ground_truth)} samples")
    logger.info(f"Model: {metadata.get('model_name', 'unknown')}")
    logger.info(f"Evaluation dataset: {metadata.get('evaluation_dataset', 'unknown')}")
    
    return predictions, ground_truth, metadata


def compute_metrics_from_predictions(
    predictions_probs: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute metrics from loaded predictions.
    
    Args:
        predictions_probs: Predicted probabilities [N, 14]
        ground_truth: Ground truth labels [N, 14]
        thresholds: Optional thresholds to apply (if None, uses 0.5)
        class_names: Optional list of class names (default: CHEXPERT_CLASSES)
    
    Returns:
        Dictionary with computed metrics
    """
    if class_names is None:
        class_names = CHEXPERT_CLASSES
    
    if thresholds is None:
        thresholds = {name: 0.5 for name in class_names}
    
    # Compute AUROC and AUPRC per label
    aurocs = compute_auroc_per_label(ground_truth, predictions_probs, class_names)
    auprcs = compute_auprc_per_label(ground_truth, predictions_probs, class_names)
    
    # Apply thresholds and compute binary metrics
    binary_preds = apply_thresholds(predictions_probs, thresholds, class_names)
    
    results = {
        'auroc_per_label': aurocs,
        'auprc_per_label': auprcs,
        'binary_predictions': binary_preds,
    }
    
    return results


def compute_calibration(
    predictions_probs: np.ndarray,
    ground_truth: np.ndarray,
    class_idx: int,
    bins: int = 10,
) -> Dict:
    """
    Compute calibration metrics for a specific class.
    
    Args:
        predictions_probs: Predicted probabilities [N, 14]
        ground_truth: Ground truth labels [N, 14]
        class_idx: Index of class to analyze
        bins: Number of bins for calibration curve
    
    Returns:
        Dictionary with calibration metrics
    """
    # Filter out uncertain labels
    valid_mask = ground_truth[:, class_idx] != -1
    y_true = ground_truth[valid_mask, class_idx]
    y_pred = predictions_probs[valid_mask, class_idx]
    
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {'error': 'Insufficient data for calibration'}
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=bins, strategy='uniform'
    )
    
    # Compute ECE (Expected Calibration Error)
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist(),
        'ece': float(ece),
    }


def compute_subgroup_metrics(
    predictions_probs: np.ndarray,
    ground_truth: np.ndarray,
    subgroups: Dict[str, np.ndarray],
    class_idx: int,
    class_name: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Compute metrics for custom subgroups.
    
    Args:
        predictions_probs: Predicted probabilities [N, 14]
        ground_truth: Ground truth labels [N, 14]
        subgroups: Dictionary mapping subgroup names to boolean masks [N]
        class_idx: Index of class to analyze
        class_name: Optional class name for logging
    
    Returns:
        Dictionary mapping subgroup names to metrics
    """
    if class_name is None:
        class_name = CHEXPERT_CLASSES[class_idx]
    
    results = {}
    
    for subgroup_name, mask in subgroups.items():
        if mask.sum() < 10:  # Skip if too few samples
            continue
        
        # Filter out uncertain labels
        valid_mask = (ground_truth[mask, class_idx] != -1)
        if valid_mask.sum() < 2:
            continue
        
        y_true_sub = ground_truth[mask, class_idx][valid_mask]
        y_pred_sub = predictions_probs[mask, class_idx][valid_mask]
        
        if len(np.unique(y_true_sub)) < 2:
            continue
        
        try:
            auroc = roc_auc_score(y_true_sub, y_pred_sub)
            auprc = average_precision_score(y_true_sub, y_pred_sub)
            
            results[subgroup_name] = {
                'auroc': float(auroc),
                'auprc': float(auprc),
                'num_samples': int(mask.sum()),
                'num_valid_samples': int(valid_mask.sum()),
            }
        except Exception as e:
            logger.warning(f"Failed to compute metrics for {subgroup_name}: {e}")
    
    return results


def compare_models(
    model_paths: List[Union[str, Path]],
    metric: str = 'auroc',
    class_name: Optional[str] = None,
) -> Dict:
    """
    Compare multiple models.
    
    Args:
        model_paths: List of paths to model output directories
        metric: Metric to compare ('auroc' or 'auprc')
        class_name: Optional class name to compare (if None, compares mean)
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for model_path in model_paths:
        model_path = Path(model_path)
        
        # Load predictions and metadata
        predictions_file = model_path / 'predictions.npz'
        metadata_file = model_path / 'metadata.json'
        results_file = model_path / 'evaluation_results.json'
        
        if not predictions_file.exists() or not metadata_file.exists():
            logger.warning(f"Skipping {model_path}: missing files")
            continue
        
        predictions, ground_truth, metadata = load_predictions(
            predictions_file, metadata_file
        )
        
        model_name = metadata.get('model_name', model_path.name)
        
        # Load saved results if available
        if results_file.exists():
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            
            if class_name:
                if metric == 'auroc':
                    value = saved_results['test_metrics']['auroc_per_label'].get(class_name)
                else:
                    value = saved_results['test_metrics']['auprc_per_label'].get(class_name)
            else:
                if metric == 'auroc':
                    value = saved_results['test_metrics'].get('mean_auroc_8_common')
                else:
                    value = saved_results['test_metrics'].get('mean_auprc_8_common')
            
            results[model_name] = {
                'value': value,
                'evaluation_dataset': metadata.get('evaluation_dataset'),
            }
        else:
            # Compute on the fly
            metrics = compute_metrics_from_predictions(
                predictions['predictions_probs'],
                ground_truth,
            )
            
            if class_name:
                if metric == 'auroc':
                    value = metrics['auroc_per_label'].get(class_name)
                else:
                    value = metrics['auprc_per_label'].get(class_name)
            else:
                # Compute mean
                if metric == 'auroc':
                    values = [v for v in metrics['auroc_per_label'].values() if not np.isnan(v)]
                else:
                    values = [v for v in metrics['auprc_per_label'].values() if not np.isnan(v)]
                value = np.mean(values) if values else None
            
            results[model_name] = {
                'value': float(value) if value is not None and not np.isnan(value) else None,
                'evaluation_dataset': metadata.get('evaluation_dataset'),
            }
    
    return results


def apply_custom_thresholds(
    predictions_probs: np.ndarray,
    threshold_strategy: str = 'fixed',
    threshold_value: float = 0.5,
    thresholds_dict: Optional[Dict[str, float]] = None,
    ground_truth: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Apply different threshold strategies to predictions.
    
    Args:
        predictions_probs: Predicted probabilities [N, 14]
        threshold_strategy: 'fixed', 'youden', 'f1', or 'custom'
        threshold_value: Fixed threshold value (for 'fixed' strategy)
        thresholds_dict: Custom thresholds dictionary (for 'custom' strategy)
        ground_truth: Ground truth labels (for 'youden' or 'f1' strategies)
        class_names: List of class names
    
    Returns:
        Binary predictions [N, 14]
    """
    if class_names is None:
        class_names = CHEXPERT_CLASSES
    
    binary_preds = np.zeros_like(predictions_probs)
    
    for i, class_name in enumerate(class_names):
        if threshold_strategy == 'fixed':
            threshold = threshold_value
        elif threshold_strategy == 'custom' and thresholds_dict:
            threshold = thresholds_dict.get(class_name, 0.5)
        elif threshold_strategy in ['youden', 'f1'] and ground_truth is not None:
            threshold = compute_optimal_threshold(
                ground_truth[:, i],
                predictions_probs[:, i],
                metric=threshold_strategy,
            )
        else:
            threshold = 0.5
        
        binary_preds[:, i] = (predictions_probs[:, i] >= threshold).astype(float)
    
    return binary_preds

