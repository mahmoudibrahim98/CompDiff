"""
Utility functions for downstream evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve

# Import CHEXPERT_CLASSES for remapping functions
# Using lazy import to avoid circular dependencies
def _get_chexpert_classes():
    """Lazy import of CHEXPERT_CLASSES to avoid circular dependencies."""
    from .dataset import CHEXPERT_CLASSES
    return CHEXPERT_CLASSES


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'f1',
    class_name: Optional[str] = None,
) -> float:
    """
    Compute optimal threshold for binary classification.
    
    Args:
        y_true: True binary labels [N]
        y_pred: Predicted probabilities [N]
        metric: Metric to optimize ('f1', 'youden')
        class_name: Optional class name for logging diagnostics
    
    Returns:
        Optimal threshold value
    """
    # Filter out uncertain labels (-1)
    valid_mask = y_true != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        if class_name:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"No valid labels for {class_name} (all uncertain)")
        return 0.5  # Default threshold
    
    # Check if we have both classes
    unique_labels = np.unique(y_true_valid)
    if len(unique_labels) < 2:
        if class_name:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Only one class present for {class_name} in validation set: {unique_labels}. "
                          f"Positive cases: {(y_true_valid == 1).sum()}, "
                          f"Negative cases: {(y_true_valid == 0).sum()}, "
                          f"Total valid: {len(y_true_valid)}")
        return 0.5  # Default threshold if only one class
    
    if metric == 'f1':
        # Find threshold that maximizes F1
        precision, recall, thresholds = precision_recall_curve(
            y_true_valid, y_pred_valid
        )
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        # Handle edge cases:
        # 1. precision_recall_curve returns len(thresholds) = len(precision) - 1
        # 2. If best_idx is out of bounds, use the last threshold
        # 3. If threshold is too extreme (>0.99 or <0.01), clamp it to reasonable range
        if best_idx >= len(thresholds):
            threshold = thresholds[-1] if len(thresholds) > 0 else 0.5
        else:
            threshold = thresholds[best_idx]
        
        # Clamp extreme thresholds to reasonable range
        # If threshold is too high, it means model predicts very low probabilities
        # If threshold is too low, it means model predicts very high probabilities
        # In both cases, we should use a more reasonable threshold
        if threshold > 0.99:
            # Model predicts very low probabilities - use a more conservative threshold
            # Check if we have positive cases
            if (y_true_valid == 1).sum() > 0:
                positive_preds = y_pred_valid[y_true_valid == 1]
                negative_preds = y_pred_valid[y_true_valid == 0]
                
                if len(positive_preds) > 0 and len(negative_preds) > 0:
                    # Use a threshold between median of positive and negative predictions
                    # This ensures we actually predict some positives
                    pos_median = np.median(positive_preds)
                    neg_median = np.median(negative_preds)
                    # Use a threshold that's closer to positive median but accounts for overlap
                    threshold = float((pos_median + neg_median) / 2)
                    # Ensure threshold is reasonable (not too high)
                    threshold = min(threshold, 0.9)
                    # Ensure threshold is not too low (should be at least median of positives)
                    threshold = max(threshold, pos_median * 0.8)
                    
                    if class_name:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Extreme threshold (>0.99) for {class_name} clamped. "
                            f"Original: {thresholds[best_idx] if best_idx < len(thresholds) else 'N/A'}, "
                            f"New: {threshold:.6f}. "
                            f"Positive pred median: {pos_median:.6f}, "
                            f"Negative pred median: {neg_median:.6f}"
                        )
                elif len(positive_preds) > 0:
                    # Only positive cases (shouldn't happen, but handle it)
                    threshold = float(np.percentile(positive_preds, 25))  # Use 25th percentile
                else:
                    threshold = 0.5
            else:
                # No positive cases in validation - use default
                threshold = 0.5
        elif threshold < 0.01:
            # Model predicts very high probabilities - use a more conservative threshold
            # Check if we have negative cases
            if (y_true_valid == 0).sum() > 0:
                positive_preds = y_pred_valid[y_true_valid == 1]
                negative_preds = y_pred_valid[y_true_valid == 0]
                
                if len(positive_preds) > 0 and len(negative_preds) > 0:
                    # Use a threshold between median of positive and negative predictions
                    pos_median = np.median(positive_preds)
                    neg_median = np.median(negative_preds)
                    threshold = float((pos_median + neg_median) / 2)
                    # Ensure threshold is reasonable (not too low)
                    threshold = max(threshold, 0.1)
                    # Ensure threshold is not too high
                    threshold = min(threshold, pos_median * 1.2)
                    
                    if class_name:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Extreme threshold (<0.01) for {class_name} clamped. "
                            f"Original: {thresholds[best_idx] if best_idx < len(thresholds) else 'N/A'}, "
                            f"New: {threshold:.6f}. "
                            f"Positive pred median: {pos_median:.6f}, "
                            f"Negative pred median: {neg_median:.6f}"
                        )
                elif len(negative_preds) > 0:
                    # Only negative cases (shouldn't happen, but handle it)
                    threshold = float(np.percentile(negative_preds, 75))  # Use 75th percentile
                else:
                    threshold = 0.5
            else:
                # No negative cases in validation - use default
                threshold = 0.5
        
        return threshold
    
    elif metric == 'youden':
        # Youden's J statistic: maximize sensitivity + specificity - 1
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true_valid, y_pred_valid)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def apply_thresholds(
    predictions: np.ndarray,
    thresholds: Dict[str, float],
    class_names: List[str],
) -> np.ndarray:
    """
    Apply per-label thresholds to predictions.
    
    Args:
        predictions: Predicted probabilities [N, num_classes]
        thresholds: Dictionary mapping class names to thresholds
        class_names: List of class names in order
    
    Returns:
        Binary predictions [N, num_classes]
    """
    binary_preds = np.zeros_like(predictions)
    
    for i, class_name in enumerate(class_names):
        if class_name in thresholds:
            threshold = thresholds[class_name]
            binary_preds[:, i] = (predictions[:, i] >= threshold).astype(float)
        else:
            # Default threshold
            binary_preds[:, i] = (predictions[:, i] >= 0.5).astype(float)
    
    return binary_preds


def compute_auroc_per_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    log_diagnostics: bool = False,
) -> Dict[str, float]:
    """
    Compute AUROC for each label.
    
    Args:
        y_true: True labels [N, num_classes]
        y_pred: Predicted probabilities [N, num_classes]
        class_names: List of class names
        log_diagnostics: If True, log diagnostic information for NaN results
    
    Returns:
        Dictionary mapping class names to AUROC scores
    """
    aurocs = {}
    
    for i, class_name in enumerate(class_names):
        # Filter out uncertain labels
        valid_mask = y_true[:, i] != -1
        if valid_mask.sum() < 2:
            if log_diagnostics:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"AUROC NaN for {class_name}: Only {valid_mask.sum()} valid labels (need ≥2)")
            aurocs[class_name] = np.nan
            continue
        
        y_true_i = y_true[valid_mask, i]
        y_pred_i = y_pred[valid_mask, i]
        
        # Check if we have both classes
        unique_labels = np.unique(y_true_i)
        if len(unique_labels) < 2:
            if log_diagnostics:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"AUROC NaN for {class_name}: Only one class present in test set. "
                    f"Unique labels: {unique_labels}, "
                    f"Positive cases: {(y_true_i == 1).sum()}, "
                    f"Negative cases: {(y_true_i == 0).sum()}, "
                    f"Total valid: {len(y_true_i)}"
                )
            aurocs[class_name] = np.nan
            continue
        
        try:
            auroc = roc_auc_score(y_true_i, y_pred_i)
            # Check for invalid AUROC values
            if np.isnan(auroc) or np.isinf(auroc):
                if log_diagnostics:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"AUROC NaN/Inf for {class_name}: "
                        f"auroc={auroc}, "
                        f"pred_range=[{y_pred_i.min():.6f}, {y_pred_i.max():.6f}], "
                        f"pred_mean={y_pred_i.mean():.6f}, "
                        f"positives={(y_true_i == 1).sum()}, negatives={(y_true_i == 0).sum()}"
                    )
                aurocs[class_name] = np.nan
            else:
                aurocs[class_name] = float(auroc)
        except ValueError as e:
            if log_diagnostics:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"AUROC ValueError for {class_name}: {e}. "
                    f"pred_range=[{y_pred_i.min():.6f}, {y_pred_i.max():.6f}], "
                    f"pred_mean={y_pred_i.mean():.6f}, "
                    f"positives={(y_true_i == 1).sum()}, negatives={(y_true_i == 0).sum()}, "
                    f"unique_preds={len(np.unique(y_pred_i))}"
                )
            aurocs[class_name] = np.nan
        except Exception as e:
            if log_diagnostics:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"AUROC Exception for {class_name}: {type(e).__name__} - {e}. "
                    f"pred_range=[{y_pred_i.min():.6f}, {y_pred_i.max():.6f}], "
                    f"positives={(y_true_i == 1).sum()}, negatives={(y_true_i == 0).sum()}"
                )
            aurocs[class_name] = np.nan
    
    return aurocs


def compute_auprc_per_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Compute AUPRC for each label.
    
    Args:
        y_true: True labels [N, num_classes]
        y_pred: Predicted probabilities [N, num_classes]
        class_names: List of class names
    
    Returns:
        Dictionary mapping class names to AUPRC scores
    """
    auprcs = {}
    
    for i, class_name in enumerate(class_names):
        # Filter out uncertain labels
        valid_mask = y_true[:, i] != -1
        if valid_mask.sum() < 2:
            auprcs[class_name] = np.nan
            continue
        
        y_true_i = y_true[valid_mask, i]
        y_pred_i = y_pred[valid_mask, i]
        
        # Check if we have both classes
        if len(np.unique(y_true_i)) < 2:
            auprcs[class_name] = np.nan
            continue
        
        try:
            auprc = average_precision_score(y_true_i, y_pred_i)
            auprcs[class_name] = float(auprc)
        except ValueError:
            auprcs[class_name] = np.nan
    
    return auprcs


def compute_f1_per_label(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Compute F1 score for each label.
    
    Args:
        y_true: True labels [N, num_classes] with values in {0, 1, -1}
        y_pred_binary: Binary predictions [N, num_classes] with values in {0, 1}
        class_names: List of class names
    
    Returns:
        Dictionary mapping class names to F1 scores
    """
    f1_scores = {}
    
    for i, class_name in enumerate(class_names):
        # Filter out uncertain labels
        valid_mask = y_true[:, i] != -1
        if valid_mask.sum() < 2:
            f1_scores[class_name] = np.nan
            continue
        
        y_true_i = y_true[valid_mask, i]
        y_pred_i = y_pred_binary[valid_mask, i]
        
        # Check if we have both classes
        if len(np.unique(y_true_i)) < 2:
            f1_scores[class_name] = np.nan
            continue
        
        try:
            f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
            f1_scores[class_name] = float(f1)
        except ValueError:
            f1_scores[class_name] = np.nan
    
    return f1_scores


def compute_weighted_f1(
    f1_per_label: Dict[str, float],
    y_true: np.ndarray,
    class_names: List[str],
    label_subset: Optional[List[str]] = None,
) -> float:
    """
    Compute weighted F1 score across labels.
    
    Args:
        f1_per_label: Dictionary mapping class names to F1 scores
        y_true: True labels [N, num_classes]
        class_names: List of class names
        label_subset: Optional subset of labels to use (e.g., COMMON_LABELS)
    
    Returns:
        Weighted F1 score
    """
    if label_subset is None:
        label_subset = class_names
    
    # Compute weights based on number of positive samples per label
    weights = []
    f1_values = []
    
    for class_name in label_subset:
        if class_name not in f1_per_label or np.isnan(f1_per_label[class_name]):
            continue
        
        class_idx = class_names.index(class_name)
        # Count positive samples (label == 1) for this class
        valid_mask = y_true[:, class_idx] != -1
        if valid_mask.sum() == 0:
            continue
        
        positive_count = (y_true[valid_mask, class_idx] == 1).sum()
        weights.append(positive_count)
        f1_values.append(f1_per_label[class_name])
    
    if len(weights) == 0 or sum(weights) == 0:
        return np.nan
    
    # Weighted average
    weights = np.array(weights)
    f1_values = np.array(f1_values)
    weighted_f1 = np.average(f1_values, weights=weights)
    
    return float(weighted_f1)


def mask_uncertain_labels(
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Create mask for uncertain labels (-1).
    
    Args:
        labels: Labels tensor [B, num_classes] with values in {0, 1, -1}
    
    Returns:
        Mask tensor [B, num_classes] where True indicates valid labels
    """
    return labels != -1


# Old CHEXPERT_CLASSES order (used in models trained before the order change)
CHEXPERT_CLASSES_OLD_ORDER = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]


def remap_predictions_old_to_new(
    predictions: np.ndarray,
    old_order: List[str],
    new_order: List[str],
) -> np.ndarray:
    """
    Remap predictions from old class order to new class order.
    
    This function is used for backward compatibility with models trained with
    the old CHEXPERT_CLASSES order. It reorders the prediction columns to match
    the new order.
    
    Args:
        predictions: Predictions array [N, num_classes] in old order
        old_order: List of class names in old order
        new_order: List of class names in new order
    
    Returns:
        Remapped predictions [N, num_classes] in new order
    
    Example:
        >>> old_preds = np.array([[0.1, 0.2, 0.3]])  # [No Finding, Enlarged Cardiomediastinum, ...]
        >>> new_preds = remap_predictions_old_to_new(old_preds, CHEXPERT_CLASSES_OLD_ORDER, CHEXPERT_CLASSES)
        >>> # new_preds[:, 0] now corresponds to 'Atelectasis' (first in new order)
    """
    if len(old_order) != len(new_order):
        raise ValueError(f"Old and new orders must have same length: {len(old_order)} != {len(new_order)}")
    
    if predictions.shape[1] != len(old_order):
        raise ValueError(
            f"Predictions must have {len(old_order)} columns (one per class), "
            f"got {predictions.shape[1]}"
        )
    
    # Verify both orders contain the same classes
    if set(old_order) != set(new_order):
        raise ValueError(
            f"Old and new orders must contain the same classes. "
            f"Missing in new: {set(old_order) - set(new_order)}, "
            f"Missing in old: {set(new_order) - set(old_order)}"
        )
    
    # Create remapping indices: for each class in new_order, find its index in old_order
    remap_indices = []
    for class_name in new_order:
        old_idx = old_order.index(class_name)
        remap_indices.append(old_idx)
    
    # Remap predictions
    remapped = predictions[:, remap_indices]
    
    return remapped


def detect_and_remap_old_predictions(
    predictions: np.ndarray,
    current_order: List[str],
    remap_if_old: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Detect if predictions are in old order and remap if needed.
    
    This function attempts to detect whether predictions are in the old order
    by checking if the model was likely trained with old order. Since we can't
    definitively detect this from predictions alone, this function should be called
    with remap_if_old=True when you know the checkpoint was trained with old order.
    
    Args:
        predictions: Predictions array [N, num_classes]
        current_order: Current class order (should match CHEXPERT_CLASSES)
        remap_if_old: If True, remap from old order to current order
    
    Returns:
        Tuple of (remapped_predictions, was_remapped)
        - remapped_predictions: Predictions in current order
        - was_remapped: True if remapping was performed, False otherwise
    """
    if not remap_if_old:
        return predictions, False
    
    # Check if current_order matches old order
    CHEXPERT_CLASSES = _get_chexpert_classes()
    
    if current_order == CHEXPERT_CLASSES_OLD_ORDER:
        # Already in old order, no remapping needed
        return predictions, False
    
    if current_order == CHEXPERT_CLASSES:
        # Current order is new order, remap from old to new
        remapped = remap_predictions_old_to_new(
            predictions,
            CHEXPERT_CLASSES_OLD_ORDER,
            CHEXPERT_CLASSES
        )
        return remapped, True
    
    # Unknown order, return as-is
    return predictions, False






