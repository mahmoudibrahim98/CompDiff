"""
Evaluation framework with prediction storage for post-hoc analysis.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import json
from datetime import datetime
from tqdm import tqdm
import logging

from .models import DenseNet121Classifier, load_checkpoint
from .dataset import CheXpertClassifierDataset, CHEXPERT_CLASSES
from .utils import (
    compute_optimal_threshold,
    compute_auroc_per_label,
    compute_auprc_per_label,
    compute_f1_per_label,
    compute_weighted_f1,
    apply_thresholds,
    mask_uncertain_labels,
    remap_predictions_old_to_new,
    CHEXPERT_CLASSES_OLD_ORDER,
)
from .statistics import (
    compute_bootstrap_cis_for_metrics,
    bootstrap_ci_for_metric,
    bootstrap_ci,
    check_auroc_eligibility,
    check_auprc_eligibility,
    check_f1_eligibility,
    check_fpr_eligibility,
    compute_fpr_jeffreys,
)
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

# 8 common labels for macro-averaging
COMMON_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion',  # Effusion
    'Pneumonia',
    'Pneumothorax',
    'No Finding',
]


def sanitize_dataset_name_for_filename(dataset_name: str) -> str:
    """
    Sanitize dataset name for use in filenames.
    
    Args:
        dataset_name: Dataset name (e.g., 'CheXpert (chexpert_filtered.csv)', 'DTest', 'synthetic_v0')
    
    Returns:
        Sanitized name safe for filenames (e.g., 'chexpert', 'dtest', 'synthetic_v0')
    """
    import re
    # Handle CheXpert specifically - extract just "chexpert"
    if 'chexpert' in dataset_name.lower():
        # Extract CSV filename if present in parentheses
        csv_match = re.search(r'\(([^)]+\.csv)\)', dataset_name)
        if csv_match:
            csv_name = csv_match.group(1).replace('.csv', '').replace('_', '-')
            return f'chexpert_{csv_name}'
        return 'chexpert'
    
    # Convert to lowercase
    sanitized = dataset_name.lower()
    # Replace spaces and parentheses with underscores
    sanitized = sanitized.replace(' ', '_').replace('(', '').replace(')', '')
    # Replace dots and other special characters with underscores
    sanitized = re.sub(r'[^\w-]', '_', sanitized)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # If empty or just underscores, use a default
    if not sanitized or sanitized == '_':
        sanitized = 'dataset'
    return sanitized


def evaluate_and_save(
    model: DenseNet121Classifier,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    model_name: str,
    checkpoint_path: str,
    evaluation_dataset: str,
    val_loader: Optional[DataLoader] = None,
    use_old_order: bool = False,
    n_bootstrap: int = 1000,
    compute_statistics: bool = True,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Evaluate model and save all predictions, labels, and metadata.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        output_dir: Output directory for saving predictions
        model_name: Name of the model (e.g., 'model_1a')
        checkpoint_path: Path to model checkpoint
        evaluation_dataset: Name of evaluation dataset (e.g., 'DTest', 'synthetic_v0')
        val_loader: Optional validation loader for threshold optimization
        use_old_order: If True, model was trained with old CHEXPERT_CLASSES order.
                       Predictions will be remapped to new order automatically.
        n_bootstrap: Number of bootstrap resamples for confidence intervals (default: 1000)
        compute_statistics: If True, compute bootstrap CIs and statistical tests (default: True)
        random_seed: Random seed for reproducibility (default: None)
    
    Returns:
        Dictionary with evaluation results including statistical analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Collect all predictions, labels, and metadata
    all_logits = []
    all_probs = []
    all_labels = []
    all_indices = []
    all_ages = []
    all_sexes = []
    all_race_ethnicities = []
    all_age_groups = []
    
    logger.info("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            logits = logits.cpu().numpy()
            
            all_logits.append(logits)
            all_probs.append(probs)
            all_labels.append(labels)
            
            # Metadata
            all_indices.append(batch['index'].cpu().numpy())
            all_ages.append(batch['age'].cpu().numpy())
            all_sexes.append(batch['sex'].cpu().numpy())
            all_race_ethnicities.append(batch['race_ethnicity'].cpu().numpy())
            all_age_groups.extend(batch['age_group'])
    
    # Concatenate all batches
    predictions_logits = np.concatenate(all_logits, axis=0)  # [N, 14]
    predictions_probs = np.concatenate(all_probs, axis=0)  # [N, 14]
    ground_truth = np.concatenate(all_labels, axis=0)  # [N, 14]
    sample_indices = np.concatenate(all_indices, axis=0)  # [N]
    
    # Check if all labels are uncertain (-1) - this means the dataset has no ground truth labels
    if (ground_truth == -1).all():
        logger.error(
            f"WARNING: All ground truth labels are uncertain (-1) for dataset '{evaluation_dataset}'. "
            f"This means the dataset has no ground truth labels. "
            f"Metrics (AUROC, AUPRC, F1) cannot be computed and will be null. "
            f"Please ensure the dataset has labels in metadata files (disease_labels or disease field)."
        )
    elif (ground_truth == -1).any():
        uncertain_count = (ground_truth == -1).sum()
        total_count = ground_truth.size
        logger.warning(
            f"Found {uncertain_count}/{total_count} uncertain labels (-1) in dataset '{evaluation_dataset}'. "
            f"These will be excluded from metric computation."
        )
    
    # Remap predictions if model was trained with old order
    if use_old_order:
        logger.info("Remapping predictions from old class order to new class order...")
        predictions_logits = remap_predictions_old_to_new(
            predictions_logits,
            CHEXPERT_CLASSES_OLD_ORDER,
            CHEXPERT_CLASSES
        )
        predictions_probs = remap_predictions_old_to_new(
            predictions_probs,
            CHEXPERT_CLASSES_OLD_ORDER,
            CHEXPERT_CLASSES
        )
        logger.info("Predictions remapped successfully.")
    
    demographics = {
        'age': np.concatenate(all_ages, axis=0).tolist(),
        'sex': np.concatenate(all_sexes, axis=0).tolist(),
        'race_ethnicity': np.concatenate(all_race_ethnicities, axis=0).tolist(),
        'age_group': all_age_groups,
    }
    
    # Optimize thresholds on validation set if provided
    thresholds = {}
    if val_loader is not None:
        logger.info("Optimizing thresholds on validation set...")
        val_logits = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating for thresholds"):
                images = batch['image'].to(device)
                labels = batch['labels'].cpu().numpy()
                logits = model(images).cpu().numpy()
                val_logits.append(logits)
                val_labels.append(labels)
        
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_probs = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
        
        # Remap validation predictions if model was trained with old order
        if use_old_order:
            logger.info("Remapping validation predictions from old class order to new class order...")
            val_logits = remap_predictions_old_to_new(
                val_logits,
                CHEXPERT_CLASSES_OLD_ORDER,
                CHEXPERT_CLASSES
            )
            val_probs = remap_predictions_old_to_new(
                val_probs,
                CHEXPERT_CLASSES_OLD_ORDER,
                CHEXPERT_CLASSES
            )
        
        # Compute optimal threshold for each label
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            threshold = compute_optimal_threshold(
                val_labels[:, i],
                val_probs[:, i],
                metric='f1',
                class_name=class_name,
            )
            thresholds[class_name] = float(threshold)
            
            # Log warning if threshold is extreme
            if threshold > 0.95 or threshold < 0.05:
                logger.warning(
                    f"Extreme threshold for {class_name}: {threshold:.6f}. "
                    f"This may indicate issues with validation set distribution or model predictions."
                )
    else:
        # Use default threshold of 0.5
        thresholds = {class_name: 0.5 for class_name in CHEXPERT_CLASSES}
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    # Diagnostic: Check prediction ranges for each label
    # Note: After label aggregation, all labels should be 0 or 1 (no more -1 uncertain labels)
    no_finding_idx = CHEXPERT_CLASSES.index('No Finding')
    for i, class_name in enumerate(CHEXPERT_CLASSES):
        valid_mask = ground_truth[:, i] != -1
        if valid_mask.sum() > 0:
            preds_valid = predictions_probs[valid_mask, i]
            labels_valid = ground_truth[valid_mask, i]
            
            # Check for any remaining -1 values (shouldn't happen after aggregation)
            uncertain_count = (labels_valid == -1).sum()
            if uncertain_count > 0:
                logger.warning(
                    f"{class_name}: Found {uncertain_count} uncertain labels (-1) in test set. "
                    f"This should not happen after label aggregation. Check dataset loading."
                )
            
            # Log statistics
            log_level = logger.info if class_name == 'No Finding' else logger.debug
            log_level(
                f"{class_name}: pred_range=[{preds_valid.min():.4f}, {preds_valid.max():.4f}], "
                f"pred_mean={preds_valid.mean():.4f}, "
                f"positives={(labels_valid == 1).sum()}/{len(labels_valid)} "
                f"({100*(labels_valid == 1).sum()/len(labels_valid):.1f}%), "
                f"negatives={(labels_valid == 0).sum()}/{len(labels_valid)} "
                f"({100*(labels_valid == 0).sum()/len(labels_valid):.1f}%), "
                f"threshold={thresholds.get(class_name, 0.5):.6f}"
            )
    
    aurocs = compute_auroc_per_label(ground_truth, predictions_probs, CHEXPERT_CLASSES, log_diagnostics=True)
    auprcs = compute_auprc_per_label(ground_truth, predictions_probs, CHEXPERT_CLASSES)
    
    # Compute binary predictions using optimized thresholds
    binary_preds = apply_thresholds(predictions_probs, thresholds, CHEXPERT_CLASSES)
    
    # Diagnostic: Check binary prediction distribution
    for i, class_name in enumerate(CHEXPERT_CLASSES):
        valid_mask = ground_truth[:, i] != -1
        if valid_mask.sum() > 0:
            binary_valid = binary_preds[valid_mask, i]
            labels_valid = ground_truth[valid_mask, i]
            positive_preds = (binary_valid == 1).sum()
            positive_labels = (labels_valid == 1).sum()
            negative_preds = (binary_valid == 0).sum()
            negative_labels = (labels_valid == 0).sum()
            
            # Log with INFO level for "No Finding" to make it visible
            log_level = logger.info if class_name == 'No Finding' else logger.debug
            log_level(
                f"{class_name} binary predictions: "
                f"{positive_preds}/{len(binary_valid)} predicted positive "
                f"({100*positive_preds/len(binary_valid):.1f}%), "
                f"{positive_labels}/{len(labels_valid)} actual positive "
                f"({100*positive_labels/len(labels_valid):.1f}%), "
                f"threshold={thresholds.get(class_name, 0.5):.6f}"
            )
    
    # Compute F1 scores per label
    f1_scores = compute_f1_per_label(ground_truth, binary_preds, CHEXPERT_CLASSES)
    
    # Log "No Finding" metrics explicitly (since it's now always evaluated after aggregation)
    no_finding_idx = CHEXPERT_CLASSES.index('No Finding')
    no_finding_auroc = aurocs.get('No Finding', np.nan)
    no_finding_auprc = auprcs.get('No Finding', np.nan)
    no_finding_f1 = f1_scores.get('No Finding', np.nan)
    no_finding_valid = (ground_truth[:, no_finding_idx] != -1).sum()
    no_finding_positives = (ground_truth[:, no_finding_idx] == 1).sum()
    no_finding_negatives = (ground_truth[:, no_finding_idx] == 0).sum()
    
    logger.info(
        f"\n{'='*60}\n"
        f"'No Finding' Evaluation Results:\n"
        f"  Total samples: {no_finding_valid}\n"
        f"  Positive (no diseases): {no_finding_positives} ({100*no_finding_positives/no_finding_valid:.1f}%)\n"
        f"  Negative (has diseases): {no_finding_negatives} ({100*no_finding_negatives/no_finding_valid:.1f}%)\n"
        f"  AUROC: {no_finding_auroc:.4f}\n"
        f"  AUPRC: {no_finding_auprc:.4f}\n"
        f"  F1 Score: {no_finding_f1:.4f}\n"
        f"{'='*60}\n"
    )
    
    # Compute macro-averaged metrics (mean of per-label metrics)
    valid_aurocs = [v for v in aurocs.values() if not np.isnan(v)]
    mean_auroc_14 = np.mean(valid_aurocs) if valid_aurocs else np.nan
    
    # Mean AUROC for 8 common labels (macro-averaged)
    common_aurocs = [aurocs[label] for label in COMMON_LABELS if not np.isnan(aurocs.get(label, np.nan))]
    mean_auroc_8 = np.mean(common_aurocs) if common_aurocs else np.nan
    
    valid_auprcs = [v for v in auprcs.values() if not np.isnan(v)]
    mean_auprc_14 = np.mean(valid_auprcs) if valid_auprcs else np.nan
    
    common_auprcs = [auprcs[label] for label in COMMON_LABELS if not np.isnan(auprcs.get(label, np.nan))]
    mean_auprc_8 = np.mean(common_auprcs) if common_auprcs else np.nan
    
    # Compute weighted F1 scores
    weighted_f1_14 = compute_weighted_f1(f1_scores, ground_truth, CHEXPERT_CLASSES, label_subset=None)
    weighted_f1_8 = compute_weighted_f1(f1_scores, ground_truth, CHEXPERT_CLASSES, label_subset=COMMON_LABELS)
    
    # Statistical analysis: Bootstrap confidence intervals
    statistical_results = {}
    if compute_statistics:
        logger.info("Computing bootstrap confidence intervals (this may take a while)...")
        bootstrap_results = compute_bootstrap_cis_for_metrics(
            ground_truth,
            predictions_probs,
            CHEXPERT_CLASSES,
            y_pred_binary=binary_preds,  # Pass binary predictions for FPR computation
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            random_seed=random_seed,
        )
        
        # Add bootstrap CIs to statistical results
        statistical_results['bootstrap_ci'] = {
            'n_bootstrap': n_bootstrap,
            'confidence_level': 0.95,
            'auroc_per_label': {},
            'auprc_per_label': {},
            'no_finding_fpr': {},  # Add FPR for "No Finding"
        }
        
        for class_name in CHEXPERT_CLASSES:
            if class_name in bootstrap_results['auroc']:
                auroc_ci = bootstrap_results['auroc'][class_name]
                statistical_results['bootstrap_ci']['auroc_per_label'][class_name] = {
                    'mean': auroc_ci['mean'] if not np.isnan(auroc_ci['mean']) else None,
                    'ci_lower': auroc_ci['ci_lower'] if not np.isnan(auroc_ci['ci_lower']) else None,
                    'ci_upper': auroc_ci['ci_upper'] if not np.isnan(auroc_ci['ci_upper']) else None,
                }
            
            if class_name in bootstrap_results['auprc']:
                auprc_ci = bootstrap_results['auprc'][class_name]
                statistical_results['bootstrap_ci']['auprc_per_label'][class_name] = {
                    'mean': auprc_ci['mean'] if not np.isnan(auprc_ci['mean']) else None,
                    'ci_lower': auprc_ci['ci_lower'] if not np.isnan(auprc_ci['ci_lower']) else None,
                    'ci_upper': auprc_ci['ci_upper'] if not np.isnan(auprc_ci['ci_upper']) else None,
                }
        
        # Extract FPR CI for "No Finding" if available
        if 'fpr' in bootstrap_results and 'No Finding' in bootstrap_results['fpr']:
            fpr_ci = bootstrap_results['fpr']['No Finding']
            statistical_results['bootstrap_ci']['no_finding_fpr'] = {
                'mean': fpr_ci['mean'] if not np.isnan(fpr_ci['mean']) else None,
                'ci_lower': fpr_ci['ci_lower'] if not np.isnan(fpr_ci['ci_lower']) else None,
                'ci_upper': fpr_ci['ci_upper'] if not np.isnan(fpr_ci['ci_upper']) else None,
                }
        
        # Compute bootstrap CIs for mean metrics
        # These should bootstrap from the actual data, not from the array of per-label metrics
        # For mean AUROC 14 labels: bootstrap samples, compute AUROC per label, then mean
        logger.info("Computing bootstrap CIs for mean AUROC/AUPRC (14 and 8 labels)...")
        
        def compute_mean_auroc_14(y_true_boot, y_pred_boot):
            """Compute mean AUROC across 14 labels from bootstrap sample."""
            try:
                aurocs_boot = []
                for i, class_name in enumerate(CHEXPERT_CLASSES):
                    valid_mask = y_true_boot[:, i] != -1
                    if valid_mask.sum() < 2:
                        continue
                    y_true_sub = y_true_boot[valid_mask, i]
                    y_pred_sub = y_pred_boot[valid_mask, i]
                    if len(np.unique(y_true_sub)) < 2:
                        continue
                    try:
                        auroc = roc_auc_score(y_true_sub, y_pred_sub)
                        if not np.isnan(auroc):
                            aurocs_boot.append(auroc)
                    except:
                        continue
                return np.mean(aurocs_boot) if aurocs_boot else np.nan
            except:
                return np.nan
        
        def compute_mean_auroc_8(y_true_boot, y_pred_boot):
            """Compute mean AUROC across 8 common labels from bootstrap sample."""
            try:
                aurocs_boot = []
                for label in COMMON_LABELS:
                    i = CHEXPERT_CLASSES.index(label)
                    valid_mask = y_true_boot[:, i] != -1
                    if valid_mask.sum() < 2:
                        continue
                    y_true_sub = y_true_boot[valid_mask, i]
                    y_pred_sub = y_pred_boot[valid_mask, i]
                    if len(np.unique(y_true_sub)) < 2:
                        continue
                    try:
                        auroc = roc_auc_score(y_true_sub, y_pred_sub)
                        if not np.isnan(auroc):
                            aurocs_boot.append(auroc)
                    except:
                        continue
                return np.mean(aurocs_boot) if aurocs_boot else np.nan
            except:
                return np.nan
        
        def compute_mean_auprc_14(y_true_boot, y_pred_boot):
            """Compute mean AUPRC across 14 labels from bootstrap sample."""
            try:
                auprcs_boot = []
                for i, class_name in enumerate(CHEXPERT_CLASSES):
                    valid_mask = y_true_boot[:, i] != -1
                    if valid_mask.sum() < 2:
                        continue
                    y_true_sub = y_true_boot[valid_mask, i]
                    y_pred_sub = y_pred_boot[valid_mask, i]
                    if len(np.unique(y_true_sub)) < 2:
                        continue
                    try:
                        auprc = average_precision_score(y_true_sub, y_pred_sub)
                        if not np.isnan(auprc):
                            auprcs_boot.append(auprc)
                    except:
                        continue
                return np.mean(auprcs_boot) if auprcs_boot else np.nan
            except:
                return np.nan
        
        def compute_mean_auprc_8(y_true_boot, y_pred_boot):
            """Compute mean AUPRC across 8 common labels from bootstrap sample."""
            try:
                auprcs_boot = []
                for label in COMMON_LABELS:
                    i = CHEXPERT_CLASSES.index(label)
                    valid_mask = y_true_boot[:, i] != -1
                    if valid_mask.sum() < 2:
                        continue
                    y_true_sub = y_true_boot[valid_mask, i]
                    y_pred_sub = y_pred_boot[valid_mask, i]
                    if len(np.unique(y_true_sub)) < 2:
                        continue
                    try:
                        auprc = average_precision_score(y_true_sub, y_pred_sub)
                        if not np.isnan(auprc):
                            auprcs_boot.append(auprc)
                    except:
                        continue
                return np.mean(auprcs_boot) if auprcs_boot else np.nan
            except:
                return np.nan
        
        # Bootstrap mean AUROC (14 labels) from actual data
        try:
            mean_auroc_14_ci = bootstrap_ci_for_metric(
                ground_truth,
                predictions_probs,
                compute_mean_auroc_14,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            statistical_results['bootstrap_ci']['mean_auroc_14_labels'] = {
                'mean': mean_auroc_14_ci[0] if not np.isnan(mean_auroc_14_ci[0]) else mean_auroc_14,
                'ci_lower': mean_auroc_14_ci[1] if not np.isnan(mean_auroc_14_ci[1]) else None,
                'ci_upper': mean_auroc_14_ci[2] if not np.isnan(mean_auroc_14_ci[2]) else None,
            }
        except Exception as e:
            logger.warning(f"Failed to compute bootstrap CI for mean AUROC (14 labels): {e}")
            statistical_results['bootstrap_ci']['mean_auroc_14_labels'] = {
                'mean': mean_auroc_14,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        # Bootstrap mean AUROC (8 common labels) from actual data
        try:
            mean_auroc_8_ci = bootstrap_ci_for_metric(
                ground_truth,
                predictions_probs,
                compute_mean_auroc_8,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            statistical_results['bootstrap_ci']['mean_auroc_8_common'] = {
                'mean': mean_auroc_8_ci[0] if not np.isnan(mean_auroc_8_ci[0]) else mean_auroc_8,
                'ci_lower': mean_auroc_8_ci[1] if not np.isnan(mean_auroc_8_ci[1]) else None,
                'ci_upper': mean_auroc_8_ci[2] if not np.isnan(mean_auroc_8_ci[2]) else None,
            }
        except Exception as e:
            logger.warning(f"Failed to compute bootstrap CI for mean AUROC (8 common): {e}")
            statistical_results['bootstrap_ci']['mean_auroc_8_common'] = {
                'mean': mean_auroc_8,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        # Bootstrap mean AUPRC (14 labels) from actual data
        try:
            mean_auprc_14_ci = bootstrap_ci_for_metric(
                ground_truth,
                predictions_probs,
                compute_mean_auprc_14,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            statistical_results['bootstrap_ci']['mean_auprc_14_labels'] = {
                'mean': mean_auprc_14_ci[0] if not np.isnan(mean_auprc_14_ci[0]) else mean_auprc_14,
                'ci_lower': mean_auprc_14_ci[1] if not np.isnan(mean_auprc_14_ci[1]) else None,
                'ci_upper': mean_auprc_14_ci[2] if not np.isnan(mean_auprc_14_ci[2]) else None,
            }
        except Exception as e:
            logger.warning(f"Failed to compute bootstrap CI for mean AUPRC (14 labels): {e}")
            statistical_results['bootstrap_ci']['mean_auprc_14_labels'] = {
                'mean': mean_auprc_14,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        # Bootstrap mean AUPRC (8 common labels) from actual data
        try:
            mean_auprc_8_ci = bootstrap_ci_for_metric(
                ground_truth,
                predictions_probs,
                compute_mean_auprc_8,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            statistical_results['bootstrap_ci']['mean_auprc_8_common'] = {
                'mean': mean_auprc_8_ci[0] if not np.isnan(mean_auprc_8_ci[0]) else mean_auprc_8,
                'ci_lower': mean_auprc_8_ci[1] if not np.isnan(mean_auprc_8_ci[1]) else None,
                'ci_upper': mean_auprc_8_ci[2] if not np.isnan(mean_auprc_8_ci[2]) else None,
            }
        except Exception as e:
            logger.warning(f"Failed to compute bootstrap CI for mean AUPRC (8 common): {e}")
            statistical_results['bootstrap_ci']['mean_auprc_8_common'] = {
                'mean': mean_auprc_8,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        # Compute bootstrap CIs for F1 scores per label
        logger.info("Computing bootstrap CIs for F1 scores per label...")
        from sklearn.metrics import f1_score
        statistical_results['bootstrap_ci']['f1_per_label'] = {}
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            valid_mask = ground_truth[:, i] != -1
            if valid_mask.sum() < 2:
                statistical_results['bootstrap_ci']['f1_per_label'][class_name] = {
                    'mean': None,
                    'ci_lower': None,
                    'ci_upper': None,
                }
                continue
            
            y_true_sub = ground_truth[valid_mask, i]
            y_pred_sub = binary_preds[valid_mask, i]
            
            if len(np.unique(y_true_sub)) < 2:
                statistical_results['bootstrap_ci']['f1_per_label'][class_name] = {
                    'mean': None,
                    'ci_lower': None,
                    'ci_upper': None,
                }
                continue
            
            def f1_fn(y_t, y_p):
                try:
                    return f1_score(y_t, y_p, zero_division=0)
                except:
                    return np.nan
            
            try:
                mean_f1, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                    y_true_sub,
                    y_pred_sub,
                    f1_fn,
                    n_bootstrap=n_bootstrap,
                    confidence_level=0.95,
                    random_seed=random_seed,
                )
                statistical_results['bootstrap_ci']['f1_per_label'][class_name] = {
                    'mean': mean_f1 if not np.isnan(mean_f1) else f1_scores.get(class_name),
                    'ci_lower': ci_lower if not np.isnan(ci_lower) else None,
                    'ci_upper': ci_upper if not np.isnan(ci_upper) else None,
                }
            except Exception as e:
                logger.warning(f"Failed to compute F1 CI for {class_name}: {e}")
                statistical_results['bootstrap_ci']['f1_per_label'][class_name] = {
                    'mean': f1_scores.get(class_name),
                    'ci_lower': None,
                    'ci_upper': None,
                }
        
        # Compute bootstrap CIs for weighted F1 scores
        logger.info("Computing bootstrap CIs for weighted F1 scores...")
        
        # Weighted F1 for 14 labels
        if not np.isnan(weighted_f1_14):
            def weighted_f1_14_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=None)
                except:
                    return np.nan
            
            try:
                mean_wf1_14, ci_lower_14, ci_upper_14, _ = bootstrap_ci_for_metric(
                    ground_truth,
                    binary_preds,
                    weighted_f1_14_fn,
                    n_bootstrap=n_bootstrap,
                    confidence_level=0.95,
                    random_seed=random_seed,
                )
                statistical_results['bootstrap_ci']['weighted_f1_14_labels'] = {
                    'mean': mean_wf1_14 if not np.isnan(mean_wf1_14) else weighted_f1_14,
                    'ci_lower': ci_lower_14 if not np.isnan(ci_lower_14) else None,
                    'ci_upper': ci_upper_14 if not np.isnan(ci_upper_14) else None,
                }
            except Exception as e:
                logger.warning(f"Failed to compute weighted F1 14 CI: {e}")
                statistical_results['bootstrap_ci']['weighted_f1_14_labels'] = {
                    'mean': weighted_f1_14,
                    'ci_lower': None,
                    'ci_upper': None,
                }
        else:
            statistical_results['bootstrap_ci']['weighted_f1_14_labels'] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        # Weighted F1 for 8 common labels
        if not np.isnan(weighted_f1_8):
            def weighted_f1_8_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=COMMON_LABELS)
                except:
                    return np.nan
            
            try:
                mean_wf1_8, ci_lower_8, ci_upper_8, _ = bootstrap_ci_for_metric(
                    ground_truth,
                    binary_preds,
                    weighted_f1_8_fn,
                    n_bootstrap=n_bootstrap,
                    confidence_level=0.95,
                    random_seed=random_seed,
                )
                statistical_results['bootstrap_ci']['weighted_f1_8_common'] = {
                    'mean': mean_wf1_8 if not np.isnan(mean_wf1_8) else weighted_f1_8,
                    'ci_lower': ci_lower_8 if not np.isnan(ci_lower_8) else None,
                    'ci_upper': ci_upper_8 if not np.isnan(ci_upper_8) else None,
                }
            except Exception as e:
                logger.warning(f"Failed to compute weighted F1 8 CI: {e}")
                statistical_results['bootstrap_ci']['weighted_f1_8_common'] = {
                    'mean': weighted_f1_8,
                    'ci_lower': None,
                    'ci_upper': None,
                }
        else:
            statistical_results['bootstrap_ci']['weighted_f1_8_common'] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        logger.info("Bootstrap confidence intervals computed")
    
    # Sanitize dataset name for use in filenames
    dataset_suffix = sanitize_dataset_name_for_filename(evaluation_dataset)
    # Only add suffix if it's not the default (DTest) to maintain backward compatibility
    if dataset_suffix != 'dtest':
        file_suffix = f'_{dataset_suffix}'
    else:
        file_suffix = ''
    
    # Save predictions
    logger.info("Saving predictions...")
    predictions_file = output_dir / f'predictions{file_suffix}.npz'
    np.savez(
        predictions_file,
        predictions_logits=predictions_logits.astype(np.float32),
        predictions_probs=predictions_probs.astype(np.float32),
        ground_truth=ground_truth.astype(np.int8),
        sample_indices=sample_indices.astype(np.int64),
    )
    logger.info(f"Saved predictions to {predictions_file}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'checkpoint_path': str(checkpoint_path),
        'evaluation_dataset': evaluation_dataset,
        'num_samples': int(len(predictions_probs)),
        'thresholds': thresholds,
        'demographics': demographics,
        'evaluation_date': datetime.now().isoformat(),
    }
    
    metadata_file = output_dir / f'metadata{file_suffix}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")
    
    # Save evaluation results
    results = {
        'model_name': model_name,
        'evaluation_dataset': evaluation_dataset,
        'test_metrics': {
            'auroc_per_label': {k: float(v) if not np.isnan(v) else None for k, v in aurocs.items()},
            'mean_auroc_14_labels': float(mean_auroc_14) if not np.isnan(mean_auroc_14) else None,
            'mean_auroc_8_common': float(mean_auroc_8) if not np.isnan(mean_auroc_8) else None,
            'auprc_per_label': {k: float(v) if not np.isnan(v) else None for k, v in auprcs.items()},
            'mean_auprc_14_labels': float(mean_auprc_14) if not np.isnan(mean_auprc_14) else None,
            'mean_auprc_8_common': float(mean_auprc_8) if not np.isnan(mean_auprc_8) else None,
            'f1_per_label': {k: float(v) if not np.isnan(v) else None for k, v in f1_scores.items()},
            'weighted_f1_14_labels': float(weighted_f1_14) if not np.isnan(weighted_f1_14) else None,
            'weighted_f1_8_common': float(weighted_f1_8) if not np.isnan(weighted_f1_8) else None,
        },
        'thresholds': thresholds,
    }
    
    # Add statistical analysis results
    if compute_statistics and statistical_results:
        results['statistical_analysis'] = statistical_results
    
    results_file = output_dir / f'evaluation_results{file_suffix}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {results_file}")
    
    return results


def compute_fairness_metrics(
    predictions_probs: np.ndarray,
    ground_truth: np.ndarray,
    demographics: Dict[str, List],
    thresholds: Dict[str, float],
    n_bootstrap: int = 1000,
    compute_statistics: bool = True,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Compute fairness metrics including 2-way and 3-way intersectional subgroups.
    
    Args:
        predictions_probs: Predicted probabilities [N, 14]
        ground_truth: Ground truth labels [N, 14]
        demographics: Dictionary with 'age', 'sex', 'race_ethnicity', 'age_group'
        thresholds: Per-label thresholds
        n_bootstrap: Number of bootstrap resamples for confidence intervals
        compute_statistics: If True, compute bootstrap CIs
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with fairness metrics including bootstrap CIs
    """
    # Convert demographics to arrays (ensure they're 1D arrays)
    # Handle different input formats - could be list, numpy array, or nested structure
    age_raw = demographics.get('age', [])
    sex_raw = demographics.get('sex', [])
    race_ethnicity_raw = demographics.get('race_ethnicity', [])
    age_group_raw = demographics.get('age_group', [])
    
    # Flatten if nested (e.g., list of lists)
    def flatten_array(arr):
        arr = np.array(arr)
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr
    
    age = flatten_array(age_raw)
    sex = flatten_array(sex_raw)
    race_ethnicity = flatten_array(race_ethnicity_raw)
    age_group_raw = flatten_array(age_group_raw)
    
    # Get the expected length from ground_truth
    n_samples_expected = len(ground_truth)
    
    # Check if arrays have correct length - if not, it might be a data issue
    if len(sex) != n_samples_expected:
        logger.error(
            f"Demographic array length mismatch: sex has {len(sex)} elements but ground_truth has {n_samples_expected} elements. "
            f"sex type: {type(sex_raw)}, sex shape: {np.array(sex_raw).shape if hasattr(sex_raw, '__len__') else 'N/A'}. "
            f"This suggests demographics were not saved correctly. Cannot compute fairness metrics."
        )
        raise ValueError(
            f"Demographic array length mismatch: sex={len(sex)}, ground_truth={n_samples_expected}. "
            f"All demographic arrays must have the same length as ground_truth. "
            f"Check that demographics were saved correctly during evaluation."
        )
    
    if len(age) != n_samples_expected or len(race_ethnicity) != n_samples_expected or len(age_group_raw) != n_samples_expected:
        logger.error(
            f"Demographic array length mismatch: age={len(age)}, race_ethnicity={len(race_ethnicity)}, "
            f"age_group={len(age_group_raw)}, ground_truth={n_samples_expected}. "
            f"Cannot compute fairness metrics."
        )
        raise ValueError(
            f"Demographic array length mismatch. All arrays must have length {n_samples_expected}. "
            f"Check that demographics were saved correctly during evaluation."
        )
    
    # Convert age_group from numeric strings ('1', '2', '3', '4') to expected format ('18-40', '40-60', '60-80', '80+')
    # Mapping: '1' -> '18-40', '2' -> '40-60', '3' -> '60-80', '4' -> '80+'
    # If already in expected format, keep as is
    age_group_map = {'1': '18-40', '2': '40-60', '3': '60-80', '4': '80+'}
    age_group = np.array([
        age_group_map.get(str(ag), ag) if str(ag) in age_group_map else ag 
        for ag in age_group_raw
    ])
    age_group = np.atleast_1d(age_group)
    
    # Apply thresholds to get binary predictions
    binary_preds = apply_thresholds(predictions_probs, thresholds, CHEXPERT_CLASSES)
    
    # 2-way intersectional: Sex × Race/Ethnicity
    sex_race_subgroups = {}
    for s in [0, 1]:  # Male, Female
        for r in [0, 1, 2, 3]:  # White, Black, Asian, Hispanic
            mask = (sex == s) & (race_ethnicity == r)
            if mask.sum() > 0:
                sex_race_subgroups[f"{'Male' if s == 0 else 'Female'}_{['White', 'Black', 'Asian', 'Hispanic'][r]}"] = mask
    
    # 2-way intersectional: Sex × Age
    sex_age_subgroups = {}
    age_groups_unique = ['18-40', '40-60', '60-80', '80+']
    for s in [0, 1]:
        for ag in age_groups_unique:
            mask = (sex == s) & (age_group == ag)
            if mask.sum() > 0:
                sex_age_subgroups[f"{ag}_{'Male' if s == 0 else 'Female'}"] = mask
    
    # 3-way intersectional: Age × Sex × Ethnicity
    intersectional_subgroups = {}
    for ag in age_groups_unique:
        for s in [0, 1]:
            for r in [0, 1, 2, 3]:
                mask = (age_group == ag) & (sex == s) & (race_ethnicity == r)
                if mask.sum() > 0:
                    subgroup_name = f"{ag}_{'Male' if s == 0 else 'Female'}_{['White', 'Black', 'Asian', 'Hispanic'][r]}"
                    intersectional_subgroups[subgroup_name] = mask
    
    # Compute AUROC per subgroup for each label
    fairness_results = {
        'sex_race_subgroups': {},
        'sex_age_subgroups': {},
        'intersectional_subgroups': {},
    }
    
    # Focus on "No Finding" label for underdiagnosis analysis
    no_finding_idx = CHEXPERT_CLASSES.index('No Finding')
    
    # Compute metrics for each subgroup
    for subgroup_name, mask in sex_race_subgroups.items():
        if mask.sum() < 10:  # Skip if too few samples
            continue
        
        subgroup_aurocs = {}
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            valid_mask = (ground_truth[mask, i] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, i][valid_mask]
            y_pred_sub = predictions_probs[mask, i][valid_mask]
            
            # Check eligibility: N_pos >= 10 AND N_neg >= 10 for AUROC
            is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
            if not is_eligible:
                continue  # Skip this label for this subgroup
            
            try:
                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                subgroup_aurocs[class_name] = float(auroc)
                
                # Compute bootstrap CI for AUROC if requested
                if compute_statistics:
                    def auroc_fn(y_t, y_p):
                        try:
                            return roc_auc_score(y_t, y_p)
                        except:
                            return np.nan
                    
                    def validate_auroc_bootstrap(y_t_boot):
                        """Validate bootstrap sample has both classes for AUROC."""
                        unique_classes = np.unique(y_t_boot)
                        return len(unique_classes) >= 2 and (1 in unique_classes) and (0 in unique_classes)
                    
                    mean_auroc, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                        y_true_sub,
                        y_pred_sub,
                        auroc_fn,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                        validate_bootstrap=validate_auroc_bootstrap,
                    )
                    
                    if class_name not in subgroup_aurocs:
                        subgroup_aurocs[class_name] = {}
                    if not isinstance(subgroup_aurocs[class_name], dict):
                        # Convert to dict format
                        subgroup_aurocs[class_name] = {'mean': subgroup_aurocs[class_name]}
                    subgroup_aurocs[class_name]['mean'] = mean_auroc if not np.isnan(mean_auroc) else None
                    subgroup_aurocs[class_name]['ci_lower'] = ci_lower if not np.isnan(ci_lower) else None
                    subgroup_aurocs[class_name]['ci_upper'] = ci_upper if not np.isnan(ci_upper) else None
            except:
                pass
        
        # Compute AUPRC per label for this subgroup
        subgroup_auprcs = {}
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            valid_mask = (ground_truth[mask, i] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, i][valid_mask]
            y_pred_sub = predictions_probs[mask, i][valid_mask]
            
            # Check eligibility: N_pos >= 10 for AUPRC
            is_eligible, n_pos, n_neg = check_auprc_eligibility(y_true_sub, min_pos=10)
            if not is_eligible:
                continue  # Skip this label for this subgroup
            
            try:
                auprc = average_precision_score(y_true_sub, y_pred_sub)
                subgroup_auprcs[class_name] = float(auprc)
                
                # Compute bootstrap CI for AUPRC if requested
                if compute_statistics:
                    def auprc_fn(y_t, y_p):
                        try:
                            return average_precision_score(y_t, y_p)
                        except:
                            return np.nan
                    
                    def validate_auprc_bootstrap(y_t_boot):
                        """Validate bootstrap sample has at least some positives for AUPRC."""
                        unique_classes = np.unique(y_t_boot)
                        return len(unique_classes) >= 2 and (1 in unique_classes)
                    
                    mean_auprc, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                        y_true_sub,
                        y_pred_sub,
                        auprc_fn,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                        validate_bootstrap=validate_auprc_bootstrap,
                    )
                    
                    if class_name not in subgroup_auprcs:
                        subgroup_auprcs[class_name] = {}
                    if not isinstance(subgroup_auprcs[class_name], dict):
                        # Convert to dict format
                        subgroup_auprcs[class_name] = {'mean': subgroup_auprcs[class_name]}
                    subgroup_auprcs[class_name]['mean'] = mean_auprc if not np.isnan(mean_auprc) else None
                    subgroup_auprcs[class_name]['ci_lower'] = ci_lower if not np.isnan(ci_lower) else None
                    subgroup_auprcs[class_name]['ci_upper'] = ci_upper if not np.isnan(ci_upper) else None
            except:
                pass
        
        # Compute aggregate metrics (mean AUROC/AUPRC for 14 and 8 labels) for this subgroup
        if subgroup_aurocs:
            # Mean AUROC (14 labels) - macro-average of per-label AUROCs
            valid_aurocs_14 = []
            for class_name, auroc_val in subgroup_aurocs.items():
                if isinstance(auroc_val, dict):
                    mean_val = auroc_val.get('mean')
                else:
                    mean_val = auroc_val
                if mean_val is not None and not np.isnan(mean_val):
                    valid_aurocs_14.append(mean_val)
            
            mean_auroc_14_subgroup = np.mean(valid_aurocs_14) if valid_aurocs_14 else np.nan
            
            # Mean AUROC (8 common labels)
            common_aurocs_8 = []
            for label in COMMON_LABELS:
                if label in subgroup_aurocs:
                    auroc_val = subgroup_aurocs[label]
                    if isinstance(auroc_val, dict):
                        mean_val = auroc_val.get('mean')
                    else:
                        mean_val = auroc_val
                    if mean_val is not None and not np.isnan(mean_val):
                        common_aurocs_8.append(mean_val)
            mean_auroc_8_subgroup = np.mean(common_aurocs_8) if common_aurocs_8 else np.nan
            
            # Compute bootstrap CI for mean AUROC if requested
            # Bootstrap from actual data, not from array of per-label AUROCs
            mean_auroc_14_ci = {'mean': mean_auroc_14_subgroup, 'ci_lower': None, 'ci_upper': None}
            mean_auroc_8_ci = {'mean': mean_auroc_8_subgroup, 'ci_lower': None, 'ci_upper': None}
            
            if compute_statistics:
                def compute_mean_auroc_14_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUROC across 14 labels for subgroup."""
                    try:
                        aurocs_boot = []
                        for i, class_name in enumerate(CHEXPERT_CLASSES):
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auroc):
                                    aurocs_boot.append(auroc)
                            except:
                                continue
                        return np.mean(aurocs_boot) if aurocs_boot else np.nan
                    except:
                        return np.nan
                
                def compute_mean_auroc_8_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUROC across 8 common labels for subgroup."""
                    try:
                        aurocs_boot = []
                        for label in COMMON_LABELS:
                            i = CHEXPERT_CLASSES.index(label)
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auroc):
                                    aurocs_boot.append(auroc)
                            except:
                                continue
                        return np.mean(aurocs_boot) if aurocs_boot else np.nan
                    except:
                        return np.nan
                
                try:
                    mean_auroc_14_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auroc_14_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auroc_14_ci = {
                        'mean': mean_auroc_14_ci_result[0] if not np.isnan(mean_auroc_14_ci_result[0]) else mean_auroc_14_subgroup,
                        'ci_lower': mean_auroc_14_ci_result[1] if not np.isnan(mean_auroc_14_ci_result[1]) else None,
                        'ci_upper': mean_auroc_14_ci_result[2] if not np.isnan(mean_auroc_14_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUROC (14 labels) for subgroup {subgroup_name}: {e}")
                
                try:
                    mean_auroc_8_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auroc_8_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auroc_8_ci = {
                        'mean': mean_auroc_8_ci_result[0] if not np.isnan(mean_auroc_8_ci_result[0]) else mean_auroc_8_subgroup,
                        'ci_lower': mean_auroc_8_ci_result[1] if not np.isnan(mean_auroc_8_ci_result[1]) else None,
                        'ci_upper': mean_auroc_8_ci_result[2] if not np.isnan(mean_auroc_8_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUROC (8 common) for subgroup {subgroup_name}: {e}")
            
            # Mean AUPRC (14 labels) - macro-average of per-label AUPRCs
            valid_auprcs_14 = []
            for class_name, auprc_val in subgroup_auprcs.items():
                if isinstance(auprc_val, dict):
                    mean_val = auprc_val.get('mean')
                else:
                    mean_val = auprc_val
                if mean_val is not None and not np.isnan(mean_val):
                    valid_auprcs_14.append(mean_val)
            
            mean_auprc_14_subgroup = np.mean(valid_auprcs_14) if valid_auprcs_14 else np.nan
            
            # Mean AUPRC (8 common labels)
            common_auprcs_8 = []
            for label in COMMON_LABELS:
                if label in subgroup_auprcs:
                    auprc_val = subgroup_auprcs[label]
                    if isinstance(auprc_val, dict):
                        mean_val = auprc_val.get('mean')
                    else:
                        mean_val = auprc_val
                    if mean_val is not None and not np.isnan(mean_val):
                        common_auprcs_8.append(mean_val)
            mean_auprc_8_subgroup = np.mean(common_auprcs_8) if common_auprcs_8 else np.nan
            
            # Compute bootstrap CI for mean AUPRC if requested
            # Bootstrap from actual data, not from array of per-label AUPRCs
            mean_auprc_14_ci = {'mean': mean_auprc_14_subgroup, 'ci_lower': None, 'ci_upper': None}
            mean_auprc_8_ci = {'mean': mean_auprc_8_subgroup, 'ci_lower': None, 'ci_upper': None}
            
            if compute_statistics:
                def compute_mean_auprc_14_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUPRC across 14 labels for subgroup."""
                    try:
                        auprcs_boot = []
                        for i, class_name in enumerate(CHEXPERT_CLASSES):
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auprc = average_precision_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auprc):
                                    auprcs_boot.append(auprc)
                            except:
                                continue
                        return np.mean(auprcs_boot) if auprcs_boot else np.nan
                    except:
                        return np.nan
                
                def compute_mean_auprc_8_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUPRC across 8 common labels for subgroup."""
                    try:
                        auprcs_boot = []
                        for label in COMMON_LABELS:
                            i = CHEXPERT_CLASSES.index(label)
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auprc = average_precision_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auprc):
                                    auprcs_boot.append(auprc)
                            except:
                                continue
                        return np.mean(auprcs_boot) if auprcs_boot else np.nan
                    except:
                        return np.nan
                
                try:
                    mean_auprc_14_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auprc_14_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auprc_14_ci = {
                        'mean': mean_auprc_14_ci_result[0] if not np.isnan(mean_auprc_14_ci_result[0]) else mean_auprc_14_subgroup,
                        'ci_lower': mean_auprc_14_ci_result[1] if not np.isnan(mean_auprc_14_ci_result[1]) else None,
                        'ci_upper': mean_auprc_14_ci_result[2] if not np.isnan(mean_auprc_14_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUPRC (14 labels) for subgroup {subgroup_name}: {e}")
                
                try:
                    mean_auprc_8_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auprc_8_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auprc_8_ci = {
                        'mean': mean_auprc_8_ci_result[0] if not np.isnan(mean_auprc_8_ci_result[0]) else mean_auprc_8_subgroup,
                        'ci_lower': mean_auprc_8_ci_result[1] if not np.isnan(mean_auprc_8_ci_result[1]) else None,
                        'ci_upper': mean_auprc_8_ci_result[2] if not np.isnan(mean_auprc_8_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUPRC (8 common) for subgroup {subgroup_name}: {e}")
            
            # Store aggregate metrics
            fairness_results['sex_race_subgroups'][subgroup_name] = subgroup_aurocs
            fairness_results['sex_race_subgroups'][subgroup_name]['mean_auroc_14_labels'] = mean_auroc_14_ci
            fairness_results['sex_race_subgroups'][subgroup_name]['mean_auroc_8_common'] = mean_auroc_8_ci
            if subgroup_auprcs:
                fairness_results['sex_race_subgroups'][subgroup_name]['auprc_per_label'] = subgroup_auprcs
                fairness_results['sex_race_subgroups'][subgroup_name]['mean_auprc_14_labels'] = mean_auprc_14_ci
                fairness_results['sex_race_subgroups'][subgroup_name]['mean_auprc_8_common'] = mean_auprc_8_ci
    
    # Compute metrics for sex_age_subgroups
    for subgroup_name, mask in sex_age_subgroups.items():
        if mask.sum() < 10:  # Skip if too few samples
            continue
        
        subgroup_aurocs = {}
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            valid_mask = (ground_truth[mask, i] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, i][valid_mask]
            y_pred_sub = predictions_probs[mask, i][valid_mask]
            
            # Check eligibility: N_pos >= 10 AND N_neg >= 10 for AUROC
            is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
            if not is_eligible:
                continue  # Skip this label for this subgroup
            
            try:
                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                subgroup_aurocs[class_name] = float(auroc)
                
                # Compute bootstrap CI for AUROC if requested
                if compute_statistics:
                    def auroc_fn(y_t, y_p):
                        try:
                            return roc_auc_score(y_t, y_p)
                        except:
                            return np.nan
                    
                    def validate_auroc_bootstrap(y_t_boot):
                        """Validate bootstrap sample has both classes for AUROC."""
                        unique_classes = np.unique(y_t_boot)
                        return len(unique_classes) >= 2 and (1 in unique_classes) and (0 in unique_classes)
                    
                    mean_auroc, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                        y_true_sub,
                        y_pred_sub,
                        auroc_fn,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                        validate_bootstrap=validate_auroc_bootstrap,
                    )
                    
                    if class_name not in subgroup_aurocs:
                        subgroup_aurocs[class_name] = {}
                    if not isinstance(subgroup_aurocs[class_name], dict):
                        # Convert to dict format
                        subgroup_aurocs[class_name] = {'mean': subgroup_aurocs[class_name]}
                    subgroup_aurocs[class_name]['mean'] = mean_auroc if not np.isnan(mean_auroc) else None
                    subgroup_aurocs[class_name]['ci_lower'] = ci_lower if not np.isnan(ci_lower) else None
                    subgroup_aurocs[class_name]['ci_upper'] = ci_upper if not np.isnan(ci_upper) else None
            except:
                pass
        
        # Compute AUPRC per label for this subgroup
        subgroup_auprcs = {}
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            valid_mask = (ground_truth[mask, i] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, i][valid_mask]
            y_pred_sub = predictions_probs[mask, i][valid_mask]
            
            # Check eligibility: N_pos >= 10 for AUPRC
            is_eligible, n_pos, n_neg = check_auprc_eligibility(y_true_sub, min_pos=10)
            if not is_eligible:
                continue  # Skip this label for this subgroup
            
            try:
                auprc = average_precision_score(y_true_sub, y_pred_sub)
                subgroup_auprcs[class_name] = float(auprc)
                
                # Compute bootstrap CI for AUPRC if requested
                if compute_statistics:
                    def auprc_fn(y_t, y_p):
                        try:
                            return average_precision_score(y_t, y_p)
                        except:
                            return np.nan
                    
                    def validate_auprc_bootstrap(y_t_boot):
                        """Validate bootstrap sample has at least some positives for AUPRC."""
                        unique_classes = np.unique(y_t_boot)
                        return len(unique_classes) >= 2 and (1 in unique_classes)
                    
                    mean_auprc, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                        y_true_sub,
                        y_pred_sub,
                        auprc_fn,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                        validate_bootstrap=validate_auprc_bootstrap,
                    )
                    
                    if class_name not in subgroup_auprcs:
                        subgroup_auprcs[class_name] = {}
                    if not isinstance(subgroup_auprcs[class_name], dict):
                        # Convert to dict format
                        subgroup_auprcs[class_name] = {'mean': subgroup_auprcs[class_name]}
                    subgroup_auprcs[class_name]['mean'] = mean_auprc if not np.isnan(mean_auprc) else None
                    subgroup_auprcs[class_name]['ci_lower'] = ci_lower if not np.isnan(ci_lower) else None
                    subgroup_auprcs[class_name]['ci_upper'] = ci_upper if not np.isnan(ci_upper) else None
            except:
                pass
        
        # Compute aggregate metrics (mean AUROC/AUPRC for 14 and 8 labels) for this subgroup
        if subgroup_aurocs:
            # Mean AUROC (14 labels)
            valid_aurocs_14 = []
            for class_name, auroc_val in subgroup_aurocs.items():
                if isinstance(auroc_val, dict):
                    mean_val = auroc_val.get('mean')
                else:
                    mean_val = auroc_val
                if mean_val is not None and not np.isnan(mean_val):
                    valid_aurocs_14.append(mean_val)
            mean_auroc_14_subgroup = np.mean(valid_aurocs_14) if valid_aurocs_14 else np.nan
            
            # Mean AUROC (8 common labels)
            common_aurocs_8 = []
            for label in COMMON_LABELS:
                if label in subgroup_aurocs:
                    auroc_val = subgroup_aurocs[label]
                    if isinstance(auroc_val, dict):
                        mean_val = auroc_val.get('mean')
                    else:
                        mean_val = auroc_val
                    if mean_val is not None and not np.isnan(mean_val):
                        common_aurocs_8.append(mean_val)
            mean_auroc_8_subgroup = np.mean(common_aurocs_8) if common_aurocs_8 else np.nan
            
            # Compute bootstrap CI for mean AUROC - bootstrap from actual data
            mean_auroc_14_ci = {'mean': mean_auroc_14_subgroup, 'ci_lower': None, 'ci_upper': None}
            mean_auroc_8_ci = {'mean': mean_auroc_8_subgroup, 'ci_lower': None, 'ci_upper': None}
            
            if compute_statistics:
                def compute_mean_auroc_14_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUROC across 14 labels for subgroup."""
                    try:
                        aurocs_boot = []
                        for i, class_name in enumerate(CHEXPERT_CLASSES):
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auroc):
                                    aurocs_boot.append(auroc)
                            except:
                                continue
                        return np.mean(aurocs_boot) if aurocs_boot else np.nan
                    except:
                        return np.nan
                
                def compute_mean_auroc_8_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUROC across 8 common labels for subgroup."""
                    try:
                        aurocs_boot = []
                        for label in COMMON_LABELS:
                            i = CHEXPERT_CLASSES.index(label)
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auroc):
                                    aurocs_boot.append(auroc)
                            except:
                                continue
                        return np.mean(aurocs_boot) if aurocs_boot else np.nan
                    except:
                        return np.nan
                
                try:
                    mean_auroc_14_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auroc_14_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auroc_14_ci = {
                        'mean': mean_auroc_14_ci_result[0] if not np.isnan(mean_auroc_14_ci_result[0]) else mean_auroc_14_subgroup,
                        'ci_lower': mean_auroc_14_ci_result[1] if not np.isnan(mean_auroc_14_ci_result[1]) else None,
                        'ci_upper': mean_auroc_14_ci_result[2] if not np.isnan(mean_auroc_14_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUROC (14 labels) for subgroup {subgroup_name}: {e}")
                
                try:
                    mean_auroc_8_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auroc_8_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auroc_8_ci = {
                        'mean': mean_auroc_8_ci_result[0] if not np.isnan(mean_auroc_8_ci_result[0]) else mean_auroc_8_subgroup,
                        'ci_lower': mean_auroc_8_ci_result[1] if not np.isnan(mean_auroc_8_ci_result[1]) else None,
                        'ci_upper': mean_auroc_8_ci_result[2] if not np.isnan(mean_auroc_8_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUROC (8 common) for subgroup {subgroup_name}: {e}")
            
            # Mean AUPRC (14 labels)
            valid_auprcs_14 = []
            for class_name, auprc_val in subgroup_auprcs.items():
                if isinstance(auprc_val, dict):
                    mean_val = auprc_val.get('mean')
                else:
                    mean_val = auprc_val
                if mean_val is not None and not np.isnan(mean_val):
                    valid_auprcs_14.append(mean_val)
            mean_auprc_14_subgroup = np.mean(valid_auprcs_14) if valid_auprcs_14 else np.nan
            
            # Mean AUPRC (8 common labels)
            common_auprcs_8 = []
            for label in COMMON_LABELS:
                if label in subgroup_auprcs:
                    auprc_val = subgroup_auprcs[label]
                    if isinstance(auprc_val, dict):
                        mean_val = auprc_val.get('mean')
                    else:
                        mean_val = auprc_val
                    if mean_val is not None and not np.isnan(mean_val):
                        common_auprcs_8.append(mean_val)
            mean_auprc_8_subgroup = np.mean(common_auprcs_8) if common_auprcs_8 else np.nan
            
            # Compute bootstrap CI for mean AUPRC - bootstrap from actual data
            mean_auprc_14_ci = {'mean': mean_auprc_14_subgroup, 'ci_lower': None, 'ci_upper': None}
            mean_auprc_8_ci = {'mean': mean_auprc_8_subgroup, 'ci_lower': None, 'ci_upper': None}
            
            if compute_statistics:
                def compute_mean_auprc_14_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUPRC across 14 labels for subgroup."""
                    try:
                        auprcs_boot = []
                        for i, class_name in enumerate(CHEXPERT_CLASSES):
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auprc = average_precision_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auprc):
                                    auprcs_boot.append(auprc)
                            except:
                                continue
                        return np.mean(auprcs_boot) if auprcs_boot else np.nan
                    except:
                        return np.nan
                
                def compute_mean_auprc_8_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUPRC across 8 common labels for subgroup."""
                    try:
                        auprcs_boot = []
                        for label in COMMON_LABELS:
                            i = CHEXPERT_CLASSES.index(label)
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auprc = average_precision_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auprc):
                                    auprcs_boot.append(auprc)
                            except:
                                continue
                        return np.mean(auprcs_boot) if auprcs_boot else np.nan
                    except:
                        return np.nan
                
                try:
                    mean_auprc_14_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auprc_14_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auprc_14_ci = {
                        'mean': mean_auprc_14_ci_result[0] if not np.isnan(mean_auprc_14_ci_result[0]) else mean_auprc_14_subgroup,
                        'ci_lower': mean_auprc_14_ci_result[1] if not np.isnan(mean_auprc_14_ci_result[1]) else None,
                        'ci_upper': mean_auprc_14_ci_result[2] if not np.isnan(mean_auprc_14_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUPRC (14 labels) for subgroup {subgroup_name}: {e}")
                
                try:
                    mean_auprc_8_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auprc_8_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auprc_8_ci = {
                        'mean': mean_auprc_8_ci_result[0] if not np.isnan(mean_auprc_8_ci_result[0]) else mean_auprc_8_subgroup,
                        'ci_lower': mean_auprc_8_ci_result[1] if not np.isnan(mean_auprc_8_ci_result[1]) else None,
                        'ci_upper': mean_auprc_8_ci_result[2] if not np.isnan(mean_auprc_8_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUPRC (8 common) for subgroup {subgroup_name}: {e}")
            
            # Store aggregate metrics
            fairness_results['sex_age_subgroups'][subgroup_name] = subgroup_aurocs
            fairness_results['sex_age_subgroups'][subgroup_name]['mean_auroc_14_labels'] = mean_auroc_14_ci
            fairness_results['sex_age_subgroups'][subgroup_name]['mean_auroc_8_common'] = mean_auroc_8_ci
            if subgroup_auprcs:
                fairness_results['sex_age_subgroups'][subgroup_name]['auprc_per_label'] = subgroup_auprcs
                fairness_results['sex_age_subgroups'][subgroup_name]['mean_auprc_14_labels'] = mean_auprc_14_ci
                fairness_results['sex_age_subgroups'][subgroup_name]['mean_auprc_8_common'] = mean_auprc_8_ci
    
    # Compute metrics for intersectional_subgroups (3-way: age × sex × ethnicity)
    for subgroup_name, mask in intersectional_subgroups.items():
        if mask.sum() < 10:  # Skip if too few samples
            continue
        
        subgroup_aurocs = {}
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            valid_mask = (ground_truth[mask, i] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, i][valid_mask]
            y_pred_sub = predictions_probs[mask, i][valid_mask]
            
            # Check eligibility: N_pos >= 10 AND N_neg >= 10 for AUROC
            is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
            if not is_eligible:
                continue  # Skip this label for this subgroup
            
            try:
                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                subgroup_aurocs[class_name] = float(auroc)
                
                # Compute bootstrap CI for AUROC if requested
                if compute_statistics:
                    def auroc_fn(y_t, y_p):
                        try:
                            return roc_auc_score(y_t, y_p)
                        except:
                            return np.nan
                    
                    def validate_auroc_bootstrap(y_t_boot):
                        """Validate bootstrap sample has both classes for AUROC."""
                        unique_classes = np.unique(y_t_boot)
                        return len(unique_classes) >= 2 and (1 in unique_classes) and (0 in unique_classes)
                    
                    mean_auroc, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                        y_true_sub,
                        y_pred_sub,
                        auroc_fn,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                        validate_bootstrap=validate_auroc_bootstrap,
                    )
                    
                    if class_name not in subgroup_aurocs:
                        subgroup_aurocs[class_name] = {}
                    if not isinstance(subgroup_aurocs[class_name], dict):
                        # Convert to dict format
                        subgroup_aurocs[class_name] = {'mean': subgroup_aurocs[class_name]}
                    subgroup_aurocs[class_name]['mean'] = mean_auroc if not np.isnan(mean_auroc) else None
                    subgroup_aurocs[class_name]['ci_lower'] = ci_lower if not np.isnan(ci_lower) else None
                    subgroup_aurocs[class_name]['ci_upper'] = ci_upper if not np.isnan(ci_upper) else None
            except:
                pass
        
        # Compute AUPRC per label for this subgroup
        subgroup_auprcs = {}
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            valid_mask = (ground_truth[mask, i] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, i][valid_mask]
            y_pred_sub = predictions_probs[mask, i][valid_mask]
            
            # Check eligibility: N_pos >= 10 for AUPRC
            is_eligible, n_pos, n_neg = check_auprc_eligibility(y_true_sub, min_pos=10)
            if not is_eligible:
                continue  # Skip this label for this subgroup
            
            try:
                auprc = average_precision_score(y_true_sub, y_pred_sub)
                subgroup_auprcs[class_name] = float(auprc)
                
                # Compute bootstrap CI for AUPRC if requested
                if compute_statistics:
                    def auprc_fn(y_t, y_p):
                        try:
                            return average_precision_score(y_t, y_p)
                        except:
                            return np.nan
                    
                    def validate_auprc_bootstrap(y_t_boot):
                        """Validate bootstrap sample has at least some positives for AUPRC."""
                        unique_classes = np.unique(y_t_boot)
                        return len(unique_classes) >= 2 and (1 in unique_classes)
                    
                    mean_auprc, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                        y_true_sub,
                        y_pred_sub,
                        auprc_fn,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                        validate_bootstrap=validate_auprc_bootstrap,
                    )
                    
                    if class_name not in subgroup_auprcs:
                        subgroup_auprcs[class_name] = {}
                    if not isinstance(subgroup_auprcs[class_name], dict):
                        # Convert to dict format
                        subgroup_auprcs[class_name] = {'mean': subgroup_auprcs[class_name]}
                    subgroup_auprcs[class_name]['mean'] = mean_auprc if not np.isnan(mean_auprc) else None
                    subgroup_auprcs[class_name]['ci_lower'] = ci_lower if not np.isnan(ci_lower) else None
                    subgroup_auprcs[class_name]['ci_upper'] = ci_upper if not np.isnan(ci_upper) else None
            except:
                pass
        
        # Compute aggregate metrics (mean AUROC/AUPRC for 14 and 8 labels) for this subgroup
        if subgroup_aurocs:
            # Mean AUROC (14 labels)
            valid_aurocs_14 = []
            for class_name, auroc_val in subgroup_aurocs.items():
                if isinstance(auroc_val, dict):
                    mean_val = auroc_val.get('mean')
                else:
                    mean_val = auroc_val
                if mean_val is not None and not np.isnan(mean_val):
                    valid_aurocs_14.append(mean_val)
            mean_auroc_14_subgroup = np.mean(valid_aurocs_14) if valid_aurocs_14 else np.nan
            
            # Mean AUROC (8 common labels)
            common_aurocs_8 = []
            for label in COMMON_LABELS:
                if label in subgroup_aurocs:
                    auroc_val = subgroup_aurocs[label]
                    if isinstance(auroc_val, dict):
                        mean_val = auroc_val.get('mean')
                    else:
                        mean_val = auroc_val
                    if mean_val is not None and not np.isnan(mean_val):
                        common_aurocs_8.append(mean_val)
            mean_auroc_8_subgroup = np.mean(common_aurocs_8) if common_aurocs_8 else np.nan
            
            # Compute bootstrap CI for mean AUROC - bootstrap from actual data
            mean_auroc_14_ci = {'mean': mean_auroc_14_subgroup, 'ci_lower': None, 'ci_upper': None}
            mean_auroc_8_ci = {'mean': mean_auroc_8_subgroup, 'ci_lower': None, 'ci_upper': None}
            
            if compute_statistics:
                def compute_mean_auroc_14_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUROC across 14 labels for subgroup."""
                    try:
                        aurocs_boot = []
                        for i, class_name in enumerate(CHEXPERT_CLASSES):
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auroc):
                                    aurocs_boot.append(auroc)
                            except:
                                continue
                        return np.mean(aurocs_boot) if aurocs_boot else np.nan
                    except:
                        return np.nan
                
                def compute_mean_auroc_8_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUROC across 8 common labels for subgroup."""
                    try:
                        aurocs_boot = []
                        for label in COMMON_LABELS:
                            i = CHEXPERT_CLASSES.index(label)
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auroc):
                                    aurocs_boot.append(auroc)
                            except:
                                continue
                        return np.mean(aurocs_boot) if aurocs_boot else np.nan
                    except:
                        return np.nan
                
                try:
                    mean_auroc_14_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auroc_14_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auroc_14_ci = {
                        'mean': mean_auroc_14_ci_result[0] if not np.isnan(mean_auroc_14_ci_result[0]) else mean_auroc_14_subgroup,
                        'ci_lower': mean_auroc_14_ci_result[1] if not np.isnan(mean_auroc_14_ci_result[1]) else None,
                        'ci_upper': mean_auroc_14_ci_result[2] if not np.isnan(mean_auroc_14_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUROC (14 labels) for subgroup {subgroup_name}: {e}")
                
                try:
                    mean_auroc_8_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auroc_8_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auroc_8_ci = {
                        'mean': mean_auroc_8_ci_result[0] if not np.isnan(mean_auroc_8_ci_result[0]) else mean_auroc_8_subgroup,
                        'ci_lower': mean_auroc_8_ci_result[1] if not np.isnan(mean_auroc_8_ci_result[1]) else None,
                        'ci_upper': mean_auroc_8_ci_result[2] if not np.isnan(mean_auroc_8_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUROC (8 common) for subgroup {subgroup_name}: {e}")
            
            # Mean AUPRC (14 labels)
            valid_auprcs_14 = []
            for class_name, auprc_val in subgroup_auprcs.items():
                if isinstance(auprc_val, dict):
                    mean_val = auprc_val.get('mean')
                else:
                    mean_val = auprc_val
                if mean_val is not None and not np.isnan(mean_val):
                    valid_auprcs_14.append(mean_val)
            mean_auprc_14_subgroup = np.mean(valid_auprcs_14) if valid_auprcs_14 else np.nan
            
            # Mean AUPRC (8 common labels)
            common_auprcs_8 = []
            for label in COMMON_LABELS:
                if label in subgroup_auprcs:
                    auprc_val = subgroup_auprcs[label]
                    if isinstance(auprc_val, dict):
                        mean_val = auprc_val.get('mean')
                    else:
                        mean_val = auprc_val
                    if mean_val is not None and not np.isnan(mean_val):
                        common_auprcs_8.append(mean_val)
            mean_auprc_8_subgroup = np.mean(common_auprcs_8) if common_auprcs_8 else np.nan
            
            # Compute bootstrap CI for mean AUPRC - bootstrap from actual data
            mean_auprc_14_ci = {'mean': mean_auprc_14_subgroup, 'ci_lower': None, 'ci_upper': None}
            mean_auprc_8_ci = {'mean': mean_auprc_8_subgroup, 'ci_lower': None, 'ci_upper': None}
            
            if compute_statistics:
                def compute_mean_auprc_14_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUPRC across 14 labels for subgroup."""
                    try:
                        auprcs_boot = []
                        for i, class_name in enumerate(CHEXPERT_CLASSES):
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auprc = average_precision_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auprc):
                                    auprcs_boot.append(auprc)
                            except:
                                continue
                        return np.mean(auprcs_boot) if auprcs_boot else np.nan
                    except:
                        return np.nan
                
                def compute_mean_auprc_8_subgroup(y_true_boot, y_pred_boot):
                    """Compute mean AUPRC across 8 common labels for subgroup."""
                    try:
                        auprcs_boot = []
                        for label in COMMON_LABELS:
                            i = CHEXPERT_CLASSES.index(label)
                            valid_mask = y_true_boot[:, i] != -1
                            if valid_mask.sum() < 2:
                                continue
                            y_true_sub = y_true_boot[valid_mask, i]
                            y_pred_sub = y_pred_boot[valid_mask, i]
                            if len(np.unique(y_true_sub)) < 2:
                                continue
                            try:
                                auprc = average_precision_score(y_true_sub, y_pred_sub)
                                if not np.isnan(auprc):
                                    auprcs_boot.append(auprc)
                            except:
                                continue
                        return np.mean(auprcs_boot) if auprcs_boot else np.nan
                    except:
                        return np.nan
                
                try:
                    mean_auprc_14_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auprc_14_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auprc_14_ci = {
                        'mean': mean_auprc_14_ci_result[0] if not np.isnan(mean_auprc_14_ci_result[0]) else mean_auprc_14_subgroup,
                        'ci_lower': mean_auprc_14_ci_result[1] if not np.isnan(mean_auprc_14_ci_result[1]) else None,
                        'ci_upper': mean_auprc_14_ci_result[2] if not np.isnan(mean_auprc_14_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUPRC (14 labels) for subgroup {subgroup_name}: {e}")
                
                try:
                    mean_auprc_8_ci_result = bootstrap_ci_for_metric(
                        ground_truth[mask],
                        predictions_probs[mask],
                        compute_mean_auprc_8_subgroup,
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=random_seed,
                    )
                    mean_auprc_8_ci = {
                        'mean': mean_auprc_8_ci_result[0] if not np.isnan(mean_auprc_8_ci_result[0]) else mean_auprc_8_subgroup,
                        'ci_lower': mean_auprc_8_ci_result[1] if not np.isnan(mean_auprc_8_ci_result[1]) else None,
                        'ci_upper': mean_auprc_8_ci_result[2] if not np.isnan(mean_auprc_8_ci_result[2]) else None,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute bootstrap CI for mean AUPRC (8 common) for subgroup {subgroup_name}: {e}")
            
            # Store aggregate metrics
            fairness_results['intersectional_subgroups'][subgroup_name] = subgroup_aurocs
            fairness_results['intersectional_subgroups'][subgroup_name]['mean_auroc_14_labels'] = mean_auroc_14_ci
            fairness_results['intersectional_subgroups'][subgroup_name]['mean_auroc_8_common'] = mean_auroc_8_ci
            if subgroup_auprcs:
                fairness_results['intersectional_subgroups'][subgroup_name]['auprc_per_label'] = subgroup_auprcs
                fairness_results['intersectional_subgroups'][subgroup_name]['mean_auprc_14_labels'] = mean_auprc_14_ci
                fairness_results['intersectional_subgroups'][subgroup_name]['mean_auprc_8_common'] = mean_auprc_8_ci
    
    # Compute AUROC parity per label and for averages WITH BOOTSTRAP CI
    # AUROC parity = difference between highest and lowest AUROC across subgroups
    logger.info("Computing AUROC parity metrics with bootstrap CIs...")
    
    # Helper function to compute AUROC parity per label with bootstrap CI
    def compute_auroc_parity_per_label_with_ci(class_idx, class_name, n_bootstrap, compute_statistics, random_seed):
        """Compute AUROC parity (max - min and p90-p10) across eligible sex_race subgroups for a specific label."""
        subgroup_aurocs = []
        subgroup_masks = []
        subgroup_names = []
        subgroup_eligibility = {}  # Track eligibility info
        
        # Collect AUROC values and masks for each eligible subgroup
        for subgroup_name, mask in sex_race_subgroups.items():
            if mask.sum() < 10:
                continue
            valid_mask = (ground_truth[mask, class_idx] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, class_idx][valid_mask]
            y_pred_sub = predictions_probs[mask, class_idx][valid_mask]
            
            # Check eligibility: N_pos >= 10 AND N_neg >= 10
            is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
            
            if not is_eligible:
                subgroup_eligibility[subgroup_name] = {'eligible': False, 'n_pos': n_pos, 'n_neg': n_neg}
                continue
            
            try:
                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                if not np.isnan(auroc):
                    subgroup_aurocs.append(auroc)
                    subgroup_masks.append(mask)
                    subgroup_names.append(subgroup_name)
                    subgroup_eligibility[subgroup_name] = {'eligible': True, 'n_pos': n_pos, 'n_neg': n_neg}
            except:
                continue
        
        if len(subgroup_aurocs) < 2:
            return None, None, None, None, None, None, None, None, subgroup_eligibility
        
        # Compute max-min gap
        parity_value_max_min = float(max(subgroup_aurocs) - min(subgroup_aurocs))
        
        # Compute robust gap (p90-p10)
        if len(subgroup_aurocs) >= 2:
            p90 = float(np.percentile(subgroup_aurocs, 90))
            p10 = float(np.percentile(subgroup_aurocs, 10))
            parity_value_robust = p90 - p10
        else:
            parity_value_robust = None
        
        # Compute bootstrap CI for both gaps
        parity_ci_max_min = {'mean': parity_value_max_min, 'ci_lower': None, 'ci_upper': None}
        parity_ci_robust = {'mean': parity_value_robust, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and len(subgroup_masks) >= 2:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n = len(ground_truth)
            bootstrap_parities_max_min = []
            bootstrap_parities_robust = []
            
            for _ in range(n_bootstrap):
                # Bootstrap resample indices
                boot_indices = np.random.choice(n, size=n, replace=True)
                
                # Compute AUROC for each eligible subgroup on bootstrap sample
                aurocs_boot = []
                for mask in subgroup_masks:
                    # Get original indices that belong to this subgroup
                    subgroup_indices = np.where(mask)[0]
                    # Find which bootstrap indices correspond to subgroup samples
                    boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                    if boot_subgroup_mask.sum() < 10:
                        continue
                    
                    # Get bootstrap data for this subgroup
                    boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                    y_true_boot_sub = ground_truth[boot_subgroup_indices, class_idx]
                    y_pred_boot_sub = predictions_probs[boot_subgroup_indices, class_idx]
                    
                    # Filter valid labels
                    valid_mask_boot = (y_true_boot_sub != -1)
                    if valid_mask_boot.sum() < 2:
                        continue
                    
                    y_true_valid = y_true_boot_sub[valid_mask_boot]
                    y_pred_valid = y_pred_boot_sub[valid_mask_boot]
                    
                    # Check eligibility for bootstrap sample (relaxed threshold: N_pos >= 5, N_neg >= 5)
                    is_eligible_boot, _, _ = check_auroc_eligibility(y_true_valid, min_pos=5, min_neg=5)
                    if not is_eligible_boot:
                        continue
                    
                    try:
                        auroc_boot = roc_auc_score(y_true_valid, y_pred_valid)
                        if not np.isnan(auroc_boot):
                            aurocs_boot.append(auroc_boot)
                    except:
                        continue
                
                if len(aurocs_boot) >= 2:
                    # Max-min gap
                    parity_boot_max_min = float(max(aurocs_boot) - min(aurocs_boot))
                    if not np.isnan(parity_boot_max_min):
                        bootstrap_parities_max_min.append(parity_boot_max_min)
                    
                    # Robust gap (p90-p10)
                    p90_boot = float(np.percentile(aurocs_boot, 90))
                    p10_boot = float(np.percentile(aurocs_boot, 10))
                    parity_boot_robust = p90_boot - p10_boot
                    if not np.isnan(parity_boot_robust):
                        bootstrap_parities_robust.append(parity_boot_robust)
            
            # Compute CI for max-min gap
            if len(bootstrap_parities_max_min) > 0:
                bootstrap_parities_max_min = np.array(bootstrap_parities_max_min)
                mean_parity_max_min = np.mean(bootstrap_parities_max_min)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_max_min = np.percentile(bootstrap_parities_max_min, lower_percentile)
                ci_upper_max_min = np.percentile(bootstrap_parities_max_min, upper_percentile)
                
                parity_ci_max_min = {
                    'mean': float(mean_parity_max_min),
                    'ci_lower': float(ci_lower_max_min),
                    'ci_upper': float(ci_upper_max_min),
                }
            else:
                logger.debug(f"No valid bootstrap parities (max-min) computed for {class_name}")
            
            # Compute CI for robust gap
            if len(bootstrap_parities_robust) > 0:
                bootstrap_parities_robust = np.array(bootstrap_parities_robust)
                mean_parity_robust = np.mean(bootstrap_parities_robust)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_robust = np.percentile(bootstrap_parities_robust, lower_percentile)
                ci_upper_robust = np.percentile(bootstrap_parities_robust, upper_percentile)
                
                parity_ci_robust = {
                    'mean': float(mean_parity_robust),
                    'ci_lower': float(ci_lower_robust),
                    'ci_upper': float(ci_upper_robust),
                }
            else:
                logger.debug(f"No valid bootstrap parities (robust) computed for {class_name}")
        
        # Return bootstrap mean if available, otherwise point estimate
        parity_mean_max_min = parity_ci_max_min.get('mean') if parity_ci_max_min.get('ci_lower') is not None else parity_value_max_min
        parity_mean_robust = parity_ci_robust.get('mean') if parity_ci_robust.get('ci_lower') is not None else parity_value_robust
        
        return (parity_value_max_min, parity_mean_max_min, parity_ci_max_min.get('ci_lower'), parity_ci_max_min.get('ci_upper'),
                parity_value_robust, parity_mean_robust, parity_ci_robust.get('ci_lower'), parity_ci_robust.get('ci_upper'),
                subgroup_eligibility)
    
    # Compute AUROC parity per label with CI
    auroc_parity_per_label_max_min = {}
    auroc_parity_per_label_robust = {}
    auroc_parity_per_label_ci_max_min = {}
    auroc_parity_per_label_ci_robust = {}
    auroc_parity_per_label_eligibility = {}
    
    for i, class_name in enumerate(CHEXPERT_CLASSES):
        (parity_val_max_min, parity_mean_max_min, ci_lower_max_min, ci_upper_max_min,
         parity_val_robust, parity_mean_robust, ci_lower_robust, ci_upper_robust,
         eligibility) = compute_auroc_parity_per_label_with_ci(
            i, class_name, n_bootstrap, compute_statistics, random_seed
        )
        
        if parity_val_max_min is not None:
            auroc_parity_per_label_max_min[class_name] = parity_val_max_min
            auroc_parity_per_label_ci_max_min[class_name] = {
                'mean': parity_mean_max_min if parity_mean_max_min is not None else parity_val_max_min,  # Use bootstrap mean
                'ci_lower': ci_lower_max_min,
                'ci_upper': ci_upper_max_min,
            }
        else:
            auroc_parity_per_label_max_min[class_name] = None
            auroc_parity_per_label_ci_max_min[class_name] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        if parity_val_robust is not None:
            auroc_parity_per_label_robust[class_name] = parity_val_robust
            auroc_parity_per_label_ci_robust[class_name] = {
                'mean': parity_mean_robust if parity_mean_robust is not None else parity_val_robust,  # Use bootstrap mean
                'ci_lower': ci_lower_robust,
                'ci_upper': ci_upper_robust,
            }
        else:
            auroc_parity_per_label_robust[class_name] = None
            auroc_parity_per_label_ci_robust[class_name] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        auroc_parity_per_label_eligibility[class_name] = eligibility
    
    # Store for backward compatibility (max-min)
    fairness_results['auroc_parity_per_label'] = {
        k: float(v) if v is not None and not np.isnan(v) else None 
        for k, v in auroc_parity_per_label_max_min.items()
    }
    fairness_results['auroc_parity_per_label_ci'] = auroc_parity_per_label_ci_max_min
    
    # Store new metrics
    fairness_results['auroc_parity_per_label_max_min'] = {
        k: float(v) if v is not None and not np.isnan(v) else None 
        for k, v in auroc_parity_per_label_max_min.items()
    }
    fairness_results['auroc_parity_per_label_max_min_ci'] = auroc_parity_per_label_ci_max_min
    fairness_results['auroc_parity_per_label_robust'] = {
        k: float(v) if v is not None and not np.isnan(v) else None 
        for k, v in auroc_parity_per_label_robust.items()
    }
    fairness_results['auroc_parity_per_label_robust_ci'] = auroc_parity_per_label_ci_robust
    fairness_results['auroc_parity_per_label_eligibility'] = auroc_parity_per_label_eligibility
    
    # Helper function to compute both max-min and robust gaps from a list of metrics
    def compute_gaps_from_metrics(metrics_list):
        """Compute max-min and robust (p90-p10) gaps from a list of metric values."""
        if len(metrics_list) < 2:
            return None, None
        max_min_gap = float(max(metrics_list) - min(metrics_list))
        p90 = float(np.percentile(metrics_list, 90))
        p10 = float(np.percentile(metrics_list, 10))
        robust_gap = p90 - p10
        return max_min_gap, robust_gap
    
    # Compute AUROC parity for sex_age_subgroups (Sex × Age) as well
    # Helper function to compute AUROC parity per label for sex_age subgroups
    def compute_auroc_parity_per_label_sex_age_with_ci(class_idx, class_name, n_bootstrap, compute_statistics, random_seed):
        """Compute AUROC parity (max-min and p90-p10) across eligible sex_age subgroups for a specific label."""
        subgroup_aurocs = []
        subgroup_masks = []
        subgroup_names = []
        subgroup_eligibility = {}
        
        # Collect AUROC values and masks for each eligible subgroup
        for subgroup_name, mask in sex_age_subgroups.items():
            if mask.sum() < 10:
                continue
            valid_mask = (ground_truth[mask, class_idx] != -1)
            if valid_mask.sum() < 2:
                continue
            
            y_true_sub = ground_truth[mask, class_idx][valid_mask]
            y_pred_sub = predictions_probs[mask, class_idx][valid_mask]
            
            # Check eligibility: N_pos >= 10 AND N_neg >= 10
            is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
            
            if not is_eligible:
                subgroup_eligibility[subgroup_name] = {'eligible': False, 'n_pos': n_pos, 'n_neg': n_neg}
                continue
            
            try:
                auroc = roc_auc_score(y_true_sub, y_pred_sub)
                if not np.isnan(auroc):
                    subgroup_aurocs.append(auroc)
                    subgroup_masks.append(mask)
                    subgroup_names.append(subgroup_name)
                    subgroup_eligibility[subgroup_name] = {'eligible': True, 'n_pos': n_pos, 'n_neg': n_neg}
            except:
                continue
        
        if len(subgroup_aurocs) < 2:
            return None, None, None, None, None, None, None, None, subgroup_eligibility
        
        # Compute max-min gap
        parity_value_max_min = float(max(subgroup_aurocs) - min(subgroup_aurocs))
        
        # Compute robust gap (p90-p10)
        if len(subgroup_aurocs) >= 2:
            p90 = float(np.percentile(subgroup_aurocs, 90))
            p10 = float(np.percentile(subgroup_aurocs, 10))
            parity_value_robust = p90 - p10
        else:
            parity_value_robust = None
        
        # Compute bootstrap CI for both gaps
        parity_ci_max_min = {'mean': parity_value_max_min, 'ci_lower': None, 'ci_upper': None}
        parity_ci_robust = {'mean': parity_value_robust, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and len(subgroup_masks) >= 2:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n = len(ground_truth)
            bootstrap_parities_max_min = []
            bootstrap_parities_robust = []
            
            for _ in range(n_bootstrap):
                # Bootstrap resample indices
                boot_indices = np.random.choice(n, size=n, replace=True)
                
                # Compute AUROC for each eligible subgroup on bootstrap sample
                aurocs_boot = []
                for mask in subgroup_masks:
                    # Get original indices that belong to this subgroup
                    subgroup_indices = np.where(mask)[0]
                    # Find which bootstrap indices correspond to subgroup samples
                    boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                    if boot_subgroup_mask.sum() < 10:
                        continue
                    
                    # Get bootstrap data for this subgroup
                    boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                    y_true_boot_sub = ground_truth[boot_subgroup_indices, class_idx]
                    y_pred_boot_sub = predictions_probs[boot_subgroup_indices, class_idx]
                    
                    # Filter valid labels
                    valid_mask_boot = (y_true_boot_sub != -1)
                    if valid_mask_boot.sum() < 2:
                        continue
                    
                    y_true_valid = y_true_boot_sub[valid_mask_boot]
                    y_pred_valid = y_pred_boot_sub[valid_mask_boot]
                    
                    # Check eligibility for bootstrap sample (relaxed threshold: N_pos >= 5, N_neg >= 5)
                    is_eligible_boot, _, _ = check_auroc_eligibility(y_true_valid, min_pos=5, min_neg=5)
                    if not is_eligible_boot:
                        continue
                    
                    try:
                        auroc_boot = roc_auc_score(y_true_valid, y_pred_valid)
                        if not np.isnan(auroc_boot):
                            aurocs_boot.append(auroc_boot)
                    except:
                        continue
                
                if len(aurocs_boot) >= 2:
                    # Max-min gap
                    parity_boot_max_min = float(max(aurocs_boot) - min(aurocs_boot))
                    if not np.isnan(parity_boot_max_min):
                        bootstrap_parities_max_min.append(parity_boot_max_min)
                    
                    # Robust gap (p90-p10)
                    p90_boot = float(np.percentile(aurocs_boot, 90))
                    p10_boot = float(np.percentile(aurocs_boot, 10))
                    parity_boot_robust = p90_boot - p10_boot
                    if not np.isnan(parity_boot_robust):
                        bootstrap_parities_robust.append(parity_boot_robust)
            
            # Compute CI for max-min gap
            if len(bootstrap_parities_max_min) > 0:
                bootstrap_parities_max_min = np.array(bootstrap_parities_max_min)
                mean_parity_max_min = np.mean(bootstrap_parities_max_min)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_max_min = np.percentile(bootstrap_parities_max_min, lower_percentile)
                ci_upper_max_min = np.percentile(bootstrap_parities_max_min, upper_percentile)
                
                parity_ci_max_min = {
                    'mean': float(mean_parity_max_min),
                    'ci_lower': float(ci_lower_max_min),
                    'ci_upper': float(ci_upper_max_min),
                }
            else:
                logger.debug(f"No valid bootstrap parities (max-min) computed for AUROC parity per label (sex-age) {class_name}")
            
            # Compute CI for robust gap
            if len(bootstrap_parities_robust) > 0:
                bootstrap_parities_robust = np.array(bootstrap_parities_robust)
                mean_parity_robust = np.mean(bootstrap_parities_robust)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_robust = np.percentile(bootstrap_parities_robust, lower_percentile)
                ci_upper_robust = np.percentile(bootstrap_parities_robust, upper_percentile)
                
                parity_ci_robust = {
                    'mean': float(mean_parity_robust),
                    'ci_lower': float(ci_lower_robust),
                    'ci_upper': float(ci_upper_robust),
                }
            else:
                logger.debug(f"No valid bootstrap parities (robust) computed for AUROC parity per label (sex-age) {class_name}")
        
        # Return bootstrap mean if available, otherwise point estimate
        parity_mean_max_min = parity_ci_max_min.get('mean') if parity_ci_max_min.get('ci_lower') is not None else parity_value_max_min
        parity_mean_robust = parity_ci_robust.get('mean') if parity_ci_robust.get('ci_lower') is not None else parity_value_robust
        
        return (parity_value_max_min, parity_mean_max_min, parity_ci_max_min.get('ci_lower'), parity_ci_max_min.get('ci_upper'),
                parity_value_robust, parity_mean_robust, parity_ci_robust.get('ci_lower'), parity_ci_robust.get('ci_upper'),
                subgroup_eligibility)
    
    # Compute AUROC parity per label for sex_age subgroups with CI
    auroc_parity_per_label_sex_age_max_min = {}
    auroc_parity_per_label_sex_age_robust = {}
    auroc_parity_per_label_sex_age_ci_max_min = {}
    auroc_parity_per_label_sex_age_ci_robust = {}
    auroc_parity_per_label_sex_age_eligibility = {}
    
    for i, class_name in enumerate(CHEXPERT_CLASSES):
        (parity_val_max_min, parity_mean_max_min, ci_lower_max_min, ci_upper_max_min,
         parity_val_robust, parity_mean_robust, ci_lower_robust, ci_upper_robust,
         eligibility) = compute_auroc_parity_per_label_sex_age_with_ci(
            i, class_name, n_bootstrap, compute_statistics, random_seed
        )
        
        if parity_val_max_min is not None:
            auroc_parity_per_label_sex_age_max_min[class_name] = parity_val_max_min
            auroc_parity_per_label_sex_age_ci_max_min[class_name] = {
                'mean': parity_mean_max_min if parity_mean_max_min is not None else parity_val_max_min,  # Use bootstrap mean
                'ci_lower': ci_lower_max_min,
                'ci_upper': ci_upper_max_min,
            }
        else:
            auroc_parity_per_label_sex_age_max_min[class_name] = None
            auroc_parity_per_label_sex_age_ci_max_min[class_name] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        if parity_val_robust is not None:
            auroc_parity_per_label_sex_age_robust[class_name] = parity_val_robust
            auroc_parity_per_label_sex_age_ci_robust[class_name] = {
                'mean': parity_mean_robust if parity_mean_robust is not None else parity_val_robust,  # Use bootstrap mean
                'ci_lower': ci_lower_robust,
                'ci_upper': ci_upper_robust,
            }
        else:
            auroc_parity_per_label_sex_age_robust[class_name] = None
            auroc_parity_per_label_sex_age_ci_robust[class_name] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
            }
        
        auroc_parity_per_label_sex_age_eligibility[class_name] = eligibility
    
    # Store for backward compatibility (max-min)
    fairness_results['auroc_parity_per_label_sex_age'] = {
        k: float(v) if v is not None and not np.isnan(v) else None 
        for k, v in auroc_parity_per_label_sex_age_max_min.items()
    }
    fairness_results['auroc_parity_per_label_sex_age_ci'] = auroc_parity_per_label_sex_age_ci_max_min
    
    # Store new metrics
    fairness_results['auroc_parity_per_label_sex_age_max_min'] = {
        k: float(v) if v is not None and not np.isnan(v) else None 
        for k, v in auroc_parity_per_label_sex_age_max_min.items()
    }
    fairness_results['auroc_parity_per_label_sex_age_max_min_ci'] = auroc_parity_per_label_sex_age_ci_max_min
    fairness_results['auroc_parity_per_label_sex_age_robust'] = {
        k: float(v) if v is not None and not np.isnan(v) else None 
        for k, v in auroc_parity_per_label_sex_age_robust.items()
    }
    fairness_results['auroc_parity_per_label_sex_age_robust_ci'] = auroc_parity_per_label_sex_age_ci_robust
    fairness_results['auroc_parity_per_label_sex_age_eligibility'] = auroc_parity_per_label_sex_age_eligibility
    
    # Compute AUROC parity for 14-label average with bootstrap CI (for sex_race)
    def compute_auroc_parity_14_with_ci(n_bootstrap, compute_statistics, random_seed):
        """Compute AUROC parity (max-min and p90-p10) of mean AUROC (14 labels) across eligible sex_race subgroups."""
        subgroup_mean_aurocs = []
        subgroup_masks = []
        subgroup_names = []
        subgroup_eligibility = {}
        
        for subgroup_name, mask in sex_race_subgroups.items():
            if mask.sum() < 10:
                continue
            
            # Compute mean AUROC for this subgroup, only including eligible labels
            aurocs = []
            n_valid_labels = 0
            for i, class_name in enumerate(CHEXPERT_CLASSES):
                valid_mask = (ground_truth[mask, i] != -1)
                if valid_mask.sum() < 2:
                    continue
                y_true_sub = ground_truth[mask, i][valid_mask]
                y_pred_sub = predictions_probs[mask, i][valid_mask]
                
                # Check eligibility: N_pos >= 10 AND N_neg >= 10
                is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
                if not is_eligible:
                    continue
                
                try:
                    auroc = roc_auc_score(y_true_sub, y_pred_sub)
                    if not np.isnan(auroc):
                        aurocs.append(auroc)
                        n_valid_labels += 1
                except:
                    continue
            
            # Only include subgroups with at least some eligible labels
            if len(aurocs) > 0:
                mean_auroc = np.mean(aurocs)
                subgroup_mean_aurocs.append(mean_auroc)
                subgroup_masks.append(mask)
                subgroup_names.append(subgroup_name)
                subgroup_eligibility[subgroup_name] = {'n_valid_labels': n_valid_labels, 'n_total_labels': len(CHEXPERT_CLASSES)}
        
        if len(subgroup_mean_aurocs) < 2:
            return None, None, None, None, None, None, None, None, subgroup_eligibility
        
        # Compute max-min gap
        parity_value_max_min = float(max(subgroup_mean_aurocs) - min(subgroup_mean_aurocs))
        
        # Compute robust gap (p90-p10)
        if len(subgroup_mean_aurocs) >= 2:
            p90 = float(np.percentile(subgroup_mean_aurocs, 90))
            p10 = float(np.percentile(subgroup_mean_aurocs, 10))
            parity_value_robust = p90 - p10
        else:
            parity_value_robust = None
        
        # Compute bootstrap CI for both gaps
        parity_ci_max_min = {'mean': parity_value_max_min, 'ci_lower': None, 'ci_upper': None}
        parity_ci_robust = {'mean': parity_value_robust, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and len(subgroup_masks) >= 2:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n = len(ground_truth)
            bootstrap_parities_max_min = []
            bootstrap_parities_robust = []
            
            for _ in range(n_bootstrap):
                # Bootstrap resample indices
                boot_indices = np.random.choice(n, size=n, replace=True)
                
                # Compute mean AUROC for each subgroup on bootstrap sample
                mean_aurocs_boot = []
                for mask in subgroup_masks:
                    # Get original indices that belong to this subgroup
                    subgroup_indices = np.where(mask)[0]
                    # Find which bootstrap indices correspond to subgroup samples
                    boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                    if boot_subgroup_mask.sum() < 10:
                        continue
                    
                    # Get bootstrap data for this subgroup
                    boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                    aurocs_boot = []
                    
                    for i in range(len(CHEXPERT_CLASSES)):
                        y_true_boot_sub = ground_truth[boot_subgroup_indices, i]
                        y_pred_boot_sub = predictions_probs[boot_subgroup_indices, i]
                        
                        # Filter valid labels
                        valid_mask_boot = (y_true_boot_sub != -1)
                        if valid_mask_boot.sum() < 2:
                            continue
                        
                        y_true_valid = y_true_boot_sub[valid_mask_boot]
                        y_pred_valid = y_pred_boot_sub[valid_mask_boot]
                        
                        # Check eligibility for bootstrap sample (relaxed threshold: N_pos >= 5, N_neg >= 5)
                        is_eligible_boot, _, _ = check_auroc_eligibility(y_true_valid, min_pos=5, min_neg=5)
                        if not is_eligible_boot:
                            continue
                        
                        try:
                            auroc_boot = roc_auc_score(y_true_valid, y_pred_valid)
                            if not np.isnan(auroc_boot):
                                aurocs_boot.append(auroc_boot)
                        except:
                            continue
                    
                    if len(aurocs_boot) > 0:
                        mean_auroc_boot = np.mean(aurocs_boot)
                        mean_aurocs_boot.append(mean_auroc_boot)
                
                if len(mean_aurocs_boot) >= 2:
                    # Max-min gap
                    parity_boot_max_min = float(max(mean_aurocs_boot) - min(mean_aurocs_boot))
                    if not np.isnan(parity_boot_max_min):
                        bootstrap_parities_max_min.append(parity_boot_max_min)
                    
                    # Robust gap (p90-p10)
                    p90_boot = float(np.percentile(mean_aurocs_boot, 90))
                    p10_boot = float(np.percentile(mean_aurocs_boot, 10))
                    parity_boot_robust = p90_boot - p10_boot
                    if not np.isnan(parity_boot_robust):
                        bootstrap_parities_robust.append(parity_boot_robust)
            
            # Compute CI for max-min gap
            if len(bootstrap_parities_max_min) > 0:
                bootstrap_parities_max_min = np.array(bootstrap_parities_max_min)
                mean_parity_max_min = np.mean(bootstrap_parities_max_min)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_max_min = np.percentile(bootstrap_parities_max_min, lower_percentile)
                ci_upper_max_min = np.percentile(bootstrap_parities_max_min, upper_percentile)
                
                parity_ci_max_min = {
                    'mean': float(mean_parity_max_min),
                    'ci_lower': float(ci_lower_max_min),
                    'ci_upper': float(ci_upper_max_min),
                }
            else:
                logger.debug("No valid bootstrap parities (max-min) computed for AUROC parity (14 labels)")
            
            # Compute CI for robust gap
            if len(bootstrap_parities_robust) > 0:
                bootstrap_parities_robust = np.array(bootstrap_parities_robust)
                mean_parity_robust = np.mean(bootstrap_parities_robust)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_robust = np.percentile(bootstrap_parities_robust, lower_percentile)
                ci_upper_robust = np.percentile(bootstrap_parities_robust, upper_percentile)
                
                parity_ci_robust = {
                    'mean': float(mean_parity_robust),
                    'ci_lower': float(ci_lower_robust),
                    'ci_upper': float(ci_upper_robust),
                }
            else:
                logger.debug("No valid bootstrap parities (robust) computed for AUROC parity (14 labels)")
        
        # Return bootstrap mean if available, otherwise point estimate
        parity_mean_max_min = parity_ci_max_min.get('mean') if parity_ci_max_min.get('ci_lower') is not None else parity_value_max_min
        parity_mean_robust = parity_ci_robust.get('mean') if parity_ci_robust.get('ci_lower') is not None else parity_value_robust
        
        return (parity_value_max_min, parity_mean_max_min, parity_ci_max_min.get('ci_lower'), parity_ci_max_min.get('ci_upper'),
                parity_value_robust, parity_mean_robust, parity_ci_robust.get('ci_lower'), parity_ci_robust.get('ci_upper'),
                subgroup_eligibility)
    
    (parity_14_val, parity_14_mean, parity_14_lower, parity_14_upper,
     parity_14_robust, parity_14_robust_mean, parity_14_robust_lower, parity_14_robust_upper,
     parity_14_eligibility) = compute_auroc_parity_14_with_ci(
        n_bootstrap, compute_statistics, random_seed
    )
    if parity_14_val is not None:
        fairness_results['auroc_parity_14_labels'] = parity_14_val  # Backward compatibility
        fairness_results['auroc_parity_14_labels_ci'] = {
            'mean': parity_14_mean if parity_14_mean is not None else parity_14_val,  # Use bootstrap mean
            'ci_lower': parity_14_lower,
            'ci_upper': parity_14_upper,
        }
        fairness_results['auroc_parity_14_labels_max_min'] = parity_14_val
        fairness_results['auroc_parity_14_labels_max_min_ci'] = {
            'mean': parity_14_mean if parity_14_mean is not None else parity_14_val,  # Use bootstrap mean
            'ci_lower': parity_14_lower,
            'ci_upper': parity_14_upper,
        }
        if parity_14_robust is not None:
            fairness_results['auroc_parity_14_labels_robust'] = parity_14_robust
            fairness_results['auroc_parity_14_labels_robust_ci'] = {
                'mean': parity_14_robust_mean if parity_14_robust_mean is not None else parity_14_robust,  # Use bootstrap mean
                'ci_lower': parity_14_robust_lower,
                'ci_upper': parity_14_robust_upper,
            }
        fairness_results['auroc_parity_14_labels_eligibility'] = parity_14_eligibility
    else:
        fairness_results['auroc_parity_14_labels'] = None
        fairness_results['auroc_parity_14_labels_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_14_labels_max_min'] = None
        fairness_results['auroc_parity_14_labels_max_min_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_14_labels_robust'] = None
        fairness_results['auroc_parity_14_labels_robust_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_14_labels_eligibility'] = {}
    
    # Compute AUROC parity for 8-label average with bootstrap CI (for sex_race)
    def compute_auroc_parity_8_with_ci(n_bootstrap, compute_statistics, random_seed):
        """Compute AUROC parity (max-min and p90-p10) of mean AUROC (8 common) across eligible sex_race subgroups."""
        subgroup_mean_aurocs = []
        subgroup_masks = []
        subgroup_names = []
        subgroup_eligibility = {}
        
        for subgroup_name, mask in sex_race_subgroups.items():
            if mask.sum() < 10:
                continue
            
            # Compute mean AUROC for 8 common labels, only including eligible labels
            aurocs = []
            n_valid_labels = 0
            for label in COMMON_LABELS:
                i = CHEXPERT_CLASSES.index(label)
                valid_mask = (ground_truth[mask, i] != -1)
                if valid_mask.sum() < 2:
                    continue
                y_true_sub = ground_truth[mask, i][valid_mask]
                y_pred_sub = predictions_probs[mask, i][valid_mask]
                
                # Check eligibility: N_pos >= 10 AND N_neg >= 10
                is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
                if not is_eligible:
                    continue
                
                try:
                    auroc = roc_auc_score(y_true_sub, y_pred_sub)
                    if not np.isnan(auroc):
                        aurocs.append(auroc)
                        n_valid_labels += 1
                except:
                    continue
            
            # Only include subgroups with at least some eligible labels
            if len(aurocs) > 0:
                mean_auroc = np.mean(aurocs)
                subgroup_mean_aurocs.append(mean_auroc)
                subgroup_masks.append(mask)
                subgroup_names.append(subgroup_name)
                subgroup_eligibility[subgroup_name] = {'n_valid_labels': n_valid_labels, 'n_total_labels': len(COMMON_LABELS)}
        
        if len(subgroup_mean_aurocs) < 2:
            return None, None, None, None, None, None, None, None, subgroup_eligibility
        
        # Compute max-min gap
        parity_value_max_min = float(max(subgroup_mean_aurocs) - min(subgroup_mean_aurocs))
        
        # Compute robust gap (p90-p10)
        if len(subgroup_mean_aurocs) >= 2:
            p90 = float(np.percentile(subgroup_mean_aurocs, 90))
            p10 = float(np.percentile(subgroup_mean_aurocs, 10))
            parity_value_robust = p90 - p10
        else:
            parity_value_robust = None
        
        # Compute bootstrap CI for both gaps
        parity_ci_max_min = {'mean': parity_value_max_min, 'ci_lower': None, 'ci_upper': None}
        parity_ci_robust = {'mean': parity_value_robust, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and len(subgroup_masks) >= 2:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n = len(ground_truth)
            bootstrap_parities_max_min = []
            bootstrap_parities_robust = []
            
            for _ in range(n_bootstrap):
                # Bootstrap resample indices
                boot_indices = np.random.choice(n, size=n, replace=True)
                
                # Compute mean AUROC for each subgroup on bootstrap sample
                mean_aurocs_boot = []
                for mask in subgroup_masks:
                    # Get original indices that belong to this subgroup
                    subgroup_indices = np.where(mask)[0]
                    # Find which bootstrap indices correspond to subgroup samples
                    boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                    if boot_subgroup_mask.sum() < 10:
                        continue
                    
                    # Get bootstrap data for this subgroup
                    boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                    aurocs_boot = []
                    
                    for label in COMMON_LABELS:
                        i = CHEXPERT_CLASSES.index(label)
                        y_true_boot_sub = ground_truth[boot_subgroup_indices, i]
                        y_pred_boot_sub = predictions_probs[boot_subgroup_indices, i]
                        
                        # Filter valid labels
                        valid_mask_boot = (y_true_boot_sub != -1)
                        if valid_mask_boot.sum() < 2:
                            continue
                        
                        y_true_valid = y_true_boot_sub[valid_mask_boot]
                        y_pred_valid = y_pred_boot_sub[valid_mask_boot]
                        
                        # Check eligibility for bootstrap sample (relaxed threshold: N_pos >= 5, N_neg >= 5)
                        is_eligible_boot, _, _ = check_auroc_eligibility(y_true_valid, min_pos=5, min_neg=5)
                        if not is_eligible_boot:
                            continue
                        
                        try:
                            auroc_boot = roc_auc_score(y_true_valid, y_pred_valid)
                            if not np.isnan(auroc_boot):
                                aurocs_boot.append(auroc_boot)
                        except:
                            continue
                    
                    if len(aurocs_boot) > 0:
                        mean_auroc_boot = np.mean(aurocs_boot)
                        mean_aurocs_boot.append(mean_auroc_boot)
                
                if len(mean_aurocs_boot) >= 2:
                    # Max-min gap
                    parity_boot_max_min = float(max(mean_aurocs_boot) - min(mean_aurocs_boot))
                    if not np.isnan(parity_boot_max_min):
                        bootstrap_parities_max_min.append(parity_boot_max_min)
                    
                    # Robust gap (p90-p10)
                    p90_boot = float(np.percentile(mean_aurocs_boot, 90))
                    p10_boot = float(np.percentile(mean_aurocs_boot, 10))
                    parity_boot_robust = p90_boot - p10_boot
                    if not np.isnan(parity_boot_robust):
                        bootstrap_parities_robust.append(parity_boot_robust)
            
            # Compute CI for max-min gap
            if len(bootstrap_parities_max_min) > 0:
                bootstrap_parities_max_min = np.array(bootstrap_parities_max_min)
                mean_parity_max_min = np.mean(bootstrap_parities_max_min)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_max_min = np.percentile(bootstrap_parities_max_min, lower_percentile)
                ci_upper_max_min = np.percentile(bootstrap_parities_max_min, upper_percentile)
                
                parity_ci_max_min = {
                    'mean': float(mean_parity_max_min),
                    'ci_lower': float(ci_lower_max_min),
                    'ci_upper': float(ci_upper_max_min),
                }
            else:
                logger.debug("No valid bootstrap parities (max-min) computed for AUROC parity (8 common)")
            
            # Compute CI for robust gap
            if len(bootstrap_parities_robust) > 0:
                bootstrap_parities_robust = np.array(bootstrap_parities_robust)
                mean_parity_robust = np.mean(bootstrap_parities_robust)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_robust = np.percentile(bootstrap_parities_robust, lower_percentile)
                ci_upper_robust = np.percentile(bootstrap_parities_robust, upper_percentile)
                
                parity_ci_robust = {
                    'mean': float(mean_parity_robust),
                    'ci_lower': float(ci_lower_robust),
                    'ci_upper': float(ci_upper_robust),
                }
            else:
                logger.debug("No valid bootstrap parities (robust) computed for AUROC parity (8 common)")
        
        # Return bootstrap mean if available, otherwise point estimate
        parity_mean_max_min = parity_ci_max_min.get('mean') if parity_ci_max_min.get('ci_lower') is not None else parity_value_max_min
        parity_mean_robust = parity_ci_robust.get('mean') if parity_ci_robust.get('ci_lower') is not None else parity_value_robust
        
        return (parity_value_max_min, parity_mean_max_min, parity_ci_max_min.get('ci_lower'), parity_ci_max_min.get('ci_upper'),
                parity_value_robust, parity_mean_robust, parity_ci_robust.get('ci_lower'), parity_ci_robust.get('ci_upper'),
                subgroup_eligibility)
    
    (parity_8_val, parity_8_mean, parity_8_lower, parity_8_upper,
     parity_8_robust, parity_8_robust_mean, parity_8_robust_lower, parity_8_robust_upper,
     parity_8_eligibility) = compute_auroc_parity_8_with_ci(
        n_bootstrap, compute_statistics, random_seed
    )
    if parity_8_val is not None:
        fairness_results['auroc_parity_8_common'] = parity_8_val  # Backward compatibility
        fairness_results['auroc_parity_8_common_ci'] = {
            'mean': parity_8_mean if parity_8_mean is not None else parity_8_val,  # Use bootstrap mean
            'ci_lower': parity_8_lower,
            'ci_upper': parity_8_upper,
        }
        fairness_results['auroc_parity_8_common_max_min'] = parity_8_val
        fairness_results['auroc_parity_8_common_max_min_ci'] = {
            'mean': parity_8_mean if parity_8_mean is not None else parity_8_val,  # Use bootstrap mean
            'ci_lower': parity_8_lower,
            'ci_upper': parity_8_upper,
        }
        if parity_8_robust is not None:
            fairness_results['auroc_parity_8_common_robust'] = parity_8_robust
            fairness_results['auroc_parity_8_common_robust_ci'] = {
                'mean': parity_8_robust_mean if parity_8_robust_mean is not None else parity_8_robust,  # Use bootstrap mean
                'ci_lower': parity_8_robust_lower,
                'ci_upper': parity_8_robust_upper,
            }
        fairness_results['auroc_parity_8_common_eligibility'] = parity_8_eligibility
    else:
        fairness_results['auroc_parity_8_common'] = None
        fairness_results['auroc_parity_8_common_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_8_common_max_min'] = None
        fairness_results['auroc_parity_8_common_max_min_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_8_common_robust'] = None
        fairness_results['auroc_parity_8_common_robust_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_8_common_eligibility'] = {}
    
    # Compute AUROC parity for sex_age subgroups (14 and 8 labels)
    def compute_auroc_parity_14_sex_age_with_ci(n_bootstrap, compute_statistics, random_seed):
        """Compute AUROC parity (max-min and p90-p10) of mean AUROC (14 labels) across eligible sex_age subgroups."""
        subgroup_mean_aurocs = []
        subgroup_masks = []
        subgroup_names = []
        subgroup_eligibility = {}
        
        for subgroup_name, mask in sex_age_subgroups.items():
            if mask.sum() < 10:
                continue
            
            # Compute mean AUROC for this subgroup, only including eligible labels
            aurocs = []
            n_valid_labels = 0
            for i, class_name in enumerate(CHEXPERT_CLASSES):
                valid_mask = (ground_truth[mask, i] != -1)
                if valid_mask.sum() < 2:
                    continue
                y_true_sub = ground_truth[mask, i][valid_mask]
                y_pred_sub = predictions_probs[mask, i][valid_mask]
                
                # Check eligibility: N_pos >= 10 AND N_neg >= 10
                is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
                if not is_eligible:
                    continue
                
                try:
                    auroc = roc_auc_score(y_true_sub, y_pred_sub)
                    if not np.isnan(auroc):
                        aurocs.append(auroc)
                        n_valid_labels += 1
                except:
                    continue
            
            # Only include subgroups with at least some eligible labels
            if len(aurocs) > 0:
                mean_auroc = np.mean(aurocs)
                subgroup_mean_aurocs.append(mean_auroc)
                subgroup_masks.append(mask)
                subgroup_names.append(subgroup_name)
                subgroup_eligibility[subgroup_name] = {'n_valid_labels': n_valid_labels, 'n_total_labels': len(CHEXPERT_CLASSES)}
        
        if len(subgroup_mean_aurocs) < 2:
            return None, None, None, None, None, None, None, None, subgroup_eligibility
        
        # Compute max-min gap
        parity_value_max_min = float(max(subgroup_mean_aurocs) - min(subgroup_mean_aurocs))
        
        # Compute robust gap (p90-p10)
        if len(subgroup_mean_aurocs) >= 2:
            p90 = float(np.percentile(subgroup_mean_aurocs, 90))
            p10 = float(np.percentile(subgroup_mean_aurocs, 10))
            parity_value_robust = p90 - p10
        else:
            parity_value_robust = None
        
        # Compute bootstrap CI for both gaps
        parity_ci_max_min = {'mean': parity_value_max_min, 'ci_lower': None, 'ci_upper': None}
        parity_ci_robust = {'mean': parity_value_robust, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and len(subgroup_masks) >= 2:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n = len(ground_truth)
            bootstrap_parities_max_min = []
            bootstrap_parities_robust = []
            
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n, size=n, replace=True)
                mean_aurocs_boot = []
                for mask in subgroup_masks:
                    subgroup_indices = np.where(mask)[0]
                    boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                    if boot_subgroup_mask.sum() < 10:
                        continue
                    
                    boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                    aurocs_boot = []
                    
                    for i in range(len(CHEXPERT_CLASSES)):
                        y_true_boot_sub = ground_truth[boot_subgroup_indices, i]
                        y_pred_boot_sub = predictions_probs[boot_subgroup_indices, i]
                        
                        valid_mask_boot = (y_true_boot_sub != -1)
                        if valid_mask_boot.sum() < 2:
                            continue
                        
                        y_true_valid = y_true_boot_sub[valid_mask_boot]
                        y_pred_valid = y_pred_boot_sub[valid_mask_boot]
                        
                        # Check eligibility for bootstrap sample (relaxed threshold: N_pos >= 5, N_neg >= 5)
                        is_eligible_boot, _, _ = check_auroc_eligibility(y_true_valid, min_pos=5, min_neg=5)
                        if not is_eligible_boot:
                            continue
                        
                        try:
                            auroc_boot = roc_auc_score(y_true_valid, y_pred_valid)
                            if not np.isnan(auroc_boot):
                                aurocs_boot.append(auroc_boot)
                        except:
                            continue
                    
                    if len(aurocs_boot) > 0:
                        mean_auroc_boot = np.mean(aurocs_boot)
                        mean_aurocs_boot.append(mean_auroc_boot)
                
                if len(mean_aurocs_boot) >= 2:
                    # Max-min gap
                    parity_boot_max_min = float(max(mean_aurocs_boot) - min(mean_aurocs_boot))
                    if not np.isnan(parity_boot_max_min):
                        bootstrap_parities_max_min.append(parity_boot_max_min)
                    
                    # Robust gap (p90-p10)
                    p90_boot = float(np.percentile(mean_aurocs_boot, 90))
                    p10_boot = float(np.percentile(mean_aurocs_boot, 10))
                    parity_boot_robust = p90_boot - p10_boot
                    if not np.isnan(parity_boot_robust):
                        bootstrap_parities_robust.append(parity_boot_robust)
            
            # Compute CI for max-min gap
            if len(bootstrap_parities_max_min) > 0:
                bootstrap_parities_max_min = np.array(bootstrap_parities_max_min)
                mean_parity_max_min = np.mean(bootstrap_parities_max_min)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_max_min = np.percentile(bootstrap_parities_max_min, lower_percentile)
                ci_upper_max_min = np.percentile(bootstrap_parities_max_min, upper_percentile)
                
                parity_ci_max_min = {
                    'mean': float(mean_parity_max_min),
                    'ci_lower': float(ci_lower_max_min),
                    'ci_upper': float(ci_upper_max_min),
                }
            else:
                logger.debug("No valid bootstrap parities (max-min) computed for AUROC parity (14 labels, sex-age)")
            
            # Compute CI for robust gap
            if len(bootstrap_parities_robust) > 0:
                bootstrap_parities_robust = np.array(bootstrap_parities_robust)
                mean_parity_robust = np.mean(bootstrap_parities_robust)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_robust = np.percentile(bootstrap_parities_robust, lower_percentile)
                ci_upper_robust = np.percentile(bootstrap_parities_robust, upper_percentile)
                
                parity_ci_robust = {
                    'mean': float(mean_parity_robust),
                    'ci_lower': float(ci_lower_robust),
                    'ci_upper': float(ci_upper_robust),
                }
            else:
                logger.debug("No valid bootstrap parities (robust) computed for AUROC parity (14 labels, sex-age)")
        
        # Return bootstrap mean if available, otherwise point estimate
        parity_mean_max_min = parity_ci_max_min.get('mean') if parity_ci_max_min.get('ci_lower') is not None else parity_value_max_min
        parity_mean_robust = parity_ci_robust.get('mean') if parity_ci_robust.get('ci_lower') is not None else parity_value_robust
        
        return (parity_value_max_min, parity_mean_max_min, parity_ci_max_min.get('ci_lower'), parity_ci_max_min.get('ci_upper'),
                parity_value_robust, parity_mean_robust, parity_ci_robust.get('ci_lower'), parity_ci_robust.get('ci_upper'),
                subgroup_eligibility)
    
    def compute_auroc_parity_8_sex_age_with_ci(n_bootstrap, compute_statistics, random_seed):
        """Compute AUROC parity (max-min and p90-p10) of mean AUROC (8 common) across eligible sex_age subgroups."""
        subgroup_mean_aurocs = []
        subgroup_masks = []
        subgroup_names = []
        subgroup_eligibility = {}
        
        for subgroup_name, mask in sex_age_subgroups.items():
            if mask.sum() < 10:
                continue
            
            # Compute mean AUROC for 8 common labels, only including eligible labels
            aurocs = []
            n_valid_labels = 0
            for label in COMMON_LABELS:
                i = CHEXPERT_CLASSES.index(label)
                valid_mask = (ground_truth[mask, i] != -1)
                if valid_mask.sum() < 2:
                    continue
                y_true_sub = ground_truth[mask, i][valid_mask]
                y_pred_sub = predictions_probs[mask, i][valid_mask]
                
                # Check eligibility: N_pos >= 10 AND N_neg >= 10
                is_eligible, n_pos, n_neg = check_auroc_eligibility(y_true_sub, min_pos=10, min_neg=10)
                if not is_eligible:
                    continue
                
                try:
                    auroc = roc_auc_score(y_true_sub, y_pred_sub)
                    if not np.isnan(auroc):
                        aurocs.append(auroc)
                        n_valid_labels += 1
                except:
                    continue
            
            # Only include subgroups with at least some eligible labels
            if len(aurocs) > 0:
                mean_auroc = np.mean(aurocs)
                subgroup_mean_aurocs.append(mean_auroc)
                subgroup_masks.append(mask)
                subgroup_names.append(subgroup_name)
                subgroup_eligibility[subgroup_name] = {'n_valid_labels': n_valid_labels, 'n_total_labels': len(COMMON_LABELS)}
        
        if len(subgroup_mean_aurocs) < 2:
            return None, None, None, None, None, None, None, None, subgroup_eligibility
        
        # Compute max-min gap
        parity_value_max_min = float(max(subgroup_mean_aurocs) - min(subgroup_mean_aurocs))
        
        # Compute robust gap (p90-p10)
        if len(subgroup_mean_aurocs) >= 2:
            p90 = float(np.percentile(subgroup_mean_aurocs, 90))
            p10 = float(np.percentile(subgroup_mean_aurocs, 10))
            parity_value_robust = p90 - p10
        else:
            parity_value_robust = None
        
        # Compute bootstrap CI for both gaps
        parity_ci_max_min = {'mean': parity_value_max_min, 'ci_lower': None, 'ci_upper': None}
        parity_ci_robust = {'mean': parity_value_robust, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and len(subgroup_masks) >= 2:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n = len(ground_truth)
            bootstrap_parities_max_min = []
            bootstrap_parities_robust = []
            
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n, size=n, replace=True)
                mean_aurocs_boot = []
                for mask in subgroup_masks:
                    subgroup_indices = np.where(mask)[0]
                    boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                    if boot_subgroup_mask.sum() < 10:
                        continue
                    
                    boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                    aurocs_boot = []
                    
                    for label in COMMON_LABELS:
                        i = CHEXPERT_CLASSES.index(label)
                        y_true_boot_sub = ground_truth[boot_subgroup_indices, i]
                        y_pred_boot_sub = predictions_probs[boot_subgroup_indices, i]
                        
                        valid_mask_boot = (y_true_boot_sub != -1)
                        if valid_mask_boot.sum() < 2:
                            continue
                        
                        y_true_valid = y_true_boot_sub[valid_mask_boot]
                        y_pred_valid = y_pred_boot_sub[valid_mask_boot]
                        
                        # Check eligibility for bootstrap sample (relaxed threshold: N_pos >= 5, N_neg >= 5)
                        is_eligible_boot, _, _ = check_auroc_eligibility(y_true_valid, min_pos=5, min_neg=5)
                        if not is_eligible_boot:
                            continue
                        
                        try:
                            auroc_boot = roc_auc_score(y_true_valid, y_pred_valid)
                            if not np.isnan(auroc_boot):
                                aurocs_boot.append(auroc_boot)
                        except:
                            continue
                    
                    if len(aurocs_boot) > 0:
                        mean_auroc_boot = np.mean(aurocs_boot)
                        mean_aurocs_boot.append(mean_auroc_boot)
                
                if len(mean_aurocs_boot) >= 2:
                    # Max-min gap
                    parity_boot_max_min = float(max(mean_aurocs_boot) - min(mean_aurocs_boot))
                    if not np.isnan(parity_boot_max_min):
                        bootstrap_parities_max_min.append(parity_boot_max_min)
                    
                    # Robust gap (p90-p10)
                    p90_boot = float(np.percentile(mean_aurocs_boot, 90))
                    p10_boot = float(np.percentile(mean_aurocs_boot, 10))
                    parity_boot_robust = p90_boot - p10_boot
                    if not np.isnan(parity_boot_robust):
                        bootstrap_parities_robust.append(parity_boot_robust)
            
            # Compute CI for max-min gap
            if len(bootstrap_parities_max_min) > 0:
                bootstrap_parities_max_min = np.array(bootstrap_parities_max_min)
                mean_parity_max_min = np.mean(bootstrap_parities_max_min)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_max_min = np.percentile(bootstrap_parities_max_min, lower_percentile)
                ci_upper_max_min = np.percentile(bootstrap_parities_max_min, upper_percentile)
                
                parity_ci_max_min = {
                    'mean': float(mean_parity_max_min),
                    'ci_lower': float(ci_lower_max_min),
                    'ci_upper': float(ci_upper_max_min),
                }
            else:
                logger.debug("No valid bootstrap parities (max-min) computed for AUROC parity (8 common, sex-age)")
            
            # Compute CI for robust gap
            if len(bootstrap_parities_robust) > 0:
                bootstrap_parities_robust = np.array(bootstrap_parities_robust)
                mean_parity_robust = np.mean(bootstrap_parities_robust)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_robust = np.percentile(bootstrap_parities_robust, lower_percentile)
                ci_upper_robust = np.percentile(bootstrap_parities_robust, upper_percentile)
                
                parity_ci_robust = {
                    'mean': float(mean_parity_robust),
                    'ci_lower': float(ci_lower_robust),
                    'ci_upper': float(ci_upper_robust),
                }
            else:
                logger.debug("No valid bootstrap parities (robust) computed for AUROC parity (8 common, sex-age)")
        
        # Return bootstrap mean if available, otherwise point estimate
        parity_mean_max_min = parity_ci_max_min.get('mean') if parity_ci_max_min.get('ci_lower') is not None else parity_value_max_min
        parity_mean_robust = parity_ci_robust.get('mean') if parity_ci_robust.get('ci_lower') is not None else parity_value_robust
        
        return (parity_value_max_min, parity_mean_max_min, parity_ci_max_min.get('ci_lower'), parity_ci_max_min.get('ci_upper'),
                parity_value_robust, parity_mean_robust, parity_ci_robust.get('ci_lower'), parity_ci_robust.get('ci_upper'),
                subgroup_eligibility)
    
    (parity_14_sex_age_val, parity_14_sex_age_mean, parity_14_sex_age_lower, parity_14_sex_age_upper,
     parity_14_sex_age_robust, parity_14_sex_age_robust_mean, parity_14_sex_age_robust_lower, parity_14_sex_age_robust_upper,
     parity_14_sex_age_eligibility) = compute_auroc_parity_14_sex_age_with_ci(
        n_bootstrap, compute_statistics, random_seed
    )
    if parity_14_sex_age_val is not None:
        fairness_results['auroc_parity_14_labels_sex_age'] = parity_14_sex_age_val  # Backward compatibility
        fairness_results['auroc_parity_14_labels_sex_age_ci'] = {
            'mean': parity_14_sex_age_mean if parity_14_sex_age_mean is not None else parity_14_sex_age_val,  # Use bootstrap mean
            'ci_lower': parity_14_sex_age_lower,
            'ci_upper': parity_14_sex_age_upper,
        }
        fairness_results['auroc_parity_14_labels_sex_age_max_min'] = parity_14_sex_age_val
        fairness_results['auroc_parity_14_labels_sex_age_max_min_ci'] = {
            'mean': parity_14_sex_age_mean if parity_14_sex_age_mean is not None else parity_14_sex_age_val,  # Use bootstrap mean
            'ci_lower': parity_14_sex_age_lower,
            'ci_upper': parity_14_sex_age_upper,
        }
        if parity_14_sex_age_robust is not None:
            fairness_results['auroc_parity_14_labels_sex_age_robust'] = parity_14_sex_age_robust
            fairness_results['auroc_parity_14_labels_sex_age_robust_ci'] = {
                'mean': parity_14_sex_age_robust_mean if parity_14_sex_age_robust_mean is not None else parity_14_sex_age_robust,  # Use bootstrap mean
                'ci_lower': parity_14_sex_age_robust_lower,
                'ci_upper': parity_14_sex_age_robust_upper,
            }
        fairness_results['auroc_parity_14_labels_sex_age_eligibility'] = parity_14_sex_age_eligibility
    else:
        fairness_results['auroc_parity_14_labels_sex_age'] = None
        fairness_results['auroc_parity_14_labels_sex_age_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_14_labels_sex_age_max_min'] = None
        fairness_results['auroc_parity_14_labels_sex_age_max_min_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_14_labels_sex_age_robust'] = None
        fairness_results['auroc_parity_14_labels_sex_age_robust_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_14_labels_sex_age_eligibility'] = {}
    
    (parity_8_sex_age_val, parity_8_sex_age_mean, parity_8_sex_age_lower, parity_8_sex_age_upper,
     parity_8_sex_age_robust, parity_8_sex_age_robust_mean, parity_8_sex_age_robust_lower, parity_8_sex_age_robust_upper,
     parity_8_sex_age_eligibility) = compute_auroc_parity_8_sex_age_with_ci(
        n_bootstrap, compute_statistics, random_seed
    )
    if parity_8_sex_age_val is not None:
        fairness_results['auroc_parity_8_common_sex_age'] = parity_8_sex_age_val  # Backward compatibility
        fairness_results['auroc_parity_8_common_sex_age_ci'] = {
            'mean': parity_8_sex_age_mean if parity_8_sex_age_mean is not None else parity_8_sex_age_val,  # Use bootstrap mean
            'ci_lower': parity_8_sex_age_lower,
            'ci_upper': parity_8_sex_age_upper,
        }
        fairness_results['auroc_parity_8_common_sex_age_max_min'] = parity_8_sex_age_val
        fairness_results['auroc_parity_8_common_sex_age_max_min_ci'] = {
            'mean': parity_8_sex_age_mean if parity_8_sex_age_mean is not None else parity_8_sex_age_val,  # Use bootstrap mean
            'ci_lower': parity_8_sex_age_lower,
            'ci_upper': parity_8_sex_age_upper,
        }
        if parity_8_sex_age_robust is not None:
            fairness_results['auroc_parity_8_common_sex_age_robust'] = parity_8_sex_age_robust
            fairness_results['auroc_parity_8_common_sex_age_robust_ci'] = {
                'mean': parity_8_sex_age_robust_mean if parity_8_sex_age_robust_mean is not None else parity_8_sex_age_robust,  # Use bootstrap mean
                'ci_lower': parity_8_sex_age_robust_lower,
                'ci_upper': parity_8_sex_age_robust_upper,
            }
        fairness_results['auroc_parity_8_common_sex_age_eligibility'] = parity_8_sex_age_eligibility
    else:
        fairness_results['auroc_parity_8_common_sex_age'] = None
        fairness_results['auroc_parity_8_common_sex_age_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_8_common_sex_age_max_min'] = None
        fairness_results['auroc_parity_8_common_sex_age_max_min_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_8_common_sex_age_robust'] = None
        fairness_results['auroc_parity_8_common_sex_age_robust_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['auroc_parity_8_common_sex_age_eligibility'] = {}
    
    # Compute F1 scores per label and weighted F1 for each demographic group
    # Start with sex_race_subgroups
    for subgroup_name, mask in sex_race_subgroups.items():
        if mask.sum() < 10:
            continue
        
        # Compute F1 per label for this subgroup
        subgroup_f1 = compute_f1_per_label(
            ground_truth[mask],
            binary_preds[mask],
            CHEXPERT_CLASSES
        )
        
        # Compute bootstrap CIs for F1 scores if requested
        f1_per_label_with_ci = {}
        if compute_statistics:
            for i, class_name in enumerate(CHEXPERT_CLASSES):
                if class_name in subgroup_f1 and not np.isnan(subgroup_f1[class_name]):
                    valid_mask = (ground_truth[mask, i] != -1)
                    if valid_mask.sum() >= 2:
                        y_true_sub = ground_truth[mask, i][valid_mask]
                        y_pred_sub = binary_preds[mask, i][valid_mask]
                        
                        def f1_fn(y_t, y_p):
                            try:
                                from sklearn.metrics import f1_score
                                return f1_score(y_t, y_p, zero_division=0)
                            except:
                                return np.nan
                        
                        mean_f1, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                            y_true_sub,
                            y_pred_sub,
                            f1_fn,
                            n_bootstrap=n_bootstrap,
                            confidence_level=0.95,
                            random_seed=random_seed,
                        )
                        
                        f1_per_label_with_ci[class_name] = {
                            'mean': mean_f1 if not np.isnan(mean_f1) else None,
                            'ci_lower': ci_lower if not np.isnan(ci_lower) else None,
                            'ci_upper': ci_upper if not np.isnan(ci_upper) else None,
                        }
                    else:
                        f1_per_label_with_ci[class_name] = {
                            'mean': float(subgroup_f1[class_name]) if not np.isnan(subgroup_f1[class_name]) else None,
                            'ci_lower': None,
                            'ci_upper': None,
                        }
                else:
                    f1_per_label_with_ci[class_name] = {
                        'mean': None,
                        'ci_lower': None,
                        'ci_upper': None,
                    }
        else:
            f1_per_label_with_ci = {
                k: {'mean': float(v) if not np.isnan(v) else None, 'ci_lower': None, 'ci_upper': None}
                for k, v in subgroup_f1.items()
            }
        
        # Compute weighted F1 for 14 labels and 8 labels
        weighted_f1_14 = compute_weighted_f1(
            subgroup_f1,
            ground_truth[mask],
            CHEXPERT_CLASSES,
            label_subset=None
        )
        weighted_f1_8 = compute_weighted_f1(
            subgroup_f1,
            ground_truth[mask],
            CHEXPERT_CLASSES,
            label_subset=COMMON_LABELS
        )
        
        # Compute bootstrap CI for weighted F1 if requested
        weighted_f1_14_ci = {'mean': weighted_f1_14, 'ci_lower': None, 'ci_upper': None}
        weighted_f1_8_ci = {'mean': weighted_f1_8, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and not np.isnan(weighted_f1_14):
            # Bootstrap weighted F1 by resampling samples
            def weighted_f1_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=None)
                except:
                    return np.nan
            
            mean_wf1_14, ci_lower_14, ci_upper_14, _ = bootstrap_ci_for_metric(
                ground_truth[mask],
                binary_preds[mask],
                weighted_f1_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            weighted_f1_14_ci = {
                'mean': mean_wf1_14 if not np.isnan(mean_wf1_14) else weighted_f1_14,
                'ci_lower': ci_lower_14 if not np.isnan(ci_lower_14) else None,
                'ci_upper': ci_upper_14 if not np.isnan(ci_upper_14) else None,
            }
        
        if compute_statistics and not np.isnan(weighted_f1_8):
            def weighted_f1_8_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=COMMON_LABELS)
                except:
                    return np.nan
            
            mean_wf1_8, ci_lower_8, ci_upper_8, _ = bootstrap_ci_for_metric(
                ground_truth[mask],
                binary_preds[mask],
                weighted_f1_8_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            weighted_f1_8_ci = {
                'mean': mean_wf1_8 if not np.isnan(mean_wf1_8) else weighted_f1_8,
                'ci_lower': ci_lower_8 if not np.isnan(ci_lower_8) else None,
                'ci_upper': ci_upper_8 if not np.isnan(ci_upper_8) else None,
            }
        
        # Store in fairness_results (merge with existing AUROC data)
        if subgroup_name in fairness_results['sex_race_subgroups']:
            fairness_results['sex_race_subgroups'][subgroup_name]['f1_per_label'] = f1_per_label_with_ci
            fairness_results['sex_race_subgroups'][subgroup_name]['weighted_f1_14_labels'] = weighted_f1_14_ci
            fairness_results['sex_race_subgroups'][subgroup_name]['weighted_f1_8_common'] = weighted_f1_8_ci
        else:
            # Create new entry if it doesn't exist
            fairness_results['sex_race_subgroups'][subgroup_name] = {
                'f1_per_label': f1_per_label_with_ci,
                'weighted_f1_14_labels': weighted_f1_14_ci,
                'weighted_f1_8_common': weighted_f1_8_ci,
            }
    
    # Compute F1 scores for sex_age_subgroups
    for subgroup_name, mask in sex_age_subgroups.items():
        if mask.sum() < 10:
            continue
        
        subgroup_f1 = compute_f1_per_label(
            ground_truth[mask],
            binary_preds[mask],
            CHEXPERT_CLASSES
        )
        
        # Compute bootstrap CIs for F1 scores if requested (same logic as sex_race_subgroups)
        f1_per_label_with_ci = {}
        if compute_statistics:
            for i, class_name in enumerate(CHEXPERT_CLASSES):
                if class_name in subgroup_f1 and not np.isnan(subgroup_f1[class_name]):
                    valid_mask = (ground_truth[mask, i] != -1)
                    if valid_mask.sum() >= 2:
                        y_true_sub = ground_truth[mask, i][valid_mask]
                        y_pred_sub = binary_preds[mask, i][valid_mask]
                        
                        def f1_fn(y_t, y_p):
                            try:
                                from sklearn.metrics import f1_score
                                return f1_score(y_t, y_p, zero_division=0)
                            except:
                                return np.nan
                        
                        mean_f1, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                            y_true_sub,
                            y_pred_sub,
                            f1_fn,
                            n_bootstrap=n_bootstrap,
                            confidence_level=0.95,
                            random_seed=random_seed,
                        )
                        
                        f1_per_label_with_ci[class_name] = {
                            'mean': mean_f1 if not np.isnan(mean_f1) else None,
                            'ci_lower': ci_lower if not np.isnan(ci_lower) else None,
                            'ci_upper': ci_upper if not np.isnan(ci_upper) else None,
                        }
                    else:
                        f1_per_label_with_ci[class_name] = {
                            'mean': float(subgroup_f1[class_name]) if not np.isnan(subgroup_f1[class_name]) else None,
                            'ci_lower': None,
                            'ci_upper': None,
                        }
                else:
                    f1_per_label_with_ci[class_name] = {
                        'mean': None,
                        'ci_lower': None,
                        'ci_upper': None,
                    }
        else:
            f1_per_label_with_ci = {
                k: {'mean': float(v) if not np.isnan(v) else None, 'ci_lower': None, 'ci_upper': None}
                for k, v in subgroup_f1.items()
            }
        
        weighted_f1_14 = compute_weighted_f1(
            subgroup_f1,
            ground_truth[mask],
            CHEXPERT_CLASSES,
            label_subset=None
        )
        weighted_f1_8 = compute_weighted_f1(
            subgroup_f1,
            ground_truth[mask],
            CHEXPERT_CLASSES,
            label_subset=COMMON_LABELS
        )
        
        # Compute bootstrap CI for weighted F1 if requested
        weighted_f1_14_ci = {'mean': weighted_f1_14, 'ci_lower': None, 'ci_upper': None}
        weighted_f1_8_ci = {'mean': weighted_f1_8, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and not np.isnan(weighted_f1_14):
            def weighted_f1_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=None)
                except:
                    return np.nan
            
            mean_wf1_14, ci_lower_14, ci_upper_14, _ = bootstrap_ci_for_metric(
                ground_truth[mask],
                binary_preds[mask],
                weighted_f1_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            weighted_f1_14_ci = {
                'mean': mean_wf1_14 if not np.isnan(mean_wf1_14) else weighted_f1_14,
                'ci_lower': ci_lower_14 if not np.isnan(ci_lower_14) else None,
                'ci_upper': ci_upper_14 if not np.isnan(ci_upper_14) else None,
            }
        
        if compute_statistics and not np.isnan(weighted_f1_8):
            def weighted_f1_8_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=COMMON_LABELS)
                except:
                    return np.nan
            
            mean_wf1_8, ci_lower_8, ci_upper_8, _ = bootstrap_ci_for_metric(
                ground_truth[mask],
                binary_preds[mask],
                weighted_f1_8_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            weighted_f1_8_ci = {
                'mean': mean_wf1_8 if not np.isnan(mean_wf1_8) else weighted_f1_8,
                'ci_lower': ci_lower_8 if not np.isnan(ci_lower_8) else None,
                'ci_upper': ci_upper_8 if not np.isnan(ci_upper_8) else None,
            }
        
        if subgroup_name in fairness_results['sex_age_subgroups']:
            fairness_results['sex_age_subgroups'][subgroup_name]['f1_per_label'] = f1_per_label_with_ci
            fairness_results['sex_age_subgroups'][subgroup_name]['weighted_f1_14_labels'] = weighted_f1_14_ci
            fairness_results['sex_age_subgroups'][subgroup_name]['weighted_f1_8_common'] = weighted_f1_8_ci
        else:
            fairness_results['sex_age_subgroups'][subgroup_name] = {
                'f1_per_label': f1_per_label_with_ci,
                'weighted_f1_14_labels': weighted_f1_14_ci,
                'weighted_f1_8_common': weighted_f1_8_ci,
            }
    
    # Compute F1 scores for intersectional_subgroups
    for subgroup_name, mask in intersectional_subgroups.items():
        if mask.sum() < 10:
            continue
        
        subgroup_f1 = compute_f1_per_label(
            ground_truth[mask],
            binary_preds[mask],
            CHEXPERT_CLASSES
        )
        
        # Compute bootstrap CIs for F1 scores if requested (same logic as sex_race_subgroups)
        f1_per_label_with_ci = {}
        if compute_statistics:
            for i, class_name in enumerate(CHEXPERT_CLASSES):
                if class_name in subgroup_f1 and not np.isnan(subgroup_f1[class_name]):
                    valid_mask = (ground_truth[mask, i] != -1)
                    if valid_mask.sum() >= 2:
                        y_true_sub = ground_truth[mask, i][valid_mask]
                        y_pred_sub = binary_preds[mask, i][valid_mask]
                        
                        def f1_fn(y_t, y_p):
                            try:
                                from sklearn.metrics import f1_score
                                return f1_score(y_t, y_p, zero_division=0)
                            except:
                                return np.nan
                        
                        mean_f1, ci_lower, ci_upper, _ = bootstrap_ci_for_metric(
                            y_true_sub,
                            y_pred_sub,
                            f1_fn,
                            n_bootstrap=n_bootstrap,
                            confidence_level=0.95,
                            random_seed=random_seed,
                        )
                        
                        f1_per_label_with_ci[class_name] = {
                            'mean': mean_f1 if not np.isnan(mean_f1) else None,
                            'ci_lower': ci_lower if not np.isnan(ci_lower) else None,
                            'ci_upper': ci_upper if not np.isnan(ci_upper) else None,
                        }
                    else:
                        f1_per_label_with_ci[class_name] = {
                            'mean': float(subgroup_f1[class_name]) if not np.isnan(subgroup_f1[class_name]) else None,
                            'ci_lower': None,
                            'ci_upper': None,
                        }
                else:
                    f1_per_label_with_ci[class_name] = {
                        'mean': None,
                        'ci_lower': None,
                        'ci_upper': None,
                    }
        else:
            f1_per_label_with_ci = {
                k: {'mean': float(v) if not np.isnan(v) else None, 'ci_lower': None, 'ci_upper': None}
                for k, v in subgroup_f1.items()
            }
        
        weighted_f1_14 = compute_weighted_f1(
            subgroup_f1,
            ground_truth[mask],
            CHEXPERT_CLASSES,
            label_subset=None
        )
        weighted_f1_8 = compute_weighted_f1(
            subgroup_f1,
            ground_truth[mask],
            CHEXPERT_CLASSES,
            label_subset=COMMON_LABELS
        )
        
        # Compute bootstrap CI for weighted F1 if requested
        weighted_f1_14_ci = {'mean': weighted_f1_14, 'ci_lower': None, 'ci_upper': None}
        weighted_f1_8_ci = {'mean': weighted_f1_8, 'ci_lower': None, 'ci_upper': None}
        
        if compute_statistics and not np.isnan(weighted_f1_14):
            def weighted_f1_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=None)
                except:
                    return np.nan
            
            mean_wf1_14, ci_lower_14, ci_upper_14, _ = bootstrap_ci_for_metric(
                ground_truth[mask],
                binary_preds[mask],
                weighted_f1_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            weighted_f1_14_ci = {
                'mean': mean_wf1_14 if not np.isnan(mean_wf1_14) else weighted_f1_14,
                'ci_lower': ci_lower_14 if not np.isnan(ci_lower_14) else None,
                'ci_upper': ci_upper_14 if not np.isnan(ci_upper_14) else None,
            }
        
        if compute_statistics and not np.isnan(weighted_f1_8):
            def weighted_f1_8_fn(y_true_boot, y_pred_boot):
                try:
                    f1_boot = compute_f1_per_label(y_true_boot, y_pred_boot, CHEXPERT_CLASSES)
                    return compute_weighted_f1(f1_boot, y_true_boot, CHEXPERT_CLASSES, label_subset=COMMON_LABELS)
                except:
                    return np.nan
            
            mean_wf1_8, ci_lower_8, ci_upper_8, _ = bootstrap_ci_for_metric(
                ground_truth[mask],
                binary_preds[mask],
                weighted_f1_8_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_seed=random_seed,
            )
            weighted_f1_8_ci = {
                'mean': mean_wf1_8 if not np.isnan(mean_wf1_8) else weighted_f1_8,
                'ci_lower': ci_lower_8 if not np.isnan(ci_lower_8) else None,
                'ci_upper': ci_upper_8 if not np.isnan(ci_upper_8) else None,
            }
        
        if subgroup_name in fairness_results['intersectional_subgroups']:
            fairness_results['intersectional_subgroups'][subgroup_name]['f1_per_label'] = f1_per_label_with_ci
            fairness_results['intersectional_subgroups'][subgroup_name]['weighted_f1_14_labels'] = weighted_f1_14_ci
            fairness_results['intersectional_subgroups'][subgroup_name]['weighted_f1_8_common'] = weighted_f1_8_ci
        else:
            fairness_results['intersectional_subgroups'][subgroup_name] = {
                'f1_per_label': f1_per_label_with_ci,
                'weighted_f1_14_labels': weighted_f1_14_ci,
                'weighted_f1_8_common': weighted_f1_8_ci,
            }
    
    # Identify worst-performing subgroup (lowest weighted F1 across all intersectional subgroups)
    worst_subgroup = None
    worst_f1_score = float('inf')
    
    for subgroup_name, subgroup_data in fairness_results['intersectional_subgroups'].items():
        if 'weighted_f1_14_labels' in subgroup_data and subgroup_data['weighted_f1_14_labels'] is not None:
            f1_value = subgroup_data['weighted_f1_14_labels']
            # Handle both old format (float) and new format (dict with 'mean')
            if isinstance(f1_value, dict):
                f1_score = f1_value.get('mean')
            elif isinstance(f1_value, (int, float)) and not np.isnan(f1_value):
                f1_score = f1_value
            else:
                continue
            
            if f1_score is not None and not np.isnan(f1_score) and f1_score < worst_f1_score:
                worst_f1_score = f1_score
                worst_subgroup = subgroup_name
    
    if worst_subgroup is not None:
        # Get CI from the worst subgroup's data
        worst_subgroup_data = fairness_results['intersectional_subgroups'].get(worst_subgroup, {})
        worst_f1_ci = None
        if 'weighted_f1_14_labels' in worst_subgroup_data:
            f1_value = worst_subgroup_data['weighted_f1_14_labels']
            if isinstance(f1_value, dict):
                worst_f1_ci = {
                    'mean': f1_value.get('mean'),
                    'ci_lower': f1_value.get('ci_lower'),
                    'ci_upper': f1_value.get('ci_upper'),
                }
        
        fairness_results['worst_performing_subgroup'] = {
            'subgroup_name': worst_subgroup,
            'weighted_f1_14_labels': float(worst_f1_score) if worst_f1_score != float('inf') else None,
            'weighted_f1_14_labels_ci': worst_f1_ci if worst_f1_ci else {
                'mean': float(worst_f1_score) if worst_f1_score != float('inf') else None,
                'ci_lower': None,
                'ci_upper': None,
            }
        }
    
    # Compute performance gap per demographic dimension WITH BOOTSTRAP CI
    # Performance gap = F1(best group) - F1(worst group) for each dimension
    logger.info("Computing performance gaps with bootstrap CIs...")
    
    # Helper function to compute performance gap with bootstrap CI
    def compute_performance_gap_with_ci(dimension_name, subgroup_f1_dict, subgroup_masks_dict, n_bootstrap, compute_statistics, random_seed):
        """Compute performance gap (max - min) across demographic groups with bootstrap CI.
        
        Args:
            dimension_name: Name of dimension ('sex', 'race', 'age')
            subgroup_f1_dict: Dict mapping subgroup names to F1 scores
            subgroup_masks_dict: Dict mapping subgroup names to masks
            n_bootstrap: Number of bootstrap samples
            compute_statistics: Whether to compute CI
            random_seed: Random seed
        """
        # Group F1 scores by dimension value
        dimension_f1_scores = {}
        dimension_masks = {}  # Maps dimension value to list of masks
        
        for subgroup_name, f1_value in subgroup_f1_dict.items():
            if f1_value is None or np.isnan(f1_value):
                continue
            
            # Extract dimension value from subgroup name
            parts = subgroup_name.split('_')
            if dimension_name == 'sex':
                dim_value = parts[0]  # "Male_Asian" -> "Male"
            elif dimension_name == 'race':
                dim_value = parts[1] if len(parts) > 1 else None  # "Male_Asian" -> "Asian"
            elif dimension_name == 'age':
                dim_value = parts[0]  # "18-40_Male" -> "18-40"
            else:
                continue
            
            if dim_value is None:
                continue
            
            if dim_value not in dimension_f1_scores:
                dimension_f1_scores[dim_value] = []
                dimension_masks[dim_value] = []
            
            dimension_f1_scores[dim_value].append(f1_value)
            if subgroup_name in subgroup_masks_dict:
                dimension_masks[dim_value].append(subgroup_masks_dict[subgroup_name])
        
        # Average F1 per dimension group
        dimension_avg_f1 = {}
        for dim_value, f1_list in dimension_f1_scores.items():
            if len(f1_list) > 0:
                dimension_avg_f1[dim_value] = np.mean(f1_list)
        
        if len(dimension_avg_f1) < 2:
            return None, None, None, None, None, None, None, None
        
        gap_value = float(max(dimension_avg_f1.values()) - min(dimension_avg_f1.values()))
        
        # Compute robust gap from point estimates
        if len(dimension_avg_f1) >= 2:
            f1_values = list(dimension_avg_f1.values())
            p90 = float(np.percentile(f1_values, 90))
            p10 = float(np.percentile(f1_values, 10))
            gap_value_robust = p90 - p10
        else:
            gap_value_robust = None
        
        # Compute bootstrap CI for both gaps
        gap_ci_max_min = {'mean': gap_value, 'ci_lower': None, 'ci_upper': None}
        gap_ci_robust = {'mean': gap_value_robust, 'ci_lower': None, 'ci_upper': None}
        if compute_statistics and len(dimension_masks) >= 2:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            n = len(ground_truth)
            bootstrap_gaps_max_min = []
            bootstrap_gaps_robust = []
            
            for _ in range(n_bootstrap):
                # Bootstrap resample indices
                boot_indices = np.random.choice(n, size=n, replace=True)
                
                # Compute weighted F1 for each dimension group on bootstrap sample
                dim_f1_boot = {}
                for dim_value, masks_list in dimension_masks.items():
                    f1_scores_boot = []
                    for mask in masks_list:
                        # Get original indices that belong to this subgroup
                        subgroup_indices = np.where(mask)[0]
                        # Find which bootstrap indices correspond to subgroup samples
                        boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                        if boot_subgroup_mask.sum() < 10:
                            continue
                        
                        # Get bootstrap data for this subgroup
                        boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                        
                        # Compute weighted F1 on bootstrap sample
                        try:
                            f1_boot = compute_f1_per_label(
                                ground_truth[boot_subgroup_indices],
                                binary_preds[boot_subgroup_indices],
                                CHEXPERT_CLASSES
                            )
                            weighted_f1_boot = compute_weighted_f1(
                                f1_boot,
                                ground_truth[boot_subgroup_indices],
                                CHEXPERT_CLASSES,
                                label_subset=None
                            )
                            if not np.isnan(weighted_f1_boot):
                                f1_scores_boot.append(weighted_f1_boot)
                        except:
                            continue
                    
                    if len(f1_scores_boot) > 0:
                        dim_f1_boot[dim_value] = np.mean(f1_scores_boot)
                
                if len(dim_f1_boot) >= 2:
                    f1_values_boot = list(dim_f1_boot.values())
                    
                    # Max-min gap
                    gap_boot_max_min = float(max(f1_values_boot) - min(f1_values_boot))
                    if not np.isnan(gap_boot_max_min):
                        bootstrap_gaps_max_min.append(gap_boot_max_min)
                    
                    # Robust gap (p90-p10)
                    p90_boot = float(np.percentile(f1_values_boot, 90))
                    p10_boot = float(np.percentile(f1_values_boot, 10))
                    gap_boot_robust = p90_boot - p10_boot
                    if not np.isnan(gap_boot_robust):
                        bootstrap_gaps_robust.append(gap_boot_robust)
            
            # Compute CI for max-min gap
            if len(bootstrap_gaps_max_min) > 0:
                bootstrap_gaps_max_min = np.array(bootstrap_gaps_max_min)
                mean_gap_max_min = np.mean(bootstrap_gaps_max_min)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_max_min = np.percentile(bootstrap_gaps_max_min, lower_percentile)
                ci_upper_max_min = np.percentile(bootstrap_gaps_max_min, upper_percentile)
                
                # Use bootstrap mean for consistency with other metrics (AUROC, AUPRC, FPR)
                gap_ci_max_min = {
                    'mean': float(mean_gap_max_min),
                    'ci_lower': float(ci_lower_max_min),
                    'ci_upper': float(ci_upper_max_min),
                }
            else:
                logger.debug(f"No valid bootstrap gaps (max-min) computed for {dimension_name}")
            
            # Compute CI for robust gap
            if len(bootstrap_gaps_robust) > 0:
                bootstrap_gaps_robust = np.array(bootstrap_gaps_robust)
                mean_gap_robust = np.mean(bootstrap_gaps_robust)
                alpha = 0.05
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower_robust = np.percentile(bootstrap_gaps_robust, lower_percentile)
                ci_upper_robust = np.percentile(bootstrap_gaps_robust, upper_percentile)
                
                # Use bootstrap mean for consistency with other metrics (AUROC, AUPRC, FPR)
                gap_ci_robust = {
                    'mean': float(mean_gap_robust),
                    'ci_lower': float(ci_lower_robust),
                    'ci_upper': float(ci_upper_robust),
                }
            else:
                logger.debug(f"No valid bootstrap gaps (robust) computed for {dimension_name}")
        
        # Return bootstrap mean if available, otherwise point estimate
        gap_mean_max_min = gap_ci_max_min.get('mean') if gap_ci_max_min.get('ci_lower') is not None else gap_value
        gap_mean_robust = gap_ci_robust.get('mean') if gap_ci_robust.get('ci_lower') is not None else gap_value_robust
        
        return (gap_value, gap_mean_max_min, gap_ci_max_min.get('ci_lower'), gap_ci_max_min.get('ci_upper'),
                gap_value_robust, gap_mean_robust, gap_ci_robust.get('ci_lower'), gap_ci_robust.get('ci_upper'))
    
    # 1. Sex dimension
    sex_f1_scores = {}
    sex_subgroup_masks = {}
    for subgroup_name, subgroup_data in fairness_results['sex_race_subgroups'].items():
        if isinstance(subgroup_data, dict):
            if 'weighted_f1_14_labels' in subgroup_data and subgroup_data['weighted_f1_14_labels'] is not None:
                f1_value = subgroup_data['weighted_f1_14_labels']
                if isinstance(f1_value, dict):
                    f1_score = f1_value.get('mean')
                elif isinstance(f1_value, (int, float)) and not np.isnan(f1_value):
                    f1_score = f1_value
                else:
                    continue
                
                if f1_score is not None and not np.isnan(f1_score):
                    sex_f1_scores[subgroup_name] = f1_score
                    if subgroup_name in sex_race_subgroups:
                        sex_subgroup_masks[subgroup_name] = sex_race_subgroups[subgroup_name]
    
    (gap_sex_val, gap_sex_mean, gap_sex_lower, gap_sex_upper,
     gap_sex_robust, gap_sex_robust_mean, gap_sex_robust_lower, gap_sex_robust_upper) = compute_performance_gap_with_ci(
        'sex', sex_f1_scores, sex_subgroup_masks, n_bootstrap, compute_statistics, random_seed
    )
    if gap_sex_val is not None:
        fairness_results['performance_gap_sex'] = gap_sex_val  # Backward compatibility
        fairness_results['performance_gap_sex_ci'] = {
            'mean': gap_sex_mean if gap_sex_mean is not None else gap_sex_val,  # Use bootstrap mean
            'ci_lower': gap_sex_lower,
            'ci_upper': gap_sex_upper,
        }
        fairness_results['performance_gap_sex_max_min'] = gap_sex_val
        fairness_results['performance_gap_sex_max_min_ci'] = {
            'mean': gap_sex_mean if gap_sex_mean is not None else gap_sex_val,  # Use bootstrap mean
            'ci_lower': gap_sex_lower,
            'ci_upper': gap_sex_upper,
        }
        if gap_sex_robust is not None:
            fairness_results['performance_gap_sex_robust'] = gap_sex_robust
            fairness_results['performance_gap_sex_robust_ci'] = {
                'mean': gap_sex_robust_mean if gap_sex_robust_mean is not None else gap_sex_robust,  # Use bootstrap mean
                'ci_lower': gap_sex_robust_lower,
                'ci_upper': gap_sex_robust_upper,
            }
    else:
        fairness_results['performance_gap_sex'] = None
        fairness_results['performance_gap_sex_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['performance_gap_sex_max_min'] = None
        fairness_results['performance_gap_sex_max_min_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['performance_gap_sex_robust'] = None
        fairness_results['performance_gap_sex_robust_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
    
    # 2. Race/Ethnicity dimension
    race_f1_scores = {}
    race_subgroup_masks = {}
    for subgroup_name, subgroup_data in fairness_results['sex_race_subgroups'].items():
        if isinstance(subgroup_data, dict):
            if 'weighted_f1_14_labels' in subgroup_data and subgroup_data['weighted_f1_14_labels'] is not None:
                f1_value = subgroup_data['weighted_f1_14_labels']
                if isinstance(f1_value, dict):
                    f1_score = f1_value.get('mean')
                elif isinstance(f1_value, (int, float)) and not np.isnan(f1_value):
                    f1_score = f1_value
                else:
                    continue
                
                if f1_score is not None and not np.isnan(f1_score):
                    race_f1_scores[subgroup_name] = f1_score
                    if subgroup_name in sex_race_subgroups:
                        race_subgroup_masks[subgroup_name] = sex_race_subgroups[subgroup_name]
    
    (gap_race_val, gap_race_mean, gap_race_lower, gap_race_upper,
     gap_race_robust, gap_race_robust_mean, gap_race_robust_lower, gap_race_robust_upper) = compute_performance_gap_with_ci(
        'race', race_f1_scores, race_subgroup_masks, n_bootstrap, compute_statistics, random_seed
    )
    if gap_race_val is not None:
        fairness_results['performance_gap_race'] = gap_race_val  # Backward compatibility
        fairness_results['performance_gap_race_ci'] = {
            'mean': gap_race_mean if gap_race_mean is not None else gap_race_val,  # Use bootstrap mean
            'ci_lower': gap_race_lower,
            'ci_upper': gap_race_upper,
        }
        fairness_results['performance_gap_race_max_min'] = gap_race_val
        fairness_results['performance_gap_race_max_min_ci'] = {
            'mean': gap_race_mean if gap_race_mean is not None else gap_race_val,  # Use bootstrap mean
            'ci_lower': gap_race_lower,
            'ci_upper': gap_race_upper,
        }
        if gap_race_robust is not None:
            fairness_results['performance_gap_race_robust'] = gap_race_robust
            fairness_results['performance_gap_race_robust_ci'] = {
                'mean': gap_race_robust_mean if gap_race_robust_mean is not None else gap_race_robust,  # Use bootstrap mean
                'ci_lower': gap_race_robust_lower,
                'ci_upper': gap_race_robust_upper,
            }
    else:
        fairness_results['performance_gap_race'] = None
        fairness_results['performance_gap_race_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['performance_gap_race_max_min'] = None
        fairness_results['performance_gap_race_max_min_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['performance_gap_race_robust'] = None
        fairness_results['performance_gap_race_robust_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
    
    # 3. Age dimension (from sex_age_subgroups)
    age_f1_scores = {}
    age_subgroup_masks = {}
    for subgroup_name, subgroup_data in fairness_results['sex_age_subgroups'].items():
        if isinstance(subgroup_data, dict):
            if 'weighted_f1_14_labels' in subgroup_data and subgroup_data['weighted_f1_14_labels'] is not None:
                f1_value = subgroup_data['weighted_f1_14_labels']
                if isinstance(f1_value, dict):
                    f1_score = f1_value.get('mean')
                elif isinstance(f1_value, (int, float)) and not np.isnan(f1_value):
                    f1_score = f1_value
                else:
                    continue
                
                if f1_score is not None and not np.isnan(f1_score):
                    age_f1_scores[subgroup_name] = f1_score
                    if subgroup_name in sex_age_subgroups:
                        age_subgroup_masks[subgroup_name] = sex_age_subgroups[subgroup_name]
    
    (gap_age_val, gap_age_mean, gap_age_lower, gap_age_upper,
     gap_age_robust, gap_age_robust_mean, gap_age_robust_lower, gap_age_robust_upper) = compute_performance_gap_with_ci(
        'age', age_f1_scores, age_subgroup_masks, n_bootstrap, compute_statistics, random_seed
    )
    if gap_age_val is not None:
        fairness_results['performance_gap_age'] = gap_age_val  # Backward compatibility
        fairness_results['performance_gap_age_ci'] = {
            'mean': gap_age_mean if gap_age_mean is not None else gap_age_val,  # Use bootstrap mean
            'ci_lower': gap_age_lower,
            'ci_upper': gap_age_upper,
        }
        fairness_results['performance_gap_age_max_min'] = gap_age_val
        fairness_results['performance_gap_age_max_min_ci'] = {
            'mean': gap_age_mean if gap_age_mean is not None else gap_age_val,  # Use bootstrap mean
            'ci_lower': gap_age_lower,
            'ci_upper': gap_age_upper,
        }
        if gap_age_robust is not None:
            fairness_results['performance_gap_age_robust'] = gap_age_robust
            fairness_results['performance_gap_age_robust_ci'] = {
                'mean': gap_age_robust_mean if gap_age_robust_mean is not None else gap_age_robust,  # Use bootstrap mean
                'ci_lower': gap_age_robust_lower,
                'ci_upper': gap_age_robust_upper,
            }
    else:
        fairness_results['performance_gap_age'] = None
        fairness_results['performance_gap_age_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['performance_gap_age_max_min'] = None
        fairness_results['performance_gap_age_max_min_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
        fairness_results['performance_gap_age_robust'] = None
        fairness_results['performance_gap_age_robust_ci'] = {
            'mean': None,
            'ci_lower': None,
            'ci_upper': None,
        }
    
    # Compute underdiagnosis rate (FPR for "No Finding") for all subgroup levels
    from downstream_eval_chest.statistics import compute_fpr_no_finding
    
    def compute_fpr_with_ci(y_true_nf, y_pred_nf, compute_statistics, n_bootstrap, random_seed, min_neg=20):
        """Helper function to compute FPR with Jeffreys smoothing and optional bootstrap CI.
        
        Args:
            y_true_nf: True labels for "No Finding" [N]
            y_pred_nf: Binary predictions for "No Finding" [N]
            compute_statistics: Whether to compute CI
            n_bootstrap: Number of bootstrap samples
            random_seed: Random seed
            min_neg: Minimum number of negatives (has finding, GT=0) required for eligibility
        
        Returns:
            Tuple of (fpr, fpr_ci) or (None, None) if insufficient data
        """
        # Check eligibility: N_neg >= min_neg
        is_eligible, n_neg = check_fpr_eligibility(y_true_nf, min_neg=min_neg)
        if not is_eligible:
            return None, None
        
        # Filter out uncertain labels
        valid_mask = y_true_nf != -1
        if valid_mask.sum() < 2:
            return None, None
        
        y_true_valid = y_true_nf[valid_mask]
        y_pred_valid = y_pred_nf[valid_mask]
        
        # Compute FPR with Jeffreys smoothing
        fpr = compute_fpr_jeffreys(y_true_valid, y_pred_valid)
        if fpr is None:
            return None, None
        
        # Compute bootstrap CI if requested
        fpr_ci = None
        if compute_statistics:
            def fpr_fn_jeffreys(y_t, y_p):
                """Compute FPR with Jeffreys smoothing."""
                fpr_val = compute_fpr_jeffreys(y_t, y_p)
                return fpr_val if fpr_val is not None else np.nan
            
            def validate_fpr_bootstrap(y_t_boot):
                """Validate bootstrap sample has enough negatives for FPR."""
                is_eligible_boot, _ = check_fpr_eligibility(y_t_boot, min_neg=10)  # Relaxed threshold for bootstrap
                return is_eligible_boot
            
            try:
                mean_fpr, ci_lower, ci_upper, n_valid = bootstrap_ci_for_metric(
                    y_true_valid,
                    y_pred_valid,
                    fpr_fn_jeffreys,
                    n_bootstrap=n_bootstrap,
                    confidence_level=0.95,
                    random_seed=random_seed,
                    validate_bootstrap=validate_fpr_bootstrap,
                )
                if not (np.isnan(mean_fpr) or np.isnan(ci_lower) or np.isnan(ci_upper)):
                    fpr_ci = {
                        'mean': mean_fpr if not np.isnan(mean_fpr) else fpr,  # Bootstrap mean for CI
                        'point_estimate': fpr,  # Point estimate computed on full dataset
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n_valid_bootstrap': n_valid,
                        'n_negatives': n_neg,
                    }
                else:
                    # Fallback to point estimate
                    fpr_ci = {
                        'mean': fpr,
                        'point_estimate': fpr,
                        'ci_lower': None,
                        'ci_upper': None,
                        'n_valid_bootstrap': 0,
                        'n_negatives': n_neg,
                    }
            except Exception as e:
                logger.debug(f"Bootstrap CI for FPR failed: {e}")
                fpr_ci = {
                    'mean': fpr,
                    'ci_lower': None,
                    'ci_upper': None,
                    'n_valid_bootstrap': 0,
                    'n_negatives': n_neg,
                }
        else:
            fpr_ci = {
                'mean': fpr,
                'ci_lower': None,
                'ci_upper': None,
                'n_negatives': n_neg,
            }
        
        return fpr, fpr_ci
    
    # Initialize underdiagnosis rates structure
    underdiagnosis_rates = {
        'level_1': {'age': {}, 'sex': {}, 'ethnicity': {}},
        'level_2': {'age_sex': {}, 'age_ethnicity': {}, 'sex_ethnicity': {}},
        'level_3': {'age_sex_ethnicity': {}}
    }
    
    # Get binary predictions and labels for "No Finding"
    y_true_nf = ground_truth[:, no_finding_idx]
    y_pred_nf = binary_preds[:, no_finding_idx]
    
    # At this point, all arrays should already have the same length (validated above)
    # No need to truncate - if there's a mismatch, it's a data error that should be caught earlier
    
    # 1-level subgroups: Age
    unique_ages = ['18-40', '40-60', '60-80', '80+']
    for ag in unique_ages:
        mask = np.array(age_group == ag)
        if mask.sum() >= 10:
            y_true_sub = y_true_nf[mask]
            y_pred_sub = y_pred_nf[mask]
            fpr, fpr_ci = compute_fpr_with_ci(y_true_sub, y_pred_sub, compute_statistics, n_bootstrap, random_seed)
            if fpr is not None:
                if fpr_ci:
                    underdiagnosis_rates['level_1']['age'][ag] = fpr_ci
                else:
                    underdiagnosis_rates['level_1']['age'][ag] = {'mean': fpr, 'ci_lower': None, 'ci_upper': None}
    
    # 1-level subgroups: Sex
    # Aggregate from sex_race_subgroups or sex_age_subgroups to get all samples for each sex
    for s in [0, 1]:  # Male, Female
        sex_name = 'Male' if s == 0 else 'Female'
        # Combine masks from all subgroups containing this sex
        sex_mask = np.zeros(len(y_true_nf), dtype=bool)
        for subgroup_name, subgroup_mask in sex_race_subgroups.items():
            if subgroup_name.startswith(sex_name + '_'):
                # Ensure mask is numpy array and has correct length
                subgroup_mask = np.atleast_1d(np.array(subgroup_mask))
                if len(subgroup_mask) == len(sex_mask):
                    sex_mask |= subgroup_mask
        for subgroup_name, subgroup_mask in sex_age_subgroups.items():
            if subgroup_name.endswith('_' + sex_name):
                # Ensure mask is numpy array and has correct length
                subgroup_mask = np.atleast_1d(np.array(subgroup_mask))
                if len(subgroup_mask) == len(sex_mask):
                    sex_mask |= subgroup_mask
        
        if sex_mask.sum() >= 10:
            y_true_sub = y_true_nf[sex_mask]
            y_pred_sub = y_pred_nf[sex_mask]
            fpr, fpr_ci = compute_fpr_with_ci(y_true_sub, y_pred_sub, compute_statistics, n_bootstrap, random_seed)
            if fpr is not None:
                if fpr_ci:
                    underdiagnosis_rates['level_1']['sex'][sex_name] = fpr_ci
                else:
                    underdiagnosis_rates['level_1']['sex'][sex_name] = {'mean': fpr, 'ci_lower': None, 'ci_upper': None}
    
    # 1-level subgroups: Ethnicity
    for r in [0, 1, 2, 3]:  # White, Black, Asian, Hispanic
        race_name = ['White', 'Black', 'Asian', 'Hispanic'][r]
        mask = np.array(race_ethnicity == r)
        if mask.sum() >= 10:
            y_true_sub = y_true_nf[mask]
            y_pred_sub = y_pred_nf[mask]
            fpr, fpr_ci = compute_fpr_with_ci(y_true_sub, y_pred_sub, compute_statistics, n_bootstrap, random_seed)
            if fpr is not None:
                if fpr_ci:
                    underdiagnosis_rates['level_1']['ethnicity'][race_name] = fpr_ci
                else:
                    underdiagnosis_rates['level_1']['ethnicity'][race_name] = {'mean': fpr, 'ci_lower': None, 'ci_upper': None}
    
    # 2-level subgroups: Age × Sex (use existing sex_age_subgroups masks)
    for subgroup_name, mask in sex_age_subgroups.items():
        # Ensure mask is numpy array
        mask = np.atleast_1d(np.array(mask))
        if len(mask) == len(y_true_nf) and mask.sum() >= 10:
            y_true_sub = y_true_nf[mask]
            y_pred_sub = y_pred_nf[mask]
            fpr, fpr_ci = compute_fpr_with_ci(y_true_sub, y_pred_sub, compute_statistics, n_bootstrap, random_seed)
            if fpr is not None:
                if fpr_ci:
                    underdiagnosis_rates['level_2']['age_sex'][subgroup_name] = fpr_ci
                else:
                    underdiagnosis_rates['level_2']['age_sex'][subgroup_name] = {'mean': fpr, 'ci_lower': None, 'ci_upper': None}
    
    # 2-level subgroups: Age × Ethnicity
    for ag in unique_ages:
        for r in [0, 1, 2, 3]:
            race_name = ['White', 'Black', 'Asian', 'Hispanic'][r]
            mask = np.array((age_group == ag) & (race_ethnicity == r))
            if mask.sum() >= 10:
                subgroup_name = f"{ag}_{race_name}"
                y_true_sub = y_true_nf[mask]
                y_pred_sub = y_pred_nf[mask]
                fpr, fpr_ci = compute_fpr_with_ci(y_true_sub, y_pred_sub, compute_statistics, n_bootstrap, random_seed)
                if fpr is not None:
                    if fpr_ci:
                        underdiagnosis_rates['level_2']['age_ethnicity'][subgroup_name] = fpr_ci
                    else:
                        underdiagnosis_rates['level_2']['age_ethnicity'][subgroup_name] = {'mean': fpr, 'ci_lower': None, 'ci_upper': None}
    
    # 2-level subgroups: Sex × Ethnicity (use existing sex_race_subgroups masks)
    for subgroup_name, mask in sex_race_subgroups.items():
        # Ensure mask is numpy array
        mask = np.atleast_1d(np.array(mask))
        if len(mask) == len(y_true_nf) and mask.sum() >= 10:
            y_true_sub = y_true_nf[mask]
            y_pred_sub = y_pred_nf[mask]
            fpr, fpr_ci = compute_fpr_with_ci(y_true_sub, y_pred_sub, compute_statistics, n_bootstrap, random_seed)
            if fpr is not None:
                if fpr_ci:
                    underdiagnosis_rates['level_2']['sex_ethnicity'][subgroup_name] = fpr_ci
                else:
                    underdiagnosis_rates['level_2']['sex_ethnicity'][subgroup_name] = {'mean': fpr, 'ci_lower': None, 'ci_upper': None}
    
    # 3-level subgroups: Age × Sex × Ethnicity
    # Only compute for eligible subgroups (N_neg >= 20)
    eligible_subgroups = {}
    for subgroup_name, mask in intersectional_subgroups.items():
        if mask.sum() >= 10:
            y_true_sub = y_true_nf[mask]
            y_pred_sub = y_pred_nf[mask]
            # Check eligibility: N_neg >= 20
            is_eligible, n_neg = check_fpr_eligibility(y_true_sub, min_neg=20)
            if is_eligible:
                fpr, fpr_ci = compute_fpr_with_ci(y_true_sub, y_pred_sub, compute_statistics, n_bootstrap, random_seed, min_neg=20)
                if fpr is not None:
                    if fpr_ci:
                        # Store point estimate separately for gap computation
                        fpr_ci['point_estimate'] = fpr
                        underdiagnosis_rates['level_3']['age_sex_ethnicity'][subgroup_name] = fpr_ci
                    else:
                        underdiagnosis_rates['level_3']['age_sex_ethnicity'][subgroup_name] = {'mean': fpr, 'ci_lower': None, 'ci_upper': None, 'n_negatives': n_neg, 'point_estimate': fpr}
                    eligible_subgroups[subgroup_name] = mask
    
    # Store underdiagnosis rates (keep backward compatibility with old format for level_3)
    # Also store flat format for backward compatibility
    flat_underdiagnosis_rates = {}
    for subgroup_name, fpr_data in underdiagnosis_rates['level_3']['age_sex_ethnicity'].items():
        if isinstance(fpr_data, dict):
            flat_underdiagnosis_rates[subgroup_name] = fpr_data.get('mean', fpr_data)
        else:
            flat_underdiagnosis_rates[subgroup_name] = fpr_data
    
    if flat_underdiagnosis_rates:
        fairness_results['underdiagnosis_rates'] = flat_underdiagnosis_rates
        # Use point estimates (not bootstrap means) for computing gap point estimate
        fpr_values_point = []
        for subgroup_name, fpr_data in underdiagnosis_rates['level_3']['age_sex_ethnicity'].items():
            if isinstance(fpr_data, dict):
                # Use point_estimate if available, otherwise fall back to mean (for backward compatibility)
                fpr_val = fpr_data.get('point_estimate', fpr_data.get('mean', None))
            else:
                fpr_val = fpr_data
            if fpr_val is not None:
                fpr_values_point.append(fpr_val)
        
        if len(fpr_values_point) >= 2:
            # Compute max-min gap from point estimates
            gap_value_max_min = float(max(fpr_values_point) - min(fpr_values_point))
            
            # Compute robust gap (p90-p10) from point estimates
            p90 = float(np.percentile(fpr_values_point, 90))
            p10 = float(np.percentile(fpr_values_point, 10))
            gap_value_robust = p90 - p10
            
            fairness_results['underdiagnosis_gap'] = gap_value_max_min  # Backward compatibility
            fairness_results['underdiagnosis_gap_max_min'] = gap_value_max_min
            fairness_results['underdiagnosis_gap_robust'] = gap_value_robust
            
            # Compute bootstrap CI for both gaps
            gap_ci_max_min = {'mean': gap_value_max_min, 'ci_lower': None, 'ci_upper': None}
            gap_ci_robust = {'mean': gap_value_robust, 'ci_lower': None, 'ci_upper': None}
            
            if compute_statistics and len(eligible_subgroups) >= 2:
                logger.info("Computing bootstrap CI for underdiagnosis gap...")
                if random_seed is not None:
                    np.random.seed(random_seed)
                
                n = len(ground_truth)
                bootstrap_gaps_max_min = []
                bootstrap_gaps_robust = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap resample indices
                    boot_indices = np.random.choice(n, size=n, replace=True)
                    
                    # Compute FPR for each eligible intersectional subgroup on bootstrap sample
                    subgroup_fprs_boot = {}
                    for subgroup_name, mask in eligible_subgroups.items():
                        # Get original indices that belong to this subgroup
                        subgroup_indices = np.where(mask)[0]
                        # Find which bootstrap indices correspond to subgroup samples
                        boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                        if boot_subgroup_mask.sum() < 10:
                            continue
                        
                        # Get bootstrap data for this subgroup
                        boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                        y_true_boot_sub = y_true_nf[boot_subgroup_indices]
                        y_pred_boot_sub = y_pred_nf[boot_subgroup_indices]
                        
                        # Check eligibility for bootstrap sample (relaxed threshold: N_neg >= 10)
                        is_eligible_boot, _ = check_fpr_eligibility(y_true_boot_sub, min_neg=10)
                        if not is_eligible_boot:
                            continue
                        
                        # Compute FPR with Jeffreys smoothing on bootstrap sample
                        try:
                            fpr_boot = compute_fpr_jeffreys(y_true_boot_sub, y_pred_boot_sub)
                            if fpr_boot is not None and not np.isnan(fpr_boot):
                                subgroup_fprs_boot[subgroup_name] = fpr_boot
                        except:
                            continue
                    
                    # Compute gaps from bootstrap FPRs
                    if len(subgroup_fprs_boot) >= 2:
                        fpr_values_boot = list(subgroup_fprs_boot.values())
                        
                        # Max-min gap
                        gap_boot_max_min = float(max(fpr_values_boot) - min(fpr_values_boot))
                        if not np.isnan(gap_boot_max_min):
                            bootstrap_gaps_max_min.append(gap_boot_max_min)
                        
                        # Robust gap (p90-p10)
                        p90_boot = float(np.percentile(fpr_values_boot, 90))
                        p10_boot = float(np.percentile(fpr_values_boot, 10))
                        gap_boot_robust = p90_boot - p10_boot
                        if not np.isnan(gap_boot_robust):
                            bootstrap_gaps_robust.append(gap_boot_robust)
                
                # Compute CI for max-min gap
                if len(bootstrap_gaps_max_min) > 0:
                    bootstrap_gaps_max_min = np.array(bootstrap_gaps_max_min)
                    mean_gap_max_min = np.mean(bootstrap_gaps_max_min)
                    alpha = 0.05
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    ci_lower_max_min = np.percentile(bootstrap_gaps_max_min, lower_percentile)
                    ci_upper_max_min = np.percentile(bootstrap_gaps_max_min, upper_percentile)
                    
                    # Use bootstrap mean for consistency with other metrics (AUROC, AUPRC, FPR)
                    gap_ci_max_min = {
                        'mean': float(mean_gap_max_min),
                        'ci_lower': float(ci_lower_max_min),
                        'ci_upper': float(ci_upper_max_min),
                    }
                else:
                    logger.debug("No valid bootstrap gaps (max-min) computed for underdiagnosis gap")
                
                # Compute CI for robust gap
                if len(bootstrap_gaps_robust) > 0:
                    bootstrap_gaps_robust = np.array(bootstrap_gaps_robust)
                    mean_gap_robust = np.mean(bootstrap_gaps_robust)
                    alpha = 0.05
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    ci_lower_robust = np.percentile(bootstrap_gaps_robust, lower_percentile)
                    ci_upper_robust = np.percentile(bootstrap_gaps_robust, upper_percentile)
                    
                    # Use bootstrap mean for consistency with other metrics (AUROC, AUPRC, FPR)
                    gap_ci_robust = {
                        'mean': float(mean_gap_robust),
                        'ci_lower': float(ci_lower_robust),
                        'ci_upper': float(ci_upper_robust),
                    }
                else:
                    logger.debug("No valid bootstrap gaps (robust) computed for underdiagnosis gap")
            
            # Store for backward compatibility
            fairness_results['underdiagnosis_gap_ci'] = gap_ci_max_min
            
            # Store new metrics
            fairness_results['underdiagnosis_gap_max_min_ci'] = gap_ci_max_min
            fairness_results['underdiagnosis_gap_robust_ci'] = gap_ci_robust
            fairness_results['underdiagnosis_gap_eligibility'] = {
                'n_eligible_subgroups': len(eligible_subgroups),
                'eligible_subgroups': list(eligible_subgroups.keys()),
            }
    
    # Store structured format for all levels
    fairness_results['underdiagnosis_rates_by_level'] = underdiagnosis_rates
    
    return fairness_results


def compute_underdiagnosis_gap_only(
    predictions_probs: np.ndarray,
    ground_truth: np.ndarray,
    demographics: Dict[str, List],
    thresholds: Dict[str, float],
    n_bootstrap: int = 1000,
    compute_statistics: bool = True,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Compute only underdiagnosis gap (and CI) without computing all other fairness metrics.
    This is a lightweight function for when you only need the underdiagnosis gap.
    
    Args:
        predictions_probs: Predicted probabilities [N, 14]
        ground_truth: Ground truth labels [N, 14]
        demographics: Dictionary with 'age', 'sex', 'race_ethnicity', 'age_group'
        thresholds: Per-label thresholds
        n_bootstrap: Number of bootstrap resamples for confidence intervals
        compute_statistics: If True, compute bootstrap CIs
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'underdiagnosis_gap' and 'underdiagnosis_gap_ci'
    """
    # Convert demographics to arrays (ensure they're 1D arrays)
    def flatten_array(arr):
        arr = np.array(arr)
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr
    
    age = flatten_array(demographics.get('age', []))
    sex = flatten_array(demographics.get('sex', []))
    race_ethnicity = flatten_array(demographics.get('race_ethnicity', []))
    age_group_raw = flatten_array(demographics.get('age_group', []))
    
    # Get the expected length from ground_truth
    n_samples_expected = len(ground_truth)
    
    # Check if arrays have correct length
    if len(sex) != n_samples_expected or len(age) != n_samples_expected or \
       len(race_ethnicity) != n_samples_expected or len(age_group_raw) != n_samples_expected:
        raise ValueError(
            f"Demographic array length mismatch. All arrays must have length {n_samples_expected}. "
            f"Check that demographics were saved correctly during evaluation."
        )
    
    # Convert age_group from numeric strings to expected format
    age_group_map = {'1': '18-40', '2': '40-60', '3': '60-80', '4': '80+'}
    age_group = np.array([
        age_group_map.get(str(ag), ag) if str(ag) in age_group_map else ag 
        for ag in age_group_raw
    ])
    age_group = np.atleast_1d(age_group)
    
    # Apply thresholds to get binary predictions
    binary_preds = apply_thresholds(predictions_probs, thresholds, CHEXPERT_CLASSES)
    
    # Create 3-way intersectional subgroups: Age × Sex × Ethnicity
    intersectional_subgroups = {}
    age_groups_unique = ['18-40', '40-60', '60-80', '80+']
    for ag in age_groups_unique:
        for s in [0, 1]:
            for r in [0, 1, 2, 3]:
                mask = (age_group == ag) & (sex == s) & (race_ethnicity == r)
                if mask.sum() > 0:
                    subgroup_name = f"{ag}_{'Male' if s == 0 else 'Female'}_{['White', 'Black', 'Asian', 'Hispanic'][r]}"
                    intersectional_subgroups[subgroup_name] = mask
    
    # Get binary predictions and labels for "No Finding"
    no_finding_idx = CHEXPERT_CLASSES.index('No Finding')
    y_true_nf = ground_truth[:, no_finding_idx]
    y_pred_nf = binary_preds[:, no_finding_idx]
    
    # Compute FPR for each eligible intersectional subgroup (N_neg >= 20)
    eligible_subgroups = {}
    subgroup_fprs = {}
    for subgroup_name, mask in intersectional_subgroups.items():
        if mask.sum() < 10:
            continue
        
        y_true_sub = y_true_nf[mask]
        y_pred_sub = y_pred_nf[mask]
        
        # Check eligibility: N_neg >= 20
        is_eligible, n_neg = check_fpr_eligibility(y_true_sub, min_neg=20)
        if not is_eligible:
            continue
        
        # Compute FPR with Jeffreys smoothing
        fpr = compute_fpr_jeffreys(y_true_sub, y_pred_sub)
        if fpr is not None and not np.isnan(fpr):
            subgroup_fprs[subgroup_name] = fpr
            eligible_subgroups[subgroup_name] = mask
    
    # Compute gaps
    if len(subgroup_fprs) < 2:
        return {
            'underdiagnosis_gap': None,
            'underdiagnosis_gap_ci': {'mean': None, 'ci_lower': None, 'ci_upper': None},
            'underdiagnosis_gap_max_min': None,
            'underdiagnosis_gap_max_min_ci': {'mean': None, 'ci_lower': None, 'ci_upper': None},
            'underdiagnosis_gap_robust': None,
            'underdiagnosis_gap_robust_ci': {'mean': None, 'ci_lower': None, 'ci_upper': None},
        }
    
    fpr_values = list(subgroup_fprs.values())
    
    # Compute max-min gap
    gap_value_max_min = float(max(fpr_values) - min(fpr_values))
    
    # Compute robust gap (p90-p10)
    p90 = float(np.percentile(fpr_values, 90))
    p10 = float(np.percentile(fpr_values, 10))
    gap_value_robust = p90 - p10
    
    # Compute bootstrap CI for both gaps
    gap_ci_max_min = {'mean': gap_value_max_min, 'ci_lower': None, 'ci_upper': None}
    gap_ci_robust = {'mean': gap_value_robust, 'ci_lower': None, 'ci_upper': None}
    
    if compute_statistics and len(eligible_subgroups) >= 2:
        logger.info("Computing bootstrap CI for underdiagnosis gap...")
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n = len(ground_truth)
        bootstrap_gaps_max_min = []
        bootstrap_gaps_robust = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample indices
            boot_indices = np.random.choice(n, size=n, replace=True)
            
            # Compute FPR for each eligible intersectional subgroup on bootstrap sample
            subgroup_fprs_boot = {}
            for subgroup_name, mask in eligible_subgroups.items():
                # Get original indices that belong to this subgroup
                subgroup_indices = np.where(mask)[0]
                # Find which bootstrap indices correspond to subgroup samples
                boot_subgroup_mask = np.isin(boot_indices, subgroup_indices)
                if boot_subgroup_mask.sum() < 10:
                    continue
                
                # Get bootstrap data for this subgroup
                boot_subgroup_indices = boot_indices[boot_subgroup_mask]
                y_true_boot_sub = y_true_nf[boot_subgroup_indices]
                y_pred_boot_sub = y_pred_nf[boot_subgroup_indices]
                
                # Check eligibility for bootstrap sample (relaxed threshold: N_neg >= 10)
                is_eligible_boot, _ = check_fpr_eligibility(y_true_boot_sub, min_neg=10)
                if not is_eligible_boot:
                    continue
                
                # Compute FPR with Jeffreys smoothing on bootstrap sample
                try:
                    fpr_boot = compute_fpr_jeffreys(y_true_boot_sub, y_pred_boot_sub)
                    if fpr_boot is not None and not np.isnan(fpr_boot):
                        subgroup_fprs_boot[subgroup_name] = fpr_boot
                except:
                    continue
            
            # Compute gaps from bootstrap FPRs
            if len(subgroup_fprs_boot) >= 2:
                fpr_values_boot = list(subgroup_fprs_boot.values())
                
                # Max-min gap
                gap_boot_max_min = float(max(fpr_values_boot) - min(fpr_values_boot))
                if not np.isnan(gap_boot_max_min):
                    bootstrap_gaps_max_min.append(gap_boot_max_min)
                
                # Robust gap (p90-p10)
                p90_boot = float(np.percentile(fpr_values_boot, 90))
                p10_boot = float(np.percentile(fpr_values_boot, 10))
                gap_boot_robust = p90_boot - p10_boot
                if not np.isnan(gap_boot_robust):
                    bootstrap_gaps_robust.append(gap_boot_robust)
        
        # Compute CI for max-min gap
        if len(bootstrap_gaps_max_min) > 0:
            bootstrap_gaps_max_min = np.array(bootstrap_gaps_max_min)
            mean_gap_max_min = np.mean(bootstrap_gaps_max_min)
            alpha = 0.05
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            ci_lower_max_min = np.percentile(bootstrap_gaps_max_min, lower_percentile)
            ci_upper_max_min = np.percentile(bootstrap_gaps_max_min, upper_percentile)
            
            # Use bootstrap mean for consistency with other metrics (AUROC, AUPRC, FPR)
            gap_ci_max_min = {
                'mean': float(mean_gap_max_min),
                'ci_lower': float(ci_lower_max_min),
                'ci_upper': float(ci_upper_max_min),
            }
        else:
            logger.debug("No valid bootstrap gaps (max-min) computed for underdiagnosis gap")
        
        # Compute CI for robust gap
        if len(bootstrap_gaps_robust) > 0:
            bootstrap_gaps_robust = np.array(bootstrap_gaps_robust)
            mean_gap_robust = np.mean(bootstrap_gaps_robust)
            alpha = 0.05
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            ci_lower_robust = np.percentile(bootstrap_gaps_robust, lower_percentile)
            ci_upper_robust = np.percentile(bootstrap_gaps_robust, upper_percentile)
            
            # Use bootstrap mean for consistency with other metrics (AUROC, AUPRC, FPR)
            gap_ci_robust = {
                'mean': float(mean_gap_robust),
                'ci_lower': float(ci_lower_robust),
                'ci_upper': float(ci_upper_robust),
            }
        else:
            logger.debug("No valid bootstrap gaps (robust) computed for underdiagnosis gap")
    
    return {
        'underdiagnosis_gap': gap_value_max_min,  # Backward compatibility
        'underdiagnosis_gap_ci': gap_ci_max_min,  # Backward compatibility
        'underdiagnosis_gap_max_min': gap_value_max_min,
        'underdiagnosis_gap_max_min_ci': gap_ci_max_min,
        'underdiagnosis_gap_robust': gap_value_robust,
        'underdiagnosis_gap_robust_ci': gap_ci_robust,
        'underdiagnosis_gap_eligibility': {
            'n_eligible_subgroups': len(eligible_subgroups),
            'eligible_subgroups': list(eligible_subgroups.keys()),
        },
    }

