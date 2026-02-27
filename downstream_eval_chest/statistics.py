"""
Statistical analysis functions for classifier evaluation.
Includes bootstrap confidence intervals, DeLong test for AUROC, and permutation test for AUPRC.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Eligibility Check Functions
# ============================================================================

def check_auroc_eligibility(y_true: np.ndarray, min_pos: int = 10, min_neg: int = 10) -> Tuple[bool, int, int]:
    """
    Check if data is eligible for AUROC computation.
    
    Args:
        y_true: True binary labels [N] (0 or 1, -1 for uncertain)
        min_pos: Minimum number of positive samples required
        min_neg: Minimum number of negative samples required
    
    Returns:
        Tuple of (is_eligible, n_positives, n_negatives)
    """
    # Filter out uncertain labels
    valid_mask = y_true != -1
    if valid_mask.sum() < 2:
        return False, 0, 0
    
    y_true_valid = y_true[valid_mask]
    n_pos = int((y_true_valid == 1).sum())
    n_neg = int((y_true_valid == 0).sum())
    
    is_eligible = (n_pos >= min_pos) and (n_neg >= min_neg)
    return is_eligible, n_pos, n_neg


def check_auprc_eligibility(y_true: np.ndarray, min_pos: int = 10) -> Tuple[bool, int, int]:
    """
    Check if data is eligible for AUPRC computation.
    
    Args:
        y_true: True binary labels [N] (0 or 1, -1 for uncertain)
        min_pos: Minimum number of positive samples required
    
    Returns:
        Tuple of (is_eligible, n_positives, n_negatives)
    """
    # Filter out uncertain labels
    valid_mask = y_true != -1
    if valid_mask.sum() < 2:
        return False, 0, 0
    
    y_true_valid = y_true[valid_mask]
    n_pos = int((y_true_valid == 1).sum())
    n_neg = int((y_true_valid == 0).sum())
    
    # AUPRC needs at least some positives and some negatives
    is_eligible = (n_pos >= min_pos) and (n_neg > 0)
    return is_eligible, n_pos, n_neg


def check_f1_eligibility(y_true: np.ndarray, min_pos: int = 10) -> Tuple[bool, int, int]:
    """
    Check if data is eligible for F1 computation.
    
    Args:
        y_true: True binary labels [N] (0 or 1, -1 for uncertain)
        min_pos: Minimum number of positive samples required
    
    Returns:
        Tuple of (is_eligible, n_positives, n_negatives)
    """
    # Filter out uncertain labels
    valid_mask = y_true != -1
    if valid_mask.sum() < 2:
        return False, 0, 0
    
    y_true_valid = y_true[valid_mask]
    n_pos = int((y_true_valid == 1).sum())
    n_neg = int((y_true_valid == 0).sum())
    
    # F1 needs at least some positives
    is_eligible = n_pos >= min_pos
    return is_eligible, n_pos, n_neg


def check_fpr_eligibility(y_true_nf: np.ndarray, min_neg: int = 20) -> Tuple[bool, int]:
    """
    Check if data is eligible for FPR (underdiagnosis rate) computation.
    
    Args:
        y_true_nf: True labels for "No Finding" [N] (0 = has finding, 1 = no finding, -1 = uncertain)
        min_neg: Minimum number of negative samples required (has finding, GT=0)
    
    Returns:
        Tuple of (is_eligible, n_negatives)
    """
    # Filter out uncertain labels
    valid_mask = y_true_nf != -1
    if valid_mask.sum() < 2:
        return False, 0
    
    y_true_valid = y_true_nf[valid_mask]
    # For FPR, negatives are samples with GT=0 (has finding)
    n_neg = int((y_true_valid == 0).sum())
    
    is_eligible = n_neg >= min_neg
    return is_eligible, n_neg


def compute_fpr_jeffreys(y_true_nf: np.ndarray, y_pred_nf: np.ndarray) -> Optional[float]:
    """
    Compute FPR for "No Finding" with Jeffreys smoothing.
    
    Jeffreys smoothing: FPR = (FP + 0.5) / (FP + TN + 1)
    This provides a Bayesian estimate that avoids 0/0 and extreme values.
    
    Args:
        y_true_nf: True labels for "No Finding" [N] (0 = has finding, 1 = no finding, -1 = uncertain)
        y_pred_nf: Binary predictions for "No Finding" [N] (0 or 1)
    
    Returns:
        Smoothed FPR value, or None if insufficient data
    """
    # Filter out uncertain labels
    valid_mask = y_true_nf != -1
    if valid_mask.sum() < 2:
        return None
    
    y_true_valid = y_true_nf[valid_mask]
    y_pred_valid = y_pred_nf[valid_mask]
    
    # Treat -1 and NaN as 0 (has finding)
    y_true_clean = np.where((y_true_valid == -1) | np.isnan(y_true_valid), 0, y_true_valid).astype(int)
    
    # Check if we have negative cases (GT = 0, has finding)
    negative_mask = (y_true_clean == 0)
    if negative_mask.sum() == 0:
        return None
    
    # Compute FP and TN
    fp = ((y_pred_valid == 1) & (y_true_clean == 0)).sum()
    tn = ((y_pred_valid == 0) & (y_true_clean == 0)).sum()
    
    if (fp + tn) == 0:
        return None
    
    # Jeffreys smoothing: (FP + 0.5) / (FP + TN + 1)
    fpr_smoothed = float((fp + 0.5) / (fp + tn + 1))
    return fpr_smoothed


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array [N]
        statistic_fn: Function that computes the statistic from data
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = data[indices]
        
        try:
            stat = statistic_fn(bootstrap_sample)
            if not np.isnan(stat) and not np.isinf(stat):
                bootstrap_stats.append(stat)
        except Exception as e:
            logger.debug(f"Bootstrap statistic computation failed: {e}")
            continue
    
    if len(bootstrap_stats) == 0:
        logger.warning("No valid bootstrap statistics computed")
        return np.nan, np.nan, np.nan
    
    bootstrap_stats = np.array(bootstrap_stats)
    mean_stat = np.mean(bootstrap_stats)
    
    # Compute percentiles for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return float(mean_stat), float(lower_bound), float(upper_bound)


def bootstrap_ci_for_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
    validate_bootstrap: Optional[Callable[[np.ndarray], bool]] = None,
) -> Tuple[float, float, float, int]:
    """
    Compute bootstrap confidence interval for a metric computed from predictions and labels.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted probabilities [N]
        metric_fn: Function that computes metric from (y_true, y_pred)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level
        random_seed: Random seed for reproducibility
        validate_bootstrap: Optional function to validate bootstrap sample (y_true_boot) -> bool
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound, n_valid_bootstrap)
        n_valid_bootstrap: Number of valid bootstrap replicates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(y_true)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Validate bootstrap sample if validation function provided
        if validate_bootstrap is not None:
            if not validate_bootstrap(y_true_boot):
                continue  # Skip this bootstrap replicate
        
        try:
            stat = metric_fn(y_true_boot, y_pred_boot)
            if not np.isnan(stat) and not np.isinf(stat):
                bootstrap_stats.append(stat)
        except Exception as e:
            logger.debug(f"Bootstrap metric computation failed: {e}")
            continue
    
    n_valid = len(bootstrap_stats)
    
    if n_valid == 0:
        logger.warning("No valid bootstrap statistics computed")
        return np.nan, np.nan, np.nan, 0
    
    bootstrap_stats = np.array(bootstrap_stats)
    mean_stat = np.mean(bootstrap_stats)
    
    # Compute percentiles for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return float(mean_stat), float(lower_bound), float(upper_bound), n_valid


def delong_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> Tuple[float, float]:
    """
    DeLong test for comparing two correlated ROC curves.
    
    This implements the DeLong test as described in:
    DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). 
    Comparing the areas under two or more correlated receiver operating 
    characteristic curves: a nonparametric approach. Biometrics, 44(3), 837-845.
    
    Args:
        y_true: True binary labels [N]
        y_pred_1: Predicted probabilities from model 1 [N]
        y_pred_2: Predicted probabilities from model 2 [N]
    
    Returns:
        Tuple of (z_statistic, p_value)
    """
    from scipy import stats
    
    # Compute AUROC for both models
    try:
        auroc_1 = roc_auc_score(y_true, y_pred_1)
        auroc_2 = roc_auc_score(y_true, y_pred_2)
    except ValueError as e:
        logger.warning(f"DeLong test: Error computing AUROC: {e}")
        return np.nan, np.nan
    
    # Compute variance using DeLong's method
    # This requires computing the covariance structure
    n = len(y_true)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    
    if n_pos == 0 or n_neg == 0:
        logger.warning("DeLong test: Need both positive and negative samples")
        return np.nan, np.nan
    
    # Sort predictions for positive and negative cases
    pos_pred_1 = y_pred_1[y_true == 1]
    neg_pred_1 = y_pred_1[y_true == 0]
    pos_pred_2 = y_pred_2[y_true == 1]
    neg_pred_2 = y_pred_2[y_true == 0]
    
    # Compute V10 and V01 (DeLong's variance components)
    # V10[i] = proportion of negatives with score < pos_pred_1[i]
    # V01[j] = proportion of positives with score > neg_pred_1[j]
    
    V10_1 = np.array([(neg_pred_1 < p).mean() for p in pos_pred_1])
    V01_1 = np.array([(pos_pred_1 > n).mean() for n in neg_pred_1])
    
    V10_2 = np.array([(neg_pred_2 < p).mean() for p in pos_pred_2])
    V01_2 = np.array([(pos_pred_2 > n).mean() for n in neg_pred_2])
    
    # Compute variance of AUROC difference
    # Var(AUROC_1 - AUROC_2) = Var(AUROC_1) + Var(AUROC_2) - 2*Cov(AUROC_1, AUROC_2)
    
    # Variance of AUROC_1
    S10_1 = np.var(V10_1) / n_pos
    S01_1 = np.var(V01_1) / n_neg
    var_auroc_1 = S10_1 + S01_1
    
    # Variance of AUROC_2
    S10_2 = np.var(V10_2) / n_pos
    S01_2 = np.var(V01_2) / n_neg
    var_auroc_2 = S10_2 + S01_2
    
    # Covariance between AUROC_1 and AUROC_2
    # Cov = E[(V10_1 - AUROC_1)(V10_2 - AUROC_2)] / n_pos + E[(V01_1 - AUROC_1)(V01_2 - AUROC_2)] / n_neg
    cov_V10 = np.mean((V10_1 - auroc_1) * (V10_2 - auroc_2)) / n_pos
    cov_V01 = np.mean((V01_1 - auroc_1) * (V01_2 - auroc_2)) / n_neg
    cov_auroc = cov_V10 + cov_V01
    
    # Variance of difference
    var_diff = var_auroc_1 + var_auroc_2 - 2 * cov_auroc
    
    if var_diff <= 0:
        logger.warning(f"DeLong test: Non-positive variance: {var_diff}")
        return np.nan, np.nan
    
    # Compute z-statistic
    diff = auroc_1 - auroc_2
    se_diff = np.sqrt(var_diff)
    z_stat = diff / se_diff
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return float(z_stat), float(p_value)


def permutation_test_auprc(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    n_permutations: int = 10000,
    random_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Permutation test for comparing two AUPRC values.
    
    Args:
        y_true: True binary labels [N]
        y_pred_1: Predicted probabilities from model 1 [N]
        y_pred_2: Predicted probabilities from model 2 [N]
        n_permutations: Number of permutations
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (observed_diff, p_value)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Compute observed AUPRC difference
    try:
        auprc_1 = average_precision_score(y_true, y_pred_1)
        auprc_2 = average_precision_score(y_true, y_pred_2)
        observed_diff = auprc_1 - auprc_2
    except ValueError as e:
        logger.warning(f"Permutation test: Error computing AUPRC: {e}")
        return np.nan, np.nan
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        logger.warning("Permutation test: Need both positive and negative samples")
        return np.nan, np.nan
    
    # Permutation test: randomly swap predictions between models
    n = len(y_true)
    permuted_diffs = []
    
    for _ in range(n_permutations):
        # Randomly decide which samples to swap
        swap_mask = np.random.rand(n) < 0.5
        
        # Create permuted predictions
        y_pred_1_perm = np.where(swap_mask, y_pred_2, y_pred_1)
        y_pred_2_perm = np.where(swap_mask, y_pred_1, y_pred_2)
        
        try:
            auprc_1_perm = average_precision_score(y_true, y_pred_1_perm)
            auprc_2_perm = average_precision_score(y_true, y_pred_2_perm)
            permuted_diff = auprc_1_perm - auprc_2_perm
            
            if not np.isnan(permuted_diff) and not np.isinf(permuted_diff):
                permuted_diffs.append(permuted_diff)
        except Exception as e:
            logger.debug(f"Permutation test: Error in permutation: {e}")
            continue
    
    if len(permuted_diffs) == 0:
        logger.warning("Permutation test: No valid permutations")
        return observed_diff, np.nan
    
    permuted_diffs = np.array(permuted_diffs)
    
    # Compute p-value: proportion of permutations with |diff| >= |observed_diff|
    # Two-tailed test
    extreme_count = (np.abs(permuted_diffs) >= np.abs(observed_diff)).sum()
    p_value = (extreme_count + 1) / (len(permuted_diffs) + 1)  # +1 for observed
    
    return float(observed_diff), float(p_value)


def compute_fpr_no_finding(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
) -> float:
    """
    Compute False Positive Rate (FPR) for "No Finding" label.
    
    FPR = FP / (FP + TN) where:
    - FP: Predicted "No Finding" (1) but ground truth has finding (0)
    - TN: Predicted has finding (0) and ground truth has finding (0)
    
    Args:
        y_true: True binary labels for "No Finding" [N] (0 = has finding, 1 = no finding)
        y_pred_binary: Binary predictions for "No Finding" [N] (0 = has finding, 1 = no finding)
    
    Returns:
        FPR value or np.nan if cannot compute
    """
    # Filter out uncertain labels (-1)
    valid_mask = y_true != -1
    if valid_mask.sum() == 0:
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred_binary[valid_mask]
    
    # Treat -1 and NaN as 0 (has finding) per the updated logic
    y_true_clean = np.where((y_true_valid == -1) | np.isnan(y_true_valid), 0, y_true_valid).astype(int)
    
    # FPR = FP / (FP + TN)
    # FP: predicted 1 (No Finding) but GT is 0 (has finding)
    # TN: predicted 0 (has finding) and GT is 0 (has finding)
    # So we need cases where GT = 0
    negative_mask = (y_true_clean == 0)
    if negative_mask.sum() == 0:
        return np.nan  # No negative cases
    
    fp = ((y_pred_valid == 1) & (y_true_clean == 0)).sum()
    tn = ((y_pred_valid == 0) & (y_true_clean == 0)).sum()
    
    if (fp + tn) == 0:
        return np.nan
    
    fpr = fp / (fp + tn)
    return float(fpr)


def compute_bootstrap_cis_for_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    y_pred_binary: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for AUROC, AUPRC per label, and FPR for "No Finding".
    
    Args:
        y_true: True labels [N, num_classes]
        y_pred: Predicted probabilities [N, num_classes]
        class_names: List of class names
        y_pred_binary: Binary predictions [N, num_classes] (optional, needed for FPR computation)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with structure:
        {
            'auroc': {
                'class_name': {'mean': float, 'ci_lower': float, 'ci_upper': float}
            },
            'auprc': {
                'class_name': {'mean': float, 'ci_lower': float, 'ci_upper': float}
            },
            'fpr': {
                'No Finding': {'mean': float, 'ci_lower': float, 'ci_upper': float}  # Only for "No Finding"
            }
        }
    """
    results = {
        'auroc': {},
        'auprc': {},
        'fpr': {},
    }
    
    for i, class_name in enumerate(class_names):
        # Filter out uncertain labels
        valid_mask = y_true[:, i] != -1
        if valid_mask.sum() < 2:
            results['auroc'][class_name] = {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
            results['auprc'][class_name] = {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
            continue
        
        y_true_i = y_true[valid_mask, i]
        y_pred_i = y_pred[valid_mask, i]
        
        # Check if we have both classes
        if len(np.unique(y_true_i)) < 2:
            results['auroc'][class_name] = {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
            results['auprc'][class_name] = {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
            continue
        
        # Bootstrap CI for AUROC with validation
        def auroc_fn(y_t, y_p):
            try:
                return roc_auc_score(y_t, y_p)
            except:
                return np.nan
        
        def validate_auroc_bootstrap(y_t_boot):
            """Validate bootstrap sample has both classes for AUROC."""
            unique_classes = np.unique(y_t_boot)
            return len(unique_classes) >= 2 and (1 in unique_classes) and (0 in unique_classes)
        
        mean_auroc, ci_lower_auroc, ci_upper_auroc, n_valid_auroc = bootstrap_ci_for_metric(
            y_true_i,
            y_pred_i,
            auroc_fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
            validate_bootstrap=validate_auroc_bootstrap,
        )
        
        results['auroc'][class_name] = {
            'mean': mean_auroc,
            'ci_lower': ci_lower_auroc,
            'ci_upper': ci_upper_auroc,
            'n_valid_bootstrap': n_valid_auroc,
        }
        
        # Bootstrap CI for AUPRC with validation
        def auprc_fn(y_t, y_p):
            try:
                return average_precision_score(y_t, y_p)
            except:
                return np.nan
        
        def validate_auprc_bootstrap(y_t_boot):
            """Validate bootstrap sample has at least some positives for AUPRC."""
            unique_classes = np.unique(y_t_boot)
            return len(unique_classes) >= 2 and (1 in unique_classes)
        
        mean_auprc, ci_lower_auprc, ci_upper_auprc, n_valid_auprc = bootstrap_ci_for_metric(
            y_true_i,
            y_pred_i,
            auprc_fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed,
            validate_bootstrap=validate_auprc_bootstrap,
        )
        
        results['auprc'][class_name] = {
            'mean': mean_auprc,
            'ci_lower': ci_lower_auprc,
            'ci_upper': ci_upper_auprc,
            'n_valid_bootstrap': n_valid_auprc,
        }
    
    # Compute FPR for "No Finding" label if binary predictions are provided
    if y_pred_binary is not None and 'No Finding' in class_names:
        no_finding_idx = class_names.index('No Finding')
        y_true_nf = y_true[:, no_finding_idx]
        y_pred_binary_nf = y_pred_binary[:, no_finding_idx]
        
        # Check eligibility for FPR computation (N_neg >= 20)
        is_eligible, n_neg = check_fpr_eligibility(y_true_nf, min_neg=20)
        
        if is_eligible:
            # Filter out uncertain labels
            valid_mask = y_true_nf != -1
            y_true_nf_valid = y_true_nf[valid_mask]
            y_pred_binary_nf_valid = y_pred_binary_nf[valid_mask]
            
            # Bootstrap CI for FPR using Jeffreys smoothing
            def fpr_fn_jeffreys(y_t, y_p):
                """Compute FPR with Jeffreys smoothing."""
                fpr = compute_fpr_jeffreys(y_t, y_p)
                return fpr if fpr is not None else np.nan
            
            def validate_fpr_bootstrap(y_t_boot):
                """Validate bootstrap sample has enough negatives for FPR."""
                valid_mask_boot = y_t_boot != -1
                if valid_mask_boot.sum() < 2:
                    return False
                y_t_boot_valid = y_t_boot[valid_mask_boot]
                y_t_clean = np.where((y_t_boot_valid == -1) | np.isnan(y_t_boot_valid), 0, y_t_boot_valid).astype(int)
                n_neg_boot = (y_t_clean == 0).sum()
                return n_neg_boot >= 10  # Lower threshold for bootstrap samples
            
            mean_fpr, ci_lower_fpr, ci_upper_fpr, n_valid_fpr = bootstrap_ci_for_metric(
                y_true_nf_valid,
                y_pred_binary_nf_valid,
                fpr_fn_jeffreys,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_seed=random_seed,
                validate_bootstrap=validate_fpr_bootstrap,
            )
            
            # Also compute point estimate with Jeffreys smoothing
            fpr_point = compute_fpr_jeffreys(y_true_nf_valid, y_pred_binary_nf_valid)
            
            results['fpr']['No Finding'] = {
                'mean': mean_fpr if not np.isnan(mean_fpr) else fpr_point,
                'ci_lower': ci_lower_fpr,
                'ci_upper': ci_upper_fpr,
                'n_valid_bootstrap': n_valid_fpr,
                'n_negatives': n_neg,
                'eligible': True,
            }
        else:
            # Insufficient data
            results['fpr']['No Finding'] = {
                'mean': None,
                'ci_lower': None,
                'ci_upper': None,
                'n_valid_bootstrap': 0,
                'n_negatives': n_neg,
                'eligible': False,
                'insufficient_data': True,
            }
    
    return results


def compare_models(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    class_names: List[str],
    n_permutations: int = 10000,
    random_seed: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare two models using DeLong test for AUROC and permutation test for AUPRC.
    
    Args:
        y_true: True labels [N, num_classes]
        y_pred_1: Predicted probabilities from model 1 [N, num_classes]
        y_pred_2: Predicted probabilities from model 2 [N, num_classes]
        class_names: List of class names
        n_permutations: Number of permutations for AUPRC test
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with structure:
        {
            'auroc_comparison': {
                'class_name': {'z_statistic': float, 'p_value': float, 'auroc_1': float, 'auroc_2': float, 'difference': float}
            },
            'auprc_comparison': {
                'class_name': {'observed_diff': float, 'p_value': float, 'auprc_1': float, 'auprc_2': float}
            }
        }
    """
    results = {
        'auroc_comparison': {},
        'auprc_comparison': {},
    }
    
    for i, class_name in enumerate(class_names):
        # Filter out uncertain labels
        valid_mask = y_true[:, i] != -1
        if valid_mask.sum() < 2:
            continue
        
        y_true_i = y_true[valid_mask, i]
        y_pred_1_i = y_pred_1[valid_mask, i]
        y_pred_2_i = y_pred_2[valid_mask, i]
        
        # Check if we have both classes
        if len(np.unique(y_true_i)) < 2:
            continue
        
        # DeLong test for AUROC
        try:
            auroc_1 = roc_auc_score(y_true_i, y_pred_1_i)
            auroc_2 = roc_auc_score(y_true_i, y_pred_2_i)
            z_stat, p_value = delong_test(y_true_i, y_pred_1_i, y_pred_2_i)
            
            results['auroc_comparison'][class_name] = {
                'z_statistic': z_stat if not np.isnan(z_stat) else None,
                'p_value': p_value if not np.isnan(p_value) else None,
                'auroc_1': float(auroc_1) if not np.isnan(auroc_1) else None,
                'auroc_2': float(auroc_2) if not np.isnan(auroc_2) else None,
                'difference': float(auroc_1 - auroc_2) if not (np.isnan(auroc_1) or np.isnan(auroc_2)) else None,
                'significant': p_value < 0.05 if not np.isnan(p_value) else None,
            }
        except Exception as e:
            logger.warning(f"DeLong test failed for {class_name}: {e}")
            results['auroc_comparison'][class_name] = {
                'z_statistic': None,
                'p_value': None,
                'auroc_1': None,
                'auroc_2': None,
                'difference': None,
                'significant': None,
            }
        
        # Permutation test for AUPRC
        try:
            auprc_1 = average_precision_score(y_true_i, y_pred_1_i)
            auprc_2 = average_precision_score(y_true_i, y_pred_2_i)
            observed_diff, p_value = permutation_test_auprc(
                y_true_i,
                y_pred_1_i,
                y_pred_2_i,
                n_permutations=n_permutations,
                random_seed=random_seed,
            )
            
            results['auprc_comparison'][class_name] = {
                'observed_diff': observed_diff if not np.isnan(observed_diff) else None,
                'p_value': p_value if not np.isnan(p_value) else None,
                'auprc_1': float(auprc_1) if not np.isnan(auprc_1) else None,
                'auprc_2': float(auprc_2) if not np.isnan(auprc_2) else None,
                'significant': p_value < 0.05 if not np.isnan(p_value) else None,
            }
        except Exception as e:
            logger.warning(f"Permutation test failed for {class_name}: {e}")
            results['auprc_comparison'][class_name] = {
                'observed_diff': None,
                'p_value': None,
                'auprc_1': None,
                'auprc_2': None,
                'significant': None,
            }
    
    return results
