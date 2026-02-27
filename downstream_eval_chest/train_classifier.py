"""
Training framework for CheXpert multi-label classification.
Supports all training strategies: real-only, synthetic-only, augmentation, pretraining.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, IterableDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import json
from tqdm import tqdm
import logging


class MixedDataset(IterableDataset):
    """
    A dataset that combines IterableDataset with regular Dataset.
    This allows concatenating WebDataset (IterableDataset) with regular Dataset.
    Interleaves samples from all datasets instead of exhausting one first.
    """
    def __init__(self, datasets: List[Union[Dataset, IterableDataset]], interleave_ratio: Optional[Dict[int, int]] = None):
        """
        Args:
            datasets: List of datasets to combine
            interleave_ratio: Optional dict mapping dataset index to ratio (e.g., {0: 1, 1: 1} for 1:1).
                            If None, uses round-robin (1:1:1...).
        """
        self.datasets = datasets
        self.is_iterable = [isinstance(ds, IterableDataset) for ds in datasets]
        # Track which dataset is which type for source identification
        self.dataset_sources = []
        for i, ds in enumerate(datasets):
            if isinstance(ds, IterableDataset):
                # Real data (WebDataset)
                self.dataset_sources.append('real')
            else:
                # Check if it's synthetic by looking at data_type attribute
                if hasattr(ds, 'data_type') and ds.data_type == 'synthetic':
                    self.dataset_sources.append('synthetic')
                else:
                    self.dataset_sources.append('real')
        
        # Setup interleaving ratios
        if interleave_ratio is None:
            # Default: round-robin (1:1:1...)
            self.interleave_ratio = {i: 1 for i in range(len(datasets))}
        else:
            self.interleave_ratio = interleave_ratio
    
    def __iter__(self):
        # Create iterators for all datasets
        iterators = []
        dataset_info = []
        for dataset_idx, dataset in enumerate(self.datasets):
            source = self.dataset_sources[dataset_idx]
            dataset_len = len(dataset) if hasattr(dataset, '__len__') else 'unknown'
            dataset_info.append(f"Dataset {dataset_idx}: {source} (len={dataset_len})")
            
            if isinstance(dataset, IterableDataset):
                # For IterableDataset, use it directly
                iterators.append((iter(dataset), source, True))
            else:
                # For regular Dataset, create a cycling iterator
                # Use itertools.cycle to repeat the dataset infinitely
                from itertools import cycle
                indices = list(range(len(dataset)))
                def make_iterator():
                    items_yielded = 0
                    for idx in cycle(indices):
                        item = dataset[idx]
                        if isinstance(item, dict):
                            item['data_source'] = source
                        items_yielded += 1
                        # Log first few items to verify iterator is working
                        if items_yielded <= 3:
                            logger.debug(f"MixedDataset: Yielding {source} item #{items_yielded} from dataset {dataset_idx}")
                        yield item
                iterators.append((make_iterator(), source, False))
        
        logger.info(f"MixedDataset initialized with {len(self.datasets)} datasets: {', '.join(dataset_info)}")
        
        # Track items yielded per dataset for verification
        items_yielded_per_dataset = {i: 0 for i in range(len(self.datasets))}
        
        # Round-robin interleaving with ratios
        # Track current position in ratio cycle for each dataset
        ratio_positions = {i: 0 for i in range(len(self.datasets))}
        
        while True:
            # One interleaving cycle: go through each dataset
            for dataset_idx in range(len(self.datasets)):
                iterator, source, is_iterable = iterators[dataset_idx]
                ratio = self.interleave_ratio.get(dataset_idx, 1)
                
                # Yield 'ratio' number of samples from this dataset
                for _ in range(ratio):
                    try:
                        item = next(iterator)
                        # Ensure data_source is set
                        if isinstance(item, dict):
                            item['data_source'] = source
                        # Track items yielded
                        items_yielded_per_dataset[dataset_idx] += 1
                        # Log first few items from each dataset to verify interleaving
                        if items_yielded_per_dataset[dataset_idx] <= 3:
                            logger.debug(f"MixedDataset: Yielding {source} item (total from this dataset: {items_yielded_per_dataset[dataset_idx]})")
                        yield item
                    except StopIteration:
                        # Should not happen for infinite iterators or cycling finite ones
                        # But handle gracefully
                        if not is_iterable:
                            # Recreate cycling iterator
                            dataset = self.datasets[dataset_idx]
                            from itertools import cycle
                            indices = list(range(len(dataset)))
                            def make_iterator():
                                for idx in cycle(indices):
                                    item = dataset[idx]
                                    if isinstance(item, dict):
                                        item['data_source'] = source
                                    yield item
                            iterators[dataset_idx] = (make_iterator(), source, False)
                            item = next(iterators[dataset_idx][0])
                            if isinstance(item, dict):
                                item['data_source'] = source
                            yield item
                        else:
                            # Infinite iterator exhausted (shouldn't happen)
                            return
    
    def __len__(self):
        """Return approximate length (sum of all dataset lengths)."""
        total = 0
        for dataset in self.datasets:
            if hasattr(dataset, '__len__'):
                total += len(dataset)
            else:
                # For IterableDataset without __len__, we can't know the exact length
                # Return a large number to indicate unknown
                return float('inf')
        return total


class LimitedIterableDataset(IterableDataset):
    """
    Wrapper to limit the number of samples from an IterableDataset.
    This is used for subsetting IterableDataset (like WebDataset) since Subset doesn't work with IterableDataset.
    """
    def __init__(self, dataset: IterableDataset, max_samples: Optional[int] = None):
        """
        Args:
            dataset: The IterableDataset to wrap
            max_samples: Maximum number of samples to yield (None for no limit)
        """
        self.dataset = dataset
        self.max_samples = max_samples
    
    def __iter__(self):
        count = 0
        for item in self.dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break
            yield item
            count += 1
    
    def __len__(self):
        """Return length if available, otherwise return max_samples if set."""
        if self.max_samples is not None:
            return self.max_samples
        if hasattr(self.dataset, '__len__'):
            return len(self.dataset)
        return float('inf')

from .models import DenseNet121Classifier, load_checkpoint
from .dataset import CheXpertClassifierDataset, CheXpertClassifierWebDataset, CHEXPERT_CLASSES
from .data_utils import (
    combine_datasets,
    create_subset,
    compute_mimic_cxr_normalization,
    compute_synthetic_normalization,
    get_synthetic_training_paths,
    get_balanced_dataset_path,
)
from .utils import (
    compute_optimal_threshold,
    compute_auroc_per_label,
    compute_auprc_per_label,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for CheXpert multi-label classification.
    """
    
    def __init__(
        self,
        model: DenseNet121Classifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        initial_lr: float = 0.0001,
        weight_decay: float = 0.05,
        max_epochs: int = 100,
        early_stopping_patience: int = 20,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Training state
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_val_auroc = -1.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.val_aurocs = []
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_samples = 0  # Track total samples processed
        skipped_batches = 0  # Track batches skipped due to all uncertain labels
        label_stats = {"total": 0, "positive": 0, "negative": 0, "uncertain": 0}  # Track label distribution
        # Track data source usage
        data_source_stats = {"real": 0, "synthetic": 0, "unknown": 0}
        # Track losses separately for real vs synthetic (reset each epoch)
        self.real_losses_epoch = []
        self.synthetic_losses_epoch = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            total_samples += batch['image'].shape[0]  # Count samples in this batch
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)  # [B, 14]
            
            # #region agent log
            import json as json_lib
            import time
            num_positive = (labels == 1).sum().item()
            num_negative = (labels == 0).sum().item()
            num_uncertain = (labels == -1).sum().item()
            total_labels = labels.numel()
            label_stats["total"] += total_labels
            label_stats["positive"] += num_positive
            label_stats["negative"] += num_negative
            label_stats["uncertain"] += num_uncertain
            
            # Track data source if available (H12: Compare real vs synthetic)
            data_source_counts = {}
            if 'data_source' in batch:
                sources = batch['data_source']
                if isinstance(sources, (list, tuple)):
                    for src in sources:
                        src_str = str(src)
                        count = data_source_counts.get(src_str, 0) + 1
                        data_source_counts[src_str] = count
                        # Update epoch-level stats
                        if src_str in data_source_stats:
                            data_source_stats[src_str] += 1
                        else:
                            data_source_stats["unknown"] += 1
                else:
                    # Single value
                    src_str = str(sources)
                    count = len(labels)
                    data_source_counts[src_str] = count
                    # Update epoch-level stats
                    if src_str in data_source_stats:
                        data_source_stats[src_str] += count
                    else:
                        data_source_stats["unknown"] += count
            else:
                # No data_source in batch - mark as unknown
                data_source_stats["unknown"] += len(labels)
            
            # Log first 5 batches and every 100th batch
            if num_batches < 5 or num_batches % 100 == 0:
                # Log to main logger with data source info
                if data_source_counts:
                    logger.info(f"Batch {num_batches}: data_source={data_source_counts}, "
                               f"real={data_source_counts.get('real', 0)}, "
                               f"synthetic={data_source_counts.get('synthetic', 0)}")
                else:
                    logger.warning(f"Batch {num_batches}: ⚠️  No data_source field in batch!")
                
                log_data = {
                    "sessionId": "debug-session",
                    "runId": "H4-training",
                    "hypothesisId": "H4",
                    "location": "train_classifier.py:129",
                    "message": "Training batch label statistics",
                    "data": {
                        "batch_idx": num_batches,
                        "batch_size": labels.shape[0],
                        "num_positive": num_positive,
                        "num_negative": num_negative,
                        "num_uncertain": num_uncertain,
                        "total_labels": total_labels,
                        "valid_labels_ratio": (num_positive + num_negative) / total_labels if total_labels > 0 else 0.0,
                        "unique_values": torch.unique(labels).tolist(),
                        "data_source_counts": data_source_counts,
                        "image_mean": images.mean().item(),
                        "image_std": images.std().item(),
                        "image_min": images.min().item(),
                        "image_max": images.max().item()
                    },
                    "timestamp": int(time.time() * 1000)
                }
                with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                    log_file.write(json_lib.dumps(log_data) + "\n")
            # #endregion
            
            # Debug: Check input shape and values (first batch only)
            if num_batches == 0:
                logger.info(f"Input image shape: {images.shape}, dtype: {images.dtype}, min: {images.min():.4f}, max: {images.max():.4f}, mean: {images.mean():.4f}")
                logger.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
                logger.info(f"Valid labels count: {(labels != -1).sum().item()}/{labels.numel()}, unique values: {torch.unique(labels).tolist()}")
                # Log data source information for first batch
                if 'data_source' in batch:
                    logger.info(f"Data source in batch: {data_source_counts}")
                else:
                    logger.warning("⚠️  WARNING: No 'data_source' field in batch! Cannot verify synthetic data usage.")
            
            # Forward pass
            logits = self.model(images)  # [B, 14]
            
            # Check for NaN in logits
            if torch.isnan(logits).any():
                logger.warning(f"NaN detected in model logits at batch {num_batches}, shape: {logits.shape}")
                continue
            
            # Compute loss (no uncertainty masking needed - labels are aggregated to 0 or 1)
            # After aggregation, all labels are 0 (negative) or 1 (positive), so we train via multi-label classification
            loss_per_sample = self.criterion(logits, labels)  # [B, 14]
            
            # Check for NaN in loss_per_sample
            if torch.isnan(loss_per_sample).any():
                logger.warning(f"NaN detected in loss_per_sample at batch {num_batches}")
                logger.warning(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                logger.warning(f"Labels stats: min={labels.min():.4f}, max={labels.max():.4f}, mean={labels.mean():.4f}")
                continue
            
            # All labels are now valid (0 or 1) after aggregation, so compute average loss over all labels
            loss = loss_per_sample.mean()
            
            # #region agent log - H12: Track loss separately for real vs synthetic
            if 'data_source' in batch:
                import json as json_lib
                import time
                # Compute average loss per sample
                avg_loss_per_sample = loss_per_sample.mean(dim=1)  # [B] - average over all 14 labels
                sources = batch['data_source'] if isinstance(batch['data_source'], (list, tuple)) else [batch['data_source']] * len(avg_loss_per_sample)
                real_losses = [avg_loss_per_sample[i].item() for i, src in enumerate(sources) if src == 'real']
                synthetic_losses = [avg_loss_per_sample[i].item() for i, src in enumerate(sources) if src == 'synthetic']
                
                # Track losses for epoch summary
                if real_losses:
                    if not hasattr(self, 'real_losses_epoch'):
                        self.real_losses_epoch = []
                    self.real_losses_epoch.extend(real_losses)
                if synthetic_losses:
                    if not hasattr(self, 'synthetic_losses_epoch'):
                        self.synthetic_losses_epoch = []
                    self.synthetic_losses_epoch.extend(synthetic_losses)
                
                # Log periodically
                if num_batches < 5 or num_batches % 100 == 0:
                    log_data = {
                        "sessionId": "debug-session",
                        "runId": "H12-loss-comparison",
                        "hypothesisId": "H12",
                        "location": "train_classifier.py:197",
                        "message": "Loss comparison: real vs synthetic",
                        "data": {
                            "batch_idx": num_batches,
                            "num_real_samples": len(real_losses),
                            "num_synthetic_samples": len(synthetic_losses),
                            "real_avg_loss": sum(real_losses) / len(real_losses) if real_losses else None,
                            "synthetic_avg_loss": sum(synthetic_losses) / len(synthetic_losses) if synthetic_losses else None,
                            "real_losses": real_losses[:10] if len(real_losses) > 10 else real_losses,
                            "synthetic_losses": synthetic_losses[:10] if len(synthetic_losses) > 10 else synthetic_losses
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                        log_file.write(json_lib.dumps(log_data) + "\n")
            # #endregion
            
            # Check for NaN in final loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf in final loss at batch {num_batches}")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            if any(torch.isnan(p.grad).any() for p in self.model.parameters() if p.grad is not None):
                logger.warning(f"NaN detected in gradients at batch {num_batches}, skipping update")
                self.optimizer.zero_grad()
                continue
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        if num_batches == 0:
            logger.error("No valid batches found in training epoch (all labels uncertain or NaN detected)")
            return float('inf')  # Return infinity to trigger early stopping
        
        # #region agent log
        import json as json_lib
        import time
        log_data = {
            "sessionId": "debug-session",
            "runId": "H4-training",
            "hypothesisId": "H4",
            "location": "train_classifier.py:189",
            "message": "Epoch label statistics summary",
            "data": {
                "num_batches": num_batches,
                "skipped_batches": skipped_batches,
                "total_samples": total_samples,
                "label_stats": label_stats,
                "valid_labels_ratio": (label_stats["positive"] + label_stats["negative"]) / label_stats["total"] if label_stats["total"] > 0 else 0.0,
                "positive_ratio": label_stats["positive"] / (label_stats["positive"] + label_stats["negative"]) if (label_stats["positive"] + label_stats["negative"]) > 0 else 0.0
            },
            "timestamp": int(time.time() * 1000)
        }
        with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
            log_file.write(json_lib.dumps(log_data) + "\n")
        # #endregion
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch training stats: {num_batches} batches, {total_samples} total samples processed, {skipped_batches} skipped batches")
        logger.info(f"Label distribution: {label_stats['positive']} positive, {label_stats['negative']} negative, {label_stats['uncertain']} uncertain out of {label_stats['total']} total")
        # Log data source statistics
        total_tracked = sum(data_source_stats.values())
        if total_tracked > 0:
            real_pct = (data_source_stats['real'] / total_tracked) * 100
            synthetic_pct = (data_source_stats['synthetic'] / total_tracked) * 100
            unknown_pct = (data_source_stats['unknown'] / total_tracked) * 100
            logger.info(f"📊 Data source distribution: Real={data_source_stats['real']} ({real_pct:.1f}%), "
                       f"Synthetic={data_source_stats['synthetic']} ({synthetic_pct:.1f}%), "
                       f"Unknown={data_source_stats['unknown']} ({unknown_pct:.1f}%)")
            if data_source_stats['synthetic'] == 0 and data_source_stats['unknown'] == 0:
                logger.warning("⚠️  WARNING: No synthetic data detected! Only real data was used.")
            elif data_source_stats['synthetic'] > 0:
                logger.info(f"✅ Synthetic data confirmed: {data_source_stats['synthetic']} synthetic samples used")
                
                # Log loss comparison between real and synthetic
                if hasattr(self, 'real_losses_epoch') and hasattr(self, 'synthetic_losses_epoch'):
                    if len(self.real_losses_epoch) > 0 and len(self.synthetic_losses_epoch) > 0:
                        real_avg_loss = np.mean(self.real_losses_epoch)
                        synthetic_avg_loss = np.mean(self.synthetic_losses_epoch)
                        loss_diff = abs(real_avg_loss - synthetic_avg_loss)
                        # Note: These values represent average loss per sample (sum across all classes),
                        # not per-class loss like train_loss. They should be roughly 14x larger than train_loss
                        # if samples have 14 valid labels. If they're much larger, samples may have fewer valid labels.
                        logger.info(f"📉 Loss comparison: Real avg={real_avg_loss:.4f}, Synthetic avg={synthetic_avg_loss:.4f}, Diff={loss_diff:.4f}")
                        logger.debug(f"   (Note: These are per-sample losses, not per-class. Expected ~14x train_loss if samples have 14 valid labels)")
                        if loss_diff > 0.1:
                            logger.warning(f"⚠️  Large loss difference detected! Model learning differently from real vs synthetic data.")
                        if synthetic_avg_loss < real_avg_loss * 0.5:
                            logger.warning(f"⚠️  Synthetic loss much lower than real loss - model may be overfitting to synthetic patterns!")
                        elif synthetic_avg_loss > real_avg_loss * 1.5:
                            logger.warning(f"⚠️  Synthetic loss much higher than real loss - synthetic data may be too difficult or noisy!")
        else:
            logger.warning("⚠️  WARNING: Could not track data source - no data_source field in batches")
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(images)
                
                # Compute loss (no uncertainty masking needed - labels are aggregated to 0 or 1)
                loss_per_sample = self.criterion(logits, labels)
                loss = loss_per_sample.mean()
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf in validation loss, skipping batch")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
        
        aurocs = compute_auroc_per_label(all_labels, all_probs, CHEXPERT_CLASSES)
        
        # Compute mean AUROC (macro-averaged over all 14 labels, excluding NaN)
        valid_aurocs = [v for v in aurocs.values() if not np.isnan(v)]
        mean_auroc = np.mean(valid_aurocs) if valid_aurocs else 0.0
        
        return avg_loss, aurocs, mean_auroc
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        
        for epoch in range(self.max_epochs):
            logger.info(f"Epoch {epoch+1}/{self.max_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_aurocs, mean_auroc = self.validate()
            self.val_losses.append(val_loss)
            self.val_aurocs.append(mean_auroc)
            
            # Update learning rate
            self.scheduler.step()
            
            logger.info(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Val AUROC={mean_auroc:.4f}"
            )
            
            # Check for improvement
            is_best = mean_auroc > self.best_val_auroc
            if is_best:
                self.best_val_auroc = mean_auroc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint every epoch (not just when improved)
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best AUROC: {self.best_val_auroc:.4f} at epoch {self.best_epoch+1}"
                )
                break
        
        logger.info(f"Training completed. Best AUROC: {self.best_val_auroc:.4f}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auroc': self.best_val_auroc,
            'val_aurocs': self.val_aurocs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Save checkpoint for every epoch
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
        
        # Also save as best checkpoint if this is the best epoch
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")


def create_training_dataset(
    strategy: str,
    real_train_path: Union[str, Path],
    synthetic_base_path: Optional[Union[str, Path]],
    mimic_mean: float,
    mimic_std: float,
    model_version: Optional[str] = None,
    dataset_name: Optional[str] = None,
    num_generations: int = 1,
    subset_fraction: Optional[float] = None,
) -> Dataset:
    """
    Create training dataset based on strategy.
    
    Args:
        strategy: Model strategy ('1a', '1b', '1c', '1d', '1e', '2b', '2c', '3a', '4a', '5a', '5b')
        real_train_path: Path to real training data
        synthetic_base_path: Base path to synthetic datasets
        mimic_mean: MIMIC-CXR normalization mean
        mimic_std: MIMIC-CXR normalization std
        model_version: 'v0' or 'v7' for synthetic data
        dataset_name: Dataset name for synthetic data
        num_generations: Number of generations to combine (for 1x, 2x, 3x)
        subset_fraction: Fraction for fine-tuning (for Model 3a)
    
    Returns:
        Combined training dataset
    """
    datasets = []
    
    # Determine real data type first
    train_path_obj = Path(real_train_path)
    if train_path_obj.is_dir() and any(train_path_obj.glob("*.tar")):
        train_data_type = 'real_wds'
    elif train_path_obj.suffix == '.csv':
        train_data_type = 'real_csv'
    else:
        train_data_type = 'real_wds' if train_path_obj.is_dir() else 'real_csv'
    
    # Real data dataset - use streaming WebDataset for WebDataset (like validation)
    if train_data_type == 'real_wds':
        # Use streaming WebDataset (same approach as validation)
        import glob
        tar_files = sorted(glob.glob(str(train_path_obj / "*.tar")))
        if not tar_files:
            tar_files = sorted(glob.glob(str(train_path_obj / "*.tar.gz")))
        if not tar_files:
            raise FileNotFoundError(f"No tar files found in: {train_path_obj}")
        
        from .dataset import CheXpertClassifierWebDataset
        real_dataset = CheXpertClassifierWebDataset(
            url_list=tar_files,
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            shuffle=True,
            is_training=True,  # Enable augmentation for training (matches CSV dataset)
            augmentation=True,
        )
    else:
        # Use regular dataset for CSV
        real_dataset = CheXpertClassifierDataset(
            data_path=train_path_obj,
            data_type=train_data_type,
            split='train',
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            is_training=True,  # Enable augmentation for training
        )
    
    if strategy in ['1a']:
        # Model 1a: Real only
        return real_dataset
    
    elif strategy in ['1b', '1c']:
        # Model 1b: Real + 1× synthetic
        # Model 1c: Real + 2× synthetic
        if synthetic_base_path is None:
            raise ValueError(f"synthetic_base_path required for strategy {strategy}")
        if model_version is None or dataset_name is None:
            raise ValueError(f"model_version and dataset_name required for strategy {strategy}")
        
        synthetic_paths = get_synthetic_training_paths(
            synthetic_base_path,
            model_version,
            dataset_name,
            num_generations=num_generations,
        )
        
        for synth_path in synthetic_paths:
            synth_dataset = CheXpertClassifierDataset(
                data_path=synth_path,
                data_type='synthetic',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,  # Enable augmentation for training
            )
            datasets.append(synth_dataset)
        
        datasets.insert(0, real_dataset)  # Real first
        
        # Log dataset information
        logger.info(f"Creating training dataset for strategy {strategy}:")
        logger.info(f"  - Real dataset: {type(real_dataset).__name__}, len={len(real_dataset) if hasattr(real_dataset, '__len__') else 'unknown'}")
        for i, synth_ds in enumerate(datasets[1:], 1):
            logger.info(f"  - Synthetic dataset {i}: {type(synth_ds).__name__}, len={len(synth_ds) if hasattr(synth_ds, '__len__') else 'unknown'}")
        
        # Check if any dataset is IterableDataset
        has_iterable = any(isinstance(ds, IterableDataset) for ds in datasets)
        if has_iterable:
            # Use MixedDataset to combine IterableDataset with regular Dataset
            logger.info(f"  Using MixedDataset (has IterableDataset: {has_iterable})")
            return MixedDataset(datasets)
        else:
            # All are regular Dataset, use ConcatDataset
            logger.info(f"  Using ConcatDataset (all regular Dataset)")
            return ConcatDataset(datasets)
    
    elif strategy in ['1d', '1e']:
        # Model 1d: Synthetic only 1×
        # Model 1e: Synthetic only 2×
        if synthetic_base_path is None:
            raise ValueError(f"synthetic_base_path required for strategy {strategy}")
        if model_version is None or dataset_name is None:
            raise ValueError(f"model_version and dataset_name required for strategy {strategy}")
        
        synthetic_paths = get_synthetic_training_paths(
            synthetic_base_path,
            model_version,
            dataset_name,
            num_generations=num_generations,
        )
        
        for synth_path in synthetic_paths:
            synth_dataset = CheXpertClassifierDataset(
                data_path=synth_path,
                data_type='synthetic',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,  # Enable augmentation for training
            )
            datasets.append(synth_dataset)
        
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    elif strategy in ['2b', '2c']:
        # Model 2b: Real + 100k balanced
        # Model 2c: 100k balanced only
        if synthetic_base_path is None:
            raise ValueError(f"synthetic_base_path required for strategy {strategy}")
        if model_version is None:
            raise ValueError(f"model_version required for strategy {strategy}")
        
        balanced_path = get_balanced_dataset_path(synthetic_base_path, model_version, dataset_name=dataset_name)
        balanced_dataset = CheXpertClassifierDataset(
            data_path=balanced_path,
            data_type='synthetic',
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            is_training=True,  # Enable augmentation for training
        )
        
        if strategy == '2b':
            datasets = [real_dataset, balanced_dataset]
            # Check if any dataset is IterableDataset
            has_iterable = any(isinstance(ds, IterableDataset) for ds in datasets)
            if has_iterable:
                return MixedDataset(datasets)
            else:
                return ConcatDataset(datasets)
        else:
            return balanced_dataset
    
    elif strategy == '3a':
        # Model 3a: Pretrain on synthetic, fine-tune on real subset
        # This is handled separately in the training function
        if subset_fraction is None:
            raise ValueError("subset_fraction required for strategy 3a")
        return create_subset(real_dataset, subset_fraction)
    
    elif strategy == '4a':
        # Model 4a: Balanced pretrain, fine-tune on real
        # This is handled separately in the training function
        return real_dataset
    
    elif strategy in ['5a', '5b']:
        # Model 5a/5b: Real train only (evaluated on synthetic test)
        return real_dataset
    
    elif strategy == '6a':
        # Model 6a: Real train only (evaluated on multiple test sets)
        return real_dataset
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def train_model(
    strategy: str,
    real_train_path: Path,
    real_val_path: Path,
    synthetic_base_path: Optional[Path] = None,
    output_dir: Path = Path('outputs/downstream_eval'),
    model_version: Optional[str] = None,
    dataset_name: Optional[str] = None,
    num_generations: int = 1,
    subset_fraction: Optional[float] = None,
    from_scratch: bool = False,
    initial_lr: float = 0.0001,
    fine_tune_lr: Optional[float] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    mimic_mean: Optional[float] = None,
    mimic_std: Optional[float] = None,
    normalization_source: str = 'real',
) -> Path:
    """
    Train a model with the specified strategy.
    
    Args:
        mimic_mean: Normalization mean (computed if None)
        mimic_std: Normalization std (computed if None)
        normalization_source: Source for normalization stats ('real' or 'synthetic').
                             Default 'real' uses MIMIC-CXR stats.
                             'synthetic' computes stats from synthetic training data.
                             Only used for synthetic-only models (strategies 1d, 1e, 2c).
    
    Returns:
        Path to best checkpoint
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training strategy: {strategy}")
    logger.info(f"Device: {device}")
    
    # Determine if this is a synthetic-only strategy
    synthetic_only_strategies = ['1d', '1e', '2c']
    use_synthetic_normalization = (
        normalization_source == 'synthetic' and 
        strategy in synthetic_only_strategies
    )
    
    # Compute or use provided normalization
    if mimic_mean is None or mimic_std is None:
        if use_synthetic_normalization:
            # Compute normalization from synthetic data
            if synthetic_base_path is None:
                raise ValueError(f"synthetic_base_path required for synthetic normalization with strategy {strategy}")
            if model_version is None or dataset_name is None:
                raise ValueError(f"model_version and dataset_name required for synthetic normalization with strategy {strategy}")
            
            logger.info("Computing synthetic normalization statistics...")
            synthetic_paths = get_synthetic_training_paths(
                synthetic_base_path,
                model_version,
                dataset_name,
                num_generations=num_generations,
            )
            
            # Use 10k samples for normalization computation (sufficient, avoids OOM)
            num_samples = 10000
            logger.info(f"Computing normalization from {num_samples} synthetic samples")
            mimic_mean, mimic_std = compute_synthetic_normalization(
                synthetic_paths,
                num_samples=num_samples,
            )
            logger.info(f"Computed synthetic normalization: mean={mimic_mean:.6f}, std={mimic_std:.6f}")
        else:
            # Default: compute from real data (MIMIC-CXR)
            logger.info("Computing MIMIC-CXR normalization statistics...")
            # Determine data type for normalization computation
            train_path_obj = Path(real_train_path)
            if train_path_obj.is_dir() and any(train_path_obj.glob("*.tar")):
                norm_data_type = 'real_wds'
            elif train_path_obj.suffix == '.csv':
                norm_data_type = 'real_csv'
            else:
                norm_data_type = 'real_wds' if train_path_obj.is_dir() else 'real_csv'
            
            # Use 10k samples for normalization computation (sufficient, avoids OOM)
            num_samples = 10000
            logger.info(f"Computing normalization from {num_samples} real samples")
            mimic_mean, mimic_std = compute_mimic_cxr_normalization(
                real_train_path,
                data_type=norm_data_type,
                split='train',
                num_samples=num_samples,
            )
            logger.info(f"Computed normalization: mean={mimic_mean:.6f}, std={mimic_std:.6f}")
    else:
        norm_source_str = "synthetic" if use_synthetic_normalization else "real"
        logger.info(f"Using provided normalization ({norm_source_str}): mean={mimic_mean:.6f}, std={mimic_std:.6f}")
    
    # Create datasets
    if strategy == '3a':
        # Pretrain on synthetic, then fine-tune on real subset
        if synthetic_base_path is None:
            raise ValueError("synthetic_base_path required for strategy 3a")
        if model_version is None or dataset_name is None:
            raise ValueError("model_version and dataset_name required for strategy 3a")
        if subset_fraction is None:
            raise ValueError("subset_fraction required for strategy 3a")
        
        # Pretraining phase
        logger.info("Phase 1: Pretraining on synthetic data")
        synthetic_paths = get_synthetic_training_paths(
            synthetic_base_path,
            model_version,
            dataset_name,
            num_generations=num_generations,
        )
        
        pretrain_datasets = []
        for synth_path in synthetic_paths:
            synth_dataset = CheXpertClassifierDataset(
                data_path=synth_path,
                data_type='synthetic',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,  # Enable augmentation for training
            )
            pretrain_datasets.append(synth_dataset)
        
        pretrain_dataset = ConcatDataset(pretrain_datasets) if len(pretrain_datasets) > 1 else pretrain_datasets[0]
        
        # Use all synthetic data for training (no validation split)
        # Check if dataset is IterableDataset
        from torch.utils.data import IterableDataset
        is_iterable_pretrain = isinstance(pretrain_dataset, IterableDataset)
        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=batch_size, 
            shuffle=False if is_iterable_pretrain else True,
            num_workers=num_workers
        )
        
        # Use real validation set for pretraining validation
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            # Use streaming WebDataset (same approach as validation)
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            pretrain_val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,  # No shuffle for validation
            )
        elif val_path_obj.suffix == '.csv':
            pretrain_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            pretrain_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        pretrain_val_loader = DataLoader(
            pretrain_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Create model (from scratch)
        model = DenseNet121Classifier(num_classes=14, from_scratch=True)
        
        # Pretrain
        # Include version in output directory for strategies that use synthetic data
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '4a', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            pretrain_output_dir = output_dir / f'model_{strategy}_{model_version}_pretrain'
        else:
            pretrain_output_dir = output_dir / f'model_{strategy}_pretrain'
        trainer = Trainer(
            model=model,
            train_loader=pretrain_loader,
            val_loader=pretrain_val_loader,
            device=device,
            output_dir=pretrain_output_dir,
            initial_lr=initial_lr,
        )
        trainer.train()
        
        # Load best pretrained model
        pretrain_checkpoint = pretrain_output_dir / 'checkpoint_best.pth'
        model = load_checkpoint(model, str(pretrain_checkpoint), device)
        
        # Fine-tuning phase
        logger.info(f"Phase 2: Fine-tuning on real data (subset fraction: {subset_fraction})")
        # Determine data type - use streaming WebDataset for WebDataset (like other strategies)
        train_path_obj = Path(real_train_path)
        if train_path_obj.is_dir() and any(train_path_obj.glob("*.tar")):
            train_data_type = 'real_wds'
        elif train_path_obj.suffix == '.csv':
            train_data_type = 'real_csv'
        else:
            train_data_type = 'real_wds' if train_path_obj.is_dir() else 'real_csv'
        
        # Use streaming WebDataset for WebDataset data (same approach as create_training_dataset)
        if train_data_type == 'real_wds':
            import glob
            tar_files = sorted(glob.glob(str(train_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(train_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {train_path_obj}")
            
            real_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=True,
                is_training=True,  # Enable augmentation for training (matches CSV dataset)
                augmentation=True,
            )
            # For IterableDataset, use LimitedIterableDataset for subsetting
            if subset_fraction < 1.0:
                total_size = len(real_dataset)
                max_samples = int(total_size * subset_fraction)
                fine_tune_dataset = LimitedIterableDataset(real_dataset, max_samples=max_samples)
            else:
                fine_tune_dataset = real_dataset
            is_iterable_finetune = True
        else:
            # Use regular dataset for CSV
            real_dataset = CheXpertClassifierDataset(
                data_path=train_path_obj,
                data_type=train_data_type,
                split='train',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,  # Enable augmentation for training
            )
            # For regular Dataset, use Subset for subsetting
            fine_tune_dataset = create_subset(real_dataset, subset_fraction)
            is_iterable_finetune = False
        
        fine_tune_loader = DataLoader(
            fine_tune_dataset, 
            batch_size=batch_size, 
            shuffle=False if is_iterable_finetune else True,
            num_workers=num_workers
        )
        
        # Use real validation set
        # Determine data type for validation - use streaming WebDataset if WebDataset
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            # Use streaming WebDataset (same approach as validation)
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,  # No shuffle for validation
            )
        elif val_path_obj.suffix == '.csv':
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Fine-tune with lower learning rate
        fine_tune_lr = fine_tune_lr if fine_tune_lr is not None else initial_lr * 0.5
        # Include version in output directory for strategies that use synthetic data
        # Format subset_fraction consistently (e.g., 1.0 instead of 1)
        subset_fraction_str = f"{subset_fraction:.1f}" if isinstance(subset_fraction, float) else str(subset_fraction)
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '4a', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            fine_tune_output_dir = output_dir / f'model_{strategy}_{model_version}_finetune_{subset_fraction_str}'
        else:
            fine_tune_output_dir = output_dir / f'model_{strategy}_finetune_{subset_fraction_str}'
        trainer = Trainer(
            model=model,
            train_loader=fine_tune_loader,
            val_loader=val_loader,
            device=device,
            output_dir=fine_tune_output_dir,
            initial_lr=fine_tune_lr,
        )
        trainer.train()
        
        return fine_tune_output_dir / 'checkpoint_best.pth'
    
    elif strategy == '4a':
        # Balanced pretrain, then fine-tune on real
        if synthetic_base_path is None:
            raise ValueError("synthetic_base_path required for strategy 4a")
        if model_version is None:
            raise ValueError("model_version required for strategy 4a")
        
        # Pretraining on balanced synthetic
        logger.info("Phase 1: Pretraining on balanced synthetic data")
        balanced_path = get_balanced_dataset_path(synthetic_base_path, model_version, dataset_name=dataset_name)
        balanced_dataset = CheXpertClassifierDataset(
            data_path=balanced_path,
            data_type='synthetic',
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            is_training=True,  # Enable augmentation for training
        )
        
        # Use all balanced data for training (no validation split)
        balanced_loader = DataLoader(
            balanced_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        # Use real validation set for pretraining validation
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            # Use streaming WebDataset (same approach as validation)
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            balanced_val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,  # No shuffle for validation
            )
        elif val_path_obj.suffix == '.csv':
            balanced_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            balanced_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        balanced_val_loader = DataLoader(
            balanced_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        model = DenseNet121Classifier(num_classes=14, from_scratch=True)
        
        # Include version in output directory for strategies that use synthetic data
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '4a', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            pretrain_output_dir = output_dir / f'model_{strategy}_{model_version}_pretrain'
        else:
            pretrain_output_dir = output_dir / f'model_{strategy}_pretrain'
        trainer = Trainer(
            model=model,
            train_loader=balanced_loader,
            val_loader=balanced_val_loader,
            device=device,
            output_dir=pretrain_output_dir,
            initial_lr=initial_lr,
        )
        trainer.train()
        
        # Load best pretrained model
        pretrain_checkpoint = pretrain_output_dir / 'checkpoint_best.pth'
        model = load_checkpoint(model, str(pretrain_checkpoint), device)
        
        # Fine-tune on real
        logger.info("Phase 2: Fine-tuning on real data")
        # Determine data type - use streaming WebDataset for WebDataset (like other strategies)
        train_path_obj = Path(real_train_path)
        if train_path_obj.is_dir() and any(train_path_obj.glob("*.tar")):
            train_data_type = 'real_wds'
        elif train_path_obj.suffix == '.csv':
            train_data_type = 'real_csv'
        else:
            train_data_type = 'real_wds' if train_path_obj.is_dir() else 'real_csv'
        
        # Use streaming WebDataset for WebDataset data (same approach as create_training_dataset)
        if train_data_type == 'real_wds':
            import glob
            tar_files = sorted(glob.glob(str(train_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(train_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {train_path_obj}")
            
            real_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=True,
                is_training=True,  # Enable augmentation for training (matches CSV dataset)
                augmentation=True,
            )
            is_iterable_finetune = True
        else:
            # Use regular dataset for CSV
            real_dataset = CheXpertClassifierDataset(
                data_path=train_path_obj,
                data_type=train_data_type,
                split='train',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,  # Enable augmentation for training
            )
            is_iterable_finetune = False
        
        train_loader = DataLoader(
            real_dataset, 
            batch_size=batch_size, 
            shuffle=False if is_iterable_finetune else True,
            num_workers=num_workers
        )
        
        # Determine data type for validation - use streaming WebDataset if WebDataset
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            # Use streaming WebDataset (same approach as validation)
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,  # No shuffle for validation
            )
        elif val_path_obj.suffix == '.csv':
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        fine_tune_lr = fine_tune_lr if fine_tune_lr is not None else initial_lr * 0.5
        # Include version in output directory for strategies that use synthetic data
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '4a', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            fine_tune_output_dir = output_dir / f'model_{strategy}_{model_version}_finetune'
        else:
            fine_tune_output_dir = output_dir / f'model_{strategy}_finetune'
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=fine_tune_output_dir,
            initial_lr=fine_tune_lr,
        )
        trainer.train()
        
        return fine_tune_output_dir / 'checkpoint_best.pth'
    
    elif strategy == '3b':
        # Model 3b: Like 3a but with ImageNet pretrained weights instead of from scratch
        # Pretrain on synthetic (starting from ImageNet), then fine-tune on real subset
        if synthetic_base_path is None:
            raise ValueError("synthetic_base_path required for strategy 3b")
        if model_version is None or dataset_name is None:
            raise ValueError("model_version and dataset_name required for strategy 3b")
        if subset_fraction is None:
            raise ValueError("subset_fraction required for strategy 3b")
        
        # Pretraining phase (starting from ImageNet weights)
        logger.info("Phase 1: Pretraining on synthetic data (initialized from ImageNet)")
        synthetic_paths = get_synthetic_training_paths(
            synthetic_base_path,
            model_version,
            dataset_name,
            num_generations=num_generations,
        )
        
        pretrain_datasets = []
        for synth_path in synthetic_paths:
            synth_dataset = CheXpertClassifierDataset(
                data_path=synth_path,
                data_type='synthetic',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,  # Enable augmentation for training
            )
            pretrain_datasets.append(synth_dataset)
        
        pretrain_dataset = ConcatDataset(pretrain_datasets) if len(pretrain_datasets) > 1 else pretrain_datasets[0]
        
        # Check if dataset is IterableDataset
        from torch.utils.data import IterableDataset
        is_iterable_pretrain = isinstance(pretrain_dataset, IterableDataset)
        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=batch_size, 
            shuffle=False if is_iterable_pretrain else True,
            num_workers=num_workers
        )
        
        # Use real validation set for pretraining validation
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            pretrain_val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,
            )
        elif val_path_obj.suffix == '.csv':
            pretrain_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            pretrain_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        pretrain_val_loader = DataLoader(
            pretrain_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Create model with ImageNet pretrained weights (key difference from 3a)
        model = DenseNet121Classifier(num_classes=14, from_scratch=False, pretrained=True)
        logger.info("Model initialized with ImageNet pretrained weights")
        
        # Pretrain
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            pretrain_output_dir = output_dir / f'model_{strategy}_{model_version}_pretrain'
        else:
            pretrain_output_dir = output_dir / f'model_{strategy}_pretrain'
        trainer = Trainer(
            model=model,
            train_loader=pretrain_loader,
            val_loader=pretrain_val_loader,
            device=device,
            output_dir=pretrain_output_dir,
            initial_lr=initial_lr,
        )
        trainer.train()
        
        # Load best pretrained model
        pretrain_checkpoint = pretrain_output_dir / 'checkpoint_best.pth'
        model = load_checkpoint(model, str(pretrain_checkpoint), device)
        
        # Fine-tuning phase
        logger.info(f"Phase 2: Fine-tuning on real data (subset fraction: {subset_fraction})")
        train_path_obj = Path(real_train_path)
        if train_path_obj.is_dir() and any(train_path_obj.glob("*.tar")):
            train_data_type = 'real_wds'
        elif train_path_obj.suffix == '.csv':
            train_data_type = 'real_csv'
        else:
            train_data_type = 'real_wds' if train_path_obj.is_dir() else 'real_csv'
        
        if train_data_type == 'real_wds':
            import glob
            tar_files = sorted(glob.glob(str(train_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(train_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {train_path_obj}")
            
            real_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=True,
                is_training=True,
                augmentation=True,
            )
            if subset_fraction < 1.0:
                total_size = len(real_dataset)
                max_samples = int(total_size * subset_fraction)
                fine_tune_dataset = LimitedIterableDataset(real_dataset, max_samples=max_samples)
            else:
                fine_tune_dataset = real_dataset
            is_iterable_finetune = True
        else:
            real_dataset = CheXpertClassifierDataset(
                data_path=train_path_obj,
                data_type=train_data_type,
                split='train',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,
            )
            fine_tune_dataset = create_subset(real_dataset, subset_fraction)
            is_iterable_finetune = False
        
        fine_tune_loader = DataLoader(
            fine_tune_dataset, 
            batch_size=batch_size, 
            shuffle=False if is_iterable_finetune else True,
            num_workers=num_workers
        )
        
        # Use real validation set
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,
            )
        elif val_path_obj.suffix == '.csv':
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        fine_tune_lr = fine_tune_lr if fine_tune_lr is not None else initial_lr * 0.5
        subset_fraction_str = f"{subset_fraction:.1f}" if isinstance(subset_fraction, float) else str(subset_fraction)
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            fine_tune_output_dir = output_dir / f'model_{strategy}_{model_version}_finetune_{subset_fraction_str}'
        else:
            fine_tune_output_dir = output_dir / f'model_{strategy}_finetune_{subset_fraction_str}'
        trainer = Trainer(
            model=model,
            train_loader=fine_tune_loader,
            val_loader=val_loader,
            device=device,
            output_dir=fine_tune_output_dir,
            initial_lr=fine_tune_lr,
        )
        trainer.train()
        
        return fine_tune_output_dir / 'checkpoint_best.pth'
    
    elif strategy == '4b':
        # Model 4b: Like 4a but with ImageNet pretrained weights instead of from scratch
        # Balanced pretrain (starting from ImageNet), then fine-tune on real
        if synthetic_base_path is None:
            raise ValueError("synthetic_base_path required for strategy 4b")
        if model_version is None:
            raise ValueError("model_version required for strategy 4b")
        
        # Pretraining on balanced synthetic (starting from ImageNet weights)
        logger.info("Phase 1: Pretraining on balanced synthetic data (initialized from ImageNet)")
        balanced_path = get_balanced_dataset_path(synthetic_base_path, model_version, dataset_name=dataset_name)
        balanced_dataset = CheXpertClassifierDataset(
            data_path=balanced_path,
            data_type='synthetic',
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            is_training=True,
        )
        
        balanced_loader = DataLoader(
            balanced_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        # Use real validation set for pretraining validation
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            balanced_val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,
            )
        elif val_path_obj.suffix == '.csv':
            balanced_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            balanced_val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        balanced_val_loader = DataLoader(
            balanced_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Create model with ImageNet pretrained weights (key difference from 4a)
        model = DenseNet121Classifier(num_classes=14, from_scratch=False, pretrained=True)
        logger.info("Model initialized with ImageNet pretrained weights")
        
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            pretrain_output_dir = output_dir / f'model_{strategy}_{model_version}_pretrain'
        else:
            pretrain_output_dir = output_dir / f'model_{strategy}_pretrain'
        trainer = Trainer(
            model=model,
            train_loader=balanced_loader,
            val_loader=balanced_val_loader,
            device=device,
            output_dir=pretrain_output_dir,
            initial_lr=initial_lr,
        )
        trainer.train()
        
        # Load best pretrained model
        pretrain_checkpoint = pretrain_output_dir / 'checkpoint_best.pth'
        model = load_checkpoint(model, str(pretrain_checkpoint), device)
        
        # Fine-tune on real
        logger.info("Phase 2: Fine-tuning on real data")
        train_path_obj = Path(real_train_path)
        if train_path_obj.is_dir() and any(train_path_obj.glob("*.tar")):
            train_data_type = 'real_wds'
        elif train_path_obj.suffix == '.csv':
            train_data_type = 'real_csv'
        else:
            train_data_type = 'real_wds' if train_path_obj.is_dir() else 'real_csv'
        
        if train_data_type == 'real_wds':
            import glob
            tar_files = sorted(glob.glob(str(train_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(train_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {train_path_obj}")
            
            real_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=True,
                is_training=True,
                augmentation=True,
            )
            is_iterable_finetune = True
        else:
            real_dataset = CheXpertClassifierDataset(
                data_path=train_path_obj,
                data_type=train_data_type,
                split='train',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                is_training=True,
            )
            is_iterable_finetune = False
        
        train_loader = DataLoader(
            real_dataset, 
            batch_size=batch_size, 
            shuffle=False if is_iterable_finetune else True,
            num_workers=num_workers
        )
        
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,
            )
        elif val_path_obj.suffix == '.csv':
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        fine_tune_lr = fine_tune_lr if fine_tune_lr is not None else initial_lr * 0.5
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '3b', '4a', '4b', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            fine_tune_output_dir = output_dir / f'model_{strategy}_{model_version}_finetune'
        else:
            fine_tune_output_dir = output_dir / f'model_{strategy}_finetune'
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=fine_tune_output_dir,
            initial_lr=fine_tune_lr,
        )
        trainer.train()
        
        return fine_tune_output_dir / 'checkpoint_best.pth'
    
    else:
        # Standard training (1a, 1b, 1c, 1d, 1e, 2b, 2c, 5a, 5b)
        train_dataset = create_training_dataset(
            strategy=strategy,
            real_train_path=real_train_path,
            synthetic_base_path=synthetic_base_path,
            mimic_mean=mimic_mean,
            mimic_std=mimic_std,
            model_version=model_version,
            dataset_name=dataset_name,
            num_generations=num_generations,
        )
        
        # Check if dataset is IterableDataset (shuffling handled by dataset pipeline)
        from torch.utils.data import IterableDataset
        is_iterable = isinstance(train_dataset, IterableDataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False if is_iterable else True,  # IterableDataset handles shuffling internally
            num_workers=num_workers
        )
        
        # Determine data type for validation - use streaming WebDataset if WebDataset
        val_path_obj = Path(real_val_path)
        if val_path_obj.is_dir() and any(val_path_obj.glob("*.tar")):
            # Use streaming WebDataset (same approach as validation)
            import glob
            tar_files = sorted(glob.glob(str(val_path_obj / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(val_path_obj / "*.tar.gz")))
            if not tar_files:
                raise FileNotFoundError(f"No tar files found in: {val_path_obj}")
            
            val_dataset = CheXpertClassifierWebDataset(
                url_list=tar_files,
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
                shuffle=False,  # No shuffle for validation
            )
        elif val_path_obj.suffix == '.csv':
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        else:
            val_dataset = CheXpertClassifierDataset(
                data_path=val_path_obj,
                data_type='real_csv',
                split='val',
                mimic_mean=mimic_mean,
                mimic_std=mimic_std,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Create model
        model = DenseNet121Classifier(
            num_classes=14,
            pretrained=not from_scratch,
            from_scratch=from_scratch,
        )
        
        # Train
        # Include version in output directory for all strategies that use synthetic data
        strategies_with_synthetic = ['1b', '1c', '1d', '1e', '2b', '2c', '3a', '4a', '5a', '5b']
        if strategy in strategies_with_synthetic and model_version:
            output_dir_model = output_dir / f'model_{strategy}_{model_version}'
        else:
            output_dir_model = output_dir / f'model_{strategy}'
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir_model,
            initial_lr=initial_lr,
        )
        trainer.train()
        
        return output_dir_model / 'checkpoint_best.pth'

