"""
Data utilities for downstream evaluation:
- Computing MIMIC-CXR normalization statistics
- Combining datasets (real + synthetic, multiple generations)
- Creating subsets for fine-tuning
- Loading synthetic test sets
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob
import json
from tqdm import tqdm

from .dataset import CheXpertClassifierDataset


def compute_synthetic_normalization(
    synthetic_paths: List[Union[str, Path]],
    num_samples: Optional[int] = None,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Compute mean and standard deviation from synthetic training data.
    
    Args:
        synthetic_paths: List of paths to synthetic training directories
        num_samples: Number of samples to use (None = all, or use subset for large datasets)
                     Default: 1000 (reduced from 10000 for faster computation)
        batch_size: Batch size for processing
    
    Returns:
        (mean, std) tuple for single-channel grayscale normalization
    """
    print("Computing synthetic normalization statistics...")
    
    # Use fewer samples by default for synthetic data (it's slower to load individual files)
    # 1000 samples is sufficient for accurate statistics and much faster
    if num_samples is None:
        num_samples = 1000
        print(f"Using default {num_samples} samples for synthetic normalization (faster than 10000)")
    
    # Create datasets for all synthetic paths
    # Use mimic_mean=0.0 and mimic_std=1.0 to get raw pixel values (no normalization)
    # This is needed because we're computing normalization stats from the data itself
    datasets = []
    for synth_path in synthetic_paths:
        synth_dataset = CheXpertClassifierDataset(
            data_path=Path(synth_path),
            data_type='synthetic',
            is_training=False,  # No augmentation for normalization computation
            mimic_mean=0.0,  # Disable normalization for stat computation
            mimic_std=1.0,   # Disable normalization for stat computation
        )
        datasets.append(synth_dataset)
    
    # Combine datasets
    if len(datasets) > 1:
        combined_dataset = ConcatDataset(datasets)
    else:
        combined_dataset = datasets[0]
    
    # Get dataset length to check if we need subsetting
    dataset_len = len(combined_dataset)
    print(f"Total synthetic samples available: {dataset_len}")
    
    # Create subset if needed (use random sampling for better statistics)
    if num_samples is not None and dataset_len > num_samples:
        print(f"Sampling {num_samples} random samples from {dataset_len} total samples")
        indices = torch.randperm(dataset_len)[:num_samples]
        combined_dataset = torch.utils.data.Subset(combined_dataset, indices)
    
    # Compute mean and std using running statistics to avoid OOM
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    total_count = 0
    
    # Use more workers for faster loading (synthetic data is I/O bound)
    # But limit to avoid too many file handles
    num_workers = min(4, batch_size)
    dataloader = torch.utils.data.DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer if available
    )
    
    print(f"Processing with batch_size={batch_size}, num_workers={num_workers}")
    processed_batches = 0
    for batch in tqdm(dataloader, desc="Computing synthetic statistics"):
        images = batch['image']  # [B, 1, 224, 224]
        # Flatten to [B*H*W]
        pixels = images.flatten()
        count = pixels.numel()
        sum_pixels += pixels.sum().item()
        sum_sq_pixels += (pixels ** 2).sum().item()
        total_count += count
        processed_batches += 1
        
        # Log progress every 10 batches
        if processed_batches % 10 == 0:
            print(f"  Processed {processed_batches} batches, {total_count // (224*224)} samples so far...")
    
    if total_count == 0:
        raise ValueError("No valid samples processed for synthetic normalization computation")
    
    mean = sum_pixels / total_count
    variance = (sum_sq_pixels / total_count) - (mean ** 2)
    std = float(np.sqrt(variance))
    
    print(f"Synthetic normalization: mean={mean:.4f}, std={std:.4f} (from {total_count // (224*224)} samples)")
    return mean, std


def compute_mimic_cxr_normalization(
    data_path: Union[str, Path],
    data_type: str = 'real_csv',
    split: str = 'train',
    num_samples: Optional[int] = None,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Compute mean and standard deviation from MIMIC-CXR training set.
    
    Args:
        data_path: Path to training data
        data_type: 'real_csv' or 'real_wds'
        split: 'train' to use training data
        num_samples: Number of samples to use (None = all, or use subset for large datasets)
        batch_size: Batch size for processing
    
    Returns:
        (mean, std) tuple for single-channel grayscale normalization
    """
    print("Computing MIMIC-CXR normalization statistics...")
    
    # Determine data type if not specified
    data_path_obj = Path(data_path)
    if data_type is None:
        if data_path_obj.is_dir() and any(data_path_obj.glob("*.tar")):
            data_type = 'real_wds'
        elif data_path_obj.suffix == '.csv':
            data_type = 'real_csv'
        else:
            data_type = 'real_wds' if data_path_obj.is_dir() else 'real_csv'
    
    # For WebDataset, use streaming approach to avoid OOM
    if data_type == 'real_wds':
        return _compute_normalization_wds_streaming(
            data_path_obj, num_samples=num_samples or 10000, batch_size=batch_size
        )
    
    # For CSV, use dataset approach
    def center_crop_fn(img):
        """Center crop to square (min dimension)."""
        if img.dim() == 3:
            C, H, W = img.shape
            min_dim = min(H, W)
            top = (H - min_dim) // 2
            left = (W - min_dim) // 2
            return img[:, top:top+min_dim, left:left+min_dim]
        return img
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Lambda(center_crop_fn),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        # Convert RGB to grayscale
        transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True) if x.shape[0] == 3 else x),
    ])
    
    dataset = CheXpertClassifierDataset(
        data_path=data_path_obj,
        data_type=data_type,
        split=split,
        transform=transform,
    )
    
    if num_samples is not None:
        # Create subset
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Compute mean and std using running statistics to avoid OOM
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    total_count = 0
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2  # Reduced workers
    )
    
    for batch in tqdm(dataloader, desc="Computing statistics"):
        images = batch['image']  # [B, 1, 224, 224]
        # Flatten to [B*H*W]
        pixels = images.flatten()
        count = pixels.numel()
        sum_pixels += pixels.sum().item()
        sum_sq_pixels += (pixels ** 2).sum().item()
        total_count += count
    
    mean = sum_pixels / total_count
    variance = (sum_sq_pixels / total_count) - (mean ** 2)
    std = float(np.sqrt(variance))
    
    print(f"MIMIC-CXR normalization: mean={mean:.4f}, std={std:.4f}")
    return mean, std


def _compute_normalization_wds_streaming(
    data_path: Path,
    num_samples: int = 10000,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Compute normalization statistics by streaming WebDataset samples (memory efficient).
    """
    import webdataset as wds
    import pickle
    from torchvision import transforms
    
    # Find tar files
    tar_files = sorted(glob.glob(str(data_path / "*.tar")))
    if not tar_files:
        tar_files = sorted(glob.glob(str(data_path / "*.tar.gz")))
    
    if not tar_files:
        raise FileNotFoundError(f"No tar files found in: {data_path}")
    
    # Create WebDataset pipeline
    dataset = wds.WebDataset(tar_files)
    
    # Transform function for preprocessing
    def center_crop_tensor(img_tensor):
        """Center crop tensor to square."""
        if img_tensor.dim() == 3:
            C, H, W = img_tensor.shape
            if H == 0 or W == 0:
                return None
            min_dim = min(H, W)
            if min_dim == 0:
                return None
            top = (H - min_dim) // 2
            left = (W - min_dim) // 2
            return img_tensor[:, top:top+min_dim, left:left+min_dim]
        elif img_tensor.dim() == 2:
            # Handle 2D tensor [H, W]
            H, W = img_tensor.shape
            if H == 0 or W == 0:
                return None
            min_dim = min(H, W)
            if min_dim == 0:
                return None
            top = (H - min_dim) // 2
            left = (W - min_dim) // 2
            return img_tensor[top:top+min_dim, left:left+min_dim].unsqueeze(0)  # [1, H, W]
        return img_tensor
    
    def resize_tensor(img_tensor, size=(224, 224)):
        """Resize tensor using interpolation."""
        import torch.nn.functional as F
        
        if img_tensor is None:
            return None
        
        # Ensure tensor has valid dimensions
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]
        elif img_tensor.dim() == 3:
            C, H, W = img_tensor.shape
            if H == 0 or W == 0:
                return None
            img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
        elif img_tensor.dim() == 4:
            pass  # Already [B, C, H, W]
        else:
            return None
        
        # Check if tensor has valid spatial dimensions
        if img_tensor.dim() < 4:
            return None
        
        _, _, H, W = img_tensor.shape
        if H == 0 or W == 0:
            return None
        
        img_tensor = F.interpolate(img_tensor, size=size, mode='bilinear', align_corners=False)
        return img_tensor.squeeze(0) if img_tensor.shape[0] == 1 else img_tensor
    
    # Running statistics
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    total_count = 0
    processed = 0
    
    # Process samples in batches
    batch_images = []
    
    for item in tqdm(dataset, desc="Computing statistics (streaming)", total=num_samples):
        if processed >= num_samples:
            break
        
        try:
            # Load image tensor
            img_tensor = pickle.loads(item['pt_image'])  # [1, H, W] or [C, H, W] in [0, 1]
            
            # Validate tensor
            if not isinstance(img_tensor, torch.Tensor):
                continue
            if img_tensor.numel() == 0:
                continue
            
            # Ensure tensor is in [0, 1] range and has valid dimensions
            if img_tensor.dim() == 2:
                # [H, W] -> [1, H, W]
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.dim() == 3:
                C, H, W = img_tensor.shape
                if C == 0 or H == 0 or W == 0:
                    continue
                # Convert to RGB if single channel
                if C == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1)
            else:
                continue  # Skip invalid dimensions
            
            # Apply preprocessing
            img_tensor = center_crop_tensor(img_tensor)
            if img_tensor is None:
                continue
            
            img_tensor = resize_tensor(img_tensor, size=(224, 224))
            if img_tensor is None:
                continue
            
            # Ensure we have [C, H, W] format
            if img_tensor.dim() != 3:
                continue
            
            # Convert to grayscale
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor.mean(dim=0, keepdim=True)  # [1, 224, 224]
            elif img_tensor.shape[0] != 1:
                continue  # Skip if not 1 or 3 channels
            
            # Final validation
            if img_tensor.shape != (1, 224, 224):
                continue
            
            batch_images.append(img_tensor)
            processed += 1
            
            # Process batch when full
            if len(batch_images) >= batch_size:
                batch_tensor = torch.stack(batch_images)  # [B, 1, 224, 224]
                pixels = batch_tensor.flatten()
                count = pixels.numel()
                sum_pixels += pixels.sum().item()
                sum_sq_pixels += (pixels ** 2).sum().item()
                total_count += count
                batch_images = []
        
        except Exception as e:
            print(f"Warning: Failed to process sample: {e}")
            continue
    
    # Process remaining samples
    if batch_images:
        batch_tensor = torch.stack(batch_images)
        pixels = batch_tensor.flatten()
        count = pixels.numel()
        sum_pixels += pixels.sum().item()
        sum_sq_pixels += (pixels ** 2).sum().item()
        total_count += count
    
    if total_count == 0:
        raise ValueError("No valid samples processed for normalization computation")
    
    mean = sum_pixels / total_count
    variance = (sum_sq_pixels / total_count) - (mean ** 2)
    std = float(np.sqrt(variance))
    
    print(f"MIMIC-CXR normalization (streaming): mean={mean:.4f}, std={std:.4f} (from {processed} samples)")
    return mean, std


def combine_datasets(
    datasets: List[Dataset],
    shuffle: bool = True,
) -> ConcatDataset:
    """
    Combine multiple datasets into one.
    
    Args:
        datasets: List of datasets to combine
        shuffle: Whether to shuffle the combined dataset
    
    Returns:
        ConcatDataset combining all input datasets
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset required")
    if len(datasets) == 1:
        return datasets[0]
    
    combined = ConcatDataset(datasets)
    
    if shuffle:
        # Create a shuffled index mapping
        indices = torch.randperm(len(combined))
        # Note: ConcatDataset doesn't support direct shuffling,
        # so we'd need a wrapper dataset for this
        # For now, we'll return the combined dataset and let DataLoader handle shuffling
    
    return combined


def create_subset(
    dataset: Dataset,
    fraction: float,
    shuffle: bool = True,
) -> torch.utils.data.Subset:
    """
    Create a subset of a dataset.
    
    Args:
        dataset: Dataset to subset
        fraction: Fraction of data to keep (e.g., 0.1 for 10%)
        shuffle: Whether to shuffle before selecting subset
    
    Returns:
        Subset dataset
    """
    total_size = len(dataset)
    subset_size = int(total_size * fraction)
    
    if shuffle:
        indices = torch.randperm(total_size)[:subset_size]
    else:
        indices = torch.arange(subset_size)
    
    return torch.utils.data.Subset(dataset, indices)


def load_synthetic_test_set(
    test_path: Union[str, Path],
    mimic_mean: Optional[float] = None,
    mimic_std: Optional[float] = None,
) -> CheXpertClassifierDataset:
    """
    Load synthetic test set from outputs_summarized directory.
    
    Args:
        test_path: Path to test_images directory (e.g., outputs_summarized/v0/0_train_baseline/test_images/step_10000/)
        mimic_mean: Mean for normalization
        mimic_std: Std for normalization
    
    Returns:
        CheXpertClassifierDataset loaded with synthetic test data
    """
    test_path = Path(test_path)
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test path does not exist: {test_path}")
    
    return CheXpertClassifierDataset(
        data_path=test_path,
        data_type='synthetic',
        transform=None,  # Will use default
        mimic_mean=mimic_mean,
        mimic_std=mimic_std,
    )


def get_synthetic_training_paths(
    base_path: Union[str, Path],
    model_version: str,  # 'v0' or 'v7'
    dataset_name: str,  # e.g., '0_train_baseline' or '6_train_hcn_age_from_promt'
    num_generations: int = 1,  # 1, 2, or 3
) -> List[Path]:
    """
    Get paths to synthetic training datasets for combining multiple generations.
    
    Args:
        base_path: Base path to synthetic_datasets directory
        model_version: 'v0' or 'v7'
        dataset_name: Dataset name (e.g., '0_train_baseline')
        num_generations: Number of generations to combine (1, 2, or 3)
    
    Returns:
        List of paths to synthetic training directories
    """
    base_path = Path(base_path)
    model_path = base_path / model_version / dataset_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Look for training directories (may be numbered or have different names)
    # Common patterns: train_0, train_1, train_2 or generation_0, etc.
    training_dirs = []
    
    # First, try to find directories matching the train-*-copy* pattern
    import glob
    train_copy_pattern = str(model_path / "train-*-copy*")
    train_copy_dirs = sorted(glob.glob(train_copy_pattern))
    for match in train_copy_dirs:
        candidate = Path(match)
        if candidate.exists() and candidate.is_dir():
            training_dirs.append(candidate)
    
    # If found train-*-copy* directories, use them (sorted by name)
    if training_dirs:
        # Sort by copy number (extract number from name)
        def get_copy_number(path):
            import re
            match = re.search(r'copy(\d+)', str(path))
            return int(match.group(1)) if match else 0
        training_dirs = sorted(training_dirs, key=get_copy_number)
        if len(training_dirs) >= num_generations:
            return training_dirs[:num_generations]
        # If not enough, continue to check other patterns
    
    # Try different naming patterns
    patterns = [
        f"train_{i}" for i in range(num_generations)
    ] + [
        f"generation_{i}" for i in range(num_generations)
    ] + [
        f"train_images_gen_{i}" for i in range(num_generations)
    ]
    
    for pattern in patterns:
        candidate = model_path / pattern
        if candidate.exists() and candidate.is_dir():
            training_dirs.append(candidate)
    
    # Remove duplicates and sort
    training_dirs = sorted(list(set(training_dirs)), key=lambda x: str(x))
    
    # If no numbered directories found, check if there's a single training directory
    if not training_dirs:
        # Check for common single directory names
        for name in ['train_images', 'training', 'train']:
            candidate = model_path / name
            if candidate.exists() and candidate.is_dir():
                # If only one generation requested, return this
                if num_generations == 1:
                    return [candidate]
                else:
                    # For multiple generations, we might need to duplicate or find subdirectories
                    # This depends on the actual structure
                    print(f"Warning: Found single training directory {candidate}, but {num_generations} generations requested")
                    return [candidate] * num_generations
    
    if len(training_dirs) < num_generations:
        raise ValueError(
            f"Found {len(training_dirs)} training directories, but {num_generations} requested. "
            f"Available: {[str(d) for d in training_dirs]}"
        )
    
    return training_dirs[:num_generations]


def get_balanced_dataset_path(
    base_path: Union[str, Path],
    model_version: str,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Get path to balanced synthetic dataset (>100k samples).
    
    Args:
        base_path: Base path to synthetic_datasets directory
        model_version: 'v0' or 'v7'
        dataset_name: Optional dataset name to search under (e.g., '0_train_baseline').
                     If provided, will search under {base_path}/{model_version}/{dataset_name}/
                     for directories matching balanced-*-copy* pattern.
    
    Returns:
        Path to balanced dataset directory
    """
    base_path = Path(base_path)
    model_path = base_path / model_version 
    
    balanced_dirs = []
    
    # If dataset_name is provided, first check under that directory for balanced-*-copy* patterns
    if dataset_name is not None:
        dataset_path = model_path / dataset_name
        if dataset_path.exists():
            import glob
            # Look for balanced-*-copy* directories
            balanced_copy_pattern = str(dataset_path / "balanced-*-copy*")
            balanced_copy_dirs = sorted(glob.glob(balanced_copy_pattern))
            for match in balanced_copy_dirs:
                candidate = Path(match)
                if candidate.exists() and candidate.is_dir():
                    balanced_dirs.append(candidate)
            
            # If found, sort by copy number and return the first one
            if balanced_dirs:
                def get_copy_number(path):
                    import re
                    match = re.search(r'copy(\d+)', str(path))
                    return int(match.group(1)) if match else 0
                balanced_dirs = sorted(balanced_dirs, key=get_copy_number)
                return balanced_dirs[0]
        else:
            # Dataset path doesn't exist, log a warning
            print(f"Warning: Dataset path does not exist: {dataset_path}")
    
    # Otherwise, look for directories with "balanced" in the name at model_version level
    for item in model_path.iterdir():
        if item.is_dir() and 'balanced' in item.name.lower():
            balanced_dirs.append(item)
    
    if not balanced_dirs:
        if dataset_name:
            searched_paths = [str(model_path), str(model_path / dataset_name)]
            raise FileNotFoundError(
                f"No balanced dataset found. Searched in:\n"
                f"  1. {model_path} (for directories with 'balanced' in name)\n"
                f"  2. {model_path / dataset_name} (for 'balanced-*-copy*' pattern)\n"
                f"Expected pattern: {model_path / dataset_name / 'balanced-*-copy*'}"
            )
        else:
            raise FileNotFoundError(f"No balanced dataset found in {model_path}")
    
    if len(balanced_dirs) > 1:
        print(f"Warning: Multiple balanced datasets found: {balanced_dirs}, using first: {balanced_dirs[0]}")
    
    return balanced_dirs[0]

