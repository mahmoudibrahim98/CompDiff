"""
Standardized image preprocessing for CheXpert classification.

This module provides a unified preprocessing pipeline for both real and synthetic data
to ensure identical transformations and eliminate distribution mismatches.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, Union
from pathlib import Path


def center_crop_pil(img: Image.Image) -> Image.Image:
    """
    Center crop PIL image to square (min dimension).
    
    This function is used consistently across all data types to ensure
    identical preprocessing between real and synthetic data.
    
    Args:
        img: PIL Image (any size, any mode)
    
    Returns:
        PIL Image cropped to square (min dimension)
    """
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))


def tensor_to_pil_image(img_tensor: torch.Tensor) -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image using standardized conversion.
    
    This ensures consistent conversion between real (pickled tensor) and
    synthetic (PNG) data sources.
    
    Args:
        img_tensor: PyTorch tensor in [0, 1] range, shape [1, H, W] or [H, W]
    
    Returns:
        PIL Image in 'L' (grayscale) mode
    """
    # Ensure tensor is 2D [H, W]
    if img_tensor.dim() == 3:
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)  # [1, H, W] -> [H, W]
        elif img_tensor.shape[0] == 3:
            # Convert RGB to grayscale
            img_tensor = img_tensor.mean(dim=0)  # [3, H, W] -> [H, W]
        else:
            raise ValueError(f"Unexpected tensor shape: {img_tensor.shape}")
    elif img_tensor.dim() == 2:
        pass  # Already [H, W]
    else:
        raise ValueError(f"Unexpected tensor dimensions: {img_tensor.dim()}")
    
    # Convert to NumPy array and scale to [0, 255]
    # Use consistent conversion: float32 [0, 1] -> uint8 [0, 255]
    img_array = (img_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    
    # Create PIL Image in grayscale mode
    img = Image.fromarray(img_array, mode='L')
    
    return img


def pil_image_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor using standardized conversion.
    
    This ensures consistent conversion for both real and synthetic data.
    
    Args:
        img: PIL Image (any mode, will be converted to grayscale)
    
    Returns:
        PyTorch tensor in [0, 1] range, shape [1, H, W]
    """
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to NumPy array
    img_array = np.array(img, dtype=np.uint8)  # [H, W], uint8 [0, 255]
    
    # Convert to tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(img_array).float() / 255.0  # [H, W], float32 [0, 1]
    
    # Add channel dimension: [H, W] -> [1, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def get_standard_transform(
    mimic_mean: float,
    mimic_std: float,
    augmentation: bool = False,
) -> transforms.Compose:
    """
    Get standardized image transform pipeline.
    
    This function provides identical preprocessing for both real and synthetic data,
    ensuring no distribution mismatches due to different transform pipelines.
    
    Args:
        mimic_mean: Normalization mean (MIMIC-CXR statistics)
        mimic_std: Normalization std (MIMIC-CXR statistics)
        augmentation: Whether to apply data augmentation (only for training)
    
    Returns:
        Composed transform pipeline
    """
    if augmentation:
        # Training transforms with data augmentation
        # Augmentation helps prevent overfitting, especially with synthetic data
        # As per paper: center crop, random horizontal flip, and 15° random rotation
        return transforms.Compose([
            transforms.Lambda(center_crop_pil),  # Center crop PIL image to square
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),  # Final center crop to 224x224
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip (anatomically valid for X-rays)
            transforms.RandomRotation(degrees=15),  # 15° random rotation (as per paper)
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel)
            transforms.ToTensor(),  # PIL to tensor, [0, 1] - single channel
            transforms.Normalize(mean=[mimic_mean], std=[mimic_std]),  # MIMIC-CXR normalization
        ])
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Lambda(center_crop_pil),  # Center crop PIL image to square
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel)
            transforms.ToTensor(),  # PIL to tensor, [0, 1] - single channel
            transforms.Normalize(mean=[mimic_mean], std=[mimic_std]),  # MIMIC-CXR normalization
        ])


def load_and_preprocess_image(
    source: Union[torch.Tensor, str, Path, Image.Image],
    transform: transforms.Compose,
    is_tensor: bool = False,
) -> torch.Tensor:
    """
    Load and preprocess an image from various sources using standardized pipeline.
    
    This function handles:
    - PyTorch tensors (from pickled WebDataset files)
    - Image file paths (PNG, JPG, etc.)
    - PIL Images
    
    All sources are converted to PIL Image first, then transformed identically.
    
    Args:
        source: Image source (tensor, file path, or PIL Image)
        transform: Transform pipeline to apply
        is_tensor: If True, source is a PyTorch tensor (for WebDataset)
    
    Returns:
        Preprocessed image tensor [1, 224, 224] after normalization
    """
    if is_tensor or isinstance(source, torch.Tensor):
        # Handle PyTorch tensor (from WebDataset)
        img_tensor = source
        
        # Ensure tensor is [1, H, W] or [H, W] in [0, 1]
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif img_tensor.dim() == 3:
            if img_tensor.shape[0] == 3:
                # [3, H, W] -> convert to grayscale -> [1, H, W]
                img_tensor = img_tensor.mean(dim=0, keepdim=True)
            # If already 1 channel, keep as is
        
        # Convert tensor to PIL Image using standardized conversion
        img = tensor_to_pil_image(img_tensor)
    
    elif isinstance(source, (str, Path)):
        # Handle file path (PNG, JPG, etc.)
        img_path = Path(source)
        try:
            img = Image.open(img_path)
            # Convert to grayscale immediately for consistency
            if img.mode != 'L':
                img = img.convert('L')
        except Exception as e:
            raise ValueError(f"Failed to load image from {img_path}: {e}")
    
    elif isinstance(source, Image.Image):
        # Handle PIL Image directly
        img = source
        # Convert to grayscale for consistency
        if img.mode != 'L':
            img = img.convert('L')
    
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")
    
    # Apply standardized transform pipeline
    processed_img = transform(img)
    
    return processed_img
