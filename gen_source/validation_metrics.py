"""
Validation metrics for model checkpoints.

This module implements comprehensive validation metrics as described in the paper:
1. Text Prompt Alignment: Disease AUROC, sex/race accuracy, age RMSE
2. Real-Synthetic Image Similarity: FID, BioViL cosine similarity, MS-SSIM
3. Intra-Prompt Image Diversity: MS-SSIM and BioViL embedding similarity

References:
- Torch X-ray Vision (XRV): https://github.com/mlmed/torchxrayvision
- Sex prediction model: https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00032-4/fulltext
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from scipy.linalg import sqrtm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import warnings
from pathlib import Path
import sys
import os
import shutil
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# FairDiffusion fairness metrics imports
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio
    )
except ImportError:
    warnings.warn("fairlearn not installed. Fairness metrics will not be available.")
    demographic_parity_difference = None
    demographic_parity_ratio = None
    equalized_odds_difference = None
    equalized_odds_ratio = None

from sklearn.utils import resample
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import age binning utilities from local dataset_wds module
# The code previously in train_code is now in gen_source
try:
    from dataset_wds import get_default_age_bins, get_age_bins_from_num_bins
except ImportError:
    warnings.warn("dataset_wds module not found. Using default age binning functions.")
    # Fallback implementations
    def get_default_age_bins():
        """Default age bins: [18, 40, 60, 80] creates bins [0-18, 18-40, 40-60, 60-80, 80+]"""
        return [18, 40, 60, 80]
    
    def get_age_bins_from_num_bins(num_bins: int):
        """Generate age bins from number of bins."""
        if num_bins == 5:
            return [18, 40, 60, 80]
        elif num_bins == 10:
            return [10, 20, 30, 40, 50, 60, 70, 80, 90]
        else:
            # Default to 5 bins
            return [18, 40, 60, 80]


def age_to_bin(age: float, bins: list) -> int:
    """
    Convert age value to bin index.
    
    Args:
        age: Age in years
        bins: List of age thresholds. For N bins, provide N-1 thresholds.
              Example: [18, 40, 60, 80] creates 5 bins: [0-18, 18-40, 40-60, 60-80, 80+]
    
    Returns:
        Age bin index (0 to len(bins), where len(bins) is the last bin)
    """
    if age < 0:
        return -1  # Invalid age
    
    for i, threshold in enumerate(bins):
        if age < threshold:
            return i
    return len(bins)  # Last bin (e.g., 80+)


class TextPromptAlignmentMetrics:
    """
    Evaluates how accurately the model adheres to provided text prompts.
    Uses pretrained classifiers from Torch X-ray Vision (XRV) library.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize XRV models for disease, sex, race, and age prediction.

        Args:
            device: Device to run inference on
        """
        self.device = device

        # Disease classification model (DenseNet-121)
        # Note: Expects 224×224 grayscale (1 channel) images
        self.disease_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.disease_model.to(device)
        self.disease_model.eval()

        # Disease labels to evaluate
        self.disease_labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Edema",
            "Pneumothorax",
            "Effusion"
        ]
        print(self.disease_model.pathologies)
        # Get indices for these diseases in XRV model output
        self.disease_indices = [
            list(self.disease_model.pathologies).index(disease)
            for disease in self.disease_labels
        ]

        # Race classification model
        # Note: Expects 320×320 grayscale (1 channel) images
        self.race_model = xrv.baseline_models.emory_hiti.RaceModel()
        self.race_model.to(device)
        self.race_model.eval()

        # Age prediction model
        # Note: Expects 320×320 grayscale (1 channel) images
        # self.age_model = xrv.models.DenseNet(weights="densenet121-res224-age")
        self.age_model = xrv.baseline_models.riken.AgeModel()
        self.age_model.to(device)
        self.age_model.eval()

        # Sex prediction model (MIRA model from torchxrayvision)
        # Note: Expects 224×224 grayscale (1 channel) images in HU range
        # Load MIRA model by default (doesn't require checkpoint)
        self.sex_model = xrv.baseline_models.mira.SexModel(weights=True)
        self.sex_model.to(device)
        self.sex_model.eval()

        # Transform for XRV models (expects 224x224)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BOX),
            transforms.Normalize(mean=[0.5], std=[0.5])  # XRV expects normalized images
        ])

    def load_sex_model(self, checkpoint_path: str = None):
        """
        Load MIRA sex prediction model from torchxrayvision.
        
        The MIRA model expects single-channel (grayscale) images in HU range [-1024, 1024],
        similar to other XRV models. MIRA model is now loaded by default in __init__,
        but this method is kept for backward compatibility.

        Args:
            checkpoint_path: Not used for MIRA model (kept for compatibility)
        """
        # MIRA model is already loaded in __init__, but reload if needed
        if self.sex_model is None:
            self.sex_model = xrv.baseline_models.mira.SexModel(weights=True)
            self.sex_model.to(self.device)
            self.sex_model.eval()

    def preprocess_images(self, images: torch.Tensor, target_size: int = 224) -> torch.Tensor:
        """
        Preprocess images for XRV models.
        Converts to grayscale and normalizes to Hounsfield Units range [-1024, 1024] as expected by XRV models.
        
        Based on XRV documentation: xrv.datasets.normalize(img, 255) converts 8-bit images to [-1024, 1024] HU range.

        Args:
            images: Batch of images [B, C, H, W] in range [0, 1] or [-1, 1]
            target_size: Target image size (224 for disease model, 320 for race/age models)

        Returns:
            Preprocessed images [B, 1, target_size, target_size] (grayscale) in HU range [-1024, 1024]
        """
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Convert RGB to grayscale if needed
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)

        # Downsample to target size using area interpolation
        images = F.interpolate(
            images,
            size=(target_size, target_size),
            mode="area"
        )

        # Convert to Hounsfield Units range [-1024, 1024] as expected by XRV models
        # This matches xrv.datasets.normalize(img, 255) behavior:
        # - Input: [0, 1] normalized image (equivalent to 8-bit [0, 255])
        # - Output: [-1024, 1024] HU range
        # Formula from XRV: HU = (img * 255 - 128) * 8
        # Simplified: HU = img * 2048 - 1024
        images = images * 2048.0 - 1024.0  # [0, 1] -> [-1024, 1024]

        return images

    def preprocess_images_for_sex_model(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for MIRA sex prediction model (512x512 -> 224x224).
        
        The MIRA model expects single-channel (grayscale) images in HU range [-1024, 1024],
        similar to other XRV models. Uses the same preprocessing as preprocess_images().

        Args:
            images: Batch of images [B, C, H, W] in range [0, 1] or [-1, 1]

        Returns:
            Preprocessed images [B, 1, 224, 224] (grayscale) in HU range [-1024, 1024]
        """
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Convert RGB to grayscale if needed (MIRA expects single channel)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)

        # Downsample to 224x224 using area interpolation
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="area"
        )

        # Convert to Hounsfield Units range [-1024, 1024] (same as other XRV models)
        # [0, 1] -> [-1024, 1024]
        images = images * 2048.0 - 1024.0

        return images

    @torch.no_grad()
    def compute_disease_auroc(
        self,
        synthetic_images: torch.Tensor,
        disease_labels: torch.Tensor,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Compute AUROC for disease classification on synthetic images.
        
        Enhanced version that:
        - Maps 5 tracked diseases to their indices in the 14-label array
        - Filters out uncertain labels (-1) before computing AUROC
        - Processes images in batches to avoid OOM

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            disease_labels: Ground truth disease labels [B, num_diseases] or [B, 14] (14-label array)
            batch_size: Batch size for processing to avoid OOM

        Returns:
            Dictionary of disease name -> AUROC score
        """
        all_predictions = []
        
        # Process in batches to avoid OOM
        num_images = len(synthetic_images)
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_images = synthetic_images[i:end_i]
            
            # Preprocess images (DenseNet expects 224x224)
            images = self.preprocess_images(batch_images, target_size=224)

            # Get predictions
            batch_predictions = self.disease_model(images)

            # Extract predictions for target diseases
            batch_predictions = batch_predictions[:, self.disease_indices].detach().cpu().numpy()
            all_predictions.append(batch_predictions)
            
            # Clear GPU cache after each batch
            del images, batch_predictions
            torch.cuda.empty_cache()
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0)
        labels = disease_labels.cpu().numpy()

        # Diagnostic logging
        print(f"[DEBUG] disease_labels shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"[DEBUG] Labels value range: min={labels.min()}, max={labels.max()}, unique values: {np.unique(labels)}")
        if labels.shape[1] == 5:
            print(f"[DEBUG] First 5 samples of 5-label format:\n{labels[:5]}")
        elif labels.shape[1] == 14:
            print(f"[DEBUG] First 5 samples of 14-label format:\n{labels[:5]}")

        # Map the 5 tracked diseases to their indices in the 14-label array
        # DISEASE_COLUMNS order: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        #                         'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
        #                         'Lung Opacity', 'No Finding', 'Pleural Effusion', 
        #                         'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        disease_name_to_14label_index = {
            "Atelectasis": 0,
            "Cardiomegaly": 1,
            "Edema": 3,
            "Pneumothorax": 12,
            "Effusion": 9,  # "Pleural Effusion" in the 14-label array
        }

        # Check if labels are in 14-label format or 5-label format
        # If shape is [B, 5], assume it's already the 5 tracked diseases
        # If shape is [B, 14], we need to map to the correct indices
        if labels.shape[1] == 14:
            # Labels are in 14-label format - map to correct indices
            use_14label_format = True
            print("[DEBUG] Detected 14-label format")
        elif labels.shape[1] == 5:
            # Labels are already in 5-label format
            use_14label_format = False
            print("[DEBUG] Detected 5-label format")
        else:
            # Unknown format - try to use as-is
            use_14label_format = False
            print(f"[DEBUG] Unknown label format with shape {labels.shape[1]}, treating as 5-label")

        # Compute AUROC for each disease
        aurocs = {}
        for i, disease in enumerate(self.disease_labels):
            if use_14label_format:
                # Get the correct index in the 14-label array
                label_idx = disease_name_to_14label_index.get(disease)
                if label_idx is None:
                    aurocs[disease] = np.nan
                    continue
                
                # Get labels from the correct position in 14-label array
                disease_labels_i = labels[:, label_idx]
                # Get predictions from the model (which outputs 5 diseases)
                disease_preds_i = predictions[:, i]
                
                # Filter out samples with uncertain labels (-1)
                # Only keep samples where label is 0 or 1
                valid_mask = (disease_labels_i == 0) | (disease_labels_i == 1)
                valid_labels = disease_labels_i[valid_mask]
                valid_preds = disease_preds_i[valid_mask]
            else:
                # Labels are already in 5-label format
                disease_labels_i = labels[:, i]
                disease_preds_i = predictions[:, i]
                
                # Filter out samples with uncertain labels (-1)
                valid_mask = (disease_labels_i == 0) | (disease_labels_i == 1)
                valid_labels = disease_labels_i[valid_mask]
                valid_preds = disease_preds_i[valid_mask]

            # Diagnostic logging for Atelectasis
            if disease == "Atelectasis":
                print(f"[DEBUG Atelectasis] label_idx={i if not use_14label_format else label_idx}, "
                      f"total_samples={len(disease_labels_i)}, "
                      f"valid_samples={len(valid_labels)}, "
                      f"unique_labels={np.unique(valid_labels) if len(valid_labels) > 0 else 'N/A'}, "
                      f"label_distribution: 0={np.sum(disease_labels_i == 0)}, 1={np.sum(disease_labels_i == 1)}, -1={np.sum(disease_labels_i == -1)}")

            # Check if we have both classes (0 and 1) after filtering
            unique_labels = np.unique(valid_labels)
            if len(unique_labels) < 2 or len(valid_labels) == 0:
                aurocs[disease] = np.nan
                if disease == "Atelectasis":
                    print(f"[DEBUG Atelectasis] Returning NaN - unique_labels={unique_labels}, valid_labels count={len(valid_labels)}")
                continue

            # Check for minimum samples per class for reliable AUROC
            # AUROC becomes unreliable with < 10 samples in the minority class
            class_counts = {label: np.sum(valid_labels == label) for label in unique_labels}
            min_class_count = min(class_counts.values())
            if min_class_count < 10:
                print(f"[WARNING] {disease}: Only {min_class_count} samples in minority class (class_counts: {class_counts}). "
                      f"AUROC may be unreliable. Consider filtering or using a different metric.")

            try:
                aurocs[disease] = roc_auc_score(valid_labels, valid_preds)
                if disease == "Atelectasis":
                    print(f"[DEBUG Atelectasis] AUROC={aurocs[disease]:.4f} (class_counts: {class_counts})")
            except ValueError as e:
                aurocs[disease] = np.nan
                if disease == "Atelectasis":
                    print(f"[DEBUG Atelectasis] ValueError in roc_auc_score: {e}")

        # Compute mean AUROC
        valid_aurocs = [v for v in aurocs.values() if not np.isnan(v)]
        aurocs["mean_auroc"] = np.mean(valid_aurocs) if valid_aurocs else np.nan

        return aurocs

    @torch.no_grad()
    def compute_sex_accuracy(
        self,
        synthetic_images: torch.Tensor,
        sex_labels: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute sex classification accuracy.
        
        Enhanced version that:
        - Uses softmax before argmax for 2-class classification (more stable)
        - Filters out invalid labels (-1) before computing accuracy
        - Processes images in batches to avoid OOM

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            sex_labels: Ground truth sex labels [B] (0=M, 1=F, -1=invalid)
            batch_size: Batch size for processing to avoid OOM

        Returns:
            Sex classification accuracy
        """
        if self.sex_model is None:
            warnings.warn("Sex model not loaded. Call load_sex_model() first.")
            return np.nan

        all_predicted_labels = []
        all_true_labels = []
        
        # Process in batches to avoid OOM
        num_images = len(synthetic_images)
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_images = synthetic_images[i:end_i]
            batch_labels = sex_labels[i:end_i]
            
            # Preprocess images for sex model (MIRA expects grayscale/1 channel in HU range)
            images = self.preprocess_images_for_sex_model(batch_images)

            # Get predictions
            raw_predictions = self.sex_model(images).detach()
            
            # Handle different output shapes from sex model
            # Sex models can output: [B, 2] (two-class), [B, 1] (sigmoid), or [B] (flat)
            if raw_predictions.dim() > 1:
                if raw_predictions.shape[1] == 2:
                    # Two-class classification: [B, 2] logits
                    # Apply softmax to convert logits to probabilities, then take argmax
                    # This matches the approach in calculate_metrics_on_real_data.py
                    probs = torch.softmax(raw_predictions, dim=1)
                    predicted_labels = torch.argmax(probs, dim=1).cpu().numpy()
                else:
                    # Single sigmoid output: [B, 1] or [B]
                    predicted_labels = (raw_predictions.squeeze() > 0.5).long().cpu().numpy()
            else:
                # Already 1D
                predicted_labels = (raw_predictions > 0.5).long().cpu().numpy()
            
            all_predicted_labels.append(predicted_labels.flatten())
            all_true_labels.append(batch_labels.cpu().numpy().flatten())
            
            # Clear GPU cache after each batch
            del images, raw_predictions
            torch.cuda.empty_cache()
        
        # Concatenate all predictions and labels
        predicted_labels = np.concatenate(all_predicted_labels, axis=0)
        true_labels = np.concatenate(all_true_labels, axis=0)
        
        # Filter out invalid labels (-1) if any
        valid_mask = true_labels >= 0
        if np.sum(valid_mask) < len(true_labels):
            # Some labels are invalid - filter them out
            true_labels = true_labels[valid_mask]
            predicted_labels = predicted_labels[valid_mask]
        
        if len(true_labels) == 0:
            return np.nan

        return accuracy_score(true_labels, predicted_labels)

    @torch.no_grad()
    def compute_race_accuracy(
        self,
        synthetic_images: torch.Tensor,
        race_labels: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute race classification accuracy using XRV model.
        
        Enhanced version that:
        - Maps model output indices to dataset label indices
        - Filters out Hispanic (3) and invalid labels (-1) before computing accuracy
        - Processes images in batches to avoid OOM

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            race_labels: Ground truth race labels [B] (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            batch_size: Batch size for processing to avoid OOM

        Returns:
            Race classification accuracy
        """
        all_predicted_labels = []
        all_true_labels = []
        
        # Process in batches to avoid OOM
        num_images = len(synthetic_images)
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_images = synthetic_images[i:end_i]
            batch_labels = race_labels[i:end_i]
            
            # Preprocess images (RaceModel expects 320x320)
            images = self.preprocess_images(batch_images, target_size=320)

            # Get predictions
            predictions = self.race_model(images)
            predicted_labels = torch.argmax(predictions.detach(), dim=1).cpu().numpy()
            
            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(batch_labels.cpu().numpy().flatten())
            
            # Clear GPU cache after each batch
            del images, predictions
            torch.cuda.empty_cache()
        
        # Concatenate all predictions and labels
        predicted_labels = np.concatenate(all_predicted_labels, axis=0)
        true_labels = np.concatenate(all_true_labels, axis=0)
        
        # Check if race model has a targets attribute (class names)
        if hasattr(self.race_model, 'targets'):
            model_targets = self.race_model.targets
            # Map model output indices to dataset label indices
            # Model outputs: ['Asian', 'Black', 'White'] (indices 0, 1, 2)
            # Dataset labels: 0=White, 1=Black, 2=Asian, 3=Hispanic
            # Mapping: Model 0='Asian' → Dataset 2, Model 1='Black' → Dataset 1, Model 2='White' → Dataset 0
            model_to_dataset_map = {}
            for model_idx, class_name in enumerate(model_targets):
                if class_name == 'White':
                    model_to_dataset_map[model_idx] = 0
                elif class_name == 'Black':
                    model_to_dataset_map[model_idx] = 1
                elif class_name == 'Asian':
                    model_to_dataset_map[model_idx] = 2
                else:
                    # Unknown class - mark as invalid
                    model_to_dataset_map[model_idx] = -1
            
            # Map predictions from model indices to dataset label indices
            mapped_predictions = np.array([model_to_dataset_map.get(pred, -1) for pred in predicted_labels])
            
            # Filter out samples where mapping failed or label is Hispanic (3) or invalid (-1)
            # Hispanic (3) is not in the model, so we can't evaluate it
            valid_mask = (true_labels >= 0) & (true_labels < 3) & (mapped_predictions >= 0)
            
            true_labels = true_labels[valid_mask]
            predicted_labels = mapped_predictions[valid_mask]
        else:
            # No targets attribute - assume direct mapping (fallback)
            # Filter out Hispanic (3) and invalid labels (-1)
            valid_mask = (true_labels >= 0) & (true_labels < 3)
            true_labels = true_labels[valid_mask]
            predicted_labels = predicted_labels[valid_mask]
        
        if len(true_labels) == 0:
            return np.nan

        return accuracy_score(true_labels, predicted_labels)

    @torch.no_grad()
    def compute_age_rmse(
        self,
        synthetic_images: torch.Tensor,
        age_labels: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute age prediction RMSE in years.
        
        Processes images in batches to avoid OOM.

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            age_labels: Ground truth ages in years [B]
            batch_size: Batch size for processing to avoid OOM

        Returns:
            RMSE in years
        """
        all_predictions = []
        all_labels = []
        
        # Process in batches to avoid OOM
        num_images = len(synthetic_images)
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_images = synthetic_images[i:end_i]
            batch_labels = age_labels[i:end_i]
            
            # Preprocess images (AgeModel expects 320x320)
            images = self.preprocess_images(batch_images, target_size=320)

            # Get predictions
            # Use flatten() instead of squeeze() to avoid creating 0-dimensional tensors
            # when batch size is 1. flatten() ensures we always have at least 1 dimension.
            predictions = self.age_model(images).flatten()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
            
            # Clear GPU cache after each batch
            del images, predictions
            torch.cuda.empty_cache()
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute RMSE
        rmse = torch.sqrt(F.mse_loss(all_predictions, all_labels))

        return rmse.item()

    @torch.no_grad()
    def compute_age_bin_accuracy(
        self,
        synthetic_images: torch.Tensor,
        age_labels: torch.Tensor,
        age_bins: Optional[list] = None,
        batch_size: int = 32
    ) -> float:
        """
        Compute age bin classification accuracy.
        
        Predicts age from images, bins the predictions, and compares with ground truth age bins.
        
        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            age_labels: Ground truth ages in years [B]
            age_bins: Age bin thresholds. If None, uses default bins [18, 40, 60, 80].
                      For N bins, provide N-1 thresholds.
            batch_size: Batch size for processing to avoid OOM
        
        Returns:
            Classification accuracy (0.0 to 1.0) for age bin prediction
        """
        if age_bins is None:
            age_bins = get_default_age_bins()
        
        all_predictions = []
        all_labels = []
        
        # Process in batches to avoid OOM
        num_images = len(synthetic_images)
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            batch_images = synthetic_images[i:end_i]
            batch_labels = age_labels[i:end_i]
            
            # Preprocess images (AgeModel expects 320x320)
            images = self.preprocess_images(batch_images, target_size=320)

            # Get predictions
            predictions = self.age_model(images).flatten()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
            
            # Clear GPU cache after each batch
            del images, predictions
            torch.cuda.empty_cache()
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Convert to numpy for binning
        predictions_np = all_predictions.numpy()
        labels_np = all_labels.numpy()
        
        # Bin the predictions and labels
        predicted_bins = np.array([age_to_bin(age, age_bins) for age in predictions_np])
        true_bins = np.array([age_to_bin(age, age_bins) for age in labels_np])
        
        # Filter out invalid ages (age < 0)
        valid_mask = (predicted_bins >= 0) & (true_bins >= 0)
        if valid_mask.sum() == 0:
            return np.nan
        
        predicted_bins = predicted_bins[valid_mask]
        true_bins = true_bins[valid_mask]
        predictions_valid = predictions_np[valid_mask]
        labels_valid = labels_np[valid_mask]
        
        # Compute classification accuracy
        accuracy = accuracy_score(true_bins, predicted_bins)
        
        # Diagnostic: Log error statistics for misclassified cases
        misclassified_mask = predicted_bins != true_bins
        if misclassified_mask.sum() > 0:
            errors = np.abs(predictions_valid - labels_valid)
            misclassified_errors = errors[misclassified_mask]
            correct_errors = errors[~misclassified_mask]
            
            # Log diagnostic info (can be enabled via logging level)
            import logging
            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Age bin accuracy diagnostics:")
                logger.debug(f"  Total samples: {len(predictions_valid)}")
                logger.debug(f"  Misclassified: {misclassified_mask.sum()} ({100*misclassified_mask.sum()/len(predictions_valid):.1f}%)")
                logger.debug(f"  Mean error (all): {np.mean(errors):.2f} years")
                logger.debug(f"  Mean error (correct bins): {np.mean(correct_errors):.2f} years")
                logger.debug(f"  Mean error (wrong bins): {np.mean(misclassified_errors):.2f} years")
                logger.debug(f"  Max error (wrong bins): {np.max(misclassified_errors):.2f} years")
                logger.debug(f"  Errors > 10 years: {(errors > 10).sum()} ({(errors > 10).sum()/len(errors)*100:.1f}%)")
                # Additional boundary analysis
                boundary_threshold = 2.0  # Consider ages within 2 years of boundary as "near boundary"
                near_boundary_mask = np.zeros(len(labels_valid), dtype=bool)
                for threshold in age_bins:
                    near_boundary_mask |= np.abs(labels_valid - threshold) < boundary_threshold
                logger.debug(f"  Samples near bin boundaries (<2 years): {near_boundary_mask.sum()} ({100*near_boundary_mask.sum()/len(labels_valid):.1f}%)")
                if near_boundary_mask.sum() > 0:
                    boundary_errors = errors[near_boundary_mask]
                    boundary_misclassified = misclassified_mask[near_boundary_mask]
                    logger.debug(f"  Misclassification rate near boundaries: {100*boundary_misclassified.sum()/len(boundary_misclassified):.1f}%")
                    logger.debug(f"  Mean error near boundaries: {np.mean(boundary_errors):.2f} years")
        
        return float(accuracy)


class RealSyntheticSimilarityMetrics:
    """
    Evaluates similarity between real and synthetic images using:
    1. FID (Fréchet Inception Distance)
    2. BioViL cosine similarity
    3. MS-SSIM (Multi-Scale Structural Similarity Index)
    """

    def __init__(self, device: str = "cuda", age_bins: Optional[List[int]] = None):
        """
        Initialize models for similarity computation.

        Args:
            device: Device to run inference on
            age_bins: Age bin thresholds. If None, uses default from dataset_wds.
                      Example: [18, 40, 60, 80] creates bins: [0-18, 18-40, 40-60, 60-80, 80+]
        """
        self.device = device
        
        # Age bins configuration for subgroup metrics
        if age_bins is None:
            age_bins = get_default_age_bins()
        self.age_bins = age_bins
        self.age_group_strings = self._generate_age_group_strings()

        # Inception v3 for FID
        self.inception_model = None  # Will be loaded lazily

        # RadImageNet ResNet50 for medical FID
        self.radimagenet_model = None  # Will be loaded lazily

        # BioViL encoder for medical image embeddings
        self.biovil_model = None  # Will be loaded lazily

    def _load_inception(self):
        """Lazy load Inception v3 model for FID computation."""
        if self.inception_model is None:
            from torchvision.models import inception_v3
            self.inception_model = inception_v3(pretrained=True, transform_input=False)
            self.inception_model.fc = nn.Identity()  # Remove final FC layer
            self.inception_model.to(self.device)
            self.inception_model.eval()

    def _load_radimagenet(self):
        """Lazy load RadImageNet ResNet50 model for medical FID computation.
        Tries pretrained_models/fid_radnet first, then torch.hub."""
        if self.radimagenet_model is None:
            import os
            import sys

            # Resolve repo root (parent of gen_source)
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            _repo_root = os.path.dirname(_this_dir)
            # pretrained_models/fid_radnet: weights (.pth) and radimagenet-models package
            _fid_radnet_dir = os.path.join(_repo_root, "pretrained_models", "fid_radnet")
            _radimagenet_package_dir = os.path.join(
                _fid_radnet_dir, "radimagenet-models-main", "radimagenet-models-main"
            )
            weights_file = os.path.join(_fid_radnet_dir, "RadImageNet-ResNet50_notop.pth")

            # Try loading from local pretrained_models/fid_radnet first
            if os.path.exists(weights_file) and os.path.isdir(_radimagenet_package_dir):
                try:
                    print("Loading RadImageNet ResNet50 from local path...")
                    if _radimagenet_package_dir not in sys.path:
                        sys.path.insert(0, _radimagenet_package_dir)
                    from radimagenet_models.models.resnet import radimagenet_resnet50
                    print(f"Found local weights: {weights_file}")
                    self.radimagenet_model = radimagenet_resnet50(model_dir=_fid_radnet_dir, progress=True)
                    # Remove final FC layer to get features
                    if hasattr(self.radimagenet_model, 'fc'):
                        self.radimagenet_model.fc = nn.Identity()
                    elif hasattr(self.radimagenet_model, 'classifier'):
                        self.radimagenet_model.classifier = nn.Identity()
                    self.radimagenet_model.to(self.device)
                    self.radimagenet_model.eval()
                    print("✓ RadImageNet ResNet50 loaded successfully from local path")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load RadImageNet from local path: {e}")
                    print("Falling back to torch.hub loading...")
            
            # Fallback to torch.hub loading with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Loading RadImageNet ResNet50 via torch.hub (attempt {attempt + 1}/{max_retries})...")
                    self.radimagenet_model = torch.hub.load(
                        "Warvito/radimagenet-models", 
                        model="radimagenet_resnet50", 
                        trust_repo=True, 
                        verbose=False
                    )
                    
                    # Remove final FC layer to get features (spatial features before pooling)
                    # We'll do spatial averaging in the embedding extraction function
                    if hasattr(self.radimagenet_model, 'fc'):
                        self.radimagenet_model.fc = nn.Identity()
                    elif hasattr(self.radimagenet_model, 'classifier'):
                        self.radimagenet_model.classifier = nn.Identity()
                    
                    self.radimagenet_model.to(self.device)
                    self.radimagenet_model.eval()
                    print("✓ RadImageNet ResNet50 loaded successfully via torch.hub")
                    return
                    
                except Exception as e:
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        print("Failed to load RadImageNet model after all attempts")
                        # Final fallback to ImageNet-pretrained ResNet50
                        try:
                            from torchvision.models import resnet50
                            print("⚠️ Falling back to ImageNet-pretrained ResNet50...")
                            fallback_model = resnet50(pretrained=True)
                            fallback_model.fc = nn.Identity()
                            self.radimagenet_model = fallback_model
                            self.radimagenet_model.to(self.device)
                            self.radimagenet_model.eval()
                            print("✓ Using ImageNet-pretrained ResNet50 as fallback")
                            warnings.warn(
                                "RadImageNet loading failed. Using ImageNet-pretrained ResNet50 as fallback. "
                                "This may give less accurate FID scores for medical images."
                            )
                        except Exception as fallback_error:
                            warnings.warn(f"Could not load ResNet50 fallback model: {fallback_error}")
                            self.radimagenet_model = None
                    else:
                        print("Retrying in 5 seconds...")
                        import time
                        time.sleep(5)

    def _load_biovil(self):
        """Lazy load BioViL model using hi-ml-multimodal library."""
        if self.biovil_model is None:
            try:
                from health_multimodal.image.utils import get_image_inference, ImageModelType

                # Load BioViL using the official hi-ml-multimodal library
                self.biovil_inference_engine = get_image_inference(
                    image_model_type=ImageModelType.BIOVIL
                )
                
                # Move model to device
                self.biovil_inference_engine.model.to(self.device)
                self.biovil_inference_engine.model.eval()
                
                # Store the transform for preprocessing
                self.biovil_transform = self.biovil_inference_engine.transform
                
                # Store model reference for convenience
                self.biovil_model = self.biovil_inference_engine.model
                
                print("✓ BioViL loaded successfully using hi-ml-multimodal library")
                    
            except ImportError as e:
                warnings.warn(
                    f"Could not import hi-ml-multimodal library: {e}\n"
                    "Please install it using: pip install hi-ml-multimodal"
                )
                self.biovil_model = None
                self.biovil_inference_engine = None
            except Exception as e:
                warnings.warn(f"Could not load BioViL model: {e}")
                self.biovil_model = None
                self.biovil_inference_engine = None

    @torch.no_grad()
    def compute_fid(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute Fréchet Inception Distance between real and synthetic images.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            batch_size: Batch size for processing

        Returns:
            FID score (lower is better, 0 is perfect match)
        """
        self._load_inception()

        # Get embeddings for both image sets
        real_embeddings = self._get_inception_embeddings(real_images, batch_size)
        synthetic_embeddings = self._get_inception_embeddings(synthetic_images, batch_size)

        # Ensure embeddings are 2D [N, D] for covariance computation
        if real_embeddings.ndim > 2:
            real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)
        if synthetic_embeddings.ndim > 2:
            synthetic_embeddings = synthetic_embeddings.reshape(synthetic_embeddings.shape[0], -1)

        # Compute mean and covariance
        mu_real = np.mean(real_embeddings, axis=0)
        sigma_real = np.cov(real_embeddings, rowvar=False)

        mu_synthetic = np.mean(synthetic_embeddings, axis=0)
        sigma_synthetic = np.cov(synthetic_embeddings, rowvar=False)

        # Compute FID
        ssdiff = np.sum((mu_real - mu_synthetic) ** 2.0)
        covmean = sqrtm(sigma_real.dot(sigma_synthetic))

        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma_real + sigma_synthetic - 2.0 * covmean)

        return float(fid)

    @torch.no_grad()
    def _get_inception_embeddings(
        self,
        images: torch.Tensor,
        batch_size: int
    ) -> np.ndarray:
        """
        Extract Inception v3 embeddings from images.

        Args:
            images: Images [N, C, H, W] in range [0, 1] or [-1, 1]
            batch_size: Batch size for processing

        Returns:
            Embeddings [N, 2048]
        """
        # Ensure Inception model is loaded
        self._load_inception()
        
        if self.inception_model is None:
            raise RuntimeError("Failed to load Inception model for FID computation")
        
        # Process in batches to avoid OOM
        # Normalize for Inception
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        # Extract embeddings in batches (process normalization in batches to avoid OOM)
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            
            # Normalize to [0, 1]
            if batch.min() < 0:
                batch = (batch + 1.0) / 2.0

            # Resize to 299x299 for Inception
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)

            # Convert grayscale to RGB if needed
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            # Normalize for Inception
            batch = (batch - mean) / std
            
            embedding = self.inception_model(batch)
            embeddings.append(embedding.cpu().numpy())
            
            # Clear cache periodically
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()

        return np.concatenate(embeddings, axis=0)

    @torch.no_grad()
    def compute_fid_radimagenet(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute Fréchet Inception Distance using RadImageNet ResNet50 embeddings.
        This is more appropriate for medical images than Inception v3.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            batch_size: Batch size for processing

        Returns:
            FID score (lower is better, 0 is perfect match)
        """
        self._load_radimagenet()

        # Get embeddings for both image sets
        print("Extracting RadImageNet embeddings from real images...")
        real_embeddings = self._get_radimagenet_embeddings(real_images, batch_size)
        print("Extracting RadImageNet embeddings from synthetic images...")
        synthetic_embeddings = self._get_radimagenet_embeddings(synthetic_images, batch_size)

        # Ensure embeddings are 2D [N, D] for covariance computation
        if real_embeddings.ndim > 2:
            real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)
        if synthetic_embeddings.ndim > 2:
            synthetic_embeddings = synthetic_embeddings.reshape(synthetic_embeddings.shape[0], -1)

        # Compute mean and covariance
        mu_real = np.mean(real_embeddings, axis=0)
        sigma_real = np.cov(real_embeddings, rowvar=False)

        mu_synthetic = np.mean(synthetic_embeddings, axis=0)
        sigma_synthetic = np.cov(synthetic_embeddings, rowvar=False)

        # Compute FID
        ssdiff = np.sum((mu_real - mu_synthetic) ** 2.0)
        covmean = sqrtm(sigma_real.dot(sigma_synthetic))

        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma_real + sigma_synthetic - 2.0 * covmean)

        return float(fid)

    @torch.no_grad()
    def _get_radimagenet_embeddings(
        self,
        images: torch.Tensor,
        batch_size: int
    ) -> np.ndarray:
        """
        Extract RadImageNet ResNet50 embeddings from images.

        Args:
            images: Images [N, C, H, W] in range [0, 1] or [-1, 1]
            batch_size: Batch size for processing

        Returns:
            Embeddings [N, 2048] (ResNet50 feature dimension)
        """
        # Ensure RadImageNet model is loaded
        self._load_radimagenet()
        
        if self.radimagenet_model is None:
            raise RuntimeError("Failed to load RadImageNet model for FID computation")
        
        # RadImageNet preprocessing: Subtract mean used during RadImageNet training [B, G, R] = [0.406, 0.456, 0.485]
        # Note: After BGR conversion, channels are [B, G, R]
        mean = torch.tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1).to(self.device)

        # Extract embeddings in batches (process normalization in batches to avoid OOM)
        embeddings = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(images), batch_size)):
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  Processing RadImageNet batch {batch_idx + 1}/{num_batches}...")
            
            batch = images[i:i+batch_size].to(self.device)
            
            # Normalize to [0, 1]
            if batch.min() < 0:
                batch = (batch + 1.0) / 2.0

            # Resize to 224x224 for ResNet50
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)

            # Convert grayscale to RGB if needed
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            # RadImageNet preprocessing: Change RGB to BGR and subtract mean
            # Reference: https://github.com/BMEII-AI/RadImageNet
            # Change order from 'RGB' to 'BGR'
            batch = batch[:, [2, 1, 0], ...]
            
            # Subtract mean
            batch = batch - mean
            with torch.no_grad():
                feature_image = self.radimagenet_model(batch)
                # Spatial average: mean over spatial dimensions [2, 3]
                # This flattens the spatial dimensions: [B, C, H, W] -> [B, C]
                if feature_image.dim() == 4:
                    embedding = feature_image.mean([2, 3], keepdim=False)
                elif feature_image.dim() == 2:
                    # Already flattened
                    embedding = feature_image
                elif feature_image.dim() == 1:
                    # Single sample, add batch dimension
                    embedding = feature_image.unsqueeze(0)
                else:
                    # Fallback: flatten all non-batch dimensions
                    embedding = feature_image.view(feature_image.size(0), -1)
            embeddings.append(embedding.cpu().numpy())
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        result = np.concatenate(embeddings, axis=0)
        # Ensure final result is 2D [N, D]
        if result.ndim > 2:
            result = result.reshape(result.shape[0], -1)
        print(f"  ✓ Extracted {result.shape[0]} embeddings with dimension {result.shape[1]}")
        return result

    @torch.no_grad()
    def compute_biovil_similarity(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between BioViL embeddings of paired real/synthetic images.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W] (paired with real)

        Returns:
            Mean cosine similarity (1.0 = identical, 0.0 = orthogonal)
        """
        self._load_biovil()

        if self.biovil_model is None:
            warnings.warn("BioViL model not available. Returning NaN.")
            return np.nan

        # Get embeddings
        real_embeddings = self._get_biovil_embeddings(real_images)
        synthetic_embeddings = self._get_biovil_embeddings(synthetic_images)

        # Compute pairwise cosine similarity
        real_embeddings = F.normalize(real_embeddings, dim=1)
        synthetic_embeddings = F.normalize(synthetic_embeddings, dim=1)

        similarities = torch.sum(real_embeddings * synthetic_embeddings, dim=1)

        return similarities.mean().item()

    @torch.no_grad()
    def _get_biovil_embeddings(self, images: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        """
        Extract BioViL embeddings from images using hi-ml-multimodal library.

        Args:
            images: Images [N, C, H, W] in range [0, 1] or [-1, 1]
            batch_size: Batch size for processing to avoid OOM errors

        Returns:
            Embeddings [N, D] in the joint image-text latent space, L2-normalized
        """
        from PIL import Image
        
        if self.biovil_inference_engine is None:
            raise RuntimeError(
                "BioViL model not loaded. Please ensure hi-ml-multimodal is installed: "
                "pip install hi-ml-multimodal"
            )

        # Normalize to [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Convert tensor to PIL Images for processing
        pil_images = []
        for i in range(images.shape[0]):
            # Get single image [C, H, W]
            img = images[i]
            
            # Convert to numpy and scale to [0, 255]
            img_np = img.cpu().numpy()
            
            # Handle grayscale (single channel)
            if img_np.shape[0] == 1:
                img_np = img_np[0]  # Remove channel dimension -> [H, W]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='L')  # Grayscale
            else:
                # RGB image - convert to grayscale for BioViL (expects chest X-rays in grayscale)
                img_np = img_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='RGB').convert('L')  # Convert to grayscale
            
            pil_images.append(pil_img)

        # Process images in batches to avoid OOM
        all_embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch_pil = pil_images[i:i + batch_size]
            
            # Apply BioViL transform (includes resize, normalize, etc.)
            batch = torch.stack([self.biovil_transform(img) for img in batch_pil])
            batch = batch.to(self.device)

            # Extract embeddings using the inference engine
            output = self.biovil_model(batch)  # Returns ImageModelOutput
            batch_embeddings = output.projected_global_embedding  # [B, joint_feature_size]
            
            # L2 normalize (recommended for similarity computation)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
            
            # Move to CPU to save GPU memory
            all_embeddings.append(batch_embeddings.cpu())
            
            # Clear GPU cache
            del batch, output, batch_embeddings
            torch.cuda.empty_cache()
        
        # Concatenate all embeddings and move back to device
        embeddings = torch.cat(all_embeddings, dim=0).to(self.device)

        return embeddings

    def compute_ms_ssim(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor
    ) -> float:
        """
        Compute Multi-Scale Structural Similarity Index between paired images.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W] (paired with real)

        Returns:
            Mean MS-SSIM (1.0 = identical, 0.0 = no similarity)
        """
        from pytorch_msssim import ms_ssim

        # Normalize to [0, 1]
        if real_images.min() < 0:
            real_images = (real_images + 1.0) / 2.0
        if synthetic_images.min() < 0:
            synthetic_images = (synthetic_images + 1.0) / 2.0

        # Compute MS-SSIM
        ms_ssim_val = ms_ssim(
            real_images,
            synthetic_images,
            data_range=1.0,
            size_average=True
        )

        return ms_ssim_val.item()

    def _generate_age_group_strings(self) -> List[str]:
        """
        Generate age group strings from age bins.
        
        Returns:
            List of age group strings like ["0-18", "18-40", "40-60", "60-80", "80+"]
        """
        if not self.age_bins:
            return ["0-18", "18-40", "40-60", "60-80", "80+"]  # Default fallback
        
        age_groups = []
        prev_threshold = 0
        for threshold in self.age_bins:
            age_groups.append(f"{prev_threshold}-{threshold}")
            prev_threshold = threshold
        # Add the last bin
        age_groups.append(f"{prev_threshold}+")
        return age_groups
    
    def _get_age_group(self, age: float) -> str:
        """
        Convert age in years to age group string using configured age bins.
        
        Args:
            age: Age in years
            
        Returns:
            Age group string (e.g., "0-18", "18-40", "40-60", "60-80", "80+") or "unknown"
        """
        if age < 0:
            return "unknown"
        
        # Find which bin this age belongs to
        for i, threshold in enumerate(self.age_bins):
            if age < threshold:
                return self.age_group_strings[i]
        
        # Age is >= last threshold, return last bin
        return self.age_group_strings[-1]

    @staticmethod
    def _get_sex_label(sex_idx: int) -> str:
        """
        Convert sex index to label string.
        
        Args:
            sex_idx: Sex index (0=Female, 1=Male, -1=invalid)
            
        Returns:
            Sex label: "Female", "Male", or "unknown"
        """
        if sex_idx == 0:
            return "Female"
        elif sex_idx == 1:
            return "Male"
        else:
            return "unknown"

    @staticmethod
    def _get_race_label(race_idx: int) -> str:
        """
        Convert race index to label string.
        
        Args:
            race_idx: Race index (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            
        Returns:
            Race label: "White", "Black", "Asian", "Hispanic", or "unknown"
        """
        race_map = {
            0: "White",
            1: "Black",
            2: "Asian",
            3: "Hispanic",
        }
        return race_map.get(race_idx, "unknown")

    @torch.no_grad()
    def compute_fid_per_subgroup(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        sex_labels: Optional[torch.Tensor] = None,
        race_labels: Optional[torch.Tensor] = None,
        age_labels: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        use_radimagenet: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute FID per subgroup (Level 1: per sex, per ethnicity, per age group).
        
        OPTIMIZED: Computes embeddings once for all images, then filters by subgroup.
        This is much more efficient than recomputing embeddings for each subgroup.
        
        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            sex_labels: Sex labels [N] (0=Female, 1=Male, -1=invalid)
            race_labels: Race labels [N] (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            age_labels: Age labels [N] (age in years, -1=invalid)
            batch_size: Batch size for processing
            use_radimagenet: If True, use RadImageNet FID instead of Inception v3
            
        Returns:
            Dictionary with structure:
            {
                "sex": {"Female": fid_value, "Male": fid_value},
                "race": {"White": fid_value, "Black": fid_value, ...},
                "age_group": {"0-18": fid_value, "18-40": fid_value, ...}
            }
        """
        results = {
            "sex": {},
            "race": {},
            "age_group": {}
        }
        
        # OPTIMIZATION: Compute embeddings once for all images
        print("  Computing embeddings for all images (once)...")
        if use_radimagenet:
            real_embeddings = self._get_radimagenet_embeddings(real_images, batch_size)
            synthetic_embeddings = self._get_radimagenet_embeddings(synthetic_images, batch_size)
        else:
            real_embeddings = self._get_inception_embeddings(real_images, batch_size)
            synthetic_embeddings = self._get_inception_embeddings(synthetic_images, batch_size)
        
        # Ensure embeddings are 2D [N, D]
        if real_embeddings.ndim > 2:
            real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)
        if synthetic_embeddings.ndim > 2:
            synthetic_embeddings = synthetic_embeddings.reshape(synthetic_embeddings.shape[0], -1)
        
        # Helper function to compute FID from pre-computed embeddings
        def compute_subgroup_fid_from_embeddings(real_emb, synth_emb, subgroup_name, group_type):
            if len(real_emb) < 2 or len(synth_emb) < 2:
                return np.nan
            
            try:
                # Compute mean and covariance from embeddings
                mu_real = np.mean(real_emb, axis=0)
                sigma_real = np.cov(real_emb, rowvar=False)
                
                mu_synthetic = np.mean(synth_emb, axis=0)
                sigma_synthetic = np.cov(synth_emb, rowvar=False)
                
                # Compute FID
                ssdiff = np.sum((mu_real - mu_synthetic) ** 2.0)
                covmean = sqrtm(sigma_real.dot(sigma_synthetic))
                
                # Handle numerical errors
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                
                fid = ssdiff + np.trace(sigma_real + sigma_synthetic - 2.0 * covmean)
                return float(fid)
            except Exception as e:
                warnings.warn(f"Failed to compute FID for {group_type} subgroup '{subgroup_name}': {e}")
                return np.nan
        
        # Ensure labels are on the same device as images
        device = real_images.device
        
        # Group by sex
        if sex_labels is not None:
            if isinstance(sex_labels, torch.Tensor):
                sex_labels_tensor = sex_labels.to(device)
            else:
                sex_labels_tensor = torch.tensor(sex_labels, dtype=torch.long, device=device)
            
            for sex_idx in [0, 1]:  # Female, Male
                mask = (sex_labels_tensor == sex_idx)
                if mask.sum().item() >= 2:  # Need at least 2 samples
                    sex_label = self._get_sex_label(sex_idx)
                    # Filter embeddings by mask (much faster than recomputing)
                    mask_np = mask.cpu().numpy()
                    real_emb_subset = real_embeddings[mask_np]
                    synth_emb_subset = synthetic_embeddings[mask_np]
                    fid = compute_subgroup_fid_from_embeddings(real_emb_subset, synth_emb_subset, sex_label, "sex")
                    if not np.isnan(fid):
                        results["sex"][sex_label] = fid
        
        # Group by race
        if race_labels is not None:
            if isinstance(race_labels, torch.Tensor):
                race_labels_tensor = race_labels.to(device)
            else:
                race_labels_tensor = torch.tensor(race_labels, dtype=torch.long, device=device)
            
            for race_idx in [0, 1, 2, 3]:  # White, Black, Asian, Hispanic
                mask = (race_labels_tensor == race_idx)
                if mask.sum().item() >= 2:  # Need at least 2 samples
                    race_label = self._get_race_label(race_idx)
                    # Filter embeddings by mask (much faster than recomputing)
                    mask_np = mask.cpu().numpy()
                    real_emb_subset = real_embeddings[mask_np]
                    synth_emb_subset = synthetic_embeddings[mask_np]
                    fid = compute_subgroup_fid_from_embeddings(real_emb_subset, synth_emb_subset, race_label, "race")
                    if not np.isnan(fid):
                        results["race"][race_label] = fid
        
        # Group by age group
        if age_labels is not None:
            if isinstance(age_labels, torch.Tensor):
                age_labels_tensor = age_labels.to(device)
            else:
                age_labels_tensor = torch.tensor(age_labels, dtype=torch.float32, device=device)
            
            age_groups = self.age_group_strings
            for age_group in age_groups:
                # Create mask for this age group
                mask = torch.zeros(len(age_labels_tensor), dtype=torch.bool, device=device)
                for i, age in enumerate(age_labels_tensor):
                    if age.item() >= 0:
                        group = self._get_age_group(age.item())
                        if group == age_group:
                            mask[i] = True
                
                if mask.sum().item() >= 2:  # Need at least 2 samples
                    # Filter embeddings by mask (much faster than recomputing)
                    mask_np = mask.cpu().numpy()
                    real_emb_subset = real_embeddings[mask_np]
                    synth_emb_subset = synthetic_embeddings[mask_np]
                    fid = compute_subgroup_fid_from_embeddings(real_emb_subset, synth_emb_subset, age_group, "age_group")
                    if not np.isnan(fid):
                        results["age_group"][age_group] = fid
        
        return results

    @torch.no_grad()
    def compute_fid_per_intersectional_subgroup(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        sex_labels: Optional[torch.Tensor] = None,
        race_labels: Optional[torch.Tensor] = None,
        age_labels: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        use_radimagenet: bool = False
    ) -> Dict[str, float]:
        """
        Compute FID per intersectional subgroup (Level 2: age group x ethnicity x sex).
        
        OPTIMIZED: Computes embeddings once for all images, then filters by subgroup.
        This is much more efficient than recomputing embeddings for each subgroup.
        
        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            sex_labels: Sex labels [N] (0=Female, 1=Male, -1=invalid)
            race_labels: Race labels [N] (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            age_labels: Age labels [N] (age in years, -1=invalid)
            batch_size: Batch size for processing
            use_radimagenet: If True, use RadImageNet FID instead of Inception v3
            
        Returns:
            Dictionary mapping intersectional subgroup names to FID values.
            Format: "age_group_sex_race" -> fid_value
            Example: "40-60_Male_White": 15.23
        """
        results = {}
        
        if sex_labels is None or race_labels is None or age_labels is None:
            warnings.warn("Missing labels for intersectional subgroup analysis. Returning empty results.")
            return results
        
        # OPTIMIZATION: Compute embeddings once for all images
        print("  Computing embeddings for all images (once) for intersectional subgroups...")
        if use_radimagenet:
            real_embeddings = self._get_radimagenet_embeddings(real_images, batch_size)
            synthetic_embeddings = self._get_radimagenet_embeddings(synthetic_images, batch_size)
        else:
            real_embeddings = self._get_inception_embeddings(real_images, batch_size)
            synthetic_embeddings = self._get_inception_embeddings(synthetic_images, batch_size)
        
        # Ensure embeddings are 2D [N, D]
        if real_embeddings.ndim > 2:
            real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)
        if synthetic_embeddings.ndim > 2:
            synthetic_embeddings = synthetic_embeddings.reshape(synthetic_embeddings.shape[0], -1)
        
        # Convert to tensors if needed
        if isinstance(sex_labels, torch.Tensor):
            sex_labels_tensor = sex_labels
        else:
            sex_labels_tensor = torch.tensor(sex_labels, dtype=torch.long)
        
        if isinstance(race_labels, torch.Tensor):
            race_labels_tensor = race_labels
        else:
            race_labels_tensor = torch.tensor(race_labels, dtype=torch.long)
        
        if isinstance(age_labels, torch.Tensor):
            age_labels_tensor = age_labels
        else:
            age_labels_tensor = torch.tensor(age_labels, dtype=torch.float32)
        
        # Ensure all tensors are on the same device
        device = real_images.device
        sex_labels_tensor = sex_labels_tensor.to(device)
        race_labels_tensor = race_labels_tensor.to(device)
        age_labels_tensor = age_labels_tensor.to(device)
        
        # Helper function to compute FID from pre-computed embeddings
        def compute_intersectional_fid_from_embeddings(real_emb, synth_emb, subgroup_name):
            if len(real_emb) < 2 or len(synth_emb) < 2:
                return np.nan
            
            try:
                # Compute mean and covariance from embeddings
                mu_real = np.mean(real_emb, axis=0)
                sigma_real = np.cov(real_emb, rowvar=False)
                
                mu_synthetic = np.mean(synth_emb, axis=0)
                sigma_synthetic = np.cov(synth_emb, rowvar=False)
                
                # Compute FID
                ssdiff = np.sum((mu_real - mu_synthetic) ** 2.0)
                covmean = sqrtm(sigma_real.dot(sigma_synthetic))
                
                # Handle numerical errors
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                
                fid = ssdiff + np.trace(sigma_real + sigma_synthetic - 2.0 * covmean)
                return float(fid)
            except Exception as e:
                warnings.warn(f"Failed to compute FID for intersectional subgroup '{subgroup_name}': {e}")
                return np.nan
        
        # Iterate over all combinations
        age_groups = self.age_group_strings
        sex_values = [0, 1]  # Female, Male
        race_values = [0, 1, 2, 3]  # White, Black, Asian, Hispanic
        
        for age_group in age_groups:
            for sex_idx in sex_values:
                for race_idx in race_values:
                    # Create mask for this intersectional subgroup
                    mask = torch.zeros(len(age_labels_tensor), dtype=torch.bool, device=device)
                    for i in range(len(age_labels_tensor)):
                        age = age_labels_tensor[i].item()
                        sex = sex_labels_tensor[i].item()
                        race = race_labels_tensor[i].item()
                        
                        # Check if this sample belongs to the subgroup
                        if (age >= 0 and self._get_age_group(age) == age_group and
                            sex == sex_idx and race == race_idx):
                            mask[i] = True
                    
                    if mask.sum().item() >= 2:  # Need at least 2 samples
                        sex_label = self._get_sex_label(sex_idx)
                        race_label = self._get_race_label(race_idx)
                        subgroup_name = f"{age_group}_{sex_label}_{race_label}"
                        
                        # Filter embeddings by mask (much faster than recomputing)
                        mask_np = mask.cpu().numpy()
                        real_emb_subset = real_embeddings[mask_np]
                        synth_emb_subset = synthetic_embeddings[mask_np]
                        
                        fid = compute_intersectional_fid_from_embeddings(real_emb_subset, synth_emb_subset, subgroup_name)
                        if not np.isnan(fid):
                            results[subgroup_name] = fid
        
        return results

    @torch.no_grad()
    def compute_ms_ssim_per_subgroup(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        sex_labels: Optional[torch.Tensor] = None,
        race_labels: Optional[torch.Tensor] = None,
        age_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute MS-SSIM per subgroup (Level 1: per sex, per ethnicity, per age group).
        
        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            sex_labels: Sex labels [N] (0=Female, 1=Male, -1=invalid)
            race_labels: Race labels [N] (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            age_labels: Age labels [N] (age in years, -1=invalid)
            
        Returns:
            Dictionary with structure:
            {
                "sex": {"Female": ms_ssim_value, "Male": ms_ssim_value},
                "race": {"White": ms_ssim_value, "Black": ms_ssim_value, ...},
                "age_group": {"0-18": ms_ssim_value, "18-40": ms_ssim_value, ...}
            }
        """
        from pytorch_msssim import ms_ssim
        
        results = {
            "sex": {},
            "race": {},
            "age_group": {}
        }
        
        # Helper function to compute MS-SSIM for a subgroup
        def compute_subgroup_ms_ssim(real_subset, synth_subset, subgroup_name, group_type):
            if len(real_subset) < 1:
                return np.nan
            
            try:
                # Normalize to [0, 1]
                real_norm = real_subset.clone()
                synth_norm = synth_subset.clone()
                if real_norm.min() < 0:
                    real_norm = (real_norm + 1.0) / 2.0
                if synth_norm.min() < 0:
                    synth_norm = (synth_norm + 1.0) / 2.0
                
                ms_ssim_val = ms_ssim(
                    real_norm,
                    synth_norm,
                    data_range=1.0,
                    size_average=True
                )
                return float(ms_ssim_val.item())
            except Exception as e:
                warnings.warn(f"Failed to compute MS-SSIM for {group_type} subgroup '{subgroup_name}': {e}")
                return np.nan
        
        # Ensure labels are on the same device as images
        device = real_images.device
        
        # Group by sex
        if sex_labels is not None:
            if isinstance(sex_labels, torch.Tensor):
                sex_labels_tensor = sex_labels.to(device)
            else:
                sex_labels_tensor = torch.tensor(sex_labels, dtype=torch.long, device=device)
            
            for sex_idx in [0, 1]:  # Female, Male
                mask = (sex_labels_tensor == sex_idx)
                if mask.sum().item() >= 1:  # Need at least 1 sample
                    sex_label = self._get_sex_label(sex_idx)
                    real_subset = real_images[mask]
                    synth_subset = synthetic_images[mask]
                    ms_ssim_val = compute_subgroup_ms_ssim(real_subset, synth_subset, sex_label, "sex")
                    if not np.isnan(ms_ssim_val):
                        results["sex"][sex_label] = ms_ssim_val
        
        # Group by race
        if race_labels is not None:
            if isinstance(race_labels, torch.Tensor):
                race_labels_tensor = race_labels.to(device)
            else:
                race_labels_tensor = torch.tensor(race_labels, dtype=torch.long, device=device)
            
            for race_idx in [0, 1, 2, 3]:  # White, Black, Asian, Hispanic
                mask = (race_labels_tensor == race_idx)
                if mask.sum().item() >= 1:  # Need at least 1 sample
                    race_label = self._get_race_label(race_idx)
                    real_subset = real_images[mask]
                    synth_subset = synthetic_images[mask]
                    ms_ssim_val = compute_subgroup_ms_ssim(real_subset, synth_subset, race_label, "race")
                    if not np.isnan(ms_ssim_val):
                        results["race"][race_label] = ms_ssim_val
        
        # Group by age group
        if age_labels is not None:
            if isinstance(age_labels, torch.Tensor):
                age_labels_tensor = age_labels.to(device)
            else:
                age_labels_tensor = torch.tensor(age_labels, dtype=torch.float32, device=device)
            
            age_groups = self.age_group_strings
            for age_group in age_groups:
                # Create mask for this age group
                mask = torch.zeros(len(age_labels_tensor), dtype=torch.bool, device=device)
                for i, age in enumerate(age_labels_tensor):
                    if age.item() >= 0:
                        group = self._get_age_group(age.item())
                        if group == age_group:
                            mask[i] = True
                
                if mask.sum().item() >= 1:  # Need at least 1 sample
                    real_subset = real_images[mask]
                    synth_subset = synthetic_images[mask]
                    ms_ssim_val = compute_subgroup_ms_ssim(real_subset, synth_subset, age_group, "age_group")
                    if not np.isnan(ms_ssim_val):
                        results["age_group"][age_group] = ms_ssim_val
        
        return results

    @torch.no_grad()
    def compute_ms_ssim_per_intersectional_subgroup(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        sex_labels: Optional[torch.Tensor] = None,
        race_labels: Optional[torch.Tensor] = None,
        age_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute MS-SSIM per intersectional subgroup (Level 2: age group x ethnicity x sex).
        
        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            sex_labels: Sex labels [N] (0=Female, 1=Male, -1=invalid)
            race_labels: Race labels [N] (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            age_labels: Age labels [N] (age in years, -1=invalid)
            
        Returns:
            Dictionary mapping intersectional subgroup names to MS-SSIM values.
            Format: "age_group_sex_race" -> ms_ssim_value
            Example: "40-60_Male_White": 0.85
        """
        from pytorch_msssim import ms_ssim
        
        results = {}
        
        if sex_labels is None or race_labels is None or age_labels is None:
            warnings.warn("Missing labels for intersectional subgroup analysis. Returning empty results.")
            return results
        
        # Convert to tensors if needed
        if isinstance(sex_labels, torch.Tensor):
            sex_labels_tensor = sex_labels
        else:
            sex_labels_tensor = torch.tensor(sex_labels, dtype=torch.long)
        
        if isinstance(race_labels, torch.Tensor):
            race_labels_tensor = race_labels
        else:
            race_labels_tensor = torch.tensor(race_labels, dtype=torch.long)
        
        if isinstance(age_labels, torch.Tensor):
            age_labels_tensor = age_labels
        else:
            age_labels_tensor = torch.tensor(age_labels, dtype=torch.float32)
        
        # Ensure all tensors are on the same device
        device = real_images.device
        sex_labels_tensor = sex_labels_tensor.to(device)
        race_labels_tensor = race_labels_tensor.to(device)
        age_labels_tensor = age_labels_tensor.to(device)
        
        # Iterate over all combinations
        age_groups = self.age_group_strings
        sex_values = [0, 1]  # Female, Male
        race_values = [0, 1, 2, 3]  # White, Black, Asian, Hispanic
        
        for age_group in age_groups:
            for sex_idx in sex_values:
                for race_idx in race_values:
                    # Create mask for this intersectional subgroup
                    mask = torch.zeros(len(age_labels_tensor), dtype=torch.bool, device=device)
                    for i in range(len(age_labels_tensor)):
                        age = age_labels_tensor[i].item()
                        sex = sex_labels_tensor[i].item()
                        race = race_labels_tensor[i].item()
                        
                        # Check if this sample belongs to the subgroup
                        if (age >= 0 and self._get_age_group(age) == age_group and
                            sex == sex_idx and race == race_idx):
                            mask[i] = True
                    
                    if mask.sum().item() >= 1:  # Need at least 1 sample
                        sex_label = self._get_sex_label(sex_idx)
                        race_label = self._get_race_label(race_idx)
                        subgroup_name = f"{age_group}_{sex_label}_{race_label}"
                        
                        real_subset = real_images[mask]
                        synth_subset = synthetic_images[mask]
                        
                        try:
                            # Normalize to [0, 1]
                            real_norm = real_subset.clone()
                            synth_norm = synth_subset.clone()
                            if real_norm.min() < 0:
                                real_norm = (real_norm + 1.0) / 2.0
                            if synth_norm.min() < 0:
                                synth_norm = (synth_norm + 1.0) / 2.0
                            
                            ms_ssim_val = ms_ssim(
                                real_norm,
                                synth_norm,
                                data_range=1.0,
                                size_average=True
                            )
                            results[subgroup_name] = float(ms_ssim_val.item())
                        except Exception as e:
                            warnings.warn(f"Failed to compute MS-SSIM for intersectional subgroup '{subgroup_name}': {e}")
        
        return results

    @torch.no_grad()
    def compute_biovil_similarity_per_subgroup(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        sex_labels: Optional[torch.Tensor] = None,
        race_labels: Optional[torch.Tensor] = None,
        age_labels: Optional[torch.Tensor] = None,
        batch_size: int = 16
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute BioViL similarity per subgroup (Level 1: per sex, per ethnicity, per age group).
        
        OPTIMIZED: Computes embeddings once for all images, then filters by subgroup.
        This is much more efficient than recomputing embeddings for each subgroup.
        
        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            sex_labels: Sex labels [N] (0=Female, 1=Male, -1=invalid)
            race_labels: Race labels [N] (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            age_labels: Age labels [N] (age in years, -1=invalid)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with structure:
            {
                "sex": {"Female": similarity_value, "Male": similarity_value},
                "race": {"White": similarity_value, "Black": similarity_value, ...},
                "age_group": {"0-18": similarity_value, "18-40": similarity_value, ...}
            }
        """
        self._load_biovil()
        
        results = {
            "sex": {},
            "race": {},
            "age_group": {}
        }
        
        if self.biovil_model is None:
            warnings.warn("BioViL model not available. Returning empty results.")
            return results
        
        # OPTIMIZATION: Compute embeddings once for all images
        print("  Computing BioViL embeddings for all images (once)...")
        real_embeddings = self._get_biovil_embeddings(real_images, batch_size=batch_size)
        synthetic_embeddings = self._get_biovil_embeddings(synthetic_images, batch_size=batch_size)
        
        # Ensure embeddings are normalized
        real_embeddings = F.normalize(real_embeddings, dim=1)
        synthetic_embeddings = F.normalize(synthetic_embeddings, dim=1)
        
        # Helper function to compute BioViL similarity from pre-computed embeddings
        def compute_subgroup_biovil_from_embeddings(real_emb, synth_emb, subgroup_name, group_type):
            if len(real_emb) < 1:
                return np.nan
            
            try:
                # Compute pairwise cosine similarity
                similarities = torch.sum(real_emb * synth_emb, dim=1)
                return float(similarities.mean().item())
            except Exception as e:
                warnings.warn(f"Failed to compute BioViL similarity for {group_type} subgroup '{subgroup_name}': {e}")
                return np.nan
        
        # Ensure labels are on the same device as images
        device = real_images.device
        
        # Group by sex
        if sex_labels is not None:
            if isinstance(sex_labels, torch.Tensor):
                sex_labels_tensor = sex_labels.to(device)
            else:
                sex_labels_tensor = torch.tensor(sex_labels, dtype=torch.long, device=device)
            
            for sex_idx in [0, 1]:  # Female, Male
                mask = (sex_labels_tensor == sex_idx)
                if mask.sum().item() >= 1:  # Need at least 1 sample
                    sex_label = self._get_sex_label(sex_idx)
                    # Filter embeddings by mask (much faster than recomputing)
                    mask_np = mask.cpu().numpy()
                    real_emb_subset = real_embeddings[mask_np]
                    synth_emb_subset = synthetic_embeddings[mask_np]
                    biovil_val = compute_subgroup_biovil_from_embeddings(real_emb_subset, synth_emb_subset, sex_label, "sex")
                    if not np.isnan(biovil_val):
                        results["sex"][sex_label] = biovil_val
        
        # Group by race
        if race_labels is not None:
            if isinstance(race_labels, torch.Tensor):
                race_labels_tensor = race_labels.to(device)
            else:
                race_labels_tensor = torch.tensor(race_labels, dtype=torch.long, device=device)
            
            for race_idx in [0, 1, 2, 3]:  # White, Black, Asian, Hispanic
                mask = (race_labels_tensor == race_idx)
                if mask.sum().item() >= 1:  # Need at least 1 sample
                    race_label = self._get_race_label(race_idx)
                    # Filter embeddings by mask (much faster than recomputing)
                    mask_np = mask.cpu().numpy()
                    real_emb_subset = real_embeddings[mask_np]
                    synth_emb_subset = synthetic_embeddings[mask_np]
                    biovil_val = compute_subgroup_biovil_from_embeddings(real_emb_subset, synth_emb_subset, race_label, "race")
                    if not np.isnan(biovil_val):
                        results["race"][race_label] = biovil_val
        
        # Group by age group
        if age_labels is not None:
            if isinstance(age_labels, torch.Tensor):
                age_labels_tensor = age_labels.to(device)
            else:
                age_labels_tensor = torch.tensor(age_labels, dtype=torch.float32, device=device)
            
            age_groups = self.age_group_strings
            for age_group in age_groups:
                # Create mask for this age group
                mask = torch.zeros(len(age_labels_tensor), dtype=torch.bool, device=device)
                for i, age in enumerate(age_labels_tensor):
                    if age.item() >= 0:
                        group = self._get_age_group(age.item())
                        if group == age_group:
                            mask[i] = True
                
                if mask.sum().item() >= 1:  # Need at least 1 sample
                    # Filter embeddings by mask (much faster than recomputing)
                    mask_np = mask.cpu().numpy()
                    real_emb_subset = real_embeddings[mask_np]
                    synth_emb_subset = synthetic_embeddings[mask_np]
                    biovil_val = compute_subgroup_biovil_from_embeddings(real_emb_subset, synth_emb_subset, age_group, "age_group")
                    if not np.isnan(biovil_val):
                        results["age_group"][age_group] = biovil_val
        
        return results

    @torch.no_grad()
    def compute_biovil_similarity_per_intersectional_subgroup(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        sex_labels: Optional[torch.Tensor] = None,
        race_labels: Optional[torch.Tensor] = None,
        age_labels: Optional[torch.Tensor] = None,
        batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Compute BioViL similarity per intersectional subgroup (Level 2: age group x ethnicity x sex).
        
        OPTIMIZED: Computes embeddings once for all images, then filters by subgroup.
        This is much more efficient than recomputing embeddings for each subgroup.
        
        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            sex_labels: Sex labels [N] (0=Female, 1=Male, -1=invalid)
            race_labels: Race labels [N] (0=White, 1=Black, 2=Asian, 3=Hispanic, -1=invalid)
            age_labels: Age labels [N] (age in years, -1=invalid)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping intersectional subgroup names to BioViL similarity values.
            Format: "age_group_sex_race" -> similarity_value
            Example: "40-60_Male_White": 0.92
        """
        self._load_biovil()
        
        results = {}
        
        if self.biovil_model is None:
            warnings.warn("BioViL model not available. Returning empty results.")
            return results
        
        if sex_labels is None or race_labels is None or age_labels is None:
            warnings.warn("Missing labels for intersectional subgroup analysis. Returning empty results.")
            return results
        
        # OPTIMIZATION: Compute embeddings once for all images
        print("  Computing BioViL embeddings for all images (once) for intersectional subgroups...")
        real_embeddings = self._get_biovil_embeddings(real_images, batch_size=batch_size)
        synthetic_embeddings = self._get_biovil_embeddings(synthetic_images, batch_size=batch_size)
        
        # Ensure embeddings are normalized
        real_embeddings = F.normalize(real_embeddings, dim=1)
        synthetic_embeddings = F.normalize(synthetic_embeddings, dim=1)
        
        # Convert to tensors if needed
        if isinstance(sex_labels, torch.Tensor):
            sex_labels_tensor = sex_labels
        else:
            sex_labels_tensor = torch.tensor(sex_labels, dtype=torch.long)
        
        if isinstance(race_labels, torch.Tensor):
            race_labels_tensor = race_labels
        else:
            race_labels_tensor = torch.tensor(race_labels, dtype=torch.long)
        
        if isinstance(age_labels, torch.Tensor):
            age_labels_tensor = age_labels
        else:
            age_labels_tensor = torch.tensor(age_labels, dtype=torch.float32)
        
        # Ensure all tensors are on the same device
        device = real_images.device
        sex_labels_tensor = sex_labels_tensor.to(device)
        race_labels_tensor = race_labels_tensor.to(device)
        age_labels_tensor = age_labels_tensor.to(device)
        
        # Iterate over all combinations
        age_groups = self.age_group_strings
        sex_values = [0, 1]  # Female, Male
        race_values = [0, 1, 2, 3]  # White, Black, Asian, Hispanic
        
        for age_group in age_groups:
            for sex_idx in sex_values:
                for race_idx in race_values:
                    # Create mask for this intersectional subgroup
                    mask = torch.zeros(len(age_labels_tensor), dtype=torch.bool, device=device)
                    for i in range(len(age_labels_tensor)):
                        age = age_labels_tensor[i].item()
                        sex = sex_labels_tensor[i].item()
                        race = race_labels_tensor[i].item()
                        
                        # Check if this sample belongs to the subgroup
                        if (age >= 0 and self._get_age_group(age) == age_group and
                            sex == sex_idx and race == race_idx):
                            mask[i] = True
                    
                    if mask.sum().item() >= 1:  # Need at least 1 sample
                        sex_label = self._get_sex_label(sex_idx)
                        race_label = self._get_race_label(race_idx)
                        subgroup_name = f"{age_group}_{sex_label}_{race_label}"
                        
                        # Filter embeddings by mask (much faster than recomputing)
                        mask_np = mask.cpu().numpy()
                        real_emb_subset = real_embeddings[mask_np]
                        synth_emb_subset = synthetic_embeddings[mask_np]
                        
                        try:
                            # Compute pairwise cosine similarity
                            similarities = torch.sum(real_emb_subset * synth_emb_subset, dim=1)
                            results[subgroup_name] = float(similarities.mean().item())
                        except Exception as e:
                            warnings.warn(f"Failed to compute BioViL similarity for intersectional subgroup '{subgroup_name}': {e}")
        
        return results


class IntraPromptDiversityMetrics:
    """
    Evaluates diversity of multiple images generated from the same prompt.
    Lower scores indicate higher diversity (more desirable).
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize diversity metrics.

        Args:
            device: Device to run inference on
        """
        self.device = device
        self.biovil_model = None

    def _load_biovil(self):
        """Lazy load BioViL model using hi-ml-multimodal library."""
        if self.biovil_model is None:
            try:
                from health_multimodal.image.utils import get_image_inference, ImageModelType

                # Load BioViL using the official hi-ml-multimodal library
                self.biovil_inference_engine = get_image_inference(
                    image_model_type=ImageModelType.BIOVIL
                )
                
                # Move model to device
                self.biovil_inference_engine.model.to(self.device)
                self.biovil_inference_engine.model.eval()
                
                # Store the transform for preprocessing
                self.biovil_transform = self.biovil_inference_engine.transform
                
                # Store model reference for convenience
                self.biovil_model = self.biovil_inference_engine.model
                
                print("✓ BioViL loaded successfully using hi-ml-multimodal library")
                    
            except ImportError as e:
                warnings.warn(
                    f"Could not import hi-ml-multimodal library: {e}\n"
                    "Please install it using: pip install hi-ml-multimodal"
                )
                self.biovil_model = None
                self.biovil_inference_engine = None
            except Exception as e:
                warnings.warn(f"Could not load BioViL model: {e}")
                self.biovil_model = None
                self.biovil_inference_engine = None

    def compute_intra_prompt_ms_ssim(
        self,
        images_per_prompt: List[torch.Tensor]
    ) -> float:
        """
        Compute mean pairwise MS-SSIM among images generated from same prompt.

        Args:
            images_per_prompt: List of image sets, where each set contains
                              multiple images [K, C, H, W] from same prompt

        Returns:
            Mean MS-SSIM across all prompts (lower = more diverse)
        """
        from pytorch_msssim import ms_ssim

        per_prompt_ssims = []

        for images in images_per_prompt:
            # Normalize to [0, 1]
            if images.min() < 0:
                images = (images + 1.0) / 2.0

            # Compute pairwise MS-SSIM
            K = len(images)
            pairwise_ssims = []

            for i in range(K):
                for j in range(i + 1, K):
                    img1 = images[i:i+1]
                    img2 = images[j:j+1]

                    ssim_val = ms_ssim(
                        img1,
                        img2,
                        data_range=1.0,
                        size_average=True
                    )
                    pairwise_ssims.append(ssim_val.item())

            # Average for this prompt
            per_prompt_ssims.append(np.mean(pairwise_ssims))

        # Average across all prompts
        return float(np.mean(per_prompt_ssims))

    @torch.no_grad()
    def compute_intra_prompt_biovil_similarity(
        self,
        images_per_prompt: List[torch.Tensor]
    ) -> float:
        """
        Compute mean pairwise BioViL cosine similarity among images from same prompt.

        Args:
            images_per_prompt: List of image sets [K, C, H, W] per prompt

        Returns:
            Mean cosine similarity (lower = more diverse)
        """
        self._load_biovil()

        if self.biovil_model is None:
            warnings.warn("BioViL model not available. Returning NaN.")
            return np.nan

        per_prompt_sims = []

        for images in images_per_prompt:
            # Get embeddings
            embeddings = self._get_biovil_embeddings(images)
            embeddings = F.normalize(embeddings, dim=1)

            # Compute pairwise cosine similarity
            K = len(embeddings)
            pairwise_sims = []

            for i in range(K):
                for j in range(i + 1, K):
                    sim = torch.sum(embeddings[i] * embeddings[j])
                    pairwise_sims.append(sim.item())

            # Average for this prompt
            per_prompt_sims.append(np.mean(pairwise_sims))

        # Average across all prompts
        return float(np.mean(per_prompt_sims))

    @torch.no_grad()
    def _get_biovil_embeddings(self, images: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        """
        Extract BioViL embeddings from images using hi-ml-multimodal library.

        Args:
            images: Images [N, C, H, W] in range [0, 1] or [-1, 1]
            batch_size: Batch size for processing to avoid OOM errors

        Returns:
            Embeddings [N, D] in the joint image-text latent space, L2-normalized
        """
        from PIL import Image
        
        if self.biovil_inference_engine is None:
            raise RuntimeError(
                "BioViL model not loaded. Please ensure hi-ml-multimodal is installed: "
                "pip install hi-ml-multimodal"
            )

        # Normalize to [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Convert tensor to PIL Images for processing
        pil_images = []
        for i in range(images.shape[0]):
            # Get single image [C, H, W]
            img = images[i]
            
            # Convert to numpy and scale to [0, 255]
            img_np = img.cpu().numpy()
            
            # Handle grayscale (single channel)
            if img_np.shape[0] == 1:
                img_np = img_np[0]  # Remove channel dimension -> [H, W]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='L')  # Grayscale
            else:
                # RGB image - convert to grayscale for BioViL (expects chest X-rays in grayscale)
                img_np = img_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='RGB').convert('L')  # Convert to grayscale
            
            pil_images.append(pil_img)

        # Process images in batches to avoid OOM
        all_embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch_pil = pil_images[i:i + batch_size]
            
            # Apply BioViL transform (includes resize, normalize, etc.)
            batch = torch.stack([self.biovil_transform(img) for img in batch_pil])
            batch = batch.to(self.device)

            # Extract embeddings using the inference engine
            output = self.biovil_model(batch)  # Returns ImageModelOutput
            batch_embeddings = output.projected_global_embedding  # [B, joint_feature_size]
            
            # L2 normalize (recommended for similarity computation)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
            
            # Move to CPU to save GPU memory
            all_embeddings.append(batch_embeddings.cpu())
            
            # Clear GPU cache
            del batch, output, batch_embeddings
            torch.cuda.empty_cache()
        
        # Concatenate all embeddings and move back to device
        embeddings = torch.cat(all_embeddings, dim=0).to(self.device)

        return embeddings


# ============================================================================
# FairDiffusion Metrics Integration
# ============================================================================

def compute_auc_fairdiffusion(pred_prob, y, num_classes=2):
    """
    Compute AUC score for binary or multiclass classification.
    Adapted from FairDiffusion classification_codebase/src/modules.py
    
    Args:
        pred_prob: Prediction probabilities [N] for binary or [N, num_classes] for multiclass
        y: Ground truth labels [N]
        num_classes: Number of classes (2 for binary, >2 for multiclass)
    
    Returns:
        AUC score
    """
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val


def num_to_onehot(nums, num_to_class):
    """Convert numeric labels to one-hot encoding."""
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec


def prob_to_label(pred_prob):
    """Convert probability predictions to one-hot label predictions."""
    max_prob_indices = np.argmax(pred_prob, axis=1)
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1
    return one_hot_vectors


def numeric_to_one_hot(y, num_classes=None):
    """Convert numeric labels to one-hot encoding."""
    y = np.asarray(y, dtype=np.int32)
    if num_classes is None:
        num_classes = np.max(y) + 1
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    return one_hot_array


def multiclass_demographic_parity(pred_prob, y, attrs):
    """Compute multiclass demographic parity difference."""
    if demographic_parity_difference is None:
        raise ImportError("fairlearn is required for demographic parity metrics")
    
    pred_one_hot = prob_to_label(pred_prob)
    gt_one_hot = numeric_to_one_hot(y)
    
    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(
            pred_one_hot[:, i],
            gt_one_hot[:, i],
            sensitive_features=attrs
        )
        scores.append(tmp_score)
    
    return np.mean(scores)


def multiclass_equalized_odds(pred_prob, y, attrs):
    """Compute multiclass equalized odds difference."""
    if equalized_odds_difference is None:
        raise ImportError("fairlearn is required for equalized odds metrics")
    
    pred_one_hot = prob_to_label(pred_prob)
    gt_one_hot = numeric_to_one_hot(y)
    
    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(
            pred_one_hot[:, i],
            gt_one_hot[:, i],
            sensitive_features=attrs
        )
        scores.append(tmp_score)
    
    return np.mean(scores)


def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    """
    Compute equity-scaled accuracy.
    Adapted from FairDiffusion classification_codebase/src/modules.py
    """
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target) / target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target) / target.shape[0]
    
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group) / gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group) / gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i] - overall_acc)
    
    es_acc = (overall_acc / (alpha * tmp + 1))
    
    return es_acc


def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    """
    Compute equity-scaled AUC.
    Adapted from FairDiffusion classification_codebase/src/modules.py
    """
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        elif num_classes > 2:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i] - overall_auc)
    
    es_auc = (overall_auc / (alpha * tmp + 1))

    return es_auc


def compute_between_group_disparity(auc_list, overall_auc):
    """
    Compute between-group disparity metrics.
    Returns (std_disparity, max_disparity) normalized by overall_auc.
    """
    if overall_auc == 0:
        return np.nan, np.nan
    return np.std(auc_list) / overall_auc, (np.max(auc_list) - np.min(auc_list)) / overall_auc


def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):
    """
    Evaluate comprehensive performance metrics including fairness metrics.
    Adapted from FairDiffusion classification_codebase/src/modules.py
    
    Args:
        preds: Predictions [N] or [N, num_classes]
        gts: Ground truth labels [N]
        attrs: Attributes array [num_attrs, N] where each row is an attribute
        num_classes: Number of classes
    
    Returns:
        Tuple of (esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity)
    """
    if attrs is None:
        return [], [], [], [], [], []
    
    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    eods = []
    between_group_disparity = []

    overall_auc = compute_auc_fairdiffusion(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i, :]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)

        try:
            es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        except Exception as e:
            es_auc = -1.
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            try:
                tmp_auc = compute_auc_fairdiffusion(preds[attr == e], gts[attr == e], num_classes=num_classes)
            except Exception as e:
                tmp_auc = -1.
            aucs_by_group.append(tmp_auc)
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            if demographic_parity_difference is not None:
                dpd = demographic_parity_difference(
                    gts,
                    pred_labels,
                    sensitive_features=attr
                )
                eod = equalized_odds_difference(
                    gts,
                    pred_labels,
                    sensitive_features=attr
                )
            else:
                dpd = np.nan
                eod = np.nan
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            eod = multiclass_equalized_odds(preds, gts, attr)

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity


def evalute_comprehensive_perf_scores(preds, gts, attrs=None, num_classes=2):
    """
    Evaluate comprehensive performance with overall accuracy and AUC.
    Adapted from FairDiffusion classification_codebase/src/modules.py
    """
    if attrs is None:
        # Compute overall metrics only
        overall_acc = accuracy_score(gts, (preds >= 0.5).astype(int)) if num_classes == 2 else accuracy_score(gts, np.argmax(preds, axis=1))
        overall_auc = compute_auc_fairdiffusion(preds, gts, num_classes=num_classes)
        return overall_acc, np.array([]), overall_auc, np.array([]), [], np.array([]), np.array([]), np.array([])
    
    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    eods = []
    between_group_disparity = []

    overall_acc = accuracy_score(gts, (preds >= 0.5).astype(int)) if num_classes == 2 else accuracy_score(gts, np.argmax(preds, axis=1))
    overall_auc = compute_auc_fairdiffusion(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i, :]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        
        try:
            es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        except Exception as e:
            es_auc = -1.
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            try:
                tmp_auc = compute_auc_fairdiffusion(preds[attr == e], gts[attr == e], num_classes=num_classes)
            except Exception as e:
                tmp_auc = -1.
            aucs_by_group.append(tmp_auc)
        aucs_by_attrs.append(np.array(aucs_by_group))
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            if demographic_parity_difference is not None:
                dpd = demographic_parity_difference(
                    gts,
                    pred_labels,
                    sensitive_features=attr
                )
                eod = equalized_odds_difference(
                    gts,
                    pred_labels,
                    sensitive_features=attr
                )
            else:
                dpd = np.nan
                eod = np.nan
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            eod = multiclass_equalized_odds(preds, gts, attr)

        dpds.append(dpd)
        eods.append(eod)

    esaccs_by_attrs = np.array(esaccs_by_attrs)
    esaucs_by_attrs = np.array(esaucs_by_attrs)
    dpds = np.array(dpds)
    eods = np.array(eods)
    between_group_disparity = np.array(between_group_disparity)

    return overall_acc, esaccs_by_attrs, overall_auc, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity


def bootstrap_performance(test_preds, test_gts, test_attrs, bootstrap_repeat_times=100, num_classes=2, num_attrs=3):
    """
    Compute bootstrap statistics for performance metrics.
    Adapted from FairDiffusion classification_codebase/src/modules.py
    """
    test_acc, test_es_acc, test_auc, test_es_auc, test_aucs_by_attrs, test_dpds, test_eods, test_between_group_disparity = (
        [], [], [], [], [[] for i in range(num_attrs)], [], [], []
    )
    
    for i in range(bootstrap_repeat_times):
        tmp_indices = np.array(list(range(0, test_gts.shape[0])))
        bootstrap_indices = resample(tmp_indices, replace=True, n_samples=test_gts.shape[0])
        bootstrap_preds = test_preds[bootstrap_indices]
        bootstrap_gts = test_gts[bootstrap_indices]
        bootstrap_attrs = []
        for x in test_attrs:
            bootstrap_attrs.append(x[bootstrap_indices])
        bootstrap_attrs = np.vstack(bootstrap_attrs)
        
        tmp_test_acc, tmp_test_es_acc, tmp_test_auc, tmp_test_es_auc, tmp_test_aucs_by_attrs, tmp_test_dpds, tmp_test_eods, tmp_test_between_group_disparity = (
            evalute_comprehensive_perf_scores(bootstrap_preds, bootstrap_gts, bootstrap_attrs, num_classes=num_classes)
        )
        
        test_acc.append(tmp_test_acc)
        test_es_acc.append(tmp_test_es_acc)
        test_auc.append(tmp_test_auc)
        test_es_auc.append(tmp_test_es_auc)
        for j in range(len(test_aucs_by_attrs)):
            test_aucs_by_attrs[j].append(tmp_test_aucs_by_attrs[j])
        test_dpds.append(tmp_test_dpds)
        test_eods.append(tmp_test_eods)
        test_between_group_disparity.append(tmp_test_between_group_disparity)
    
    test_acc = np.vstack(test_acc)
    test_es_acc = np.vstack(test_es_acc)
    test_auc = np.vstack(test_auc)
    test_es_auc = np.vstack(test_es_auc)
    for j in range(len(test_aucs_by_attrs)):
        test_aucs_by_attrs[j] = np.vstack(test_aucs_by_attrs[j])
    test_dpds = np.vstack(test_dpds)
    test_eods = np.vstack(test_eods)
    test_between_group_disparity = np.array(test_between_group_disparity)[None, :, :]
    test_between_group_disparity = np.vstack(test_between_group_disparity)

    acc = np.mean(test_acc, axis=0)[0]
    es_acc = np.mean(test_es_acc, axis=0)
    auc_val = np.mean(test_auc, axis=0)[0]
    es_auc = np.mean(test_es_auc, axis=0)
    aucs_by_attrs = []
    for j in range(len(test_aucs_by_attrs)):
        aucs_by_attrs.append(np.mean(test_aucs_by_attrs[j], axis=0))
    dpds = np.mean(test_dpds, axis=0)
    eods = np.mean(test_eods, axis=0)
    between_group_disparity = np.mean(test_between_group_disparity, axis=0)

    acc_std = np.std(test_acc, axis=0)[0]
    es_acc_std = np.std(test_es_acc, axis=0)
    auc_std = np.std(test_auc, axis=0)[0]
    es_auc_std = np.std(test_es_auc, axis=0)
    aucs_by_attrs_std = []
    for j in range(len(test_aucs_by_attrs)):
        aucs_by_attrs_std.append(np.std(test_aucs_by_attrs[j], axis=0))
    dpds_std = np.std(test_dpds, axis=0)
    eods_std = np.std(test_eods, axis=0)
    between_group_disparity_std = np.std(test_between_group_disparity, axis=0)

    return (acc, es_acc, auc_val, es_auc, aucs_by_attrs, dpds, eods, between_group_disparity,
            acc_std, es_acc_std, auc_std, es_auc_std, aucs_by_attrs_std, dpds_std, eods_std, between_group_disparity_std)


class CustomImageDataset(TorchDataset):
    """Custom dataset for loading images from directory."""
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def compute_fairness_metrics_fairdiffusion(dir1, dir2, device="cuda", batch_size=100):
    """
    Compute FID, MIFID, and IS metrics between two image directories.
    Adapted from FairDiffusion examples/text_to_image/evaluate_fairdiffusion.py
    
    Args:
        dir1: Directory containing generated images
        dir2: Directory containing real/reference images
        device: Device to run computation on
        batch_size: Batch size for processing
    
    Returns:
        Dictionary with 'fid', 'mifid', 'is' keys
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).type(torch.uint8))
    ])
    
    dataset1 = CustomImageDataset(directory=dir1, transform=transform)
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    dataset2 = CustomImageDataset(directory=dir2, transform=transform)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    # FID
    fid = FrechetInceptionDistance().to(device_obj)
    fid.set_dtype(torch.float64)
    for batch in dataloader1:
        fid.update(batch.to(device_obj), real=False)
    for batch in dataloader2:
        fid.update(batch.to(device_obj), real=True)
    fid_metric = fid.compute().item()

    # MIFID
    mifid = MemorizationInformedFrechetInceptionDistance().to(device_obj)
    for batch in dataloader1:
        mifid.update(batch.to(device_obj), real=False)
    for batch in dataloader2:
        mifid.update(batch.to(device_obj), real=True)
    mifid_metric = mifid.compute().item()

    # IS
    inception = InceptionScore().to(device_obj)
    for batch in dataloader1:
        inception.update(batch.to(device_obj))
    inception_metric = inception.compute()

    return {'fid': fid_metric, 'mifid': mifid_metric, 'is': inception_metric}


def compute_groupwise_metrics_fairdiffusion(
    generated_dir, actual_dir, prompts_path, demographic_groups, 
    device="cuda", batch_size=100, temp_dir_prefix="metrics_calculation"
):
    """
    Compute metrics for different demographic groups.
    Adapted from FairDiffusion examples/text_to_image/evaluate_fairdiffusion.py
    
    Args:
        generated_dir: Directory containing generated images
        actual_dir: Directory containing real images
        prompts_path: Path to file containing prompts (one per line)
        demographic_groups: List of demographic group strings to filter by
        device: Device to run computation on
        batch_size: Batch size for processing
        temp_dir_prefix: Prefix for temporary directories
    
    Returns:
        Dictionary mapping group names to metrics dictionaries
    """
    # Read prompts and categorize images
    with open(prompts_path, 'r') as file:
        prompts = file.readlines()
    
    group_metrics = {}
    
    # For each demographic group, filter images and calculate metrics
    for group in demographic_groups:
        group_prompts = [idx for idx, prompt in enumerate(prompts) if group in prompt]
        if len(group_prompts) == 0:
            continue
        
        tmp_dir_actual = f"{temp_dir_prefix}_tmp/actual/{group}"
        tmp_dir_generated = f"{temp_dir_prefix}_tmp/generated/{group}"
        os.makedirs(tmp_dir_actual, exist_ok=True)
        os.makedirs(tmp_dir_generated, exist_ok=True)
        
        # Filter and save images for the current group
        for idx in group_prompts:
            actual_file = os.path.join(actual_dir, f"{idx}.jpg")
            generated_file = os.path.join(generated_dir, f"{idx}.jpg")
            if os.path.exists(actual_file):
                shutil.copy(actual_file, os.path.join(tmp_dir_actual, f"{idx}.jpg"))
            if os.path.exists(generated_file):
                shutil.copy(generated_file, os.path.join(tmp_dir_generated, f"{idx}.jpg"))
        
        # Compute metrics for the current group
        if len(os.listdir(tmp_dir_generated)) > 0 and len(os.listdir(tmp_dir_actual)) > 0:
            metrics_dict = compute_fairness_metrics_fairdiffusion(
                dir1=tmp_dir_generated, 
                dir2=tmp_dir_actual,
                device=device,
                batch_size=batch_size
            )
            group_metrics[group] = metrics_dict
        
        # Clean up temporary directories after calculation
        if os.path.exists(tmp_dir_actual):
            shutil.rmtree(tmp_dir_actual)
        if os.path.exists(tmp_dir_generated):
            shutil.rmtree(tmp_dir_generated)
    
    return group_metrics


class ValidationMetricsRunner:
    """
    Main runner for computing all validation metrics on a validation set.
    Coordinates text alignment, similarity, and diversity metrics.
    """

    def __init__(
        self,
        device: str = "cuda",
        sex_model_path: Optional[str] = None,
        age_bins: Optional[List[int]] = None
    ):
        """
        Initialize all metric computers.

        Args:
            device: Device to run inference on
            sex_model_path: Path to sex prediction model checkpoint
            age_bins: Age bin thresholds for subgroup metrics. If None, uses default from dataset_wds.
                      Example: [18, 40, 60, 80] creates bins: [0-18, 18-40, 40-60, 60-80, 80+]
        """
        self.device = device
        self.age_bins = age_bins  # Store age_bins for use in age bin accuracy computation

        self.text_alignment = TextPromptAlignmentMetrics(device)
        if sex_model_path:
            self.text_alignment.load_sex_model(sex_model_path)

        self.similarity = RealSyntheticSimilarityMetrics(device, age_bins=age_bins)
        self.diversity = IntraPromptDiversityMetrics(device)

    def compute_all_metrics(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        disease_labels: torch.Tensor,
        sex_labels: torch.Tensor,
        race_labels: torch.Tensor,
        age_labels: torch.Tensor,
        images_per_prompt: Optional[List[torch.Tensor]] = None,
        compute_fairdiffusion_metrics: bool = False,
        prompts: Optional[List[str]] = None,
        demographic_groups: Optional[List[str]] = None,
        temp_dir_prefix: str = "metrics_calculation"
    ) -> Dict[str, float]:
        """
        Compute all validation metrics.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            disease_labels: Disease labels [N, 5] for target diseases
            sex_labels: Sex labels [N]
            race_labels: Race labels [N]
            age_labels: Age labels [N]
            images_per_prompt: Optional list for diversity metrics

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # 1. Text Prompt Alignment Metrics
        print("Computing text prompt alignment metrics...")

        disease_aurocs = self.text_alignment.compute_disease_auroc(
            synthetic_images, disease_labels
        )
        metrics.update(disease_aurocs)

        metrics["sex_accuracy"] = self.text_alignment.compute_sex_accuracy(
            synthetic_images, sex_labels
        )

        metrics["race_accuracy"] = self.text_alignment.compute_race_accuracy(
            synthetic_images, race_labels
        )

        metrics["age_rmse"] = self.text_alignment.compute_age_rmse(
            synthetic_images, age_labels
        )

        metrics["age_bin_accuracy"] = self.text_alignment.compute_age_bin_accuracy(
            synthetic_images, age_labels, age_bins=self.age_bins
        )

        # 2. Real-Synthetic Image Similarity Metrics
        print("Computing real-synthetic similarity metrics...")

        metrics["fid"] = self.similarity.compute_fid(
            real_images, synthetic_images
        )

        metrics["biovil_similarity"] = self.similarity.compute_biovil_similarity(
            real_images, synthetic_images
        )

        metrics["ms_ssim"] = self.similarity.compute_ms_ssim(
            real_images, synthetic_images
        )

        # 3. Intra-Prompt Diversity Metrics (if provided)
        if images_per_prompt is not None:
            print("Computing intra-prompt diversity metrics...")

            metrics["intra_prompt_ms_ssim"] = self.diversity.compute_intra_prompt_ms_ssim(
                images_per_prompt
            )

            metrics["intra_prompt_biovil_similarity"] = self.diversity.compute_intra_prompt_biovil_similarity(
                images_per_prompt
            )

        # 4. FairDiffusion Metrics (if requested)
        if compute_fairdiffusion_metrics:
            print("Computing FairDiffusion metrics...")
            
            # Compute fairness metrics using classification predictions
            # For this, we need predictions from a classifier model
            # This would typically be done separately, but we can add a placeholder here
            # The actual implementation would require classifier predictions
            
            # If prompts and demographic groups are provided, compute groupwise metrics
            if prompts is not None and demographic_groups is not None:
                # Save images temporarily for groupwise metric computation
                import tempfile
                with tempfile.TemporaryDirectory(prefix=temp_dir_prefix) as temp_dir:
                    generated_dir = os.path.join(temp_dir, "generated")
                    actual_dir = os.path.join(temp_dir, "actual")
                    prompts_file = os.path.join(temp_dir, "prompts.txt")
                    
                    os.makedirs(generated_dir, exist_ok=True)
                    os.makedirs(actual_dir, exist_ok=True)
                    
                    # Save images
                    for idx in range(len(synthetic_images)):
                        # Convert tensor to PIL and save
                        syn_img = synthetic_images[idx]
                        if syn_img.dim() == 3:
                            syn_img = syn_img.unsqueeze(0)
                        syn_img_np = syn_img.squeeze().cpu().numpy()
                        if syn_img_np.shape[0] == 1:  # Grayscale
                            syn_img_np = syn_img_np[0]
                            syn_img_pil = Image.fromarray((syn_img_np * 255).clip(0, 255).astype(np.uint8), mode='L').convert('RGB')
                        else:  # RGB
                            syn_img_np = syn_img_np.transpose(1, 2, 0)
                            syn_img_pil = Image.fromarray((syn_img_np * 255).clip(0, 255).astype(np.uint8), mode='RGB')
                        syn_img_pil.save(os.path.join(generated_dir, f"{idx}.jpg"))
                        
                        real_img = real_images[idx]
                        if real_img.dim() == 3:
                            real_img = real_img.unsqueeze(0)
                        real_img_np = real_img.squeeze().cpu().numpy()
                        if real_img_np.shape[0] == 1:  # Grayscale
                            real_img_np = real_img_np[0]
                            real_img_pil = Image.fromarray((real_img_np * 255).clip(0, 255).astype(np.uint8), mode='L').convert('RGB')
                        else:  # RGB
                            real_img_np = real_img_np.transpose(1, 2, 0)
                            real_img_pil = Image.fromarray((real_img_np * 255).clip(0, 255).astype(np.uint8), mode='RGB')
                        real_img_pil.save(os.path.join(actual_dir, f"{idx}.jpg"))
                    
                    # Save prompts
                    with open(prompts_file, 'w', encoding='utf-8') as f:
                        for prompt in prompts:
                            f.write(prompt + "\n")
                    
                    # Compute groupwise metrics
                    group_metrics = compute_groupwise_metrics_fairdiffusion(
                        generated_dir=generated_dir,
                        actual_dir=actual_dir,
                        prompts_path=prompts_file,
                        demographic_groups=demographic_groups,
                        device=self.device,
                        temp_dir_prefix=temp_dir_prefix
                    )
                    
                    # Add groupwise metrics to results
                    for group, group_metric_dict in group_metrics.items():
                        for metric_name, metric_value in group_metric_dict.items():
                            metrics[f"{metric_name}_{group.replace(' ', '_')}"] = metric_value

        return metrics

    def print_metrics_summary(self, metrics: Dict[str, float]):
        """
        Print formatted summary of validation metrics.

        Args:
            metrics: Dictionary of computed metrics
        """
        print("\n" + "="*60)
        print("VALIDATION METRICS SUMMARY")
        print("="*60)

        print("\n1. TEXT PROMPT ALIGNMENT")
        print("-" * 60)
        print(f"  Disease Classification (AUROC):")
        for disease in ["Atelectasis", "Cardiomegaly", "Edema", "Pneumothorax", "Pleural Effusion"]:
            if disease in metrics:
                print(f"    {disease:20s}: {metrics[disease]:.4f}")
        if "mean_auroc" in metrics:
            print(f"    {'Mean AUROC':20s}: {metrics['mean_auroc']:.4f}")

        print(f"\n  Demographic Attributes:")
        if "sex_accuracy" in metrics:
            print(f"    Sex Accuracy:   {metrics['sex_accuracy']:.4f}")
        if "race_accuracy" in metrics:
            print(f"    Race Accuracy:  {metrics['race_accuracy']:.4f}")
        if "age_rmse" in metrics:
            print(f"    Age RMSE:       {metrics['age_rmse']:.2f} years")

        print("\n2. REAL-SYNTHETIC IMAGE SIMILARITY")
        print("-" * 60)
        if "fid" in metrics:
            print(f"  FID Score:               {metrics['fid']:.2f} (lower is better)")
        if "biovil_similarity" in metrics:
            print(f"  BioViL Cosine Similarity: {metrics['biovil_similarity']:.4f} (1.0 = perfect)")
        if "ms_ssim" in metrics:
            print(f"  MS-SSIM:                 {metrics['ms_ssim']:.4f} (1.0 = identical)")

        print("\n3. INTRA-PROMPT DIVERSITY")
        print("-" * 60)
        if "intra_prompt_ms_ssim" in metrics:
            print(f"  Intra-Prompt MS-SSIM:     {metrics['intra_prompt_ms_ssim']:.4f} (lower = more diverse)")
        if "intra_prompt_biovil_similarity" in metrics:
            print(f"  Intra-Prompt BioViL Sim:  {metrics['intra_prompt_biovil_similarity']:.4f} (lower = more diverse)")

        print("\n4. FAIRDIFFUSION METRICS")
        print("-" * 60)
        # Check for FairDiffusion groupwise metrics
        fairdiffusion_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['fid_', 'mifid_', 'is_'])}
        if fairdiffusion_metrics:
            for metric_name, metric_value in sorted(fairdiffusion_metrics.items()):
                print(f"  {metric_name:30s}: {metric_value:.4f}")

        print("\n" + "="*60)
