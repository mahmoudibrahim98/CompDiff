"""
Dataset classes for loading CheXpert classification data.
Supports both real data (WebDataset or CSV) and synthetic data (image directories).
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import webdataset as wds
from torchvision import transforms
import glob
import json
import logging
import pickle

# Import standardized preprocessing functions
from .preprocessing import (
    center_crop_pil,
    tensor_to_pil_image,
    pil_image_to_tensor,
    get_standard_transform,
    load_and_preprocess_image,
)

logger = logging.getLogger(__name__)

# 14 CheXpert classes in order (must match DISEASE_COLUMNS order in 2_create_validation_dataset.py)
CHEXPERT_CLASSES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices'
]

# Index of "No Finding" in CHEXPERT_CLASSES
NO_FINDING_IDX = CHEXPERT_CLASSES.index('No Finding')


def aggregate_labels(labels: List[float]) -> List[float]:
    """
    Aggregate labels according to the paper's approach:
    - Convert all non-positive labels (negative, uncertain, not mentioned) to 0
    - Set 'No Finding' to 1 if none of the disease labels are positive
    
    Args:
        labels: List of 14 label values (1.0 for positive, 0.0 for negative, -1.0 for uncertain/not mentioned)
    
    Returns:
        List of 14 aggregated label values (1.0 for positive, 0.0 for negative)
    """
    if len(labels) != len(CHEXPERT_CLASSES):
        raise ValueError(f"Expected {len(CHEXPERT_CLASSES)} labels, got {len(labels)}")
    
    # Create a copy to avoid modifying the original
    aggregated = [0.0] * len(labels)
    
    # Disease indices (all except "No Finding")
    disease_indices = [i for i in range(len(CHEXPERT_CLASSES)) if i != NO_FINDING_IDX]
    
    # Check if any disease label is positive
    has_positive_disease = False
    for idx in disease_indices:
        label_val = float(labels[idx])  # Convert to float for comparison
        if label_val == 1.0 or label_val == 1:
            # Positive label stays as 1
            aggregated[idx] = 1.0
            has_positive_disease = True
        else:
            # All non-positive (negative, uncertain, not mentioned) become 0
            aggregated[idx] = 0.0
    
    # Set "No Finding" to 1 if none of the disease labels are positive
    if not has_positive_disease:
        aggregated[NO_FINDING_IDX] = 1.0
    else:
        aggregated[NO_FINDING_IDX] = 0.0
    
    return aggregated


class CheXpertClassifierWebDataset(IterableDataset):
    """
    WebDataset for CheXpert classification using streaming approach (like validation).
    Inherits from IterableDataset to stream through data naturally without preloading.
    """
    
    def __init__(
        self,
        url_list: List[str],
        mimic_mean: float = 0.5,
        mimic_std: float = 0.5,
        shuffle: bool = True,
        is_training: bool = False,
        augmentation: bool = True,
    ):
        """
        Initialize WebDataset for streaming.
        
        Args:
            url_list: List of tar file paths
            mimic_mean: Normalization mean
            mimic_std: Normalization std
            shuffle: Whether to shuffle the dataset
            is_training: Whether this is training data (enables augmentation)
            augmentation: Whether to apply data augmentation (only if is_training=True)
        """
        self.url_list = url_list
        self.mimic_mean = mimic_mean
        self.mimic_std = mimic_std
        self.is_training = is_training
        self.augmentation = augmentation and is_training  # Only augment during training
        
        # Create WebDataset pipeline (same approach as RGFineTuningWebDataset)
        if shuffle:
            self.webdataset = wds.DataPipeline(
                wds.SimpleShardList(url_list),
                wds.shuffle(100),
                wds.tarfile_to_samples(),
                wds.shuffle(1000),
            )
        else:
            self.webdataset = wds.DataPipeline(
                wds.SimpleShardList(url_list),
                wds.tarfile_to_samples(),
            )
        
        # Use standardized transform pipeline (ensures identical preprocessing for real and synthetic data)
        self.image_transforms = get_standard_transform(
            mimic_mean=self.mimic_mean,
            mimic_std=self.mimic_std,
            augmentation=self.augmentation,
        )
    
    def __len__(self):
        """
        Get dataset size from _size.txt files.
        
        Note: For IterableDataset with multiple workers, this returns the total dataset size.
        The actual number of samples processed per worker will be approximately size/num_workers
        due to worker distribution in __iter__.
        """
        size_list = [f.split(".tar")[0] + "_size.txt" for f in self.url_list]
        ds_size = 0
        for size_file in size_list:
            try:
                with open(size_file, "r") as file:
                    line = file.readline().strip()
                    ds_size += int(line)
            except:
                pass
        return ds_size
    
    def wds_item_to_sample(self, item):
        """Convert WebDataset item to sample (same approach as validation)."""
        import json
        
        sample = {}
        
        # Load image tensor from WebDataset
        img_tensor = pickle.loads(item["pt_image"])  # [1, H, W] or [H, W] in [0, 1]
        
        # Use standardized preprocessing pipeline
        # This ensures identical conversion and transforms for real and synthetic data
        sample['image'] = load_and_preprocess_image(
            source=img_tensor,
            transform=self.image_transforms,
            is_tensor=True,
        )
        
        # Extract labels and demographics from validation_metadata
        labels = [-1] * len(CHEXPERT_CLASSES)
        age = -1
        sex = -1
        race_ethnicity = -1
        age_group = 'unknown'
        
        if 'validation_metadata' in item:
            try:
                metadata = json.loads(item['validation_metadata'].decode('utf-8'))
                
                # Extract disease labels (14 CheXpert classes)
                # disease_labels is a list/array of 14 values, not a dictionary
                if 'disease_labels' in metadata:
                    disease_labels = metadata['disease_labels']
                    # disease_labels is already a list of 14 values in CHEXPERT_CLASSES order
                    if isinstance(disease_labels, list) and len(disease_labels) == len(CHEXPERT_CLASSES):
                        raw_labels = [float(val) for val in disease_labels]
                        # Aggregate labels: convert non-positive to 0, set No Finding based on disease labels
                        labels = aggregate_labels(raw_labels)
                    else:
                        logger.warning(f"Unexpected disease_labels format: {type(disease_labels)}, length: {len(disease_labels) if isinstance(disease_labels, list) else 'N/A'}")
                
                # Extract demographics
                if 'age' in metadata:
                    age = float(metadata['age'])
                elif 'anchor_age' in metadata:
                    age = float(metadata['anchor_age'])
                
                if 'sex_idx' in metadata:
                    sex = int(metadata['sex_idx'])
                elif 'sex' in metadata:
                    sex_val = metadata['sex']
                    sex = 0 if sex_val == 0 or sex_val == 'M' or sex_val == 'Male' else (1 if sex_val == 1 or sex_val == 'F' or sex_val == 'Female' else -1)
                
                if 'race_idx' in metadata:
                    race_ethnicity = int(metadata['race_idx'])
                elif 'race' in metadata or 'ethnicity' in metadata:
                    race_val = metadata.get('race', metadata.get('ethnicity', ''))
                    if isinstance(race_val, str):
                        race_str = race_val.lower()
                        if 'asian' in race_str:
                            race_ethnicity = 0
                        elif 'black' in race_str:
                            race_ethnicity = 1
                        elif 'hispanic' in race_str or 'latino' in race_str:
                            race_ethnicity = 2
                        elif 'white' in race_str:
                            race_ethnicity = 3
                
                if 'age_group' in metadata:
                    age_group = str(metadata['age_group'])
                elif 'age_bin' in metadata:
                    age_group = str(metadata['age_bin'])
            except Exception as e:
                logger.warning(f"Failed to parse validation_metadata: {e}")
        
        # Convert labels to tensor
        sample['labels'] = torch.tensor(labels, dtype=torch.float32)
        sample['age'] = torch.tensor(age, dtype=torch.float32)
        sample['sex'] = torch.tensor(sex, dtype=torch.long)
        sample['race_ethnicity'] = torch.tensor(race_ethnicity, dtype=torch.long)
        sample['age_group'] = age_group
        sample['index'] = torch.tensor(-1, dtype=torch.long)  # Not applicable for streaming
        sample['data_source'] = 'real'  # WebDataset is always real data
        
        return sample
    
    def __iter__(self):
        """Iterate through WebDataset (same approach as RGFineTuningWebDataset)."""
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        
        self.source = iter(self.webdataset)
        for i, item in enumerate(self.source):
            # Distribute samples across workers
            if i % num_workers == worker_id:
                try:
                    yield self.wds_item_to_sample(item)
                except Exception as e:
                    logger.warning(f"Failed to process WebDataset item: {e}")
                    continue


class CheXpertClassifierDataset(Dataset):
    """
    Dataset for CheXpert multi-label classification.
    
    Supports loading from:
    - Real data: WebDataset tar files or CSV with image paths
    - Synthetic data: Image directories with metadata
    - CheXpert dataset: CSV with relative paths and base image directory
    
    Args:
        data_path: Path to data (CSV file, WebDataset directory, or image directory)
        data_type: 'real_csv', 'real_wds', 'synthetic', or 'chexpert_csv'
        split: 'train', 'val', or 'test' (for real data)
        transform: Optional transform to apply to images
        mimic_mean: Mean for MIMIC-CXR normalization (default: computed if None)
        mimic_std: Std for MIMIC-CXR normalization (default: computed if None)
        is_training: Whether this is training data (enables augmentation)
        augmentation: Whether to apply data augmentation (only if is_training=True)
        image_base_path: Base path for images (required for 'chexpert_csv' type)
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        data_type: str = 'real_csv',
        split: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        mimic_mean: Optional[float] = None,
        mimic_std: Optional[float] = None,
        is_training: bool = False,
        augmentation: bool = True,
        image_base_path: Optional[Union[str, Path]] = None,
    ):
        self.data_path = Path(data_path)
        self.data_type = data_type
        self.split = split
        self.is_training = is_training
        self.augmentation = augmentation and is_training  # Only augment during training
        self.image_base_path = Path(image_base_path) if image_base_path else None
        
        # Default MIMIC-CXR normalization (will be computed if not provided)
        # These are placeholder values - should be computed from MIMIC-CXR train set
        self.mimic_mean = mimic_mean if mimic_mean is not None else 0.5
        self.mimic_std = mimic_std if mimic_std is not None else 0.5
        
        # Setup transform
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # Load data based on type
        if data_type == 'real_csv' or data_type == 'chexpert_csv':
            self._load_from_csv()
        elif data_type == 'real_wds':
            self._load_from_wds()
        elif data_type == 'synthetic':
            self._load_from_synthetic()
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def _get_default_transform(self):
        """Get default preprocessing with optional augmentation for training."""
        # Use standardized transform pipeline (ensures identical preprocessing for real and synthetic data)
        return get_standard_transform(
            mimic_mean=self.mimic_mean,
            mimic_std=self.mimic_std,
            augmentation=self.augmentation,
        )
    
    def _load_from_csv(self):
        """Load data from CSV file with image paths and labels."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Filter by split if specified (but not for CheXpert evaluation - use all data)
        is_chexpert_format = self.data_type == 'chexpert_csv' or (
            'Path' in df.columns and self.image_base_path is not None
        )
        
        # For CheXpert format, don't filter by split - use all data
        if not is_chexpert_format and self.split and 'split' in df.columns:
            df = df[df['split'] == self.split]
        
        self.df = df
        self.samples = []
        
        # Extract image paths and labels
        for idx, row in df.iterrows():
            # Handle image path
            if is_chexpert_format:
                # CheXpert format: 'Path' column contains relative paths
                relative_path = row.get('Path', None)
                if pd.isna(relative_path):
                    continue
                # Construct full path
                image_path = self.image_base_path / relative_path
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}, skipping")
                    continue
            else:
                # Standard format: 'image' column contains absolute or relative paths
                image_path = row.get('image', None)
                if pd.isna(image_path):
                    # Try constructing path from other columns
                    if 'dicom_id' in row and 'study_id' in row:
                        # Construct path similar to dataset structure
                        continue  # Skip if can't find image
                    else:
                        continue
                image_path = Path(image_path)
                if not image_path.exists():
                    # Try relative to CSV directory
                    image_path = self.data_path.parent / image_path
                    if not image_path.exists():
                        continue  # Skip if can't find image
            
            # Extract 14 CheXpert labels
            raw_labels = []
            for class_name in CHEXPERT_CLASSES:
                label_val = row.get(class_name, -1)
                if pd.isna(label_val):
                    label_val = -1
                else:
                    try:
                        label_val = float(label_val)
                        if label_val not in [0, 1, -1]:
                            label_val = -1
                    except (ValueError, TypeError):
                        label_val = -1
                raw_labels.append(label_val)
            
            # Aggregate labels: convert non-positive to 0, set No Finding based on disease labels
            labels = aggregate_labels(raw_labels)
            
            # Extract demographics
            if is_chexpert_format:
                # CheXpert format: use Age, Sex, PRIMARY_RACE, ETHNICITY columns
                age = row.get('AGE_AT_CXR', None)
                if pd.isna(age):
                    age = row.get('Age', None)
                if pd.isna(age):
                    age = -1
                else:
                    age = float(age)
                
                sex_str = str(row.get('GENDER', row.get('Sex', ''))).lower()
                # Check 'female' first since 'male' is a substring of 'female'
                if 'female' in sex_str or sex_str == 'f':
                    sex = 1
                elif 'male' in sex_str or sex_str == 'm':
                    sex = 0
                else:
                    sex = -1
                
                # Map race/ethnicity from CheXpert format using only demo_group
                # Correct mapping: WHITE=0, BLACK=1, ASIAN=2, HISPANIC=3 (matches old evaluation)
                demo_group = str(row.get('demo_group', '')).lower()
                
                race_ethnicity = -1
                # Use demo_group only (most reliable source)
                if 'white' in demo_group or 'whites' in demo_group:
                    race_ethnicity = 0
                elif 'black' in demo_group:
                    race_ethnicity = 1
                elif 'asian' in demo_group:
                    race_ethnicity = 2
                elif 'hispanic' in demo_group or 'latino' in demo_group:
                    race_ethnicity = 3
                
                # Compute age_bin from age using standard bins [18, 40, 60, 80]
                # bins: <18=0, 18-40=1, 40-60=2, 60-80=3, 80+=4
                # Evaluation expects: '1'='18-40', '2'='40-60', '3'='60-80', '4'='80+'
                if age < 0:
                    age_group = 'unknown'
                else:
                    age_bins = [18, 40, 60, 80]
                    age_bin_idx = len(age_bins)  # Default to last bin (80+)
                    for i, threshold in enumerate(age_bins):
                        if age < threshold:
                            age_bin_idx = i
                            break
                    
                    # Convert to string format expected by evaluation: '1', '2', '3', '4'
                    # Skip bin 0 (<18) as it shouldn't exist in test data
                    if age_bin_idx == 0:
                        age_group = 'unknown'  # <18 shouldn't exist
                    elif age_bin_idx == 1:
                        age_group = '1'  # 18-40
                    elif age_bin_idx == 2:
                        age_group = '2'  # 40-60
                    elif age_bin_idx == 3:
                        age_group = '3'  # 60-80
                    else:  # age_bin_idx == 4
                        age_group = '4'  # 80+
            else:
                # Standard format
                age = row.get('anchor_age', None)
                if pd.isna(age):
                    age = -1
                else:
                    age = float(age)
                
                sex_str = str(row.get('gender', '')).lower()
                # Check 'female' first since 'male' is a substring of 'female'
                if 'female' in sex_str or sex_str == 'f':
                    sex = 1
                elif 'male' in sex_str or sex_str == 'm':
                    sex = 0
                else:
                    sex = -1
                
                ethnicity_str = str(row.get('ethnicity', '')).lower()
                race_ethnicity = -1
                if 'asian' in ethnicity_str:
                    race_ethnicity = 0
                elif 'black' in ethnicity_str:
                    race_ethnicity = 1
                elif 'hispanic' in ethnicity_str or 'latino' in ethnicity_str:
                    race_ethnicity = 2
                elif 'white' in ethnicity_str:
                    race_ethnicity = 3
                
                age_group = row.get('age_group', None)
                if pd.isna(age_group):
                    age_group = 'unknown'
            
            self.samples.append({
                'image_path': str(image_path),
                'labels': labels,
                'age': age,
                'sex': sex,
                'race_ethnicity': race_ethnicity,
                'age_group': age_group,
                'index': idx,
            })
    
    def _load_from_wds(self):
        """Load data from WebDataset tar files."""
        import pickle
        
        # Find tar files
        if self.data_path.is_dir():
            tar_files = sorted(glob.glob(str(self.data_path / "*.tar")))
            if not tar_files:
                tar_files = sorted(glob.glob(str(self.data_path / "*.tar.gz")))
        else:
            tar_files = [str(self.data_path)]
        
        if not tar_files:
            raise FileNotFoundError(f"No tar files found in: {self.data_path}")
        
        self._tar_files = tar_files
        
        # Create WebDataset pipeline for iteration
        self._wds_pipeline = wds.DataPipeline(
            wds.SimpleShardList(tar_files),
            wds.tarfile_to_samples(),
        )
        
        # Store metadata only (not images) - will load images on-demand
        self.samples = []
        self._samples_loaded = False
        self._wds_item_cache = {}  # Cache for loaded items (limited size)
    
    def _load_from_synthetic(self):
        """Load synthetic data from image directory with metadata."""
        if not self.data_path.is_dir():
            raise ValueError(f"Synthetic data path must be a directory: {self.data_path}")
        
        # Look for images in subdirectories (gpu_0, gpu_1, etc.) or directly
        image_dirs = []
        if any(Path(self.data_path / d).is_dir() and d.startswith('gpu_') for d in os.listdir(self.data_path)):
            # Has gpu_* subdirectories
            image_dirs = sorted([self.data_path / d for d in os.listdir(self.data_path) 
                               if Path(self.data_path / d).is_dir() and d.startswith('gpu_')])
        else:
            # Images directly in directory
            image_dirs = [self.data_path]
        
        self.samples = []
        
        for img_dir in image_dirs:
            # Check if labels.pkl exists in this directory
            labels_pkl_path = img_dir / 'labels.pkl'
            labels_dict = None
            if labels_pkl_path.exists():
                try:
                    with open(labels_pkl_path, 'rb') as f:
                        labels_dict = pickle.load(f)
                    logger.info(f"Loaded labels from {labels_pkl_path}")
                except Exception as e:
                    logger.warning(f"Failed to load labels.pkl from {labels_pkl_path}: {e}")
            
            # Find all images
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(glob.glob(str(img_dir / ext)))
            
            for img_path in sorted(image_files):
                # Try to find corresponding metadata
                img_stem = Path(img_path).stem
                # Handle synthetic image naming: synthetic_000001.png -> metadata_000001.json
                # Strip 'synthetic_' or 'real_' prefix if present
                numeric_part = None
                if img_stem.startswith('synthetic_'):
                    numeric_part = img_stem.replace('synthetic_', '')
                    metadata_path = img_dir / f"metadata_{numeric_part}.json"
                elif img_stem.startswith('real_'):
                    numeric_part = img_stem.replace('real_', '')
                    metadata_path = img_dir / f"metadata_{numeric_part}.json"
                else:
                    # Fallback: try both with and without prefix
                    metadata_path = img_dir / f"metadata_{img_stem}.json"
                    # Try to extract numeric part from filename
                    try:
                        numeric_part = img_stem
                    except:
                        pass
                
                labels = [-1] * 14
                age = -1
                sex = -1
                race_ethnicity = -1
                age_group = 'unknown'
                
                # First, try to load from labels.pkl if available
                if labels_dict is not None and numeric_part is not None:
                    try:
                        # Extract index from numeric part (e.g., "000000" -> 0)
                        # Handle both zero-padded (e.g., "000000") and non-padded (e.g., "0") formats
                        img_index = int(numeric_part)
                        
                        # Get disease labels from pickle file
                        if 'disease' in labels_dict and img_index < len(labels_dict['disease']):
                            disease_tensor = labels_dict['disease'][img_index]
                            # Convert tensor to list of floats
                            if hasattr(disease_tensor, 'cpu'):
                                disease_tensor = disease_tensor.cpu()
                            if hasattr(disease_tensor, 'numpy'):
                                disease_tensor = disease_tensor.numpy()
                            if hasattr(disease_tensor, 'tolist'):
                                raw_labels = disease_tensor.tolist()
                            else:
                                raw_labels = list(disease_tensor)
                            
                            # Ensure we have 14 labels
                            if len(raw_labels) == len(CHEXPERT_CLASSES):
                                raw_labels = [float(val) for val in raw_labels]
                                # Aggregate labels: convert non-positive to 0, set No Finding based on disease labels
                                labels = aggregate_labels(raw_labels)
                        
                        # Get demographics from pickle file
                        if 'age' in labels_dict and img_index < len(labels_dict['age']):
                            age_val = labels_dict['age'][img_index]
                            if hasattr(age_val, 'cpu'):
                                age_val = age_val.cpu()
                            if hasattr(age_val, 'item'):
                                age = float(age_val.item())
                            elif hasattr(age_val, 'numpy'):
                                age = float(age_val.numpy())
                            else:
                                age = float(age_val)
                        
                        if 'sex' in labels_dict and img_index < len(labels_dict['sex']):
                            sex_val = labels_dict['sex'][img_index]
                            if hasattr(sex_val, 'cpu'):
                                sex_val = sex_val.cpu()
                            if hasattr(sex_val, 'item'):
                                sex_val = sex_val.item()
                            elif hasattr(sex_val, 'numpy'):
                                sex_val = sex_val.numpy()
                            sex = 0 if sex_val == 0 or sex_val == 'M' or sex_val == 'Male' else (1 if sex_val == 1 or sex_val == 'F' or sex_val == 'Female' else -1)
                        
                        if 'race' in labels_dict and img_index < len(labels_dict['race']):
                            race_val = labels_dict['race'][img_index]
                            if hasattr(race_val, 'cpu'):
                                race_val = race_val.cpu()
                            if hasattr(race_val, 'item'):
                                race_val = race_val.item()
                            elif hasattr(race_val, 'numpy'):
                                race_val = race_val.numpy()
                            
                            if isinstance(race_val, str):
                                race_str = race_val.lower()
                                if 'asian' in race_str:
                                    race_ethnicity = 0
                                elif 'black' in race_str:
                                    race_ethnicity = 1
                                elif 'hispanic' in race_str or 'latino' in race_str:
                                    race_ethnicity = 2
                                elif 'white' in race_str:
                                    race_ethnicity = 3
                            elif isinstance(race_val, (int, float)):
                                # Assume numeric encoding: 0=Asian, 1=Black, 2=Hispanic, 3=White
                                race_ethnicity = int(race_val) if 0 <= race_val <= 3 else -1
                        
                        # Age group can be computed from age if not directly available
                        if age >= 0:
                            if age < 18:
                                age_group = 'unknown'  # <18 shouldn't exist
                            elif age < 40:
                                age_group = '1'  # 18-40
                            elif age < 60:
                                age_group = '2'  # 40-60
                            elif age < 80:
                                age_group = '3'  # 60-80
                            else:
                                age_group = '4'  # 80+
                        
                    except (ValueError, IndexError, KeyError) as e:
                        logger.debug(f"Failed to load from labels.pkl for image {img_path} (index {numeric_part}): {e}")
                        # Fall through to try metadata.json
                
                # Fallback to metadata.json if labels.pkl didn't work or doesn't exist
                if metadata_path.exists() and all(l == -1 for l in labels):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # #region agent log
                        import json as json_lib
                        log_data = {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H1",
                            "location": "dataset.py:433",
                            "message": "Synthetic metadata keys check",
                            "data": {
                                "metadata_keys": list(metadata.keys()),
                                "has_disease": "disease" in metadata,
                                "has_disease_labels": "disease_labels" in metadata,
                                "img_stem": str(img_stem)
                            },
                            "timestamp": int(__import__("time").time() * 1000)
                        }
                        with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                            log_file.write(json_lib.dumps(log_data) + "\n")
                        # #endregion
                        
                        # Extract labels from metadata
                        # Try both 'disease_labels' (list format) and 'disease' (dict format)
                        if 'disease_labels' in metadata:
                            # #region agent log
                            log_data = {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H2",
                                "location": "dataset.py:450",
                                "message": "Found disease_labels in metadata",
                                "data": {
                                    "disease_labels_type": str(type(metadata['disease_labels'])),
                                    "disease_labels_value": metadata['disease_labels'] if isinstance(metadata['disease_labels'], list) and len(metadata['disease_labels']) <= 20 else str(metadata['disease_labels'])[:100],
                                    "disease_labels_len": len(metadata['disease_labels']) if isinstance(metadata['disease_labels'], (list, dict)) else "N/A"
                                },
                                "timestamp": int(__import__("time").time() * 1000)
                            }
                            with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                                log_file.write(json_lib.dumps(log_data) + "\n")
                            # #endregion
                            
                            disease_labels = metadata['disease_labels']
                            # Handle list format (same as WebDataset)
                            if isinstance(disease_labels, list) and len(disease_labels) == len(CHEXPERT_CLASSES):
                                raw_labels = [float(val) for val in disease_labels]
                                # Aggregate labels: convert non-positive to 0, set No Finding based on disease labels
                                labels = aggregate_labels(raw_labels)
                                # #region agent log - H6: Check for invalid label values
                                invalid_labels = [i for i, v in enumerate(labels) if v not in [-1.0, 0.0, 1.0]]
                                sample_idx = len(self.samples)
                                # Log first 100 samples, every 1000th sample, or if invalid labels found
                                if invalid_labels or sample_idx < 100 or sample_idx % 1000 == 0:
                                    log_data = {
                                        "sessionId": "debug-session",
                                        "runId": "H6-synthetic-labels",
                                        "hypothesisId": "H6",
                                        "location": "dataset.py:489",
                                        "message": "Synthetic label values check",
                                        "data": {
                                            "sample_idx": sample_idx,
                                            "img_stem": str(img_stem),
                                            "labels": labels,
                                            "has_invalid": len(invalid_labels) > 0,
                                            "invalid_indices": invalid_labels,
                                            "num_positive": sum(1 for l in labels if l == 1.0),
                                            "num_negative": sum(1 for l in labels if l == 0.0),
                                            "num_uncertain": sum(1 for l in labels if l == -1.0),
                                            "unique_values": sorted(set(labels))
                                        },
                                        "timestamp": int(__import__("time").time() * 1000)
                                    }
                                    with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                                        log_file.write(json_lib.dumps(log_data) + "\n")
                                # #endregion
                            # Handle dict format (legacy)
                            elif isinstance(disease_labels, dict):
                                for i, class_name in enumerate(CHEXPERT_CLASSES):
                                    if class_name in disease_labels:
                                        val = disease_labels[class_name]
                                        labels[i] = 1 if val == 1 else (0 if val == 0 else -1)
                        elif 'disease' in metadata:
                            # #region agent log
                            log_data = {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H1",
                                "location": "dataset.py:470",
                                "message": "Found disease (dict) in metadata, not disease_labels",
                                "data": {
                                    "disease_type": str(type(metadata['disease'])),
                                    "disease_keys": list(metadata['disease'].keys()) if isinstance(metadata['disease'], dict) else "N/A"
                                },
                                "timestamp": int(__import__("time").time() * 1000)
                            }
                            with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                                log_file.write(json_lib.dumps(log_data) + "\n")
                            # #endregion
                            
                            disease_labels = metadata['disease']
                            for i, class_name in enumerate(CHEXPERT_CLASSES):
                                if class_name in disease_labels:
                                    val = disease_labels[class_name]
                                    labels[i] = 1.0 if val == 1 else (0.0 if val == 0 else -1.0)
                                else:
                                    labels[i] = -1.0
                            
                            # Aggregate labels: convert non-positive to 0, set No Finding based on disease labels
                            labels = aggregate_labels(labels)
                        
                        # Extract demographics
                        if 'age' in metadata or 'age_continuous' in metadata:
                            age = float(metadata.get('age', metadata.get('age_continuous', -1)))
                        # Handle both 'sex' and 'sex_idx' fields
                        if 'sex_idx' in metadata:
                            sex = int(metadata['sex_idx'])
                        elif 'sex' in metadata:
                            sex_val = metadata['sex']
                            sex = 0 if sex_val == 0 or sex_val == 'M' or sex_val == 'Male' else (1 if sex_val == 1 or sex_val == 'F' or sex_val == 'Female' else -1)
                        # Handle both 'race_idx' and 'race'/'ethnicity' fields
                        if 'race_idx' in metadata:
                            race_ethnicity = int(metadata['race_idx'])
                        elif 'race' in metadata or 'ethnicity' in metadata:
                            race_val = metadata.get('race', metadata.get('ethnicity', ''))
                            if isinstance(race_val, str):
                                race_str = race_val.lower()
                                if 'asian' in race_str:
                                    race_ethnicity = 0
                                elif 'black' in race_str:
                                    race_ethnicity = 1
                                elif 'hispanic' in race_str or 'latino' in race_str:
                                    race_ethnicity = 2
                                elif 'white' in race_str:
                                    race_ethnicity = 3
                        if 'age_group' in metadata:
                            age_group = str(metadata['age_group'])
                        # Compute age_group from age if not in metadata (use numeric format for consistency)
                        elif age >= 0:
                            if age < 18:
                                age_group = 'unknown'  # <18 shouldn't exist
                            elif age < 40:
                                age_group = '1'  # 18-40
                            elif age < 60:
                                age_group = '2'  # 40-60
                            elif age < 80:
                                age_group = '3'  # 60-80
                            else:
                                age_group = '4'  # 80+
                    except Exception as e:
                        print(f"Warning: Failed to load metadata from {metadata_path}: {e}")
                        # #region agent log
                        import json as json_lib
                        log_data = {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H5",
                            "location": "dataset.py:490",
                            "message": "Failed to load metadata",
                            "data": {
                                "error": str(e),
                                "metadata_path": str(metadata_path),
                                "all_labels_uncertain": all(l == -1 for l in labels)
                            },
                            "timestamp": int(__import__("time").time() * 1000)
                        }
                        with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                            log_file.write(json_lib.dumps(log_data) + "\n")
                        # #endregion
                
                # #region agent log
                import json as json_lib
                log_data = {
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H3",
                    "location": "dataset.py:500",
                    "message": "Final labels after parsing",
                    "data": {
                        "labels": labels,
                        "num_uncertain": sum(1 for l in labels if l == -1),
                        "num_positive": sum(1 for l in labels if l == 1),
                        "num_negative": sum(1 for l in labels if l == 0),
                        "all_uncertain": all(l == -1 for l in labels),
                        "has_metadata": metadata_path.exists()
                    },
                    "timestamp": int(__import__("time").time() * 1000)
                }
                with open("/home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2/.cursor/debug.log", "a") as log_file:
                    log_file.write(json_lib.dumps(log_data) + "\n")
                # #endregion
                
                self.samples.append({
                    'image_path': img_path,
                    'labels': labels,
                    'age': age,
                    'sex': sex,
                    'race_ethnicity': race_ethnicity,
                    'age_group': age_group,
                    'index': len(self.samples),
                })
    
    def __len__(self) -> int:
        if self.data_type == 'real_wds':
            if not self._samples_loaded:
                # Try to get size from _size.txt files (faster than iterating)
                size_files = sorted(glob.glob(str(self.data_path / "*_size.txt")))
                if size_files:
                    total_size = 0
                    for size_file in size_files:
                        try:
                            with open(size_file, 'r') as f:
                                line = f.readline().strip()
                                total_size += int(line)
                        except:
                            pass
                    if total_size > 0:
                        return total_size
                # Fallback: need to load samples to count
                if not self._samples_loaded:
                    self._load_wds_samples()
            return len(self.samples)
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        if self.data_type == 'real_wds':
            # Handle WebDataset differently - it's iterable
            if not self._samples_loaded:
                self._load_wds_samples()
        
        sample = self.samples[idx]
        
        # For WebDataset, load image on-demand
        if self.data_type == 'real_wds':
            import pickle
            # Try to load from cache first
            item_idx = sample['item_index']
            if item_idx in self._wds_item_cache:
                pt_image_bytes = self._wds_item_cache[item_idx]
                img_tensor = pickle.loads(pt_image_bytes)  # [1, H, W], [C, H, W], or [H, W] in [0, 1]
            else:
                # Cache miss - need to iterate to this position (slow but memory efficient)
                # This is a fallback - ideally we'd have better indexing
                logger.warning(f"Cache miss for index {idx}, iterating to position (this is slow)")
                # Recreate pipeline and iterate to position
                img_tensor = None
                temp_pipeline = wds.DataPipeline(
                    wds.SimpleShardList(self._tar_files),
                    wds.tarfile_to_samples(),
                )
                for i, item in enumerate(temp_pipeline):
                    if i == item_idx:
                        if 'pt_image' in item:
                            img_tensor = pickle.loads(item['pt_image'])
                            break
                
                # Fallback if not found
                if img_tensor is None:
                    logger.warning(f"Could not load image for index {idx}, using fallback")
                    img_tensor = torch.zeros(1, 224, 224)
            
            # Ensure tensor is [1, H, W] format (grayscale) - model expects 1-channel input
            if img_tensor.dim() == 2:
                # [H, W] -> [1, H, W]
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.dim() == 3:
                C, H, W = img_tensor.shape
                if C == 3:
                    # [3, H, W] -> convert to grayscale -> [1, H, W]
                    img_tensor = img_tensor.mean(dim=0, keepdim=True)
                elif C == 1:
                    # Already 1 channel, keep as is
                    pass
                else:
                    logger.warning(f"Unexpected channels: {img_tensor.shape}, using fallback")
                    img_tensor = torch.zeros(1, 224, 224)
            else:
                logger.warning(f"Invalid tensor dimensions: {img_tensor.shape}, using fallback")
                img_tensor = torch.zeros(1, 224, 224)
            
            # Final validation: Ensure tensor is [1, H, W] format
            C, H, W = img_tensor.shape
            if C != 1 or H < 10 or W < 10:
                logger.warning(f"Invalid tensor after conversion: {img_tensor.shape}, using fallback")
                img_tensor = torch.zeros(1, 224, 224)
                C, H, W = img_tensor.shape
            
            # Use standardized preprocessing pipeline
            # This ensures identical conversion and transforms for real and synthetic data
            try:
                img = load_and_preprocess_image(
                    source=img_tensor,
                    transform=self.transform,
                    is_tensor=True,
                )
            except Exception as e:
                logger.warning(f"Failed to preprocess image tensor: {e}, using fallback")
                # Return a black image as fallback (normalized tensor)
                fallback_img = Image.new('L', (224, 224), color=0)
                img = self.transform(fallback_img) if self.transform else torch.zeros(1, 224, 224)
        else:
            # Load image from path (for CSV or synthetic data)
            img_path = sample['image_path']
            if isinstance(img_path, str):
                img_path = Path(img_path)
            
            # Use standardized preprocessing pipeline
            # This ensures identical conversion and transforms for real and synthetic data
            try:
                img = load_and_preprocess_image(
                    source=img_path,
                    transform=self.transform,
                    is_tensor=False,
                )
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}, using fallback")
                # Return a black image as fallback (normalized tensor)
                fallback_img = Image.new('L', (224, 224), color=0)
                img = self.transform(fallback_img) if self.transform else torch.zeros(1, 224, 224)
        
        # At this point, img should be a tensor (from load_and_preprocess_image or transform)
        # Ensure it's a tensor for consistency
        if not isinstance(img, torch.Tensor):
            # Fallback: if somehow we still have a PIL Image, convert it
            logger.warning("Image is not a tensor after preprocessing, converting...")
            if isinstance(img, Image.Image):
                img = self.transform(img) if self.transform else pil_image_to_tensor(img)
            else:
                img = torch.zeros(1, 224, 224)
        
        # Convert labels to tensor
        labels = torch.tensor(sample['labels'], dtype=torch.float32)
        
        return {
            'image': img,
            'labels': labels,
            'age': torch.tensor(sample['age'], dtype=torch.float32),
            'sex': torch.tensor(sample['sex'], dtype=torch.long),
            'race_ethnicity': torch.tensor(sample['race_ethnicity'], dtype=torch.long),
            'age_group': sample['age_group'],
            'index': torch.tensor(sample['index'], dtype=torch.long),
            'data_source': 'synthetic' if self.data_type == 'synthetic' else 'real',
        }
    
    def _load_wds_samples(self):
        """Load metadata from WebDataset (memory efficient - doesn't store image tensors)."""
        import pickle
        import json
        from tqdm import tqdm
        
        logger.info(f"Loading metadata from WebDataset: {self.data_path}")
        self.samples = []
        
        # Store raw items for on-demand image loading
        self._wds_items = []
        
        # Iterate through WebDataset to collect metadata only
        for item in tqdm(self._wds_pipeline, desc="Loading WebDataset metadata"):
            try:
                # Load validation metadata (contains labels and demographics)
                labels = [-1] * 14
                age = -1
                sex = -1
                race_ethnicity = -1
                age_group = 'unknown'
                
                if 'validation_metadata' in item:
                    try:
                        metadata = json.loads(item['validation_metadata'].decode('utf-8'))
                        
                        # Extract disease labels (14 CheXpert classes)
                        # disease_labels is a list/array of 14 values, not a dictionary
                        if 'disease_labels' in metadata:
                            disease_labels = metadata['disease_labels']
                            # disease_labels is already a list of 14 values in CHEXPERT_CLASSES order
                            if isinstance(disease_labels, list) and len(disease_labels) == len(CHEXPERT_CLASSES):
                                raw_labels = [float(val) for val in disease_labels]
                                # Aggregate labels: convert non-positive to 0, set No Finding based on disease labels
                                labels = aggregate_labels(raw_labels)
                            else:
                                logger.warning(f"Unexpected disease_labels format in _load_wds_samples: {type(disease_labels)}, length: {len(disease_labels) if isinstance(disease_labels, list) else 'N/A'}")
                        
                        # Extract demographics
                        if 'age' in metadata:
                            age = float(metadata['age'])
                        elif 'anchor_age' in metadata:
                            age = float(metadata['anchor_age'])
                        
                        if 'sex_idx' in metadata:
                            sex = int(metadata['sex_idx'])
                        elif 'sex' in metadata:
                            sex_val = metadata['sex']
                            sex = 0 if sex_val == 0 or sex_val == 'M' or sex_val == 'Male' else (1 if sex_val == 1 or sex_val == 'F' or sex_val == 'Female' else -1)
                        
                        if 'race_idx' in metadata:
                            race_ethnicity = int(metadata['race_idx'])
                        elif 'race' in metadata or 'ethnicity' in metadata:
                            race_val = metadata.get('race', metadata.get('ethnicity', ''))
                            if isinstance(race_val, str):
                                race_str = race_val.lower()
                                if 'asian' in race_str:
                                    race_ethnicity = 0
                                elif 'black' in race_str:
                                    race_ethnicity = 1
                                elif 'hispanic' in race_str or 'latino' in race_str:
                                    race_ethnicity = 2
                                elif 'white' in race_str:
                                    race_ethnicity = 3
                        
                        if 'age_group' in metadata:
                            age_group = str(metadata['age_group'])
                        elif 'age_bin' in metadata:
                            age_group = str(metadata['age_bin'])
                    except Exception as e:
                        logger.warning(f"Failed to parse validation_metadata: {e}")
                
                # Store only metadata - images will be loaded on-demand
                # Store item in limited cache (LRU-style, keep most recent 1000 items to balance memory and performance)
                item_idx = len(self.samples)
                if 'pt_image' in item:
                    # Use LRU-style caching: keep most recent 1000 items
                    if len(self._wds_item_cache) >= 1000:
                        # Remove oldest item (FIFO)
                        oldest_key = next(iter(self._wds_item_cache))
                        del self._wds_item_cache[oldest_key]
                    self._wds_item_cache[item_idx] = item['pt_image']
                
                self.samples.append({
                    'item_index': item_idx,
                    'labels': labels,
                    'age': age,
                    'sex': sex,
                    'race_ethnicity': race_ethnicity,
                    'age_group': age_group,
                    'index': item_idx,
                })
            except Exception as e:
                logger.warning(f"Failed to load sample from WebDataset: {e}")
                continue
        
        self._samples_loaded = True
        logger.info(f"Loaded metadata for {len(self.samples)} samples from WebDataset")

