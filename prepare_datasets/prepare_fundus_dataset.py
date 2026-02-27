"""
Create WebDataset for validation with all necessary metadata for Fundus data.

This script creates validation tar files that include:
1. Image tensors (.pt_image)
2. Prompt text (.prompt_metadata)  
3. Validation metadata (.validation_metadata)
   - Condition labels (glaucoma, cdr_status, rnflt_status, near_vision_refraction_stage)
   - Demographics (age, gender, race, ethnicity)
   - Any other metadata needed for validation metrics

The format is compatible with the validation system in train_loop.py.
"""

import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
import tarfile
import io
import json
from pathlib import Path
import torch
import pandas as pd

# ============================================================================
# CONFIGURATION (overridden by CLI arguments; defaults are arbitrary paths)
# ============================================================================

# Maximum samples per tar file (can be overridden by CLI)
DEFAULT_MAX_SAMPLES_PER_TAR = 1000

# Condition labels to extract for Fundus data
CONDITION_COLUMNS = [
    'glaucoma',
    'cdr_status',
    'rnflt_status',
    'near_vision_refraction_stage',
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_condition_labels(row, condition_columns):
    """
    Extract condition labels from dataframe row.
    
    Returns:
        dict: Condition labels with their values
    """
    labels = {}
    for condition in condition_columns:
        if condition in row:
            val = row[condition]
            if pd.isna(val):
                labels[condition] = None
            else:
                labels[condition] = str(val)
        else:
            labels[condition] = None
    return labels


def extract_demographics(row):
    """
    Extract demographic information from dataframe row.
    
    Returns:
        dict with keys: age, sex, race, ethnicity, sex_idx, race_idx, age_bin
    """
    demographics = {}
    
    # Age
    if 'age' in row:
        demographics['age'] = float(row['age']) if not pd.isna(row['age']) else -1.0
    else:
        demographics['age'] = -1.0
    
    # Age bin (for HCN/DemographicEncoder)
    age = demographics['age']
    if age < 0:
        demographics['age_bin'] = -1
    else:
        # Default age bins - can be overridden by importing from dataset_wds
        # or by setting AGE_BINS environment variable as comma-separated values
        age_bins_str = os.environ.get('AGE_BINS', None)
        if age_bins_str:
            bins = [int(x.strip()) for x in age_bins_str.split(',')]
        else:
            # Default: [18, 40, 60, 80] for 5 bins
            bins = [18, 40, 60, 80]
        
        # Find the bin index
        bin_idx = len(bins)  # Default to last bin
        for i, threshold in enumerate(bins):
            if age < threshold:
                bin_idx = i
                break
        demographics['age_bin'] = bin_idx
    
    # Gender/Sex
    if 'gender' in row:
        gender = str(row['gender']).lower()
        if gender in ['m', 'male']:
            demographics['sex'] = 'M'
            demographics['sex_idx'] = 0
        elif gender in ['f', 'female']:
            demographics['sex'] = 'F'
            demographics['sex_idx'] = 1
        else:
            demographics['sex'] = 'UNKNOWN'
            demographics['sex_idx'] = -1
    else:
        demographics['sex'] = 'UNKNOWN'
        demographics['sex_idx'] = -1
    
    # Race
    if 'race' in row:
        race = str(row['race']).lower()
        # Map to indices
        race_map = {
            'white': 0,
            'black': 1,
            'asian': 2,
            'other': 3,
            'unknown': -1
        }
        demographics['race'] = race
        demographics['race_idx'] = race_map.get(race, -1)
    else:
        demographics['race'] = 'UNKNOWN'
        demographics['race_idx'] = -1
    
    # Ethnicity
    if 'ethnicity' in row:
        ethnicity = str(row['ethnicity']).lower()
        # Map to indices
        ethnicity_map = {
            'non-hispanic': 0,
            'hispanic': 1,
            'unknown': -1
        }
        demographics['ethnicity'] = ethnicity
        demographics['ethnicity_idx'] = ethnicity_map.get(ethnicity, -1)
    else:
        demographics['ethnicity'] = 'UNKNOWN'
        demographics['ethnicity_idx'] = -1
    
    return demographics


def generate_diffusion_prompt_from_row(row):
    """
    Generate descriptive diffusion prompts modeled after the FairGenMed style.
    
    Args:
        row: DataFrame row with patient data
        
    Returns:
        str: Formatted prompt string
    """
    # Mapping for race and gender (sex) to match FairGenMed conventions
    race_map = {
        'asian': 'Asian',
        'black': 'Black',
        'white': 'White'
    }
    gender_map = {
        'female': 'Female',
        'male': 'Male'
    }

    # Get demographic attributes in prompt format (handled robustly for missing/bad values)
    race_str = race_map.get(str(row['race']).strip().lower(), str(row['race']).capitalize())
    gender_str = gender_map.get(str(row['gender']).strip().lower(), str(row['gender']).capitalize())
    
    # Format age as "X year old" or "X years old"
    age = row.get('age', None)
    if age is not None and not pd.isna(age):
        try:
            age_int = int(float(age))
            if age_int == 1:
                age_str = "1 year old"
            else:
                age_str = f"{age_int} years old"
        except (ValueError, TypeError):
            age_str = "unknown age"
    else:
        age_str = "unknown age"

    # Glaucoma yes/no (v1 uses yes/no, v2 can use glaucoma/non-glaucoma)
    glaucoma_val = str(row.get('glaucoma', '')).strip().lower()
    if glaucoma_val in ['1', 'yes', 'y', 'true']:
        y_glau_label = 'glaucoma'
        y_glau_label_v1 = 'yes'
    else:
        y_glau_label = 'non-glaucoma'
        y_glau_label_v1 = 'no'

    # Vision loss severity (may want to substitute 'normal vision' for 'normal vision loss')
    md_severity = str(row.get('md_severity', '')).strip().lower()
    md_str = f"{md_severity} vision loss" if md_severity not in ['normal', '', 'nan'] else "normal vision"

    # Cup-to-disc ratio
    cdr_status = str(row.get('cdr_status', '')).strip().lower()
    cdr_map = {
        'normal': 'normal cup-disc ratio',
        'borderline': 'borderline cup-disc ratio',
        'borderline abnormal': 'borderline cup-disc ratio',
        'abnormal': 'abnormal cup-disc ratio'
    }
    cdr_str = cdr_map.get(cdr_status, f"{cdr_status} cup-disc ratio" if cdr_status else "")

    # RNFL thickness
    rnflt_status = str(row.get('rnflt_status', '')).strip().lower()
    rnflt_str = f"{rnflt_status} RNFL thickness" if rnflt_status else ""

    # Near vision refraction
    refraction_stage = str(row.get('near_vision_refraction_stage', '')).strip().lower()
    se_map = {
        'positive': 'hyperopia',
        'neutral': 'emmetropia',
        'negative': 'myopia'
    }
    se_str = se_map.get(refraction_stage, refraction_stage) if refraction_stage else ""

    # Build disease/condition description like FairGenMed reference
    y_set = []
    y_set.append(y_glau_label_v1)
    y_set.append(md_str)
    if cdr_str: y_set.append(cdr_str)
    if rnflt_str: y_set.append(rnflt_str)
    if se_str: y_set.append(se_str)

    y_str = ", ".join(filter(None, [y_glau_label, md_str, cdr_str, se_str]))
    # Replace "normal vision loss" with "normal vision" (if present)
    if "normal vision loss" in y_str:
        y_str = y_str.replace("normal vision loss", "normal vision")

    patient_attr = f"{race_str}, {gender_str}, {age_str}"
    prompt = f"SLO fundus image of a {patient_attr} patient with the following conditions: {y_str}"

    return prompt


def create_validation_metadata(row, condition_columns):
    """
    Create complete validation metadata dictionary.
    
    Returns:
        dict containing all metadata needed for validation
    """
    metadata = {}
    
    # Condition labels
    condition_labels = extract_condition_labels(row, condition_columns)
    metadata['condition_labels'] = condition_labels
    
    # Demographics
    demographics = extract_demographics(row)
    metadata.update(demographics)
    
    # Additional metadata (optional)
    if 'filename' in row:
        metadata['filename'] = str(row['filename'])
    if 'md' in row:
        metadata['md'] = float(row['md']) if not pd.isna(row['md']) else None
    if 'md_severity' in row:
        metadata['md_severity'] = str(row['md_severity']) if not pd.isna(row['md_severity']) else None
    
    return metadata


def load_fundus_image(npz_path):
    """
    Load fundus image from .npz file.
    
    Matches third_party normalization: min-max normalization to [0, 255] then ToTensor to [0, 1]
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        torch.Tensor: Image tensor (3, H, W) for RGB, float32, [0, 1]
    """
    from PIL import Image
    
    data = np.load(npz_path)
    image = data['slo_fundus']  # Shape: (200, 200, 3) - RGB fundus image
    
    # Convert to PIL Image for compatibility with third_party normalization
    # Handle both [0, 1] and [0, 255] input ranges
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode='RGB')
    
    # Apply third_party-style normalization (min-max to [0, 255])
    image_array = np.array(pil_image)
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    # Avoid division by zero
    if max_val - min_val > 0:
        normalized_array = (image_array - min_val) / (max_val - min_val) * 255
    else:
        normalized_array = image_array
    normalized_image = Image.fromarray(normalized_array.astype(np.uint8))
    
    # Convert to tensor [0, 1] using ToTensor (matches third_party)
    from torchvision import transforms
    tensor = transforms.ToTensor()(normalized_image)  # (3, H, W), [0, 1], float32
    
    return tensor


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def create_webdataset_with_validation_metadata(
    fundus_data,
    split_to_create="val",
    split_to_img_dir=None,
    split_to_dir=None,
    max_samples_per_tar=DEFAULT_MAX_SAMPLES_PER_TAR,
):
    """
    Create WebDataset tar files with validation metadata.

    Args:
        fundus_data: DataFrame with all the data (must have 'description' column)
        split_to_create: Which split to create ("train", "val", or "test")
        split_to_img_dir: dict mapping split name to Path of image directory
        split_to_dir: dict mapping split name to Path of output directory
        max_samples_per_tar: maximum number of samples per output tar file
    """
    if split_to_img_dir is None or split_to_dir is None:
        raise ValueError("split_to_img_dir and split_to_dir must be provided")
    print(f"\n{'='*60}")
    print(f"Creating WebDataset for split: {split_to_create}")
    print(f"{'='*60}\n")
    
    # Map split names: 'training' -> 'train', 'validation' -> 'val', 'test' -> 'test'
    split_mapping = {
        'training': 'train',
        'validation': 'val',
        'test': 'test'
    }
    
    # Filter data for this split
    # The 'use' column has values: 'training', 'test', 'validation'
    use_value = None
    for key, value in split_mapping.items():
        if value == split_to_create:
            use_value = key
            break
    
    if use_value is None:
        raise ValueError(f"Invalid split_to_create: {split_to_create}")
    
    data_split = fundus_data[fundus_data["use"] == use_value]
    data_split = data_split.reset_index(drop=True)
    
    print(f"Total samples in {split_to_create}: {len(data_split)}")
    
    # Get image directory for this split
    img_dir = split_to_img_dir[split_to_create]
    
    total_count = 0
    tar_sample_count = 0
    tar_idx = 0
    tar = None
    tar_path = None
    
    # Iterate through samples
    for idx, row in tqdm(data_split.iterrows(), total=len(data_split), desc=f"Processing {split_to_create}"):
        
        # Open new tar if needed
        if tar_sample_count == 0:
            tar_path = split_to_dir[split_to_create] / f"{split_to_create}_{tar_idx}.tar"
            tar = tarfile.open(tar_path, "w")
            print(f"\nOpened new tar: {tar_path}")
        
        # Get image path
        filename = row['filename']
        img_path = img_dir / filename
        
        if not img_path.exists():
            print(f"Missing image file: {img_path}")
            continue
        
        try:
            # ================================================================
            # 1. PROCESS IMAGE - PyTorch tensor (3, H, W), float32, [0, 1]
            # ================================================================
            arr_tensor = load_fundus_image(img_path)
            
            # Serialize image tensor
            img_bytes = io.BytesIO()
            pickle.dump(arr_tensor, img_bytes)
            img_bytes.seek(0)
            
            # ================================================================
            # 2. PROCESS PROMPT - UTF-8 text
            # ================================================================
            # Use the 'description' column for prompts
            if 'description' not in row:
                raise ValueError("'description' column not found in dataframe. Please create it first.")
            
            prompt_text = str(row['description'])
            prompt_bytes = prompt_text.encode("utf-8")
            prompt_stream = io.BytesIO(prompt_bytes)
            
            # ================================================================
            # 3. CREATE VALIDATION METADATA - JSON
            # ================================================================
            validation_metadata = create_validation_metadata(row, CONDITION_COLUMNS)
            
            # Serialize metadata as JSON
            metadata_json = json.dumps(validation_metadata)
            metadata_bytes = metadata_json.encode("utf-8")
            metadata_stream = io.BytesIO(metadata_bytes)
            
            # ================================================================
            # 4. ADD ALL FILES TO TAR
            # ================================================================
            tar_key = f"{tar_sample_count+1:06d}"
            
            # Add image tensor
            ptinfo = tarfile.TarInfo(f"{tar_key}.pt_image")
            ptinfo.size = img_bytes.getbuffer().nbytes
            img_bytes.seek(0)
            tar.addfile(ptinfo, img_bytes)
            
            # Add prompt
            promptinfo = tarfile.TarInfo(f"{tar_key}.prompt_metadata")
            promptinfo.size = prompt_stream.getbuffer().nbytes
            prompt_stream.seek(0)
            tar.addfile(promptinfo, prompt_stream)
            
            # Add validation metadata
            metainfo = tarfile.TarInfo(f"{tar_key}.validation_metadata")
            metainfo.size = metadata_stream.getbuffer().nbytes
            metadata_stream.seek(0)
            tar.addfile(metainfo, metadata_stream)
            
            tar_sample_count += 1
            total_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ================================================================
        # 5. CLOSE TAR IF FULL
        # ================================================================
        if tar_sample_count >= max_samples_per_tar:
            tar.close()
            
            # Write size file
            size_txt_path = split_to_dir[split_to_create] / f"{split_to_create}_{tar_idx}_size.txt"
            with open(size_txt_path, "w") as f:
                f.write(str(tar_sample_count) + "\n")
            
            print(f"Closed {tar_path} with {tar_sample_count} samples")
            
            tar_sample_count = 0
            tar_idx += 1
    
    # ================================================================
    # 6. CLOSE FINAL TAR
    # ================================================================
    if tar_sample_count > 0 and tar is not None:
        tar.close()
        
        size_txt_path = split_to_dir[split_to_create] / f"{split_to_create}_{tar_idx}_size.txt"
        with open(size_txt_path, "w") as f:
            f.write(str(tar_sample_count) + "\n")
        
        print(f"Closed {tar_path} with {tar_sample_count} samples")
    
    print(f"\n{'='*60}")
    print(f"Split {split_to_create}: {total_count} samples in {tar_idx+1} tar file(s)")
    print(f"{'='*60}\n")
    
    # ================================================================
    # 7. VERIFY TAR FILES
    # ================================================================
    print(f"\nVerifying tar files for {split_to_create}:")
    splitdir = split_to_dir[split_to_create]
    size_files = sorted(splitdir.glob(f"{split_to_create}_*_size.txt"))
    total_in_split = 0
    
    for sizefile in size_files:
        try:
            with open(sizefile) as f:
                count = int(f.read().strip())
                total_in_split += count
                print(f"  {sizefile.name}: {count} samples")
        except Exception as e:
            print(f"  {sizefile.name}: Error reading: {e}")
    
    print(f"\nTotal samples verified: {total_in_split}")


# ============================================================================
# CLI and MAIN
# ============================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description="Create WebDataset tar files for Fundus (FairGenMed) data with validation metadata."
    )
    parser.add_argument(
        "--fundus_base_dir",
        type=Path,
        default=Path("data/fundus"),
        help="Base directory containing Training/, Validation/, Test/ and data_summary.csv (default: data/fundus)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/fundus_webdataset"),
        help="Output directory for WebDataset tar files (default: data/fundus_webdataset)",
    )
    parser.add_argument(
        "--max_samples_per_tar",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_TAR,
        help=f"Maximum samples per tar file (default: {DEFAULT_MAX_SAMPLES_PER_TAR})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to create (default: train val test)",
    )
    return parser


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  Validation Dataset Creator for Fundus Data                        ║
    ╚════════════════════════════════════════════════════════════════════╝

    This script creates WebDataset tar files with complete validation metadata.

    Required columns in data_summary.csv:
    - use: 'training', 'validation', or 'test'
    - filename: name of .npz file (e.g., 'data_00001.npz')
    - description: prompt text (created in process.ipynb, or auto-generated)
    - glaucoma, cdr_status, rnflt_status, near_vision_refraction_stage: condition labels
    - age, gender, race, ethnicity: demographic attributes

    Output format per sample:
    - {key}.pt_image: PyTorch tensor (3, H, W), float32, [0, 1] - RGB fundus image
    - {key}.prompt_metadata: UTF-8 text prompt
    - {key}.validation_metadata: JSON with condition labels & demographics

    """)

    args = get_parser().parse_args()
    fundus_base_dir = args.fundus_base_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not fundus_base_dir.exists():
        raise FileNotFoundError(
            f"Fundus base directory not found: {fundus_base_dir}. "
            "Download the FairGenMed dataset and set --fundus_base_dir to its path."
        )

    split_to_img_dir = {
        "train": fundus_base_dir / "Training",
        "val": fundus_base_dir / "Validation",
        "test": fundus_base_dir / "Test",
    }
    split_to_dir = {
        "train": output_dir / "training_data",
        "val": output_dir / "val_data",
        "test": output_dir / "test_data",
    }
    for d in split_to_dir.values():
        d.mkdir(parents=True, exist_ok=True)

    data_summary_path = fundus_base_dir / "data_summary.csv"
    if not data_summary_path.exists():
        raise FileNotFoundError(f"data_summary.csv not found in {fundus_base_dir}")
    fundus_data = pd.read_csv(data_summary_path)

    if "description" not in fundus_data.columns:
        print("Warning: 'description' column not found. Creating FairGenMed-style prompts...")
        fundus_data["description"] = fundus_data.apply(generate_diffusion_prompt_from_row, axis=1)
        print("Description column created.")

    for split in args.splits:
        create_webdataset_with_validation_metadata(
            fundus_data,
            split_to_create=split,
            split_to_img_dir=split_to_img_dir,
            split_to_dir=split_to_dir,
            max_samples_per_tar=args.max_samples_per_tar,
        )

    print("\n✓ Dataset creation complete!")
