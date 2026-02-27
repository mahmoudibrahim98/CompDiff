"""
Create WebDataset for validation with all necessary metadata.

This script creates validation tar files that include:
1. Image tensors (.pt_image)
2. Prompt text (.prompt_metadata)  
3. Validation metadata (.validation_metadata) - NEW!
   - Disease labels (5 diseases)
   - Demographics (age, sex, race)
   - Any other metadata needed for validation metrics

The format is compatible with the validation system in train_loop.py.
"""

import argparse
import os
import pickle
from PIL import Image
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

DEFAULT_MAX_SAMPLES_PER_TAR = 1000

# Disease labels to extract (modify based on your data)
DISEASE_COLUMNS = [
    'Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices',
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_disease_labels(row, disease_columns):
    """
    Extract disease labels from dataframe row.
    
    Returns:
        list of floats: Binary labels (1.0 for positive, 0.0 for negative, -1.0 for uncertain/missing)
    """
    labels = []
    for disease in disease_columns:
        if disease in row:
            val = row[disease]
            # Handle different formats: 1/0, True/False, 1.0/0.0/-1.0
            if pd.isna(val):
                labels.append(-1.0)
            elif val == 1 or val == 1.0 or val == True:
                labels.append(1.0)
            elif val == 0 or val == 0.0 or val == False:
                labels.append(0.0)
            else:
                labels.append(-1.0)  # Uncertain
        else:
            labels.append(-1.0)  # Missing
    return labels


def extract_demographics(row):
    """
    Extract demographic information from dataframe row.
    
    Returns:
        dict with keys: age, sex, race, sex_idx, race_idx, age_bin
    """
    demographics = {}
    
    # Age
    if 'anchor_age' in row:
        demographics['age'] = float(row['anchor_age']) if not pd.isna(row['anchor_age']) else -1.0
    else:
        demographics['age'] = -1.0
    
    # Age bin (for HCN/DemographicEncoder)
    # Default bins: [18, 40, 60, 80] creates 5 bins: [0-18, 18-40, 40-60, 60-80, 80+]
    # To use different bins, modify the bins list below or import from dataset_wds
    age = demographics['age']
    if age < 0:
        demographics['age_bin'] = -1
    else:
        # Default age bins - can be overridden by importing from dataset_wds
        # or by setting AGE_BINS environment variable as comma-separated values
        import os
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
    
    # Sex
    if 'gender' in row:
        sex = str(row['gender']).upper()
        if sex in ['M', 'MALE']:
            demographics['sex'] = 'M'
            demographics['sex_idx'] = 0
        elif sex in ['F', 'FEMALE']:
            demographics['sex'] = 'F'
            demographics['sex_idx'] = 1
        else:
            demographics['sex'] = 'UNKNOWN'
            demographics['sex_idx'] = -1
    else:
        demographics['sex'] = 'UNKNOWN'
        demographics['sex_idx'] = -1
    
    # Race (modify categories based on your data)
    if 'ethnicity' in row:
        race = str(row['ethnicity']).upper()
        # Map to indices (modify based on your data)
        race_map = {
            'WHITE': 0,
            'BLACK': 1,
            'ASIAN': 2,
            'HISPANIC': 3,
            'OTHER': 3,
            'UNKNOWN': -1
        }
        demographics['race'] = race
        demographics['race_idx'] = race_map.get(race, -1)
    else:
        demographics['race'] = 'UNKNOWN'
        demographics['race_idx'] = -1
    
    return demographics


def create_validation_metadata(row, disease_columns):
    """
    Create complete validation metadata dictionary.
    
    Returns:
        dict containing all metadata needed for validation
    """
    metadata = {}
    
    # Disease labels
    metadata['disease_labels'] = extract_disease_labels(row, disease_columns)
    
    # Demographics
    demographics = extract_demographics(row)
    metadata.update(demographics)
    
    # Additional metadata (optional)
    if 'study_id' in row:
        metadata['study_id'] = str(row['study_id'])
    if 'subject_id' in row:
        metadata['subject_id'] = str(row['subject_id'])
    if 'image_id' in row:
        metadata['image_id'] = str(row['image_id'])
    return metadata


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def create_webdataset_with_validation_metadata(
    PA_data,
    split_to_create="val",
    split_to_dir=None,
    max_samples_per_tar=DEFAULT_MAX_SAMPLES_PER_TAR,
    source_dir=None,
):
    """
    Create WebDataset tar files with validation metadata.

    Args:
        PA_data: DataFrame with all the data (columns: split, image, final_sentence, disease labels, demographics)
        split_to_create: Which split to create ("train", "val", or "test")
        split_to_dir: dict mapping split name to Path of output directory
        max_samples_per_tar: maximum number of samples per output tar file
        source_dir: optional Path; if set, image paths in CSV are joined with this (for relative paths)
    """
    if split_to_dir is None:
        raise ValueError("split_to_dir must be provided")
    print(f"\n{'='*60}")
    print(f"Creating WebDataset for split: {split_to_create}")
    print(f"{'='*60}\n")
    
    # Filter data for this split
    data_split = PA_data[PA_data["split"] == split_to_create]
    data_split = data_split.reset_index(drop=True)
    
    print(f"Total samples in {split_to_create}: {len(data_split)}")
    
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
        
        # Get image path (optionally relative to source_dir)
        img_path = Path(row["image"])
        if source_dir is not None:
            img_path = (source_dir / img_path).resolve()
        if not img_path.exists():
            print(f"Missing image file: {img_path}")
            continue
        
        try:
            # ================================================================
            # 1. PROCESS IMAGE - PyTorch tensor (1, H, W), float32, [0, 1]
            # ================================================================
            with Image.open(img_path) as img:
                img = img.convert("L")  # Grayscale
                arr = np.array(img)
                arr = arr.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Convert to tensor (1, H, W)
            arr_tensor = torch.from_numpy(arr)
            
            # Serialize image tensor
            img_bytes = io.BytesIO()
            pickle.dump(arr_tensor, img_bytes)
            img_bytes.seek(0)
            
            # ================================================================
            # 2. PROCESS PROMPT - UTF-8 text
            # ================================================================
            prompt_bytes = row['final_sentence'].encode("utf-8")
            prompt_stream = io.BytesIO(prompt_bytes)
            
            # ================================================================
            # 3. CREATE VALIDATION METADATA - JSON
            # ================================================================
            validation_metadata = create_validation_metadata(row, DISEASE_COLUMNS)
            
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
            
            # Add validation metadata (NEW!)
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
        description="Create WebDataset tar files for Chest (MIMIC-CXR) data with validation metadata."
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=None,
        help="Root directory of MIMIC-CXR (e.g. containing 'files/'). If set, CSV 'image' column is relative to this path.",
    )
    parser.add_argument(
        "--split_csv",
        type=Path,
        default=Path("data/chest/split_data.csv"),
        help="Path to CSV with columns: split, image, final_sentence, disease labels, demographics (default: data/chest/split_data.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/chest_webdataset"),
        help="Output directory for WebDataset tar files (default: data/chest_webdataset)",
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
    ║  Validation Dataset Creator for Chest (MIMIC-CXR)                  ║
    ╚════════════════════════════════════════════════════════════════════╝

    This script creates WebDataset tar files with complete validation metadata.

    Required columns in split CSV:
    - split: 'train', 'val', or 'test'
    - image: path to image file (absolute or relative)
    - final_sentence: prompt text
    - Atelectasis, Cardiomegaly, Edema, Pneumothorax, Pleural Effusion, etc.: disease labels
    - anchor_age, gender, ethnicity: demographic attributes

    Output format per sample:
    - {key}.pt_image: PyTorch tensor (1, H, W), float32, [0, 1]
    - {key}.prompt_metadata: UTF-8 text prompt
    - {key}.validation_metadata: JSON with disease labels & demographics

    """)

    args = get_parser().parse_args()
    split_csv = args.split_csv.resolve()
    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else None

    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {split_csv}. "
            "Create a split CSV from MIMIC-CXR metadata and set --split_csv to its path."
        )

    split_to_dir = {
        "train": output_dir / "training_data",
        "val": output_dir / "val_data",
        "test": output_dir / "test_data",
    }
    for d in split_to_dir.values():
        d.mkdir(parents=True, exist_ok=True)

    PA_data = pd.read_csv(split_csv)

    for split in args.splits:
        create_webdataset_with_validation_metadata(
            PA_data,
            split_to_create=split,
            split_to_dir=split_to_dir,
            max_samples_per_tar=args.max_samples_per_tar,
            source_dir=source_dir,
        )

    print("\n✓ Dataset creation complete!")



