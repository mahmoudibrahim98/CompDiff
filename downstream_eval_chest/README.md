# Downstream Classifier Evaluation Framework

This framework implements comprehensive downstream evaluation for training DenseNet-121 classifiers on 14 CheXpert classes, with support for multiple training strategies and post-hoc analysis.

## Features

- **Multiple Training Strategies**: Real-only, synthetic-only, augmentation, supervised pretraining
- **Comprehensive Evaluation**: AUROC, AUPRC, fairness metrics (2-way and 3-way intersectional)
- **Prediction Storage**: Saves all predictions, labels, and metadata for post-hoc analysis
- **Post-Hoc Analysis**: Compute new metrics without retraining

## Quick Start

### Training a Model

Run from the repository root so that `downstream_eval_chest` is on the Python path:

```bash
python downstream_eval_chest/train_downstream_classifier.py \
    --strategy 1a \
    --real_train_path demo_chest/training_data \
    --real_val_path demo_chest/val_data \
    --real_test_path demo_chest/test_data \
    --output_dir outputs/downstream_eval \
    --compute_normalization
```

Use WebDataset directories (with `.tar` and `*_size.txt` files) or CSV paths depending on your data. For the demo, `demo_chest/` contains WebDataset shards.

### Analyzing Saved Predictions

```bash
python downstream_eval_chest/analyze_predictions.py \
    --predictions outputs/downstream_eval/model_1a \
    --compute-calibration \
    --compute-custom-fairness
```

## Training Strategies

- **Model 1a**: Real only (62,094 samples) - Eval: DTest (output: `model_1a`)
- **Model 1b**: Real + 1× synthetic - Eval: DTest (output: `model_1b_v0` or `model_1b_v7`)
- **Model 1c**: Real + 2× synthetic - Eval: DTest (output: `model_1c_v0` or `model_1c_v7`)
- **Model 1d**: Synthetic only 1× - Eval: DTest (output: `model_1d_v0` or `model_1d_v7`)
- **Model 1e**: Synthetic only 2× - Eval: DTest (output: `model_1e_v0` or `model_1e_v7`)
- **Model 2b**: Real + 100k balanced synthetic - Eval: DTest (output: `model_2b_v0` or `model_2b_v7`)
- **Model 2c**: 100k balanced synthetic only - Eval: DTest (output: `model_2c_v0` or `model_2c_v7`)
- **Model 3a**: Pretrain on synthetic → fine-tune on real subsets - Eval: DTest (output: `model_3a_v0` or `model_3a_v7`)
- **Model 4a**: Balanced pretrain → fine-tune on real - Eval: DTest (output: `model_4a_v0` or `model_4a_v7`)
- **Model 5a**: Real train - Eval: Synthetic test v0 (auto-sets `model_version=v0`, output: `model_5a_v0`)
- **Model 5b**: Real train - Eval: Synthetic test v7 (auto-sets `model_version=v7`, output: `model_5b_v7`)

**Note**: All strategies that use synthetic data (1b-5b) include the version (v0 or v7) in their output directory names and model names to maintain clear differentiation.

## Synthetic Data Versions (v0 vs v7)

The framework supports two sources of synthetic data:

- **v0**: Baseline model - dataset name: `0_train_baseline`
- **v7**: HCN model with age from prompt - dataset name: `6_train_hcn_age_from_promt`

### How to Specify Version

1. **Via command-line argument**:
   ```bash
   --model_version v0  # or v7
   ```

2. **Via config file**:
   ```yaml
   model_version: "v0"  # or "v7"
   ```

3. **Auto-detection**:
   - Strategies 5a and 5b automatically set `model_version` (5a → v0, 5b → v7)
   - If `dataset_name` is not provided, defaults are used:
     - v0 → `0_train_baseline`
     - v7 → `6_train_hcn_age_from_promt`

### Example: Training Same Strategy with Both Versions

```bash
# Train Model 1b with v0
python downstream_eval_chest/train_downstream_classifier.py \
    --strategy 1b \
    --model_version v0 \
    --dataset_name 0_train_baseline \
    --real_train_path demo_chest/training_data \
    --real_val_path demo_chest/val_data \
    --synthetic_base_path synthetic_datasets

# Train Model 1b with v7
python downstream_eval_chest/train_downstream_classifier.py \
    --strategy 1b \
    --model_version v7 \
    --dataset_name 6_train_hcn_age_from_promt \
    --real_train_path demo_chest/training_data \
    --real_val_path demo_chest/val_data \
    --synthetic_base_path synthetic_datasets
```

## Output Structure

After training and evaluation, each model produces:

```
outputs/downstream_eval/model_1a/
├── checkpoint_best.pth          # Best model checkpoint
├── predictions.npz               # All predictions (logits, probs, labels)
├── metadata.json                 # Demographics, thresholds, model info
├── config.yaml                   # Training configuration
└── evaluation_results.json       # Evaluation metrics
```

## Post-Hoc Analysis

The saved predictions can be used to compute new metrics without retraining:

```python
from downstream_eval_chest.analyze_predictions import (
    load_predictions,
    compute_metrics_from_predictions,
    compute_calibration,
    compute_subgroup_metrics,
)

# Load predictions
predictions, ground_truth, metadata = load_predictions(
    'outputs/downstream_eval/model_1a/predictions.npz',
    'outputs/downstream_eval/model_1a/metadata.json'
)

# Compute new metrics
metrics = compute_metrics_from_predictions(
    predictions['predictions_probs'],
    ground_truth,
    thresholds=metadata['thresholds'],
)

# Compute calibration
calibration = compute_calibration(
    predictions['predictions_probs'],
    ground_truth,
    class_idx=0,  # Atelectasis
)
```

## Evaluating on CheXpert Dataset

You can evaluate trained models on the CheXpert dataset using the `--evaluate_on_chexpert` flag:

```bash
python downstream_eval_chest/train_downstream_classifier.py \
    --strategy 1a \
    --skip_training \
    --checkpoint_path outputs/downstream_eval/model_1a/checkpoint_best.pth \
    --real_train_path demo_chest/training_data \
    --real_val_path demo_chest/val_data \
    --evaluate_on_chexpert \
    --chexpert_csv_path /path/to/chexpert_filtered.csv \
    --chexpert_image_base_path /path/to/CheXpert-v1.0 \
    --output_dir outputs/downstream_eval
```

The script will automatically:
- Detect available splits in the CSV (prefers 'test', falls back to 'val')
- Load images using the relative paths from the CSV combined with the base path
- Extract labels and demographics from CheXpert-specific column names
- Save evaluation results with `evaluation_dataset='CheXpert'` in the metadata

## Data Format

### Real Data
- CSV format with columns: image path, 14 CheXpert labels, demographics
- Or WebDataset tar files

### CheXpert Dataset
- CSV format with `Path` column containing relative paths (e.g., `CheXpert-v1.0/train/patient00004/study1/view1_frontal.jpg`)
- Requires `--chexpert_image_base_path` to specify the base directory for images
- Columns: `Path`, `Sex`/`GENDER`, `Age`/`AGE_AT_CXR`, `PRIMARY_RACE`, `ETHNICITY`, `demo_group`, `split`, and all 14 CheXpert labels
- Automatically handles CheXpert-specific column names and demographics mapping

### Synthetic Data
- Image directories with metadata JSON files
- Structure: `synthetic_datasets/v0/0_train_baseline/` or `v7/6_train_hcn_age_from_promt/`

## Preprocessing

All images are preprocessed with:
1. Center crop to square (min dimension)
2. Resize to 224×224
3. Normalize with MIMIC-CXR mean/std (computed from training data)

## Evaluation Metrics

- **Performance**: AUROC, AUPRC per label (all 14 labels stored)
- **Fairness**: 
  - 2-way intersectional: Sex × Race, Sex × Age
  - 3-way intersectional: Age × Sex × Ethnicity
  - AUROC parity, underdiagnosis rates/gaps

## Statistical Analysis

- Bootstrap confidence intervals (1000 resamples)
- DeLong test for AUROC comparisons
- Permutation test for AUPRC comparisons

