# Missing Items Checklist

This document lists code, models, and directories that are referenced but not present in the codebase.

## ✅ Removed Features

### 1. HCN V9 Continuous Age
**Status:** ✅ Removed  
**Previously referenced in:** `models.py`, `train_loop.py`, `run_validation_monitor_debug.py`, `generate_synthetic_dataset.py`

**Note:** All references to continuous age encoding (V9) have been removed from the codebase.

---

### 2. FiLM Conditioning (V5)
**Status:** ✅ Removed  
**Previously referenced in:** `models.py`, `train.py`, `train_loop.py`, `config.py`

**Note:** All references to FiLM conditioning have been removed from the codebase.

---

## ⚠️ Optional/Placeholder Items

### 3. Dataset Paths (Config Files)
**Status:** ⚠️ Placeholder paths need to be updated  
**Location:** All config files in `configs/` directory

**Paths that need updating:**
- `url_root`: Training dataset path (WebDataset format)
- `wds_dataset_path`: Dataset path for generation
- `validation_csv`: Validation CSV file path
- `validation_images_dir`: Validation images directory
- `test_dir`: Test dataset directory
- `train_dir`: Training dataset directory
- `GT_dir` / `gt_dir`: Ground truth dataset directory
- `gt_wds_dataset_path`: Ground truth WebDataset path

**Example placeholder:**
```yaml
url_root: "path/to/your/training_data"  # Update with your dataset path
```

---

### 4. Model Checkpoint Paths
**Status:** ⚠️ Placeholder paths need to be updated  
**Location:** Config files

**Paths that need updating:**
- `generation_checkpoint_path`: Path to trained model checkpoint
- `validation_sex_model_path`: Path to sex classifier checkpoint (optional, used for validation)
- `demographic_encoder_pretrained_path`: Path to pretrained demographic encoder (optional)

**Note:** The sex classifier checkpoint is referenced in some configs but may not be required if using torchxrayvision's built-in sex model.

---

## ⚠️ Missing Dependencies

### 4. `safetensors` Package
**Status:** ⚠️ Missing from requirements.txt  
**Used in:**
- `gen_source/run_validation_monitor_debug.py` (multiple locations)
- `gen_source/generate_synthetic_dataset.py` (multiple locations)

**Required for:** Loading model checkpoints saved in safetensors format (default format used by accelerate)

**Fix:** Add to `requirements.txt`:
```txt
safetensors>=0.3.0
```

---

## ✅ External Dependencies (Auto-downloaded)

### 5. TorchXrayVision Models
**Status:** ✅ Automatically downloaded on first use  
**Used in:** `gen_source/validation_metrics.py`

**Models:**
- Disease classification: `densenet121-res224-all` (auto-downloaded)
- Race classification: `emory_hiti.RaceModel()` (auto-downloaded)
- Age prediction: `riken.AgeModel()` (auto-downloaded)
- Sex prediction: `mira.SexModel()` (auto-downloaded)

**Note:** These models are automatically downloaded from torchxrayvision when first used. No manual download required.

---

## 📋 Summary

### ✅ Removed Features (No Longer Needed)
1. ✅ HCN V9 Continuous Age - All references removed
2. ✅ FiLM Conditioning (V5) - All references removed

### Fixed
3. ✅ `safetensors` package - Added to requirements.txt

### Important (Should Document)
4. ⚠️ Update all placeholder paths in config files
5. ⚠️ Document dataset format requirements (WebDataset)

### Optional (Nice to Have)
6. ✅ External models auto-download (no action needed)

---

## 🔧 Recommendations

1. **For Publication:**
   - Either implement `hcn_v9_continuous_age.py` and `film.py`, OR
   - Document that these are optional features and remove references if not used in paper
   - Update all placeholder paths in example configs
   - Add a `SETUP.md` guide for dataset preparation

2. **For Users:**
   - Create example configs with clear path placeholders
   - Add dataset format documentation
   - Provide instructions for obtaining/downloading required datasets

3. **Code Cleanup:**
   - Consider making V9 and FiLM imports optional with graceful fallbacks
   - Add try/except blocks around optional feature imports
