#!/usr/bin/env python
"""
Independent validation monitoring script.

This script monitors the output directory for new checkpoints and automatically
runs validation on them. It can be run in parallel with training.

Usage:
    accelerate launch gen_source/run_validation_monitor.py --config_file configs/your_config.yaml
"""

import os
import sys
import time
import json
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from typing import Optional, Dict, List
import argparse
import yaml
from dataclasses import dataclass

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    # Fallback for older versions
    RepositoryNotFoundError = Exception

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_metrics import ValidationMetricsRunner
from dataset_wds import RGFineTuningImageDirectoryDataset, RGFineTuningWebDataset, get_age_bins_from_num_bins, set_default_age_bins


# Canonical age bins for validation reporting. Using a single definition here ensures that
# intersectional FID subgroups (e.g. "18-40_Female_White") are identical across all runs
# (v0, v7, fairdiffusion). Otherwise v7 uses [20,40,60,80] (from hcn_num_age_bins) and
# v0/fairdiffusion use default [18,40,60,80], producing different subgroup names.
VALIDATION_AGE_BINS = [18, 40, 60, 80]  # Bins: 0-18, 18-40, 40-60, 60-80, 80+

logger = get_logger(__name__, log_level="INFO")


@dataclass
class ValidationConfig:
    """Configuration for validation monitoring."""
    # Path to training config
    config_file: str

    # Monitoring settings
    check_interval: int = 1800  # Check for new checkpoints every N seconds
    manifest_file: str = "validation_manifest.json"  # Track validated checkpoints

    # Override validation settings from training config if needed
    num_validation_samples: Optional[int] = None
    val_batch_size: Optional[int] = None
    validation_guidance_scale: Optional[float] = None
    validation_num_inference_steps: Optional[int] = None


def is_step_in_validation_schedule(
    step: int,
    base_step: Optional[int],
    offsets: Optional[List[int]] = None,
    min_step: Optional[int] = None,
) -> bool:
    """Return True if a checkpoint step should be validated under the custom schedule."""
    if step <= 0:
        return False

    if min_step is not None and step < min_step:
        return False

    if base_step is None or base_step <= 0:
        return True

    if not offsets:
        return step % base_step == 0

    for offset in offsets:
        try:
            offset_int = int(offset)
        except (TypeError, ValueError):
            continue

        candidate = step - offset_int
        if candidate <= 0:
            continue
        if candidate % base_step == 0:
            return True

    return False


def load_config(file_path):
    """Load YAML config file."""
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None


class CheckpointMonitor:
    """Monitors output directory for new checkpoints."""

    def __init__(self, output_dir: str, manifest_file: str):
        self.output_dir = Path(output_dir)
        self.manifest_file = self.output_dir / manifest_file
        self.validated_checkpoints = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict]:
        """Load manifest of already validated checkpoints."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        # Empty file, return empty dict
                        return {}
                    return json.loads(content)
            except json.JSONDecodeError as e:
                # Invalid JSON, log warning and return empty dict
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Manifest file {self.manifest_file} contains invalid JSON: {e}. Starting with empty manifest.")
                return {}
        return {}

    def _save_manifest(self):
        """Save manifest of validated checkpoints."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.validated_checkpoints, f, indent=2)

    def get_new_checkpoints(self) -> List[Path]:
        """Get list of new checkpoints that haven't been validated."""
        if not self.output_dir.exists():
            return []

        # Find all checkpoint directories
        checkpoints = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                # Extract step number
                try:
                    step = int(item.name.split("-")[1])
                    # Check if not already validated or in progress
                    if item.name not in self.validated_checkpoints:
                        # Check if checkpoint is fully written (has expected files)
                        if self._is_checkpoint_complete(item):
                            checkpoints.append((step, item))
                    else:
                        # Check if validation is in progress (not completed)
                        status = self.validated_checkpoints[item.name]
                        if status.get("status") == "in_progress":
                            # Skip checkpoints that are currently being validated
                            continue
                except (ValueError, IndexError):
                    continue

        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        return [ckpt[1] for ckpt in checkpoints]

    def _is_checkpoint_complete(self, checkpoint_dir: Path) -> bool:
        """Check if checkpoint has been fully written."""
        # Check for key files that indicate checkpoint is complete
        # Accelerate saves as model.safetensors (or model.bin), not pytorch_model.bin
        has_model = (checkpoint_dir / "model.safetensors").exists() or (checkpoint_dir / "model.bin").exists() or (checkpoint_dir / "pytorch_model.bin").exists()
        has_random_states = (checkpoint_dir / "random_states_0.pkl").exists()
        return has_model and has_random_states

    def mark_in_progress(self, checkpoint_dir: Path):
        """Mark checkpoint as being validated (in progress)."""
        self.validated_checkpoints[checkpoint_dir.name] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "success": None,
            "metrics": None,
        }
        self._save_manifest()

    def mark_validated(self, checkpoint_dir: Path, metrics: Dict, success: bool = True, num_images: Optional[int] = None):
        """Mark checkpoint as validated (completed)."""
        entry = self.validated_checkpoints.get(checkpoint_dir.name, {})
        entry.update({
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "metrics": metrics if success else None,
            "num_images": num_images,
        })
        self.validated_checkpoints[checkpoint_dir.name] = entry
        self._save_manifest()

    def mark_skipped(self, checkpoint_dir: Path, reason: str):
        """Mark checkpoint as skipped (e.g., not part of validation schedule)."""
        entry = self.validated_checkpoints.get(checkpoint_dir.name, {})
        entry.update({
            "status": "skipped",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
        })
        self.validated_checkpoints[checkpoint_dir.name] = entry
        self._save_manifest()


class StepImageMonitor:
    """Monitors validation_images directory for new step directories with pre-generated images."""

    def __init__(self, images_base_dir: str, manifest_file: str):
        self.images_base_dir = Path(images_base_dir)
        self.manifest_file = self.images_base_dir.parent / manifest_file
        self.validated_checkpoints = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict]:
        """Load manifest of already validated checkpoints."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        # Empty file, return empty dict
                        return {}
                    return json.loads(content)
            except json.JSONDecodeError as e:
                # Invalid JSON, log warning and return empty dict
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Manifest file {self.manifest_file} contains invalid JSON: {e}. Starting with empty manifest.")
                return {}
        return {}

    def _save_manifest(self):
        """Save manifest of validated checkpoints."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.validated_checkpoints, f, indent=2)

    def get_new_steps(self) -> List[tuple]:
        """Get list of new step directories that haven't been validated.
        
        Returns:
            List of tuples (step_number, step_dir_path)
        """
        if not self.images_base_dir.exists():
            return []

        # Find all step directories (validation images use step_ format)
        checkpoints = []
        for item in self.images_base_dir.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                # Extract step number
                try:
                    step = int(item.name.split("_")[1])
                    # Use checkpoint- format for manifest keys (unified with checkpoint directories)
                    checkpoint_key = f"checkpoint-{step}"
                    
                    # Check if not already validated or in progress
                    if checkpoint_key not in self.validated_checkpoints:
                        # Check if step directory has images (has gpu subdirectories)
                        if self._is_step_complete(item):
                            checkpoints.append((step, item))
                    else:
                        # Check if validation is in progress (not completed)
                        status = self.validated_checkpoints[checkpoint_key]
                        if status.get("status") == "in_progress":
                            # Skip checkpoints that are currently being validated
                            continue
                except (ValueError, IndexError):
                    continue

        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def _is_step_complete(self, step_dir: Path) -> bool:
        """Check if step directory has images ready for validation."""
        # Check if there are gpu subdirectories with images
        gpu_dirs = list(step_dir.glob("gpu_*"))
        if not gpu_dirs:
            return False
        
        # Check if at least one gpu directory has synthetic images
        for gpu_dir in gpu_dirs:
            synth_files = list(gpu_dir.glob("synthetic_*.png"))
            if synth_files:
                return True
        return False

    def mark_in_progress(self, step: int):
        """Mark checkpoint as being validated (in progress)."""
        # Use checkpoint- format for manifest keys (unified with checkpoint directories)
        checkpoint_key = f"checkpoint-{step}"
        self.validated_checkpoints[checkpoint_key] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "success": None,
            "metrics": None,
        }
        self._save_manifest()

    def mark_validated(self, step: int, metrics: Dict, success: bool = True, num_images: Optional[int] = None):
        """Mark checkpoint as validated (completed)."""
        # Use checkpoint- format for manifest keys (unified with checkpoint directories)
        checkpoint_key = f"checkpoint-{step}"
        entry = self.validated_checkpoints.get(checkpoint_key, {})
        entry.update({
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "metrics": metrics if success else None,
            "num_images": num_images,
        })
        self.validated_checkpoints[checkpoint_key] = entry
        self._save_manifest()

    def mark_skipped(self, step: int, reason: str):
        """Mark checkpoint (step directory) as skipped."""
        checkpoint_key = f"checkpoint-{step}"
        entry = self.validated_checkpoints.get(checkpoint_key, {})
        entry.update({
            "status": "skipped",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
        })
        self.validated_checkpoints[checkpoint_key] = entry
        self._save_manifest()


class ValidationRunner:
    """Runs validation on a checkpoint."""

    def __init__(
        self,
        accelerator: Accelerator,
        args,
        logger,
    ):
        self.accelerator = accelerator
        self.args = args
        self.logger = logger
        self.load_images_from_dir = args.get("load_images_from_dir", False)
        self.strip_demographics_in_validation = args.get("strip_demographics_in_validation", True)
        self.keep_age_in_prompt_validation = args.get("keep_age_in_prompt_validation", False)

        # Load models (frozen during validation)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.noise_scheduler = None
        self.weight_dtype = None

        # Models that will be prepared once and reused for all checkpoints
        self.unet = None
        self.hcn = None

        # Validation infrastructure
        self.validation_dataloader = None
        self.metrics_runner = None
        self._metrics_runner_initialized = False  # Initialize flag for lazy metrics runner initialization

        # Always set canonical validation age_bins so intersectional subgroup names are
        # consistent across runs (v0, v7, fairdiffusion), including when load_images_from_dir.
        self.args["age_bins"] = VALIDATION_AGE_BINS
        set_default_age_bins(VALIDATION_AGE_BINS)

        # Only initialize models if we're generating images (not loading from directory)
        if not self.load_images_from_dir:
            self._initialize_fixed_models()
            self._initialize_trainable_models()  # Initialize UNet and HCN once
            self._initialize_validation_data()
        else:
            logger.info("Skipping model initialization (loading images from directory)")
            # Metrics runner will be initialized lazily when needed
    def _rank_prefix(self):
        return f"[Rank {self.accelerator.process_index}/{self.accelerator.num_processes}]"

    def _initialize_fixed_models(self):
        """Initialize models that don't change between checkpoints."""
        logger.info("Loading fixed models (tokenizer, text_encoder, vae, scheduler)...")

        # Tokenizer
        # Handle potential 404 error when transformers tries to check for chat templates
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.args["pretrained_model_name_or_path"],
                subfolder="tokenizer",
                revision=self.args.get("revision"),
                use_auth_token=self.args.get("use_auth_token"),
            )
        except (RepositoryNotFoundError, Exception) as e:
            # If there's an error (e.g., 404 for chat templates), try with local_files_only
            if isinstance(e, RepositoryNotFoundError) or "404" in str(e) or "RepositoryNotFoundError" in str(type(e)):
                logger.warning(f"Got repository error when loading tokenizer, trying with local cache: {e}")
                try:
                    self.tokenizer = CLIPTokenizer.from_pretrained(
                        self.args["pretrained_model_name_or_path"],
                        subfolder="tokenizer",
                        revision=self.args.get("revision"),
                        use_auth_token=self.args.get("use_auth_token"),
                        local_files_only=True,
                    )
                except Exception as e2:
                    # If that fails, try without subfolder (fallback)
                    logger.warning(f"Local files only failed, trying alternative loading: {e2}")
                    self.tokenizer = CLIPTokenizer.from_pretrained(
                        self.args["pretrained_model_name_or_path"],
                        revision=self.args.get("revision"),
                        use_auth_token=self.args.get("use_auth_token"),
                    )
            else:
                raise

        # Text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="text_encoder",
            revision=self.args.get("revision"),
            use_auth_token=self.args.get("use_auth_token"),
        )
        
        # VAE (frozen, always fp32)
        self.vae = AutoencoderKL.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="vae",
            revision=self.args.get("revision"),
            use_auth_token=self.args.get("use_auth_token"),
        )
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="scheduler",
        )

        # Don't prepare text_encoder here if it's being trained - it will be loaded from checkpoint
        # Don't prepare VAE at all - it should stay frozen and not be part of checkpoint loading
        if not self.args.get("train_text_encoder", False):
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
            self.text_encoder = self.accelerator.prepare(self.text_encoder)
        
        # Move VAE to device manually (don't use prepare() to avoid checkpoint loading issues)
        self.vae = self.vae.to(self.accelerator.device)

        # Set weight dtype
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        # VAE always stays in fp32
        self.vae.to(dtype=torch.float32)

        logger.info("✓ Fixed models loaded")

    def _initialize_validation_data(self):
        """Initialize validation dataloader and metrics runner."""
        logger.info("Loading validation data...")

        # Use canonical validation age_bins so intersectional FID subgroups are comparable
        # across all runs (v0, v7, fairdiffusion). Run-specific bins (e.g. v7's [20,40,60,80]
        # from hcn_num_age_bins) would produce different subgroup names (0-20, 20-40 vs 0-18, 18-40).
        age_bins = VALIDATION_AGE_BINS
        self.args["age_bins"] = age_bins
        set_default_age_bins(age_bins)
        if self.accelerator.is_local_main_process:
            logger.info(f"Using canonical validation age_bins for comparable subgroups: {age_bins}")

        # Load validation dataset
        if self.args.get("use_wds_dataset", False):
            # WebDataset - use validation data path
            val_path = self.args.get("validation_csv") or self.args.get("validation_images_dir")
            if val_path is None:
                raise ValueError("validation_csv or validation_images_dir must be specified for WebDataset validation")
            
            # If val_path is a directory, construct tar file paths
            import os
            import glob
            if os.path.isdir(val_path):
                # Find all tar files in the directory
                tar_files = sorted(glob.glob(os.path.join(val_path, "*.tar")))
                if not tar_files:
                    raise ValueError(f"No tar files found in validation directory: {val_path}")
                val_urls = tar_files
            else:
                val_urls = [val_path] if isinstance(val_path, str) else val_path
            
            validation_dataset = RGFineTuningWebDataset(
                url_list=val_urls,
                tokenizer=self.tokenizer,
                use_hcn=self.args.get("use_hcn", False),
                use_demographic_encoder=self.args.get("use_demographic_encoder", False),
                include_text=True,
                age_bins=age_bins,
            )
        else:
            # Directory dataset
            validation_dataset = RGFineTuningImageDirectoryDataset(
                image_dir_path=self.args["validation_images_dir"],
                text_dir_path=self.args["validation_csv"],
                tokenizer=self.tokenizer,
                use_hcn=self.args.get("use_hcn", False),
                use_demographic_encoder=self.args.get("use_demographic_encoder", False),
                include_text=True,
                age_bins=age_bins,
            )

        # Create dataloader
        # Use num_workers=0 for validation to avoid shared memory issues in distributed settings
        # With 8 GPUs, using 4 workers per GPU would create 32 worker processes
        # This can exhaust shared memory (/dev/shm) and cause "Cannot allocate memory" errors
        validation_num_workers = self.args.get("validation_num_workers", 0)
        self.validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1,  # We'll batch during generation
            shuffle=False,
            num_workers=validation_num_workers,
        )

        try:
            dataset_size = len(validation_dataset)
            logger.info(f"✓ Validation dataset loaded ({dataset_size} samples)")
        except:
            logger.info(f"✓ Validation dataset loaded")

        # NOTE: Metrics runner initialization is EXPENSIVE and blocks Rank 0
        # We initialize it lazily on first use to avoid blocking other ranks
        # during ValidationRunner initialization
        self.metrics_runner = None
        self._metrics_runner_initialized = False

    def _initialize_trainable_models(self):
        """Initialize UNet and HCN architectures (called once during __init__)."""
        logger.info("Initializing trainable model architectures (UNet, HCN)...")
        
        # Initialize UNet architecture
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="unet",
            revision=self.args.get("revision"),
            use_auth_token=self.args.get("use_auth_token"),
        )
        
        # Initialize HCN if enabled
        # CRITICAL: Must initialize HCN on ALL ranks consistently before prepare()
        # DDP requires all ranks to have the same model structure
        self.hcn = None
        
        if self.args.get("use_hcn", False):
            # Initialize HCN on all ranks - if it fails on any rank, we need to know
            # All ranks must execute this code path (no is_main_process check)
            try:
                # Check if using ordinal age loss
                elif self.args.get("hcn_age_loss_mode", "ce") != "ce":
                    from hcn_v8_ordinal import HierarchicalConditionerV8Ordinal
                    # Determine if auxiliary loss should be enabled based on hcn_aux_weight
                    hcn_aux_weight = self.args.get("hcn_aux_weight", 0.0)
                    use_aux_loss = hcn_aux_weight > 0.0
                    age_loss_mode = self.args.get("hcn_age_loss_mode", "ordinal")
                    soft_ce_sigma = self.args.get("hcn_soft_ce_sigma", 0.75)
                    self.hcn = HierarchicalConditionerV8Ordinal(
                        num_age_bins=self.args.get("hcn_num_age_bins", 5),
                        num_sex=self.args.get("hcn_num_sex", 2),
                        num_race=self.args.get("hcn_num_race", 4),
                        d_node=self.args.get("hcn_d_node", 256),
                        d_ctx=self.args.get("hcn_d_ctx", 1024),
                        dropout=self.args.get("hcn_dropout", 0.1),
                        use_uncertainty=self.args.get("hcn_use_uncertainty", True),
                        use_aux_loss=use_aux_loss,
                        aux_hidden_dim=self.args.get("hcn_aux_hidden_dim", 512),
                        age_loss_mode=age_loss_mode,
                        soft_ce_sigma=soft_ce_sigma,
                    )
                    if self.accelerator.is_local_main_process:
                        logger.info(f"✓ HCN with Ordinal Age Loss initialized on all ranks (mode: {age_loss_mode})")
                else:
                    # Default: standard HCN - use the load_hcn function from models
                    from models import load_hcn
                    # Create a minimal args object for load_hcn
                    class Args:
                        pass
                    args = Args()
                    for key, val in self.args.items():
                        setattr(args, key, val)
                    # Ensure use_hcn is set
                    args.use_hcn = True
                    self.hcn = load_hcn(args, logger)
                    if self.accelerator.is_local_main_process:
                        logger.info(f"✓ HCN architecture initialized on all ranks")
                    # Determine if auxiliary loss should be enabled based on hcn_aux_weight
                    # This must match the training code logic in models.py:load_hcn()
                    hcn_aux_weight = self.args.get("hcn_aux_weight", 0.0)
                    use_aux_loss = hcn_aux_weight > 0.0
                    self.hcn = HierarchicalConditioner(
                        num_age_bins=self.args.get("hcn_num_age_bins", 5),
                        num_sex=self.args.get("hcn_num_sex", 2),
                        num_race=self.args.get("hcn_num_race", 4),
                        d_node=self.args.get("hcn_d_node", 256),
                        d_ctx=self.args.get("hcn_d_ctx", 1024),
                        dropout=self.args.get("hcn_dropout", 0.1),
                        use_uncertainty=self.args.get("hcn_use_uncertainty", True),
                        use_aux_loss=use_aux_loss,
                    )
                if self.accelerator.is_local_main_process:
                    logger.info(f"✓ HCN architecture initialized on all ranks")
            except Exception as e:
                logger.error(f"Rank {self.accelerator.process_index}: Failed to initialize HCN: {e}")
                raise RuntimeError(
                    f"HCN initialization failed on rank {self.accelerator.process_index}. "
                    f"DDP requires HCN to be initialized consistently on all ranks. Error: {e}"
                )
        
        # Initialize DemographicEncoder (V4) if enabled
        self.demographic_encoder = None
        
        if self.args.get("use_demographic_encoder", False):
            try:
                from demographic_encoder import DemographicEncoder
                self.demographic_encoder = DemographicEncoder(
                    num_age_bins=self.args.get("demo_num_age_bins", 5),
                    num_sex=self.args.get("demo_num_sex", 2),
                    num_race=self.args.get("demo_num_race", 4),
                    d_hidden=self.args.get("demo_d_hidden", 256),
                    d_output=self.args.get("demo_d_output", 1024),
                    mode=self.args.get("demo_mode", "single"),  # 'single' or 'separate'
                    classifier_depth=self.args.get("demo_classifier_depth", "shallow"),  # 'shallow' or 'deep'
                    aux_hidden_dim=self.args.get("demo_aux_hidden_dim", 512),
                    dropout=self.args.get("demo_dropout", 0.1),
                )
                if self.accelerator.is_local_main_process:
                    logger.info(f"✓ DemographicEncoder (V4) architecture initialized on all ranks")
            except Exception as e:
                logger.error(f"Rank {self.accelerator.process_index}: Failed to initialize DemographicEncoder: {e}")
                raise RuntimeError(
                    f"DemographicEncoder initialization failed on rank {self.accelerator.process_index}. "
                    f"DDP requires DemographicEncoder to be initialized consistently on all ranks. Error: {e}"
                )
        
        # For validation, we DON'T wrap models in DDP (no gradient sync needed)
        # We'll just move them to the correct device and load checkpoint states manually
        # This avoids the prepare() hang issue in distributed validation
        if self.accelerator.is_local_main_process:
            logger.info("Moving models to device (skipping DDP wrapping for validation)...")
        
        # Move models to accelerator device
        self.unet = self.unet.to(self.accelerator.device)
        if self.hcn is not None:
            self.hcn = self.hcn.to(self.accelerator.device)
        if self.demographic_encoder is not None:
            self.demographic_encoder = self.demographic_encoder.to(self.accelerator.device)
        if self.args.get("train_text_encoder", False):
            self.text_encoder = self.text_encoder.to(self.accelerator.device)
        
        # Set to eval mode
        self.unet.eval()
        self.unet.requires_grad_(False)
        if self.hcn is not None:
            self.hcn.eval()
            self.hcn.requires_grad_(False)
        if self.demographic_encoder is not None:
            self.demographic_encoder.eval()
            self.demographic_encoder.requires_grad_(False)
        if self.args.get("train_text_encoder", False):
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        
        if self.accelerator.is_local_main_process:
            logger.info("✓ Trainable models initialized")


    def _ensure_metrics_runner(self):
        """Lazy initialization of metrics runner (on ALL ranks for streaming)."""
        if not self._metrics_runner_initialized:  # ← FIXED! No is_main_process check
            logger.info(f"{self._rank_prefix()} Initializing validation metrics runner...")
            device_str = str(self.accelerator.device)
            # Always use canonical validation age_bins so intersectional FID subgroup keys
            # match across v0, v7, fairdiffusion (and when load_images_from_dir skips _initialize_validation_data).
            age_bins = VALIDATION_AGE_BINS
            self.metrics_runner = ValidationMetricsRunner(
                device=device_str,
                sex_model_path=self.args.get("validation_sex_model_path"),
                age_bins=age_bins,
            )
            self._metrics_runner_initialized = True
            logger.info(f"{self._rank_prefix()} ✓ Metrics runner initialized")
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load checkpoint weights into models manually (without DDP)."""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        # Since we're not using prepare() (to avoid DDP hangs), we load checkpoints manually
        # Load model weights from safetensors or pytorch bin files
        try:
            import safetensors.torch
            checkpoint_path = Path(checkpoint_dir)
            
            # Load UNet
            if (checkpoint_path / "model.safetensors").exists():
                logger.info("Loading UNet from model.safetensors...")
                state_dict = safetensors.torch.load_file(checkpoint_path / "model.safetensors")
                self.unet.load_state_dict(state_dict, strict=False)
            elif (checkpoint_path / "pytorch_model.bin").exists():
                logger.info("Loading UNet from pytorch_model.bin...")
                state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=self.accelerator.device)
                self.unet.load_state_dict(state_dict, strict=False)
            
            # Load text encoder if trained
            if self.args.get("train_text_encoder", False):
                if (checkpoint_path / "model_1.safetensors").exists():
                    logger.info("Loading text encoder from model_1.safetensors...")
                    state_dict = safetensors.torch.load_file(checkpoint_path / "model_1.safetensors")
                    self.text_encoder.load_state_dict(state_dict, strict=False)
                elif (checkpoint_path / "pytorch_model_1.bin").exists():
                    logger.info("Loading text encoder from pytorch_model_1.bin...")
                    state_dict = torch.load(checkpoint_path / "pytorch_model_1.bin", map_location=self.accelerator.device)
                    self.text_encoder.load_state_dict(state_dict, strict=False)
            
            # Load HCN if enabled
            if self.hcn is not None:
                if (checkpoint_path / "model_2.safetensors").exists():
                    logger.info("Loading HCN from model_2.safetensors...")
                    state_dict = safetensors.torch.load_file(checkpoint_path / "model_2.safetensors")
                    self.hcn.load_state_dict(state_dict, strict=False)
                elif (checkpoint_path / "pytorch_model_2.bin").exists():
                    logger.info("Loading HCN from pytorch_model_2.bin...")
                    state_dict = torch.load(checkpoint_path / "pytorch_model_2.bin", map_location=self.accelerator.device)
                    self.hcn.load_state_dict(state_dict, strict=False)
            
            # Load DemographicEncoder (V4) if enabled
            if self.demographic_encoder is not None:
                # Try loading from demographic_encoder subdirectory first (saved by pipeline.py)
                demo_encoder_path = checkpoint_path / "demographic_encoder"
                if demo_encoder_path.exists():
                    logger.info(f"Loading DemographicEncoder from {demo_encoder_path}...")
                    try:
                        from demographic_encoder import DemographicEncoder
                        self.demographic_encoder = DemographicEncoder.from_pretrained(str(demo_encoder_path))
                        self.demographic_encoder = self.demographic_encoder.to(self.accelerator.device)
                        self.demographic_encoder.eval()
                        self.demographic_encoder.requires_grad_(False)
                        logger.info("✓ DemographicEncoder loaded from checkpoint")
                    except Exception as e:
                        logger.warning(f"Failed to load DemographicEncoder from {demo_encoder_path}: {e}")
                        logger.info("Trying to load from model_2 or model_3...")
                        # Fallback: try model_2 or model_3 (depending on whether HCN is also present)
                        if self.hcn is None:
                            # No HCN, so DemographicEncoder should be in model_2
                            if (checkpoint_path / "model_2.safetensors").exists():
                                state_dict = safetensors.torch.load_file(checkpoint_path / "model_2.safetensors")
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)
                            elif (checkpoint_path / "pytorch_model_2.bin").exists():
                                state_dict = torch.load(checkpoint_path / "pytorch_model_2.bin", map_location=self.accelerator.device)
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)
                        else:
                            # HCN is in model_2, so DemographicEncoder should be in model_3
                            if (checkpoint_path / "model_3.safetensors").exists():
                                state_dict = safetensors.torch.load_file(checkpoint_path / "model_3.safetensors")
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)
                            elif (checkpoint_path / "pytorch_model_3.bin").exists():
                                state_dict = torch.load(checkpoint_path / "pytorch_model_3.bin", map_location=self.accelerator.device)
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)
            
            logger.info("✓ Checkpoint state loaded")
        except FileNotFoundError as e:
            # Check if the error is about EMA model (model_3) - we don't need it for validation
            if "pytorch_model_3" in str(e) or "model_3" in str(e):
                logger.warning(f"EMA model file not found (this is OK for validation): {e}")
                # Try to load without EMA by manually loading the models we need
                # The main models should already be loaded, but let's try to continue
                logger.info("Attempting to load models manually...")
                try:
                    # Load main models directly from safetensors if available
                    import safetensors.torch
                    checkpoint_path = Path(checkpoint_dir)
                    
                    # Load UNet
                    if (checkpoint_path / "model.safetensors").exists():
                        logger.info("Loading UNet from model.safetensors...")
                        state_dict = safetensors.torch.load_file(checkpoint_path / "model.safetensors")
                        missing, unexpected = self.accelerator.unwrap_model(self.unet).load_state_dict(state_dict, strict=False)
                        if missing:
                            logger.warning(f"Missing keys in UNet: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in UNet: {len(unexpected)} keys")
                    
                    # Load text encoder if trained
                    if self.args.get("train_text_encoder", False) and (checkpoint_path / "model_1.safetensors").exists():
                        logger.info("Loading text encoder from model_1.safetensors...")
                        state_dict = safetensors.torch.load_file(checkpoint_path / "model_1.safetensors")
                        missing, unexpected = self.accelerator.unwrap_model(self.text_encoder).load_state_dict(state_dict, strict=False)
                        if missing:
                            logger.warning(f"Missing keys in text encoder: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in text encoder: {len(unexpected)} keys")
                    
                    # Load HCN if enabled
                    if self.hcn is not None and (checkpoint_path / "model_2.safetensors").exists():
                        logger.info("Loading HCN from model_2.safetensors...")
                        state_dict = safetensors.torch.load_file(checkpoint_path / "model_2.safetensors")
                        missing, unexpected = self.accelerator.unwrap_model(self.hcn).load_state_dict(state_dict, strict=False)
                        if missing:
                            logger.warning(f"Missing keys in HCN: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in HCN: {len(unexpected)} keys")
                    
                    logger.info("✓ Checkpoint models loaded manually (skipped EMA model)")
                except Exception as manual_load_error:
                    logger.error(f"Failed to load models manually: {manual_load_error}")
                    raise e  # Re-raise original error if manual load fails
            else:
                # Re-raise if it's a different FileNotFoundError
                raise

        # Set to eval mode
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        if self.hcn is not None:
            self.hcn.eval()
            self.hcn.requires_grad_(False)

        logger.info("✓ Checkpoint loaded successfully")


    def _generate_validation_images(self, unet, text_encoder, vae, global_step):
        """Generate validation images (distributed across GPUs)."""
        all_synthetic_images = []
        all_real_images = []
        all_labels = {
            "disease": [],
            "sex": [],
            "race": [],
            "age": [],
            "prompts": [],
        }

        num_images_per_prompt = self.args.get("validation_num_images_per_prompt", 4)
        max_prompts = self.args.get("num_validation_samples", 100)
        generation_batch_size = self.args.get("val_batch_size", 1)

        # Handle -1 as "use all samples"
        use_all_samples = (max_prompts == -1)

        # Collect all prompts first
        all_prompts_data = []
        num_prompts_collected = 0

        for batch in self.validation_dataloader:
            if not use_all_samples and num_prompts_collected >= max_prompts:
                break

            batch_size = batch["pixel_values"].shape[0]
            if use_all_samples:
                samples_to_take = batch_size
            else:
                samples_to_take = min(batch_size, max_prompts - num_prompts_collected)

            for i in range(samples_to_take):
                prompt = batch.get("text", [""])[i] if isinstance(batch.get("text"), list) else batch.get("text", "")
                all_prompts_data.append({
                    "prompt": prompt,
                    "batch": batch,
                    "batch_idx": i,
                })

            num_prompts_collected += samples_to_take

        if self.accelerator.is_local_main_process:
            if use_all_samples:
                logger.info(f"Collected {len(all_prompts_data)} prompts from validation dataset (using all samples)")
            else:
                logger.info(f"Collected {len(all_prompts_data)} prompts from validation dataset (limit: {max_prompts})")
            total_images = len(all_prompts_data) * num_images_per_prompt
            logger.info(f"Will generate {total_images} images total")
            logger.info(f"  - {num_images_per_prompt} image(s) per prompt")
            logger.info(f"  - Distributed across {self.accelerator.num_processes} GPU(s)")

        # Distribute prompts across GPUs
        num_gpus = self.accelerator.num_processes
        gpu_idx = self.accelerator.process_index

        prompts_for_this_gpu = []
        for i, prompt_idx in enumerate(range(len(all_prompts_data))):
            if i % num_gpus == gpu_idx:
                prompts_for_this_gpu.append(prompt_idx)

        # Generate images for assigned prompts
        images_to_generate = []
        for prompt_idx in prompts_for_this_gpu:
            for image_idx in range(num_images_per_prompt):
                images_to_generate.append((prompt_idx, image_idx))

        if self.accelerator.is_local_main_process:
            logger.info(f"GPU {gpu_idx}: Generating {len(images_to_generate)} images ({len(prompts_for_this_gpu)} prompts × {num_images_per_prompt} images)")

        # Generate with progress bar
        with torch.no_grad():
            progress_bar = tqdm(
                total=len(images_to_generate),
                desc=f"Generating images (GPU {gpu_idx})",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch_start in range(0, len(images_to_generate), generation_batch_size):
                batch_end = min(batch_start + generation_batch_size, len(images_to_generate))
                batch_images = images_to_generate[batch_start:batch_end]

                # Generate batch
                synthetic_batch, real_batch, labels_batch = self._generate_image_batch(
                    batch_images, all_prompts_data, unet, text_encoder, vae, global_step, batch_start
                )

                # Store results
                all_synthetic_images.extend(synthetic_batch)
                all_real_images.extend(real_batch)
                for key in labels_batch:
                    all_labels[key].extend(labels_batch[key])

                progress_bar.update(len(batch_images))

            progress_bar.close()

        return all_synthetic_images, all_real_images, all_labels

    def _generate_image_batch(self, batch_images, all_prompts_data, unet, text_encoder, vae, global_step, batch_start_idx):
        """Generate a batch of images."""
        # Prepare prompts
        prompts = []
        batch_data_list = []

        for (prompt_idx, image_idx) in batch_images:
            prompt_data = all_prompts_data[prompt_idx]
            prompts.append(prompt_data["prompt"])
            batch_data_list.append((prompt_data["batch"], prompt_data["batch_idx"]))

        local_batch_size = len(prompts)

        # Strip demographics from prompts if requested
        # Can be enabled independently of use_hcn, use_demographic_encoder, or demo_use_dropout modes
        use_hcn = self.args.get("use_hcn", False)
        use_demographic_encoder = self.args.get("use_demographic_encoder", False)
        demo_use_dropout = self.args.get("demo_use_dropout", False)
        should_strip = self.strip_demographics_in_validation
        
        # Strip demographics if requested
        if should_strip:
            from dataset_wds import extract_clinical_text
            # Strip demographics from prompts (same as training)
            # Note: Validation always applies if enabled (no probability check) to be deterministic
            # This matches training behavior when use_hcn=True or demo_text_dropout_prob=1.0
            prompts = [extract_clinical_text(prompt, keep_age=self.keep_age_in_prompt_validation) for prompt in prompts]
            if self.accelerator.is_local_main_process and batch_start_idx == 0:
                self.logger.info(
                    f"✓ Stripped demographics from validation prompts "
                    f"(strip_demographics_in_validation=True, keep_age_in_prompt_validation={self.keep_age_in_prompt_validation}, use_hcn={use_hcn}, use_demographic_encoder={use_demographic_encoder}, demo_use_dropout={demo_use_dropout})"
                )
        elif (use_hcn or use_demographic_encoder or demo_use_dropout) and self.accelerator.is_local_main_process and batch_start_idx == 0:
            self.logger.info(
                "⚠️ Keeping demographics in validation prompts despite demographic stripping being enabled during training "
                "(strip_demographics_in_validation=False)"
            )

        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.accelerator.device)

        # Get text embeddings
        prompt_embeds = text_encoder(input_ids=text_input_ids, return_dict=False)
        text_embeddings = prompt_embeds[0]

        # Add HCN conditioning if available
        hcn_time_emb = None  # V6 timestep injection
        if self.hcn is not None:
            text_embeddings, hcn_time_emb = self._add_hcn_conditioning(text_embeddings, batch_data_list)
        
        # Add DemographicEncoder (V4) conditioning if available
        if self.demographic_encoder is not None:
            text_embeddings = self._add_demographic_encoder_conditioning(text_embeddings, batch_data_list)

        # Create unconditional embeddings
        uncond_inputs = self.tokenizer(
            [""] * local_batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.accelerator.device)
        uncond_embeds = text_encoder(input_ids=uncond_input_ids, return_dict=False)
        uncond_embeddings = uncond_embeds[0]

        # Ensure unconditional embeddings match the sequence length of conditional ones.
        # We intentionally DO NOT add demographic conditioning here; instead we append zero
        # demographic tokens so classifier-free guidance still works mathematically.
        # Note: For V6 timestep injection mode, we don't concatenate tokens, so skip this.
        if self.hcn is not None and hcn_time_emb is None:
            # V1 token mode: add zero token to match conditional sequence length
            zero_ctx = torch.zeros(
                (uncond_embeddings.shape[0], 1, uncond_embeddings.shape[-1]),
                device=uncond_embeddings.device,
                dtype=uncond_embeddings.dtype,
            )
            uncond_embeddings = torch.cat([uncond_embeddings, zero_ctx], dim=1)
        
        if self.demographic_encoder is not None:
            zero_ctx = torch.zeros(
                (uncond_embeddings.shape[0], 1, uncond_embeddings.shape[-1]),
                device=uncond_embeddings.device,
                dtype=uncond_embeddings.dtype,
            )
            uncond_embeddings = torch.cat([uncond_embeddings, zero_ctx], dim=1)

        # Concatenate for classifier-free guidance
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        # Prepare latents
        latents_shape = (
            local_batch_size,
            unet.config.in_channels,
            self.args["resolution"] // 8,
            self.args["resolution"] // 8,
        )

        # Generate latents with unique seeds (validation_seed allows different runs via --seed)
        base_seed = self.args.get("validation_seed", 0)
        generators = [
            torch.Generator(device=self.accelerator.device).manual_seed(
                base_seed + global_step * 10000 + self.accelerator.process_index * 1000 + batch_start_idx + i
            )
            for i in range(local_batch_size)
        ]
        latents = torch.stack([
            torch.randn(
                (1, unet.config.in_channels, self.args["resolution"] // 8, self.args["resolution"] // 8),
                generator=gen,
                device=self.accelerator.device,
                dtype=text_embeddings.dtype,
            )[0] for gen in generators
        ])

        init_noise_sigma = getattr(self.noise_scheduler, 'init_noise_sigma', 1.0)
        latents = latents * init_noise_sigma

        # Diffusion loop
        self.noise_scheduler.set_timesteps(self.args.get("validation_num_inference_steps", 50))
        timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        self.noise_scheduler.timesteps = timesteps

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            timestep_tensor = t.expand(latent_model_input.shape[0]) if t.ndim == 0 else t.repeat(latent_model_input.shape[0] // t.shape[0])

            # V6 timestep injection: add HCN time embedding to UNet's internal timestep embedding
            if hcn_time_emb is not None:
                # For classifier-free guidance, we need to handle both unconditional and conditional
                # Unconditional: use zero time_emb, Conditional: use actual time_emb
                zero_time_emb = torch.zeros_like(hcn_time_emb)
                combined_time_emb = torch.cat([zero_time_emb, hcn_time_emb], dim=0)
                
                from train_loop import TimestepInjectionContext
                with TimestepInjectionContext(unet, combined_time_emb):
                    noise_pred = unet(
                        latent_model_input,
                        timestep_tensor,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
            else:
                noise_pred = unet(
                    latent_model_input,
                    timestep_tensor,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_scale = self.args.get("validation_guidance_scale", 7.5)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = 1 / 0.18215 * latents
        latents_for_decode = latents.to(dtype=torch.float32)
        images = vae.decode(latents_for_decode).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        # Store results
        synthetic_images = [images[i].cpu() for i in range(local_batch_size)]
        # Real images from dataset are in [-1, 1] range, convert to [0, 1] for saving (same as synthetic)
        real_images_raw = [batch_data_list[i][0]["pixel_values"][batch_data_list[i][1]].cpu() for i in range(local_batch_size)]
        real_images = [(img / 2 + 0.5).clamp(0, 1) for img in real_images_raw]

        labels = {
            "disease": [],
            "sex": [],
            "race": [],
            "age": [],
            "prompts": prompts,
        }

        for i in range(local_batch_size):
            batch, batch_idx = batch_data_list[i]
            if "disease_labels" in batch:
                labels["disease"].append(batch["disease_labels"][batch_idx].cpu())
            if "sex_idx" in batch:
                labels["sex"].append(batch["sex_idx"][batch_idx].cpu())
            if "race_idx" in batch:
                labels["race"].append(batch["race_idx"][batch_idx].cpu())
            if "age" in batch:
                labels["age"].append(batch["age"][batch_idx].cpu())

        return synthetic_images, real_images, labels

    def _add_hcn_conditioning(self, text_embeddings, batch_data_list):
        """Add HCN conditioning to text embeddings."""
        if self.hcn is None:
            return text_embeddings

        age_indices = []
        sex_indices = []
        race_indices = []

        for (batch_data, batch_idx) in batch_data_list:
            has_demographics = "sex_idx" in batch_data and "race_idx" in batch_data
            has_age = "age_idx" in batch_data
            
            if has_demographics and has_age:
                sex_idx = batch_data["sex_idx"][batch_idx] if batch_data["sex_idx"].dim() > 0 else batch_data["sex_idx"]
                race_idx = batch_data["race_idx"][batch_idx] if batch_data["race_idx"].dim() > 0 else batch_data["race_idx"]
                
                sex_indices.append(sex_idx)
                race_indices.append(race_idx)
                
                if "age_idx" in batch_data:
                    age_idx = batch_data["age_idx"][batch_idx] if batch_data["age_idx"].dim() > 0 else batch_data["age_idx"]
                    age_indices.append(age_idx)

        if sex_indices and race_indices:
            sex_indices = torch.stack(sex_indices).squeeze().to(self.accelerator.device)
            race_indices = torch.stack(race_indices).squeeze().to(self.accelerator.device)
            
            if sex_indices.dim() == 0:
                sex_indices = sex_indices.unsqueeze(0)
            if race_indices.dim() == 0:
                race_indices = race_indices.unsqueeze(0)

            hcn_unwrapped = self.accelerator.unwrap_model(self.hcn)
            hcn_unwrapped.eval()
            
            # HCN returns 5 values: ctx, mu, logsigma, aux_logits, time_emb
            # Call with appropriate arguments
            if age_indices:
                # V7/V8: Check if V8 Ordinal (uses positional args) or V7/V8 (uses keyword args)
                age_indices = torch.stack(age_indices).squeeze().to(self.accelerator.device)
                if age_indices.dim() == 0:
                    age_indices = age_indices.unsqueeze(0)
                
                # V10: Check if age encoding is enabled
                encode_age = getattr(hcn_unwrapped, 'encode_age', True)
                age_indices_for_hcn = age_indices if encode_age else None
                
                # Check if V8 Ordinal by checking if forward signature expects (age, sex, race) positional
                # V8 Ordinal: forward(age_idx, sex_idx, race_idx)
                # V7/V8: forward(sex_idx, race_idx, age_idx=None)
                # We can check by trying to see if it's HierarchicalConditionerV8Ordinal
                from hcn_v8_ordinal import HierarchicalConditionerV8Ordinal
                if isinstance(hcn_unwrapped, HierarchicalConditionerV8Ordinal):
                    # V8 Ordinal: positional arguments (age, sex, race)
                    # Note: V8 Ordinal doesn't support encode_age=False, so always pass age
                    hcn_ctx, _, _, _, time_emb = hcn_unwrapped(age_indices, sex_indices, race_indices)
                else:
                    # V7/V8: keyword arguments (sex, race, age)
                    hcn_ctx, _, _, _, time_emb = hcn_unwrapped(
                        sex_idx=sex_indices,
                        race_idx=race_indices,
                        age_idx=age_indices_for_hcn,
                    )
            else:
                # No age data available
                hcn_ctx, _, _, _, time_emb = hcn_unwrapped(
                    sex_idx=sex_indices,
                    race_idx=race_indices,
                    age_idx=None,
                )

            # V6 timestep injection mode: hcn_ctx is None, time_emb is used
            # V1 token mode: hcn_ctx is [B, 1, d_ctx], time_emb is None
            if hcn_ctx is not None:
                text_embeddings = torch.cat([text_embeddings, hcn_ctx], dim=1)
            
            # Return time_emb for V6 mode (will be used in diffusion loop)
            return text_embeddings, time_emb

        return text_embeddings, None
    
    def _add_demographic_encoder_conditioning(self, text_embeddings, batch_data_list):
        """Add DemographicEncoder (V4) conditioning to text embeddings."""
        if self.demographic_encoder is None:
            return text_embeddings

        age_indices = []
        sex_indices = []
        race_indices = []

        for (batch_data, batch_idx) in batch_data_list:
            if "age_idx" in batch_data and "sex_idx" in batch_data and "race_idx" in batch_data:
                age_idx = batch_data["age_idx"][batch_idx] if batch_data["age_idx"].dim() > 0 else batch_data["age_idx"]
                sex_idx = batch_data["sex_idx"][batch_idx] if batch_data["sex_idx"].dim() > 0 else batch_data["sex_idx"]
                race_idx = batch_data["race_idx"][batch_idx] if batch_data["race_idx"].dim() > 0 else batch_data["race_idx"]

                age_indices.append(age_idx)
                sex_indices.append(sex_idx)
                race_indices.append(race_idx)

        if age_indices:
            age_indices = torch.stack(age_indices).squeeze().to(self.accelerator.device)
            sex_indices = torch.stack(sex_indices).squeeze().to(self.accelerator.device)
            race_indices = torch.stack(race_indices).squeeze().to(self.accelerator.device)

            if age_indices.dim() == 0:
                age_indices = age_indices.unsqueeze(0)
            if sex_indices.dim() == 0:
                sex_indices = sex_indices.unsqueeze(0)
            if race_indices.dim() == 0:
                race_indices = race_indices.unsqueeze(0)

            demo_encoder_unwrapped = self.accelerator.unwrap_model(self.demographic_encoder) if hasattr(self.accelerator, 'unwrap_model') else self.demographic_encoder
            demo_encoder_unwrapped.eval()
            # Returns tokens: [B, 1, d] if mode='single', [B, 3, d] if mode='separate'
            demo_tokens, _ = demo_encoder_unwrapped(age_indices, sex_indices, race_indices)

            text_embeddings = torch.cat([text_embeddings, demo_tokens], dim=1)

        return text_embeddings

    def _load_validation_images_from_disk(self, global_step):
        """Load all validation images from disk (saved by all GPUs)."""
        # Use test_images directory in test mode, train_images in train mode, GT_images in GT mode, validation_images otherwise
        output_dir = Path(self.args["output_dir"])
        if self.args.get("test_mode", False):
            images_dir_name = "test_images"
        elif self.args.get("train_mode", False):
            images_dir_name = "train_images"
        elif self.args.get("gt_mode", False):
            images_dir_name = "GT_images"
        else:
            images_dir_name = "validation_images"
        validation_images_dir = output_dir / images_dir_name / f"step_{global_step}"
        return self._load_validation_images_from_custom_dir(validation_images_dir)
    
    def _load_validation_images_from_custom_dir(self, images_dir):
        """Load all validation images from a custom directory.
        
        Args:
            images_dir: Path to directory containing subdirectories like 'gpu_0', 'gpu_1', etc.
                       Each subdirectory should contain synthetic_*.png, real_*.png, and labels.pkl files.
        """
        from PIL import Image
        import torchvision.transforms as transforms
        import pickle
        import glob
        
        images_dir = Path(images_dir)
        
        if not images_dir.exists():
            logger.warning(f"Validation images directory not found: {images_dir}")
            return None, None, None
        
        # Find all GPU directories
        gpu_dirs = sorted(glob.glob(str(images_dir / "gpu_*")))
        
        if not gpu_dirs:
            logger.warning(f"No GPU directories found in {images_dir}")
            return None, None, None
        
        logger.info(f"Loading validation images from {len(gpu_dirs)} GPU directories...")
        
        # Collect all *paired* image paths and labels (synthetic + real + labels)
        # We only keep samples that have BOTH synthetic_*.png AND real_*.png so that
        # similarity and subgroup metrics always see perfectly aligned tensors.
        synthetic_image_paths = []
        real_image_paths = []
        all_labels_dict = {
            "disease": [],
            "sex": [],
            "race": [],
            "age": [],
        }

        # Transform to convert PIL to tensor (images are saved as PNG)
        # IMPORTANT: Ensure consistent resolution - images should be 512x512
        transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),  # Ensure 512x512
            transforms.ToTensor(),  # Converts PIL to tensor [0, 1]
        ])
        
        for gpu_dir in gpu_dirs:
            gpu_dir = Path(gpu_dir)

            # Load labels if available
            labels_path = gpu_dir / "labels.pkl"
            gpu_labels = None
            if labels_path.exists():
                with open(labels_path, 'rb') as f:
                    gpu_labels = pickle.load(f)

            # Find all synthetic and real images and index them by numeric id
            synth_files = sorted(gpu_dir.glob("synthetic_*.png"))
            real_files = sorted(gpu_dir.glob("real_*.png"))

            def _idx_from_name(p: Path, prefix: str) -> int:
                try:
                    # e.g., synthetic_000123.png -> 123
                    stem = p.stem  # synthetic_000123
                    return int(stem.replace(prefix, ""))
                except Exception:
                    return -1

            synth_map = {}
            for p in synth_files:
                idx = _idx_from_name(p, "synthetic_")
                if idx >= 0:
                    synth_map[idx] = p

            real_map = {}
            for p in real_files:
                idx = _idx_from_name(p, "real_")
                if idx >= 0:
                    real_map[idx] = p

            common_indices = sorted(set(synth_map.keys()) & set(real_map.keys()))

            if common_indices:
                logger.info(
                    f"GPU dir {gpu_dir.name}: using {len(common_indices)} paired samples "
                    f"(synthetic={len(synth_files)}, real={len(real_files)})"
                )

            # For each index that has BOTH synthetic and real images, add the pair and
            # the corresponding labels (if present) in a perfectly aligned way.
            for idx in common_indices:
                synthetic_image_paths.append(synth_map[idx])
                real_image_paths.append(real_map[idx])

                if gpu_labels is not None:
                    # Each labels list is indexed by the same running index that was
                    # used when saving synthetic_XXXXXX / real_XXXXXX.
                    if gpu_labels.get("disease") and idx < len(gpu_labels["disease"]):
                        all_labels_dict["disease"].append(gpu_labels["disease"][idx])
                    if gpu_labels.get("sex") and idx < len(gpu_labels["sex"]):
                        all_labels_dict["sex"].append(gpu_labels["sex"][idx])
                    if gpu_labels.get("race") and idx < len(gpu_labels["race"]):
                        all_labels_dict["race"].append(gpu_labels["race"][idx])
                    if gpu_labels.get("age") and idx < len(gpu_labels["age"]):
                        all_labels_dict["age"].append(gpu_labels["age"][idx])
        
        logger.info(f"Found {len(synthetic_image_paths)} synthetic images and {len(real_image_paths)} real images")
        
        # Load images in batches to avoid memory issues
        def load_images_batch(image_paths, batch_size=32, crop_black_borders=False):
            """Load images in batches.
            
            Args:
                image_paths: List of image paths
                batch_size: Batch size for loading
                crop_black_borders: If True, crop black borders (for real images with SquarePad artifacts)
            """
            all_images = []
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        
                        # Crop black borders if requested (for real images with SquarePad artifacts)
                        if crop_black_borders:
                            import numpy as np
                            img_array = np.array(img)
                            # Find non-black regions (threshold at 5 to handle near-black pixels)
                            non_black = np.any(img_array > 5, axis=2)
                            if non_black.any():
                                rows = np.where(non_black.any(axis=1))[0]
                                cols = np.where(non_black.any(axis=0))[0]
                                if len(rows) > 0 and len(cols) > 0:
                                    top, bottom = rows[0], rows[-1] + 1
                                    left, right = cols[0], cols[-1] + 1
                                    # Crop to content area
                                    img = img.crop((left, top, right, bottom))
                        
                        img_tensor = transform(img)
                        # Convert to [C, H, W] format (already from ToTensor)
                        all_images.append(img_tensor)
                    except Exception as e:
                        logger.warning(f"Failed to load image {path}: {e}")
            return all_images
        
        # Load synthetic images (no border cropping needed - already square)
        synthetic_images = load_images_batch(synthetic_image_paths, crop_black_borders=False)
        # Load real images with black border cropping to remove SquarePad artifacts
        real_images = load_images_batch(real_image_paths, crop_black_borders=True) if real_image_paths else None
        
        # Stack labels if available
        disease_labels = torch.stack(all_labels_dict["disease"]) if all_labels_dict["disease"] else None
        sex_labels = torch.stack(all_labels_dict["sex"]) if all_labels_dict["sex"] else None
        race_labels = torch.stack(all_labels_dict["race"]) if all_labels_dict["race"] else None
        age_labels = torch.stack(all_labels_dict["age"]) if all_labels_dict["age"] else None
        
        return synthetic_images, real_images, {
            "disease": disease_labels,
            "sex": sex_labels,
            "race": race_labels,
            "age": age_labels,
        }


    def _move_metrics_models_to_gpu(self):
        """Move validation metric models to GPU."""
        if hasattr(self.metrics_runner, 'similarity'):
            if hasattr(self.metrics_runner.similarity, 'fid_model') and self.metrics_runner.similarity.fid_model is not None:
                self.metrics_runner.similarity.fid_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.similarity, 'biovil_model') and self.metrics_runner.similarity.biovil_model is not None:
                self.metrics_runner.similarity.biovil_model.to(self.accelerator.device)
        if hasattr(self.metrics_runner, 'text_alignment'):
            if hasattr(self.metrics_runner.text_alignment, 'disease_model') and self.metrics_runner.text_alignment.disease_model is not None:
                self.metrics_runner.text_alignment.disease_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.text_alignment, 'sex_model') and self.metrics_runner.text_alignment.sex_model is not None:
                self.metrics_runner.text_alignment.sex_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.text_alignment, 'race_model') and self.metrics_runner.text_alignment.race_model is not None:
                self.metrics_runner.text_alignment.race_model.to(self.accelerator.device)
            if hasattr(self.metrics_runner.text_alignment, 'age_model') and self.metrics_runner.text_alignment.age_model is not None:
                self.metrics_runner.text_alignment.age_model.to(self.accelerator.device)

    def _save_validation_images(self, all_synthetic_images, all_real_images, all_labels, global_step):
        """Save validation images and labels to disk if requested."""
        if not all_synthetic_images:
            return
        
        try:
            from torchvision.utils import save_image
            import pickle
            
            # Create output directory for validation images
            # Use test_images directory in test mode, train_images in train mode, GT_images in GT mode, validation_images otherwise
            output_dir = Path(self.args["output_dir"])
            if self.args.get("test_mode", False):
                images_dir_name = "test_images"
            elif self.args.get("train_mode", False):
                images_dir_name = "train_images"
            elif self.args.get("gt_mode", False):
                images_dir_name = "GT_images"
            else:
                images_dir_name = "validation_images"
            validation_images_dir = output_dir / images_dir_name / f"step_{global_step}"
            validation_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images from this GPU
            gpu_idx = self.accelerator.process_index
            gpu_dir = validation_images_dir / f"gpu_{gpu_idx}"
            gpu_dir.mkdir(parents=True, exist_ok=True)
            
            prompts = all_labels.get("prompts", [""] * len(all_synthetic_images))
            
            # Save labels for later loading
            labels_to_save = {
                "disease": all_labels.get("disease", []),
                "sex": all_labels.get("sex", []),
                "race": all_labels.get("race", []),
                "age": all_labels.get("age", []),
                "prompts": prompts,
            }
            
            for i, (synth_img, real_img) in enumerate(zip(all_synthetic_images, all_real_images)):
                # Save synthetic image
                synth_path = gpu_dir / f"synthetic_{i:06d}.png"
                save_image(synth_img, synth_path)
                
                # Save real image if available
                if real_img is not None:
                    real_path = gpu_dir / f"real_{i:06d}.png"
                    save_image(real_img, real_path)
                
                # Save prompt if available
                if i < len(prompts) and prompts[i]:
                    prompt_path = gpu_dir / f"prompt_{i:06d}.txt"
                    with open(prompt_path, 'w') as f:
                        f.write(str(prompts[i]))
            
            # Save labels as pickle file
            labels_path = gpu_dir / "labels.pkl"
            with open(labels_path, 'wb') as f:
                pickle.dump(labels_to_save, f)

            # Write a small "done" marker so rank 0 can safely wait until all GPUs
            # have completely finished writing before loading for metrics.
            done_marker = gpu_dir / "done.txt"
            try:
                with open(done_marker, "w") as f:
                    f.write(f"num_images={len(all_synthetic_images)}\n")
            except Exception as e:
                logger.warning(f"Failed to write done marker {done_marker}: {e}")

            logger.info(f"Saved {len(all_synthetic_images)} validation images to {gpu_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save validation images: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    def _move_metrics_models_to_cpu(self):
        """Move validation metric models to CPU."""
        if hasattr(self.metrics_runner, 'similarity'):
            if hasattr(self.metrics_runner.similarity, 'fid_model') and self.metrics_runner.similarity.fid_model is not None:
                self.metrics_runner.similarity.fid_model.to('cpu')
            if hasattr(self.metrics_runner.similarity, 'biovil_model') and self.metrics_runner.similarity.biovil_model is not None:
                self.metrics_runner.similarity.biovil_model.to('cpu')
        if hasattr(self.metrics_runner, 'text_alignment'):
            if hasattr(self.metrics_runner.text_alignment, 'disease_model') and self.metrics_runner.text_alignment.disease_model is not None:
                self.metrics_runner.text_alignment.disease_model.to('cpu')
            if hasattr(self.metrics_runner.text_alignment, 'sex_model') and self.metrics_runner.text_alignment.sex_model is not None:
                self.metrics_runner.text_alignment.sex_model.to('cpu')
            if hasattr(self.metrics_runner.text_alignment, 'race_model') and self.metrics_runner.text_alignment.race_model is not None:
                self.metrics_runner.text_alignment.race_model.to('cpu')
            if hasattr(self.metrics_runner.text_alignment, 'age_model') and self.metrics_runner.text_alignment.age_model is not None:
                self.metrics_runner.text_alignment.age_model.to('cpu')



    # --- ADD: per-batch update ---------------------------------------------------------

    def _compute_metrics_batched(self, all_synthetic_images, all_real_images, all_labels, global_step):
        """
        Compute ALL metrics on loaded images in batches (main process only).
        No distributed operations - everything happens on one GPU.
        """
        import numpy as np
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        metrics = {}
        device = self.accelerator.device
        num_images = len(all_synthetic_images)
        batch_size = self.args.get("validation_metrics_batch_size", 8)
        
        logger.info(f"Computing metrics on {num_images} images (batch_size={batch_size})")
        
        # For large datasets, avoid stacking all images at once to prevent OOM
        # Instead, keep images as lists and process in batches
        # Only stack a small sample for diagnostics
        sample_size = min(100, num_images)
        sample_synth = torch.stack(all_synthetic_images[:sample_size]).to(device)
        sample_real = torch.stack(all_real_images[:sample_size]).to(device) if all_real_images else None
        
        # Diagnostic: Log shapes and value ranges from sample
        logger.info(f"Image shapes (from sample of {sample_size}):")
        logger.info(f"  Synthetic: {sample_synth.shape}, dtype: {sample_synth.dtype}")
        logger.info(f"  Real: {sample_real.shape if sample_real is not None else None}, dtype: {sample_real.dtype if sample_real is not None else None}")
        logger.info(f"Image value ranges (from sample):")
        logger.info(f"  Synthetic: min={sample_synth.min():.4f}, max={sample_synth.max():.4f}, mean={sample_synth.mean():.4f}, std={sample_synth.std():.4f}")
        if sample_real is not None:
            logger.info(f"  Real: min={sample_real.min():.4f}, max={sample_real.max():.4f}, mean={sample_real.mean():.4f}, std={sample_real.std():.4f}")
        
        # Check for shape mismatches using sample
        if sample_real is not None:
            if sample_synth.shape != sample_real.shape:
                logger.warning(f"⚠️ SHAPE MISMATCH: Synthetic {sample_synth.shape} vs Real {sample_real.shape}")
            if sample_synth.shape[1] != sample_real.shape[1]:
                logger.warning(f"⚠️ CHANNEL MISMATCH: Synthetic has {sample_synth.shape[1]} channels, Real has {sample_real.shape[1]} channels")
            if sample_synth.shape[2:] != sample_real.shape[2:]:
                logger.warning(f"⚠️ SPATIAL DIMENSION MISMATCH: Synthetic {sample_synth.shape[2:]} vs Real {sample_real.shape[2:]}")
        
        # Free sample tensors to save memory
        del sample_synth, sample_real
        torch.cuda.empty_cache()
        
        # Extract labels
        disease_labels = all_labels.get("disease")
        sex_labels = all_labels.get("sex")
        race_labels = all_labels.get("race")
        age_labels = all_labels.get("age")

        # Move labels to device if they exist
        if disease_labels is not None:
            disease_labels = disease_labels.to(device)
        if sex_labels is not None:
            sex_labels = sex_labels.to(device)
        if race_labels is not None:
            race_labels = race_labels.to(device)
        if age_labels is not None:
            age_labels = age_labels.to(device)

        # ------------------------------------------------------------------
        # IMPORTANT: Align label tensors with image tensors for subgroup
        # metrics. In some datasets only a subset of images have demographic
        # labels, e.g. 650 labeled out of 1219 total. The subgroup routines
        # in validation_metrics.py expect labels and images to have the same
        # length. We therefore:
        #   - Process images in batches to avoid OOM
        #   - Create "labeled" views (prefix slices) for subgroup/text
        #     alignment metrics, using the minimum length across available
        #     label tensors and images.
        # ------------------------------------------------------------------
        label_lengths = []
        for lbl in [disease_labels, sex_labels, race_labels, age_labels]:
            if lbl is not None:
                label_lengths.append(lbl.shape[0])

        if label_lengths:
            min_label_len = min(label_lengths + [num_images])
        else:
            min_label_len = num_images

        if min_label_len < num_images:
            logger.warning(
                f"Subgroup/text-alignment metrics: only {min_label_len} out of {num_images} images "
                f"have complete labels. Using the first {min_label_len} images for label-dependent metrics."
            )

        # Truncate labels to min_label_len if needed
        if disease_labels is not None and disease_labels.shape[0] > min_label_len:
            disease_labels = disease_labels[:min_label_len]
        if sex_labels is not None and sex_labels.shape[0] > min_label_len:
            sex_labels = sex_labels[:min_label_len]
        if race_labels is not None and race_labels.shape[0] > min_label_len:
            race_labels = race_labels[:min_label_len]
        if age_labels is not None and age_labels.shape[0] > min_label_len:
            age_labels = age_labels[:min_label_len]
        
        # ================================================================
        # ACCUMULATORS
        # ================================================================
        # Similarity metrics accumulators (BioViL and MS-SSIM need per-batch averaging)
        biovil_sims = []
        msssim_vals = []
        
        # ================================================================
        # BATCH PROCESSING FOR BIOVIL AND MS-SSIM
        # Process images in batches from lists to avoid OOM
        # FID will be computed separately using compute_fid() which handles batching internally
        # Text alignment metrics will be computed using methods from validation_metrics.py
        # ================================================================
        logger.info("Processing batches for BioViL and MS-SSIM computation...")
        
        for i in range(0, num_images, batch_size):
            end_i = min(i + batch_size, num_images)
            # Stack only the current batch to save memory
            batch_synth = torch.stack(all_synthetic_images[i:end_i]).to(device)
            batch_real = torch.stack(all_real_images[i:end_i]).to(device) if all_real_images else None
            
            # ------------------------------------------------------------
            # SIMILARITY METRICS (BioViL, MS-SSIM)
            # FID is computed separately below using compute_fid()
            # ------------------------------------------------------------
            if hasattr(self.metrics_runner, 'similarity') and batch_real is not None:
                try:
                    # BioViL similarity
                    biovil = self.metrics_runner.similarity.compute_biovil_similarity(
                        batch_real, batch_synth
                    )
                    if not np.isnan(biovil):
                        biovil_sims.append(biovil)
                except Exception as e:
                    logger.warning(f"BioViL error: {e}")
                
                try:
                    # MS-SSIM
                    msssim = self.metrics_runner.similarity.compute_ms_ssim(
                        batch_real, batch_synth
                    )
                    if not np.isnan(msssim):
                        msssim_vals.append(msssim)
                except Exception as e:
                    logger.warning(f"MS-SSIM error: {e}")
            
            # Free batch tensors to save memory
            del batch_synth
            if batch_real is not None:
                del batch_real
            torch.cuda.empty_cache()
        
        logger.info("Batch processing complete, aggregating metrics...")
        
        # ================================================================
        # AGGREGATE SIMILARITY METRICS
        # ================================================================
        
        # FID (Inception v3) - process in chunks to avoid OOM, but use ALL images
        if hasattr(self.metrics_runner, 'similarity') and all_real_images:
            try:
                logger.info("Computing FID (Inception v3) using enhanced method from validation_metrics.py...")
                # Process in chunks to avoid OOM when stacking images
                # Extract embeddings incrementally, then compute FID from all embeddings
                fid_chunk_size = 500  # Stack up to 500 images at a time to avoid OOM
                
                if num_images <= fid_chunk_size:
                    # Small enough to stack all at once
                    real_tensor = torch.stack(all_real_images).to(device)
                    synth_tensor = torch.stack(all_synthetic_images).to(device)
                    metrics["val/fid"] = self.metrics_runner.similarity.compute_fid(
                        real_tensor, synth_tensor, batch_size=batch_size
                    )
                    del real_tensor, synth_tensor
                else:
                    # Process in chunks: extract embeddings incrementally, then compute FID
                    logger.info(f"Processing {num_images} images in chunks of {fid_chunk_size} for FID computation...")
                    
                    # Extract embeddings in chunks
                    real_embeddings_list = []
                    synth_embeddings_list = []
                    
                    for chunk_start in range(0, num_images, fid_chunk_size):
                        chunk_end = min(chunk_start + fid_chunk_size, num_images)
                        chunk_size = chunk_end - chunk_start
                        
                        # Stack chunk and extract embeddings
                        real_chunk = torch.stack(all_real_images[chunk_start:chunk_end]).to(device)
                        synth_chunk = torch.stack(all_synthetic_images[chunk_start:chunk_end]).to(device)
                        
                        # Extract embeddings using internal method (processes in batches internally)
                        real_emb_chunk = self.metrics_runner.similarity._get_inception_embeddings(real_chunk, batch_size)
                        synth_emb_chunk = self.metrics_runner.similarity._get_inception_embeddings(synth_chunk, batch_size)
                        
                        real_embeddings_list.append(real_emb_chunk)
                        synth_embeddings_list.append(synth_emb_chunk)
                        
                        del real_chunk, synth_chunk
                        torch.cuda.empty_cache()
                        
                        if (chunk_start // fid_chunk_size + 1) % 5 == 0 or chunk_end == num_images:
                            logger.info(f"  Processed {chunk_end}/{num_images} images...")
                    
                    # Concatenate all embeddings
                    real_embeddings = np.concatenate(real_embeddings_list, axis=0)
                    synth_embeddings = np.concatenate(synth_embeddings_list, axis=0)
                    
                    del real_embeddings_list, synth_embeddings_list
                    
                    # Ensure embeddings are 2D [N, D]
                    if real_embeddings.ndim > 2:
                        real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)
                    if synth_embeddings.ndim > 2:
                        synth_embeddings = synth_embeddings.reshape(synth_embeddings.shape[0], -1)
                    
                    # Compute FID from accumulated embeddings
                    from scipy.linalg import sqrtm
                    mu_real = np.mean(real_embeddings, axis=0)
                    sigma_real = np.cov(real_embeddings, rowvar=False)
                    
                    mu_synthetic = np.mean(synth_embeddings, axis=0)
                    sigma_synthetic = np.cov(synth_embeddings, rowvar=False)
                    
                    # Compute FID
                    ssdiff = np.sum((mu_real - mu_synthetic) ** 2.0)
                    covmean = sqrtm(sigma_real.dot(sigma_synthetic))
                    
                    # Handle numerical errors
                    if np.iscomplexobj(covmean):
                        covmean = covmean.real
                    
                    metrics["val/fid"] = float(ssdiff + np.trace(sigma_real + sigma_synthetic - 2.0 * covmean))
                    
                    del real_embeddings, synth_embeddings
                    torch.cuda.empty_cache()
                
                logger.info(f"FID (Inception v3): {metrics['val/fid']:.4f}")
            except Exception as e:
                logger.warning(f"FID (Inception v3) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # FID (RadImageNet ResNet50) - medical image specific FID - process in chunks but use ALL images
        if hasattr(self.metrics_runner, 'similarity') and all_real_images:
            try:
                logger.info("Computing FID (RadImageNet ResNet50) using enhanced method from validation_metrics.py...")
                # Process in chunks to avoid OOM when stacking images
                # Extract embeddings incrementally, then compute FID from all embeddings
                fid_chunk_size = 500  # Stack up to 500 images at a time to avoid OOM
                
                if num_images <= fid_chunk_size:
                    real_tensor = torch.stack(all_real_images).to(device)
                    synth_tensor = torch.stack(all_synthetic_images).to(device)
                    metrics["val/fid_radimagenet"] = self.metrics_runner.similarity.compute_fid_radimagenet(
                        real_tensor, synth_tensor, batch_size=batch_size
                    )
                    del real_tensor, synth_tensor
                else:
                    # Process in chunks: extract embeddings incrementally, then compute FID
                    logger.info(f"Processing {num_images} images in chunks of {fid_chunk_size} for FID RadImageNet computation...")
                    
                    # Extract embeddings in chunks
                    real_embeddings_list = []
                    synth_embeddings_list = []
                    
                    for chunk_start in range(0, num_images, fid_chunk_size):
                        chunk_end = min(chunk_start + fid_chunk_size, num_images)
                        chunk_size = chunk_end - chunk_start
                        
                        # Stack chunk and extract embeddings
                        real_chunk = torch.stack(all_real_images[chunk_start:chunk_end]).to(device)
                        synth_chunk = torch.stack(all_synthetic_images[chunk_start:chunk_end]).to(device)
                        
                        # Extract embeddings using internal method (processes in batches internally)
                        real_emb_chunk = self.metrics_runner.similarity._get_radimagenet_embeddings(real_chunk, batch_size)
                        synth_emb_chunk = self.metrics_runner.similarity._get_radimagenet_embeddings(synth_chunk, batch_size)
                        
                        real_embeddings_list.append(real_emb_chunk)
                        synth_embeddings_list.append(synth_emb_chunk)
                        
                        del real_chunk, synth_chunk
                        torch.cuda.empty_cache()
                        
                        if (chunk_start // fid_chunk_size + 1) % 5 == 0 or chunk_end == num_images:
                            logger.info(f"  Processed {chunk_end}/{num_images} images...")
                    
                    # Concatenate all embeddings
                    real_embeddings = np.concatenate(real_embeddings_list, axis=0)
                    synth_embeddings = np.concatenate(synth_embeddings_list, axis=0)
                    
                    del real_embeddings_list, synth_embeddings_list
                    
                    # Ensure embeddings are 2D [N, D]
                    if real_embeddings.ndim > 2:
                        real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)
                    if synth_embeddings.ndim > 2:
                        synth_embeddings = synth_embeddings.reshape(synth_embeddings.shape[0], -1)
                    
                    # Compute FID from accumulated embeddings
                    from scipy.linalg import sqrtm
                    mu_real = np.mean(real_embeddings, axis=0)
                    sigma_real = np.cov(real_embeddings, rowvar=False)
                    
                    mu_synthetic = np.mean(synth_embeddings, axis=0)
                    sigma_synthetic = np.cov(synth_embeddings, rowvar=False)
                    
                    # Compute FID
                    ssdiff = np.sum((mu_real - mu_synthetic) ** 2.0)
                    covmean = sqrtm(sigma_real.dot(sigma_synthetic))
                    
                    # Handle numerical errors
                    if np.iscomplexobj(covmean):
                        covmean = covmean.real
                    
                    metrics["val/fid_radimagenet"] = float(ssdiff + np.trace(sigma_real + sigma_synthetic - 2.0 * covmean))
                    
                    del real_embeddings, synth_embeddings
                    torch.cuda.empty_cache()
                
                logger.info(f"FID (RadImageNet ResNet50): {metrics['val/fid_radimagenet']:.4f}")
            except Exception as e:
                logger.warning(f"FID (RadImageNet ResNet50) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # SUBGROUP METRICS (Level 1 & 2: per sex, per ethnicity, per age group, and intersectional)
        # ================================================================
        compute_subgroup_metrics = self.args.get("compute_subgroup_metrics", False)
        
        if not compute_subgroup_metrics:
            logger.info("Subgroup metrics computation is disabled (set compute_subgroup_metrics=true in config to enable)")
        
        # Create labeled tensors on-demand for subgroup metrics (only for labeled subset to save memory)
        # Process in chunks to avoid OOM, similar to FID
        real_labeled = None
        synth_labeled = None
        sex_labels_chunk = None
        race_labels_chunk = None
        age_labels_chunk = None
        if compute_subgroup_metrics and all_real_images and min_label_len > 0:
            # Process in chunks to avoid OOM
            subgroup_chunk_size = min(2000, min_label_len)  # Use 2000 for subgroup metrics (larger than FID to get better subgroup coverage)
            if min_label_len <= subgroup_chunk_size:
                # Small enough to stack all at once
                logger.info(f"Stacking {min_label_len} labeled images for subgroup metrics...")
                synth_labeled = torch.stack(all_synthetic_images[:min_label_len]).to(device)
                real_labeled = torch.stack(all_real_images[:min_label_len]).to(device)
                sex_labels_chunk = sex_labels[:min_label_len] if sex_labels is not None else None
                race_labels_chunk = race_labels[:min_label_len] if race_labels is not None else None
                age_labels_chunk = age_labels[:min_label_len] if age_labels is not None else None
            else:
                # Too large - process first chunk and warn
                logger.warning(f"Dataset too large for subgroup metrics ({min_label_len} labeled images). Processing first {subgroup_chunk_size} images for subgroup metrics.")
                synth_labeled = torch.stack(all_synthetic_images[:subgroup_chunk_size]).to(device)
                real_labeled = torch.stack(all_real_images[:subgroup_chunk_size]).to(device)
                sex_labels_chunk = sex_labels[:subgroup_chunk_size] if sex_labels is not None else None
                race_labels_chunk = race_labels[:subgroup_chunk_size] if race_labels is not None else None
                age_labels_chunk = age_labels[:subgroup_chunk_size] if age_labels is not None else None
        
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                # Clear cache before FID computation to avoid OOM
                torch.cuda.empty_cache()
                logger.info("Computing FID per subgroup (Level 1: per sex, per ethnicity, per age group)...")
                subgroup_fids = self.metrics_runner.similarity.compute_fid_per_subgroup(
                    real_labeled, synth_labeled,
                    sex_labels=sex_labels_chunk,
                    race_labels=race_labels_chunk,
                    age_labels=age_labels_chunk,
                    batch_size=batch_size,
                    use_radimagenet=False  # Use Inception v3 for Level 1
                )
                
                # Add subgroup FIDs to metrics dictionary
                for group_type, group_results in subgroup_fids.items():
                    for subgroup_name, fid_value in group_results.items():
                        metric_key = f"val/fid_subgroup_{group_type}_{subgroup_name}"
                        metrics[metric_key] = float(fid_value)
                        logger.info(f"  FID ({group_type}: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup FID computation complete ({sum(len(v) for v in subgroup_fids.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup FID (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # FID (RadImageNet) per subgroup (Level 1)
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                # Clear cache before FID computation to avoid OOM
                torch.cuda.empty_cache()
                logger.info("Computing FID (RadImageNet) per subgroup (Level 1)...")
                subgroup_fids_radimagenet = self.metrics_runner.similarity.compute_fid_per_subgroup(
                    real_labeled, synth_labeled,
                    sex_labels=sex_labels_chunk,
                    race_labels=race_labels_chunk,
                    age_labels=age_labels_chunk,
                    batch_size=batch_size,
                    use_radimagenet=True  # Use RadImageNet for Level 1
                )
                
                # Add subgroup FIDs to metrics dictionary
                for group_type, group_results in subgroup_fids_radimagenet.items():
                    for subgroup_name, fid_value in group_results.items():
                        metric_key = f"val/fid_radimagenet_subgroup_{group_type}_{subgroup_name}"
                        metrics[metric_key] = float(fid_value)
                        logger.info(f"  FID RadImageNet ({group_type}: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup FID (RadImageNet) computation complete ({sum(len(v) for v in subgroup_fids_radimagenet.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup FID RadImageNet (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # INTERSECTIONAL SUBGROUP FID METRICS (Level 2: age group x ethnicity x sex)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                # Clear cache before FID computation to avoid OOM
                torch.cuda.empty_cache()
                logger.info("Computing FID per intersectional subgroup (Level 2: age group x ethnicity x sex)...")
                intersectional_fids = self.metrics_runner.similarity.compute_fid_per_intersectional_subgroup(
                    real_labeled, synth_labeled,
                    sex_labels=sex_labels_chunk,
                    race_labels=race_labels_chunk,
                    age_labels=age_labels_chunk,
                    batch_size=batch_size,
                    use_radimagenet=False  # Use Inception v3 for Level 2
                )
                
                # Add intersectional FIDs to metrics dictionary
                for subgroup_name, fid_value in intersectional_fids.items():
                    metric_key = f"val/fid_intersectional_{subgroup_name}"
                    metrics[metric_key] = float(fid_value)
                    logger.info(f"  FID (intersectional: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup FID computation complete ({len(intersectional_fids)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup FID (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # FID (RadImageNet) per intersectional subgroup (Level 2)
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                # Clear cache before FID computation to avoid OOM
                torch.cuda.empty_cache()
                logger.info("Computing FID (RadImageNet) per intersectional subgroup (Level 2)...")
                intersectional_fids_radimagenet = self.metrics_runner.similarity.compute_fid_per_intersectional_subgroup(
                    real_labeled, synth_labeled,
                    sex_labels=sex_labels_chunk,
                    race_labels=race_labels_chunk,
                    age_labels=age_labels_chunk,
                    batch_size=batch_size,
                    use_radimagenet=True  # Use RadImageNet for Level 2
                )
                
                # Add intersectional FIDs to metrics dictionary
                for subgroup_name, fid_value in intersectional_fids_radimagenet.items():
                    metric_key = f"val/fid_radimagenet_intersectional_{subgroup_name}"
                    metrics[metric_key] = float(fid_value)
                    logger.info(f"  FID RadImageNet (intersectional: {subgroup_name}): {fid_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup FID (RadImageNet) computation complete ({len(intersectional_fids_radimagenet)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup FID RadImageNet (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # BioViL Similarity
        if biovil_sims:
            metrics["val/biovil_similarity"] = float(np.mean(biovil_sims))
            logger.info(f"BioViL Similarity: {metrics['val/biovil_similarity']:.4f}")
        
        # MS-SSIM
        if msssim_vals:
            metrics["val/ms_ssim"] = float(np.mean(msssim_vals))
            logger.info(f"MS-SSIM: {metrics['val/ms_ssim']:.4f}")
        
        # ================================================================
        # SUBGROUP MS-SSIM METRICS (Level 1: per sex, per ethnicity, per age group)
        # Process in smaller batches to avoid OOM when indexing large tensors
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                logger.info("Computing MS-SSIM per subgroup (Level 1: per sex, per ethnicity, per age group)...")
                # Process in smaller batches to avoid OOM when creating masks and indexing
                # Instead of passing the full stacked tensors, process subgroup by subgroup
                # This avoids creating large boolean masks on large tensors
                ms_ssim_subgroup_batch_size = min(500, synth_labeled.shape[0])  # Process up to 500 images at a time
                num_labeled = synth_labeled.shape[0]
                
                # Collect results per subgroup
                subgroup_ms_ssim = {"sex": {}, "race": {}, "age": {}}
                
                # Process in batches to avoid OOM
                for batch_start in range(0, num_labeled, ms_ssim_subgroup_batch_size):
                    batch_end = min(batch_start + ms_ssim_subgroup_batch_size, num_labeled)
                    batch_synth = synth_labeled[batch_start:batch_end]
                    batch_real = real_labeled[batch_start:batch_end]
                    
                    # Get corresponding label chunks
                    batch_sex = sex_labels_chunk[batch_start:batch_end] if sex_labels_chunk is not None else None
                    batch_race = race_labels_chunk[batch_start:batch_end] if race_labels_chunk is not None else None
                    batch_age = age_labels_chunk[batch_start:batch_end] if age_labels_chunk is not None else None
                    
                    # Compute MS-SSIM for this batch
                    try:
                        batch_subgroup_ms_ssim = self.metrics_runner.similarity.compute_ms_ssim_per_subgroup(
                            batch_real, batch_synth,
                            sex_labels=batch_sex,
                            race_labels=batch_race,
                            age_labels=batch_age
                        )
                        
                        # Aggregate results (average across batches for each subgroup)
                        for group_type, group_results in batch_subgroup_ms_ssim.items():
                            for subgroup_name, ms_ssim_value in group_results.items():
                                if subgroup_name not in subgroup_ms_ssim[group_type]:
                                    subgroup_ms_ssim[group_type][subgroup_name] = []
                                subgroup_ms_ssim[group_type][subgroup_name].append(ms_ssim_value)
                    except Exception as e:
                        logger.warning(f"MS-SSIM subgroup batch {batch_start}-{batch_end} failed: {e}")
                    
                    # Free batch tensors
                    del batch_synth, batch_real
                    torch.cuda.empty_cache()
                
                # Average results across batches for each subgroup
                for group_type in subgroup_ms_ssim:
                    for subgroup_name in list(subgroup_ms_ssim[group_type].keys()):
                        values = subgroup_ms_ssim[group_type][subgroup_name]
                        if values:
                            avg_value = np.mean([v for v in values if not np.isnan(v)])
                            subgroup_ms_ssim[group_type][subgroup_name] = avg_value if not np.isnan(avg_value) else np.nan
                        else:
                            del subgroup_ms_ssim[group_type][subgroup_name]
                
                # Add subgroup MS-SSIM to metrics dictionary
                for group_type, group_results in subgroup_ms_ssim.items():
                    for subgroup_name, ms_ssim_value in group_results.items():
                        if not np.isnan(ms_ssim_value):
                            metric_key = f"val/ms_ssim_subgroup_{group_type}_{subgroup_name}"
                            metrics[metric_key] = float(ms_ssim_value)
                            logger.info(f"  MS-SSIM ({group_type}: {subgroup_name}): {ms_ssim_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup MS-SSIM computation complete ({sum(len(v) for v in subgroup_ms_ssim.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup MS-SSIM (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # INTERSECTIONAL SUBGROUP MS-SSIM METRICS (Level 2: age group x ethnicity x sex)
        # Process in smaller batches to avoid OOM when indexing large tensors
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                logger.info("Computing MS-SSIM per intersectional subgroup (Level 2: age group x ethnicity x sex)...")
                # Process in smaller batches to avoid OOM when creating masks and indexing
                ms_ssim_intersectional_batch_size = min(500, synth_labeled.shape[0])  # Process up to 500 images at a time
                num_labeled = synth_labeled.shape[0]
                
                # Collect results per intersectional subgroup
                intersectional_ms_ssim = {}
                
                # Process in batches to avoid OOM
                for batch_start in range(0, num_labeled, ms_ssim_intersectional_batch_size):
                    batch_end = min(batch_start + ms_ssim_intersectional_batch_size, num_labeled)
                    batch_synth = synth_labeled[batch_start:batch_end]
                    batch_real = real_labeled[batch_start:batch_end]
                    
                    # Get corresponding label chunks
                    batch_sex = sex_labels_chunk[batch_start:batch_end] if sex_labels_chunk is not None else None
                    batch_race = race_labels_chunk[batch_start:batch_end] if race_labels_chunk is not None else None
                    batch_age = age_labels_chunk[batch_start:batch_end] if age_labels_chunk is not None else None
                    
                    # Compute MS-SSIM for this batch
                    try:
                        batch_intersectional_ms_ssim = self.metrics_runner.similarity.compute_ms_ssim_per_intersectional_subgroup(
                            batch_real, batch_synth,
                            sex_labels=batch_sex,
                            race_labels=batch_race,
                            age_labels=batch_age
                        )
                        
                        # Aggregate results (average across batches for each intersectional subgroup)
                        for subgroup_name, ms_ssim_value in batch_intersectional_ms_ssim.items():
                            if subgroup_name not in intersectional_ms_ssim:
                                intersectional_ms_ssim[subgroup_name] = []
                            intersectional_ms_ssim[subgroup_name].append(ms_ssim_value)
                    except Exception as e:
                        logger.warning(f"MS-SSIM intersectional batch {batch_start}-{batch_end} failed: {e}")
                    
                    # Free batch tensors
                    del batch_synth, batch_real
                    torch.cuda.empty_cache()
                
                # Average results across batches for each intersectional subgroup
                for subgroup_name in list(intersectional_ms_ssim.keys()):
                    values = intersectional_ms_ssim[subgroup_name]
                    if values:
                        avg_value = np.mean([v for v in values if not np.isnan(v)])
                        intersectional_ms_ssim[subgroup_name] = avg_value if not np.isnan(avg_value) else np.nan
                    else:
                        del intersectional_ms_ssim[subgroup_name]
                
                # Add intersectional MS-SSIM to metrics dictionary
                for subgroup_name, ms_ssim_value in intersectional_ms_ssim.items():
                    if not np.isnan(ms_ssim_value):
                        metric_key = f"val/ms_ssim_intersectional_{subgroup_name}"
                        metrics[metric_key] = float(ms_ssim_value)
                        logger.info(f"  MS-SSIM (intersectional: {subgroup_name}): {ms_ssim_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup MS-SSIM computation complete ({len(intersectional_ms_ssim)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup MS-SSIM (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # SUBGROUP BIOVIL SIMILARITY METRICS (Level 1: per sex, per ethnicity, per age group)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                logger.info("Computing BioViL similarity per subgroup (Level 1: per sex, per ethnicity, per age group)...")
                subgroup_biovil = self.metrics_runner.similarity.compute_biovil_similarity_per_subgroup(
                    real_labeled, synth_labeled,
                    sex_labels=sex_labels_chunk,
                    race_labels=race_labels_chunk,
                    age_labels=age_labels_chunk,
                    batch_size=batch_size
                )
                
                # Add subgroup BioViL similarity to metrics dictionary
                for group_type, group_results in subgroup_biovil.items():
                    for subgroup_name, biovil_value in group_results.items():
                        metric_key = f"val/biovil_similarity_subgroup_{group_type}_{subgroup_name}"
                        metrics[metric_key] = float(biovil_value)
                        logger.info(f"  BioViL Similarity ({group_type}: {subgroup_name}): {biovil_value:.4f}")
                
                logger.info(f"✓ Level 1 subgroup BioViL similarity computation complete ({sum(len(v) for v in subgroup_biovil.values())} subgroups)")
            except Exception as e:
                logger.warning(f"Subgroup BioViL similarity (Level 1) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ================================================================
        # INTERSECTIONAL SUBGROUP BIOVIL SIMILARITY METRICS (Level 2: age group x ethnicity x sex)
        # ================================================================
        if compute_subgroup_metrics and hasattr(self.metrics_runner, 'similarity') and real_labeled is not None:
            try:
                logger.info("Computing BioViL similarity per intersectional subgroup (Level 2: age group x ethnicity x sex)...")
                intersectional_biovil = self.metrics_runner.similarity.compute_biovil_similarity_per_intersectional_subgroup(
                    real_labeled, synth_labeled,
                    sex_labels=sex_labels_chunk,
                    race_labels=race_labels_chunk,
                    age_labels=age_labels_chunk,
                    batch_size=batch_size
                )
                
                # Add intersectional BioViL similarity to metrics dictionary
                for subgroup_name, biovil_value in intersectional_biovil.items():
                    metric_key = f"val/biovil_similarity_intersectional_{subgroup_name}"
                    metrics[metric_key] = float(biovil_value)
                    logger.info(f"  BioViL Similarity (intersectional: {subgroup_name}): {biovil_value:.4f}")
                
                logger.info(f"✓ Level 2 intersectional subgroup BioViL similarity computation complete ({len(intersectional_biovil)} subgroups)")
            except Exception as e:
                logger.warning(f"Intersectional subgroup BioViL similarity (Level 2) computation error: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # Clean up subgroup metrics tensors if they're no longer needed
        # Note: We keep them for text alignment metrics below, but clean up after all metrics are done
        # (cleanup happens at the end of this function)
        
        # ================================================================
        # AGGREGATE TEXT ALIGNMENT METRICS
        # Use enhanced methods from validation_metrics.py
        # ================================================================
        
        # Use the enhanced methods from validation_metrics.py for text alignment metrics
        # These methods handle all the filtering and mapping logic internally
        if hasattr(self.metrics_runner, 'text_alignment'):
            ta = self.metrics_runner.text_alignment
            
            # Disease AUROC - use enhanced method with batching
            if disease_labels is not None and synth_labeled is not None:
                try:
                    logger.info("Computing disease AUROC using enhanced method from validation_metrics.py (with batching)...")
                    # CRITICAL: Ensure disease_labels matches synth_labeled size exactly
                    # synth_labeled has shape [N, C, H, W] where N is the number of labeled images
                    num_labeled_images = synth_labeled.shape[0]
                    if disease_labels.shape[0] != num_labeled_images:
                        logger.warning(
                            f"Disease labels shape mismatch: disease_labels has {disease_labels.shape[0]} samples, "
                            f"but synth_labeled has {num_labeled_images} images. Truncating disease_labels to match."
                        )
                        disease_labels = disease_labels[:num_labeled_images]
                    
                    # Verify shapes match
                    if disease_labels.shape[0] != num_labeled_images:
                        raise ValueError(
                            f"Shape mismatch after truncation: disease_labels has {disease_labels.shape[0]} samples, "
                            f"synth_labeled has {num_labeled_images} images"
                        )
                    
                    disease_aurocs = ta.compute_disease_auroc(synth_labeled, disease_labels, batch_size=batch_size)
                    for disease_name, auroc in disease_aurocs.items():
                        if disease_name == "mean_auroc":
                            metrics["val/mean_auroc"] = float(auroc) if not np.isnan(auroc) else np.nan
                        else:
                            metrics[f"val/{disease_name}"] = float(auroc) if not np.isnan(auroc) else np.nan
                    logger.info(f"Disease AUROC computation complete. Mean AUROC: {disease_aurocs.get('mean_auroc', np.nan):.4f}")
                except Exception as e:
                    import traceback
                    logger.error(f"Disease AUROC error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Sex Accuracy - use enhanced method with batching
            if sex_labels is not None and synth_labeled is not None:
                try:
                    logger.info("Computing sex accuracy using enhanced method from validation_metrics.py (with batching)...")
                    # Ensure sex_labels matches synth_labeled size
                    num_labeled_images = synth_labeled.shape[0]
                    if sex_labels.shape[0] != num_labeled_images:
                        logger.warning(f"Sex labels shape mismatch: truncating to {num_labeled_images}")
                        sex_labels = sex_labels[:num_labeled_images]
                    sex_acc = ta.compute_sex_accuracy(synth_labeled, sex_labels, batch_size=batch_size)
                    if not np.isnan(sex_acc):
                        metrics["val/sex_accuracy"] = float(sex_acc)
                        logger.info(f"Sex Accuracy: {sex_acc:.4f}")
                    else:
                        logger.warning("Sex accuracy computation returned NaN")
                except Exception as e:
                    logger.warning(f"Sex accuracy error: {e}")
            
            # Race Accuracy - use enhanced method with batching
            if race_labels is not None and synth_labeled is not None:
                try:
                    logger.info("Computing race accuracy using enhanced method from validation_metrics.py (with batching)...")
                    # Ensure race_labels matches synth_labeled size
                    num_labeled_images = synth_labeled.shape[0]
                    if race_labels.shape[0] != num_labeled_images:
                        logger.warning(f"Race labels shape mismatch: truncating to {num_labeled_images}")
                        race_labels = race_labels[:num_labeled_images]
                    race_acc = ta.compute_race_accuracy(synth_labeled, race_labels, batch_size=batch_size)
                    if not np.isnan(race_acc):
                        metrics["val/race_accuracy"] = float(race_acc)
                        logger.info(f"Race Accuracy: {race_acc:.4f}")
                    else:
                        logger.warning("Race accuracy computation returned NaN")
                except Exception as e:
                    logger.warning(f"Race accuracy error: {e}")
            
            # Age RMSE - use enhanced method with batching
            if age_labels is not None and synth_labeled is not None:
                try:
                    logger.info("Computing age RMSE using enhanced method from validation_metrics.py (with batching)...")
                    # Ensure age_labels matches synth_labeled size
                    num_labeled_images = synth_labeled.shape[0]
                    if age_labels.shape[0] != num_labeled_images:
                        logger.warning(f"Age labels shape mismatch: truncating to {num_labeled_images}")
                        age_labels = age_labels[:num_labeled_images]
                    age_rmse = ta.compute_age_rmse(synth_labeled, age_labels, batch_size=batch_size)
                    metrics["val/age_rmse"] = float(age_rmse)
                    logger.info(f"Age RMSE: {age_rmse:.4f}")
                except Exception as e:
                    logger.warning(f"Age RMSE error: {e}")
                
                # Age Bin Accuracy - use enhanced method with batching
                try:
                    logger.info("Computing age bin accuracy using enhanced method from validation_metrics.py (with batching)...")
                    # age_labels already truncated above
                    age_bin_acc = ta.compute_age_bin_accuracy(
                        synth_labeled, age_labels, age_bins=self.metrics_runner.age_bins, batch_size=batch_size
                    )
                    if not np.isnan(age_bin_acc):
                        metrics["val/age_bin_accuracy"] = float(age_bin_acc)
                        logger.info(f"Age bin accuracy: {age_bin_acc:.4f}")
                    else:
                        logger.warning("Age bin accuracy computation returned NaN")
                except Exception as e:
                    logger.warning(f"Age bin accuracy error: {e}")
        
        logger.info("="*60)
        logger.info(f"Total metrics computed: {len(metrics)}")
        logger.info("="*60)
        
        # Cleanup: free labeled tensors if they were created
        if real_labeled is not None:
            del real_labeled
        if synth_labeled is not None:
            del synth_labeled
        torch.cuda.empty_cache()
        
        return metrics

    def run_validation(self, global_step: int) -> Dict:
        """
        Hybrid validation approach:
        - Distributed: Image generation (parallelized across all GPUs) - SKIPPED if load_images_from_dir is set
        - Centralized: Metrics computation (only rank 0, no collective ops)
        
        This avoids all distributed synchronization issues while keeping speed benefits.
        
        If load_images_from_dir is set, skips generation and loads pre-generated images instead.
        """
        if self.accelerator.is_main_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running validation at step {global_step}")
            if self.load_images_from_dir:
                logger.info(f"Mode: Loading pre-generated images from {self.load_images_from_dir}")
            else:
                logger.info(f"Mode: Generating images")
            logger.info(f"{'='*60}")

        # Initialize metrics runner ONLY on rank 0
        if self.accelerator.is_main_process:
            self._ensure_metrics_runner()
            if self.metrics_runner is not None:
                self._move_metrics_models_to_gpu()

        # Only set models to eval if we're generating (not needed for loading)
        if not self.load_images_from_dir:
            self.unet.eval()
            self.text_encoder.eval()
            if self.hcn is not None:
                self.hcn.eval()

        try:
            # ==============================================================
            # PHASE 1: GENERATE OR SKIP (depending on load_images_from_dir)
            # ==============================================================
            if self.load_images_from_dir:
                # Skip generation - will load from directory instead
                if self.accelerator.is_main_process:
                    logger.info(f"Rank 0: Skipping generation, will load images from {self.load_images_from_dir}")
                else:
                    logger.info(f"[Rank {self.accelerator.process_index}] Skipping generation (loading mode)")
            else:
                # Normal generation flow
                unwrapped_unet = self.accelerator.unwrap_model(self.unet).eval()
                unwrapped_text_encoder = self.accelerator.unwrap_model(self.text_encoder).eval()
                unwrapped_vae = self.accelerator.unwrap_model(self.vae).eval().to(dtype=torch.float32)

                logger.info(f"[Rank {self.accelerator.process_index}] Starting distributed generation")
                
                all_synthetic_images, all_real_images, all_labels = self._generate_validation_images(
                    unwrapped_unet, unwrapped_text_encoder, unwrapped_vae, global_step
                )
                
                logger.info(f"[Rank {self.accelerator.process_index}] Generated {len(all_synthetic_images)} images")

                # ==============================================================
                # PHASE 2: SAVE TO DISK (all ranks save their portion)
                # ==============================================================
                self._save_validation_images(
                    all_synthetic_images, all_real_images, all_labels, global_step
                )
                
                logger.info(f"[Rank {self.accelerator.process_index}] Saved images to disk")

            # ==============================================================
            # PHASE 2.5: VERIFY ALL GPUS COMPLETED (rank 0 checks)
            # ==============================================================
            if self.accelerator.is_main_process:
                # Wait for all GPUs to finish saving by checking for "done" markers
                import time
                if self.args.get("test_mode", False):
                    images_dir_name = "test_images"
                elif self.args.get("train_mode", False):
                    images_dir_name = "train_images"
                elif self.args.get("gt_mode", False):
                    images_dir_name = "GT_images"
                else:
                    images_dir_name = "validation_images"
                validation_images_dir = Path(self.args["output_dir"]) / images_dir_name / f"step_{global_step}"
                expected_gpu_dirs = [validation_images_dir / f"gpu_{i}" for i in range(self.accelerator.num_processes)]
                max_wait_seconds = 600
                poll_interval = 5
                waited = 0

                logger.info(f"Rank 0: Waiting for done markers from all GPUs for step {global_step}...")
                while waited < max_wait_seconds:
                    done_info = []
                    all_done = True
                    for i, d in enumerate(expected_gpu_dirs):
                        done_file = d / "done.txt"
                        if done_file.exists():
                            done_info.append(f"gpu_{i}:done")
                        else:
                            all_done = False
                            done_info.append(f"gpu_{i}:pending")
                    logger.info(f"Done status: {', '.join(done_info)} (waited {waited}s)")

                    if all_done:
                        logger.info(f"All GPUs reported done for step {global_step}, proceeding to metrics.")
                        break

                    time.sleep(poll_interval)
                    waited += poll_interval

                if waited >= max_wait_seconds:
                    logger.warning(
                        f"Timeout waiting for all GPUs to finish saving images for step {global_step}. "
                        f"Proceeding with whatever images are present; metrics may use fewer than the "
                        f"expected {self.accelerator.num_processes} * images_per_gpu."
                    )

            # ==============================================================
            # PHASE 3: CENTRALIZED METRICS (rank 0 only, no collectives)
            # ==============================================================
            metrics = {}
            num_images = 0
            
            if self.accelerator.is_main_process:
                logger.info("Rank 0: Loading images from disk for metrics computation")
                
                # Load images - _load_validation_images_from_disk handles both cases correctly
                # It constructs the step-specific path whether using load_images_from_dir or standard location
                loaded_synth, loaded_real, loaded_labels = self._load_validation_images_from_disk(global_step)
                
                if loaded_synth and len(loaded_synth) > 0:
                    num_images = len(loaded_synth)
                    logger.info(f"Rank 0: Loaded {num_images} total images")
                    
                    # Compute metrics on loaded images (batched to avoid OOM)
                    metrics = self._compute_metrics_batched(
                        loaded_synth, loaded_real, loaded_labels, global_step
                    )
                    
                    logger.info("Rank 0: Metrics computation complete")
                    logger.info("="*60)
                    for key, value in metrics.items():
                        logger.info(f"{key}: {value:.4f}")
                    logger.info("="*60)
                else:
                    logger.warning("Rank 0: No images found, skipping metrics")
            else:
                logger.info(f"[Rank {self.accelerator.process_index}] Waiting for rank 0 to finish metrics")

            # ==============================================================
            # PHASE 4: FINAL SYNC & CLEANUP
            # ==============================================================
            logger.info(f"[Rank {self.accelerator.process_index}] Validation complete")

            if self.accelerator.is_main_process and self.metrics_runner is not None:
                self._move_metrics_models_to_cpu()

            torch.cuda.empty_cache()
            
            # Return metrics dict with num_images added for manifest tracking
            # Store num_images in metrics dict so it's accessible to callers
            if self.accelerator.is_main_process:
                metrics["_num_images"] = num_images
            
            return metrics

        except Exception as e:
            logger.error(f"[Rank {self.accelerator.process_index}] Validation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}


def _wait_for_new_checkpoint_task(sync_dir: Path, last_step: int, check_interval: int, accelerator: Accelerator):
    """
    Poll for a new checkpoint assignment written by the main process.
    Returns (step, checkpoint_dir_str) or (None, None) if nothing new is found
    within roughly check_interval seconds.
    """
    cmd_file = sync_dir / "current_checkpoint.json"
    waited = 0
    sleep_interval = 1

    while waited < check_interval:
        if cmd_file.exists():
            try:
                with open(cmd_file, "r") as f:
                    data = json.load(f)
                step = int(data.get("step", -1))
                checkpoint_dir = data.get("checkpoint_dir")
            except Exception:
                step = -1
                checkpoint_dir = None

            if step > last_step and checkpoint_dir:
                # All ranks will use this same step/checkpoint_dir
                return step, checkpoint_dir

        time.sleep(sleep_interval)
        waited += sleep_interval

    # No new assignment within this interval
    return None, None


def main():
    """Main validation monitoring loop."""
    parser = argparse.ArgumentParser(description="Validation monitoring script")
    parser.add_argument("--config_file", type=str, required=True, help="Path to training config file")
    parser.add_argument("--check_interval", type=int, default=60, help="Check for new checkpoints every N seconds")
    parser.add_argument("--manifest_file", type=str, default="validation_manifest.json", help="Manifest file to track validated checkpoints")
    parser.add_argument("--training_run_id", type=str, default=None, help="Training run ID to resume logging (for wandb)")
    parser.add_argument("--training_run_name", type=str, default=None, help="Training run name to resume logging")
    parser.add_argument("--load_images_from_dir", action="store_true", help="Load pre-generated images from validation_images directory instead of generating them. Images should be in output_dir/validation_images/step_X/ subdirectories.")
    parser.add_argument("--stop_at_step", type=int, default=None, help="If set, stop validation after validating this global step (inclusive).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for validation image generation. Different seeds produce different samples; same seed gives reproducible runs. Can also be set in config as validation_seed.")
    # Test mode arguments
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode: validate specific checkpoints")
    parser.add_argument("--checkpoints", type=str, default=None, help="Comma-separated list of checkpoint numbers to validate in test mode (e.g., '30000,50000,70000')")
    parser.add_argument("--test_continuous", action="store_true", help="In test mode: continuously monitor for checkpoints matching the specified list. If not set, only processes existing checkpoints once.")
    # Train mode arguments
    parser.add_argument("--train_mode", action="store_true", help="Run in train mode: validate specific checkpoints on train dataset")
    parser.add_argument("--train_continuous", action="store_true", help="In train mode: continuously monitor for checkpoints matching the specified list. If not set, only processes existing checkpoints once.")
    # GT mode arguments
    parser.add_argument("--gt_mode", action="store_true", help="Run in GT mode: generate and validate Ground Truth set from specified checkpoints")
    parser.add_argument("--gt_continuous", action="store_true", help="In GT mode: continuously monitor for checkpoints matching the specified list. If not set, only processes existing checkpoints once.")
    args = parser.parse_args()

    # Load training config
    config = load_config(args.config_file)
    if config is None:
        raise ValueError(f"Failed to load config from {args.config_file}")
    
    # Determine if we're in test mode, train mode, or GT mode (mutually exclusive)
    test_mode = args.test_mode
    train_mode = args.train_mode
    gt_mode = args.gt_mode
    
    if sum([test_mode, train_mode, gt_mode]) > 1:
        raise ValueError("--test_mode, --train_mode, and --gt_mode cannot be used together. Please choose one.")
    
    if test_mode:
        if args.checkpoints is None:
            raise ValueError("--checkpoints must be provided when using --test_mode")
        # Parse checkpoint numbers
        try:
            checkpoint_numbers = [int(c.strip()) for c in args.checkpoints.split(",") if c.strip()]
            if not checkpoint_numbers:
                raise ValueError("No valid checkpoint numbers provided")
        except ValueError as e:
            raise ValueError(f"Invalid checkpoint numbers format: {args.checkpoints}. Expected comma-separated integers. Error: {e}")
        
        # Extract test_dir from config file (where test images are located, equivalent to validation_images_dir)
        test_dir = config.get("test_dir")
        if test_dir is None:
            raise ValueError("test_dir must be specified in config file when using --test_mode")
        
        # Override validation_images_dir with test_dir for test mode
        # This makes test_dir work like validation_images_dir - pointing to the test dataset directory
        config["validation_images_dir"] = test_dir
        # Also override validation_csv to test_dir for test mode
        # This is critical because when use_wds_dataset=True, validation_csv is checked first (line 469)
        # We always override it in test mode to ensure we use the test dataset, not validation
        config["validation_csv"] = test_dir
        
        # Mark that we're in test mode (for saving images to test_images directory)
        config["test_mode"] = True
        
        print(f"Test mode: Using test directory from config: {test_dir}")
        print(f"Test mode: Overriding validation_images_dir to: {test_dir}")
        
        # Override manifest file to test_manifest.json
        if args.manifest_file == "validation_manifest.json":  # Only override if using default
            args.manifest_file = "test_manifest.json"
        
        print(f"Test mode: Will validate checkpoints: {checkpoint_numbers}")
        print(f"Test mode: Using manifest file: {args.manifest_file}")
    
    if train_mode:
        if args.checkpoints is None:
            raise ValueError("--checkpoints must be provided when using --train_mode")
        # Parse checkpoint numbers
        try:
            checkpoint_numbers = [int(c.strip()) for c in args.checkpoints.split(",") if c.strip()]
            if not checkpoint_numbers:
                raise ValueError("No valid checkpoint numbers provided")
        except ValueError as e:
            raise ValueError(f"Invalid checkpoint numbers format: {args.checkpoints}. Expected comma-separated integers. Error: {e}")
        
        # Extract train_dir from config file (where train images are located, equivalent to validation_images_dir)
        train_dir = config.get("train_dir")
        if train_dir is None:
            raise ValueError("train_dir must be specified in config file when using --train_mode")
        
        # Override validation_images_dir with train_dir for train mode
        # This makes train_dir work like validation_images_dir - pointing to the train dataset directory
        config["validation_images_dir"] = train_dir
        # Also override validation_csv to train_dir for train mode
        # This is critical because when use_wds_dataset=True, validation_csv is checked first (line 469)
        # We always override it in train mode to ensure we use the train dataset, not validation
        config["validation_csv"] = train_dir
        
        # Mark that we're in train mode (for saving images to train_images directory)
        config["train_mode"] = True
        
        print(f"Train mode: Using train directory from config: {train_dir}")
        print(f"Train mode: Overriding validation_images_dir to: {train_dir}")
        print(f"Train mode: Will validate checkpoints: {checkpoint_numbers}")
        
        # Override manifest file to train_manifest.json
        if args.manifest_file == "validation_manifest.json":  # Only override if using default
            args.manifest_file = "train_manifest.json"
        
        print(f"Train mode: Using manifest file: {args.manifest_file}")
    
    if gt_mode:
        if args.checkpoints is None:
            raise ValueError("--checkpoints must be provided when using --gt_mode")
        # Parse checkpoint numbers
        try:
            checkpoint_numbers = [int(c.strip()) for c in args.checkpoints.split(",") if c.strip()]
            if not checkpoint_numbers:
                raise ValueError("No valid checkpoint numbers provided")
        except ValueError as e:
            raise ValueError(f"Invalid checkpoint numbers format: {args.checkpoints}. Expected comma-separated integers. Error: {e}")
        
        # Extract GT_dir from config file (where GT images are located, equivalent to validation_images_dir)
        gt_dir = config.get("GT_dir") or config.get("gt_dir")
        if gt_dir is None:
            raise ValueError("GT_dir or gt_dir must be specified in config file when using --gt_mode")
        
        # Override validation_images_dir with GT_dir for GT mode
        # This makes GT_dir work like validation_images_dir - pointing to the GT dataset directory
        config["validation_images_dir"] = gt_dir
        # Also override validation_csv to GT_dir for GT mode
        # This is critical because when use_wds_dataset=True, validation_csv is checked first (line 469)
        # We always override it in GT mode to ensure we use the GT dataset, not validation
        config["validation_csv"] = gt_dir
        
        # Mark that we're in GT mode (for saving images to GT_images directory)
        config["gt_mode"] = True
        
        print(f"GT mode: Using GT directory from config: {gt_dir}")
        print(f"GT mode: Overriding validation_images_dir to: {gt_dir}")
        print(f"GT mode: Will validate checkpoints: {checkpoint_numbers}")
        
        # Override manifest file to GT_manifest.json
        if args.manifest_file == "validation_manifest.json":  # Only override if using default
            args.manifest_file = "GT_manifest.json"
        
        print(f"GT mode: Using manifest file: {args.manifest_file}")
    
    # Add load_images_from_dir flag and stop_at_step to config if provided
    # If load_images_from_dir is set, we monitor step directories instead of checkpoints
    load_images_from_dir = args.load_images_from_dir
    if load_images_from_dir:
        config["load_images_from_dir"] = True
        # Use print before Accelerator is initialized (logger requires Accelerator)
        print(f"Image loading mode: Will monitor and load pre-generated images from validation_images directory")

    # Determine stop_at_step: priority is max_actual_train_steps from config,
    # then stop_at_step from command line, then max_train_steps from config
    stop_at_step = None
    if config.get("max_actual_train_steps") is not None:
        stop_at_step = config.get("max_actual_train_steps")
        print(f"Using max_actual_train_steps={stop_at_step} from config as stop_at_step")
    elif args.stop_at_step is not None:
        stop_at_step = args.stop_at_step
        print(f"Using stop_at_step={stop_at_step} from command line")
    elif config.get("max_train_steps") is not None:
        stop_at_step = config.get("max_train_steps")
        print(f"Using max_train_steps={stop_at_step} from config as stop_at_step")
    
    if stop_at_step is not None:
        config["stop_at_step"] = int(stop_at_step)

    # Seed for validation image generation (enables reproducible or varied runs)
    if args.seed is not None:
        config["validation_seed"] = args.seed
        print(f"Validation seed set to {args.seed} (use different --seed for different runs)")

    # Initialize accelerator
    # Ensure GPU is used if available (validation requires GPU for generation, unless loading from directory)
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        log_with=config.get("report_to", "wandb"),
    )
    
    # Now we can use logger after Accelerator is initialized
    if load_images_from_dir:
        logger.info(f"Image loading mode: Will monitor and load pre-generated images from validation_images directory")
    
    # Verify GPU is available (only required if generating images, not if loading from directory)
    # load_images_from_dir is already set above, but get from config to be safe
    if not load_images_from_dir:
        load_images_from_dir = config.get("load_images_from_dir", False)
    if not load_images_from_dir:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available! Validation requires GPU for image generation.")
            logger.error("Please run this script in a SLURM job with GPU allocation or ensure CUDA is available.")
            logger.error("Alternatively, use --load_images_from_dir to load pre-generated images.")
            raise RuntimeError("CUDA not available - validation requires GPU for generation")
        
        if accelerator.device.type == "cpu":
            logger.error(f"Accelerator initialized on CPU (device: {accelerator.device})")
            logger.error("This script requires GPU for image generation. Please check:")
            logger.error("  1. GPU is allocated (e.g., via SLURM: --gres=gpu:1)")
            logger.error("  2. CUDA_VISIBLE_DEVICES is set correctly")
            logger.error("  3. accelerate launch is configured for GPU")
            logger.error("Alternatively, use --load_images_from_dir to load pre-generated images.")
            raise RuntimeError("Accelerator on CPU - GPU required for validation")
        
        logger.info(f"✓ Accelerator initialized on {accelerator.device} (GPU available)")
    else:
        logger.info(f"✓ Accelerator initialized on {accelerator.device}")
        logger.info("  Note: GPU not required since loading pre-generated images")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Try to load training run info to resume same run
    training_run_info_file = Path(config["output_dir"]) / "training_run_info.json"
    training_run_id = args.training_run_id
    training_run_name = args.training_run_name
    training_project = None

    if training_run_info_file.exists():
        try:
            with open(training_run_info_file, 'r') as f:
                run_info = json.load(f)
                if training_run_id is None:
                    training_run_id = run_info.get("run_id")
                if training_run_name is None:
                    training_run_name = run_info.get("run_name")
                training_project = run_info.get("project")  # Get project from training run info
                logger.info(f"Found training run info: run_id={training_run_id}, run_name={training_run_name}, project={training_project}")
        except Exception as e:
            logger.warning(f"Could not load training run info: {e}")

    # Initialize trackers for logging
    tracker_init_kwargs = {}
    if config.get("report_to") == "wandb":
        import wandb
        # Use project from training run info if available, otherwise use logging_dir or default
        wandb_project = training_project if training_project else config.get("logging_dir", "validation")
        
        # Prepare wandb init kwargs
        wandb_init_kwargs = {
            "project": wandb_project,
            "name": training_run_name if training_run_name else f"validation-{config['output_dir'].split('/')[-1]}",
            "tags": ["validation", "monitoring"],
        }

        # If we have a training run ID, try to resume it
        if training_run_id:
            try:
                logger.info(f"Attempting to resume wandb run: {training_run_id} in project: {wandb_project}")
                # Resume the run using wandb directly - MUST use the correct project from training
                wandb.init(
                    id=training_run_id,
                    resume="allow",
                    project=wandb_project,  # Use the project from training run info
                    name=training_run_name,  # Keep the same run name
                    tags=wandb_init_kwargs.get("tags", []),
                )
                logger.info(f"✓ Successfully resumed wandb run {training_run_id} in project {wandb_project}")
                
                # Define validation metrics to allow out-of-order logging
                # This allows us to log validation metrics at any step, even if training has progressed further
                # Using summary="last" allows metrics to be updated at any step
                validation_metric_names = [
                    "val/fid", "val/fid_radimagenet", "val/biovil_similarity", "val/ms_ssim", "val/mean_auroc",
                    "val/sex_accuracy", "val/race_accuracy", "val/age_rmse", "val/age_bin_accuracy",
                    "val/Atelectasis", "val/Consolidation", "val/Infiltration", 
                    "val/Pneumothorax", "val/Edema", "val/Emphysema", "val/Fibrosis",
                    "val/Effusion", "val/Pneumonia", "val/Pleural_Thickening",
                    "val/Cardiomegaly", "val/Nodule", "val/Mass", "val/Hernia",
                    "val/Lung Lesion", "val/Fracture", "val/Lung Opacity",
                    "val/Enlarged Cardiomediastinum"
                ]
                for metric_name in validation_metric_names:
                    # Define metric to allow out-of-order logging
                    # By not specifying step constraints, wandb allows logging at any step
                    # summary="last" keeps the most recent value for summary
                    try:
                        wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                    except:
                        # If step_metric doesn't work, try without it
                        try:
                            wandb.define_metric(metric_name, summary="last")
                        except:
                            # Metric might already be defined, that's OK
                            pass
                logger.info("✓ Defined validation metrics for out-of-order logging")
                
                # Initialize accelerator trackers - when wandb is already initialized,
                # accelerator will detect and use the existing run
                # Pass minimal kwargs to avoid re-initialization
                if wandb.run is not None:
                    # Wandb is already initialized, accelerator will use it
                    tracker_init_kwargs["wandb"] = {}  # Empty dict - accelerator will detect existing run
                else:
                    # Fallback: provide init kwargs (without 'project' since it's passed as project_name)
                    tracker_init_kwargs["wandb"] = {
                        "id": training_run_id,
                        "resume": "allow",
                        # Don't include "project" here - it's passed as project_name parameter
                    }
                accelerator.init_trackers(
                    project_name=wandb_project,
                    config=config,
                    init_kwargs=tracker_init_kwargs,
                )
                logger.info("✓ Accelerator trackers initialized with resumed wandb run")
            except Exception as e:
                logger.warning(f"Could not resume training run: {e}")
                logger.info("Creating new wandb run for validation")
                # Remove 'project' from wandb_init_kwargs since it's passed as project_name parameter
                wandb_kwargs_for_tracker = {k: v for k, v in wandb_init_kwargs.items() if k != "project"}
                tracker_init_kwargs["wandb"] = wandb_kwargs_for_tracker
                accelerator.init_trackers(
                    project_name=wandb_init_kwargs["project"],
                    config=config,
                    init_kwargs=tracker_init_kwargs,
                )
                # Define validation metrics for new run too
                if wandb.run is not None:
                    validation_metric_names = [
                        "val/fid", "val/biovil_similarity", "val/ms_ssim", "val/mean_auroc",
                        "val/sex_accuracy", "val/race_accuracy", "val/age_rmse",
                        "val/Atelectasis", "val/Consolidation", "val/Infiltration", 
                        "val/Pneumothorax", "val/Edema", "val/Emphysema", "val/Fibrosis",
                        "val/Effusion", "val/Pneumonia", "val/Pleural_Thickening",
                        "val/Cardiomegaly", "val/Nodule", "val/Mass", "val/Hernia",
                        "val/Lung Lesion", "val/Fracture", "val/Lung Opacity",
                        "val/Enlarged Cardiomediastinum"
                    ]
                    for metric_name in validation_metric_names:
                        wandb.define_metric(metric_name, summary="last")
        else:
            # Remove 'project' from wandb_init_kwargs since it's passed as project_name parameter
            wandb_kwargs_for_tracker = {k: v for k, v in wandb_init_kwargs.items() if k != "project"}
            tracker_init_kwargs["wandb"] = wandb_kwargs_for_tracker
            accelerator.init_trackers(
                project_name=wandb_init_kwargs["project"],
                config=config,
                init_kwargs=tracker_init_kwargs,
            )
            # Define validation metrics for new run
            if wandb.run is not None:
                validation_metric_names = [
                    "val/fid", "val/fid_radimagenet", "val/biovil_similarity", "val/ms_ssim", "val/mean_auroc",
                    "val/sex_accuracy", "val/race_accuracy", "val/age_rmse", "val/age_bin_accuracy",
                    "val/Atelectasis", "val/Consolidation", "val/Infiltration", 
                    "val/Pneumothorax", "val/Edema", "val/Emphysema", "val/Fibrosis",
                    "val/Effusion", "val/Pneumonia", "val/Pleural_Thickening",
                    "val/Cardiomegaly", "val/Nodule", "val/Mass", "val/Hernia",
                    "val/Lung Lesion", "val/Fracture", "val/Lung Opacity",
                    "val/Enlarged Cardiomediastinum"
                ]
                for metric_name in validation_metric_names:
                    wandb.define_metric(metric_name, summary="last")
    else:
        # For tensorboard or other trackers
        accelerator.init_trackers(
            project_name=config.get("logging_dir", "validation"),
            config=config,
        )

    logger.info("✓ Logging trackers initialized")

    # Initialize runner
    runner = ValidationRunner(
        accelerator=accelerator,
        args=config,
        logger=logger,
    )
    
    # NOTE: DO NOT add wait_for_everyone() here - it causes hangs
    # Rank 0 initializes metrics runner which delays it
    # The monitoring loop will naturally synchronize when all ranks
    # call load_checkpoint() and then prepare()

    logger.info("="*60)
    logger.info("Validation monitoring started")
    logger.info(f"Output directory: {config['output_dir']}")
    logger.info(f"Check interval: {args.check_interval}s")
    logger.info("="*60)

    # Prepare validation schedule helper
    validation_interval = config.get("validation_steps", None)
    if validation_interval is not None:
        try:
            validation_interval = int(validation_interval)
        except (TypeError, ValueError):
            logger.warning(f"Invalid validation_steps value '{validation_interval}', disabling interval-based schedule")
            validation_interval = None

    schedule_offsets = config.get("validation_schedule_offsets")
    schedule_min_step = config.get("validation_schedule_min_step", None)
    if schedule_min_step is not None:
        try:
            schedule_min_step = int(schedule_min_step)
        except (TypeError, ValueError):
            logger.warning(f"Invalid validation_schedule_min_step '{schedule_min_step}', ignoring minimum step constraint")
            schedule_min_step = None

    if schedule_offsets is None:
        normalized_offsets = []
    elif isinstance(schedule_offsets, list):
        normalized_offsets = []
        for offset in schedule_offsets:
            try:
                normalized_offsets.append(int(offset))
            except (TypeError, ValueError):
                logger.warning(f"Invalid validation schedule offset '{offset}' - skipping")
    else:
        try:
            normalized_offsets = [int(schedule_offsets)]
        except (TypeError, ValueError):
            logger.warning(f"Invalid validation schedule offset '{schedule_offsets}' - ignoring custom schedule")
            normalized_offsets = []

    if normalized_offsets:
        logger.info(f"Custom validation schedule enabled with base_step={validation_interval}, offsets={normalized_offsets}, min_step={schedule_min_step}")
    elif validation_interval:
        logger.info(f"Validation will run every {validation_interval} steps (no custom offsets)")
    else:
        logger.info("Validation interval not specified; all checkpoints will be validated")

    def should_validate(step: int) -> bool:
        return is_step_in_validation_schedule(step, validation_interval, normalized_offsets, schedule_min_step)

    # Main monitoring loop
    try:
        # If loading from directory, monitor step directories instead of checkpoints
        # BUT: In test mode, train mode, or GT mode, always use CheckpointMonitor to process specific checkpoints
        if load_images_from_dir and not test_mode and not train_mode and not gt_mode:
            # Initialize step image monitor
            output_dir = Path(config["output_dir"])
            # Use test_images directory in test mode, train_images in train mode, GT_images in GT mode, validation_images otherwise
            if config.get("test_mode", False):
                images_dir_name = "test_images"
            elif config.get("train_mode", False):
                images_dir_name = "train_images"
            elif config.get("gt_mode", False):
                images_dir_name = "GT_images"
            else:
                images_dir_name = "validation_images"
            images_base_dir = output_dir / images_dir_name
            
            if not images_base_dir.exists():
                logger.error(f"Validation images directory not found: {images_base_dir}")
                raise ValueError(f"Validation images directory not found: {images_base_dir}")
            
            monitor = StepImageMonitor(
                images_base_dir=str(images_base_dir),
                manifest_file=args.manifest_file,
            )
            
            logger.info("="*60)
            logger.info("Monitoring pre-generated images")
            logger.info(f"Image base directory: {images_base_dir}")
            logger.info(f"Manifest file: {args.manifest_file}")
            logger.info("="*60)
            
            # Monitoring loop for step directories
            last_validated_step = -1
            while True:
                # Optional: stop after a specific step has been validated
                stop_at_step = config.get("stop_at_step")
                if stop_at_step is not None and last_validated_step >= stop_at_step:
                    if accelerator.is_local_main_process:
                        logger.info(f"Reached stop_at_step={stop_at_step} in image-loading mode, exiting validation monitor.")
                    break

                # Check for new step directories
                new_steps = monitor.get_new_steps()
                
                if new_steps:
                    logger.info(f"Found {len(new_steps)} new step directory(ies): {[f'step_{s[0]}' for s in new_steps]}")
                    
                    for step, step_dir in new_steps:
                        # If requested, skip steps beyond stop_at_step
                        stop_at_step = config.get("stop_at_step")
                        if stop_at_step is not None and step > stop_at_step:
                            if accelerator.is_local_main_process:
                                logger.info(f"Skipping step {step} > stop_at_step={stop_at_step}")
                            continue

                        try:
                            if not should_validate(step):
                                logger.info(f"Skipping step directory {step_dir.name} (step {step}) - not in validation schedule")
                                if accelerator.is_main_process:
                                    monitor.mark_skipped(step, "not_in_schedule")
                                continue

                            logger.info(f"\n{'='*60}")
                            logger.info(f"Validating step directory: {step_dir.name} (step {step})")
                            logger.info(f"{'='*60}")
                            
                            # Mark as in progress EARLY - before starting validation
                            # Only main process updates manifest to avoid race conditions
                            if accelerator.is_main_process:
                                monitor.mark_in_progress(step)
                                logger.info(f"✓ Marked checkpoint as in progress in manifest")
                            
                            # Run validation (loads images from step directory)
                            metrics = runner.run_validation(step)
                            
                            # Log metrics to wandb/tensorboard
                            if metrics and accelerator.is_main_process:
                                try:
                                    # Extract num_images before filtering metrics for logging
                                    num_images = metrics.get("_num_images", 0)
                                    # Filter out _num_images from metrics before logging (it's just for tracking)
                                    metrics_for_logging = {k: v for k, v in metrics.items() if k != "_num_images"}
                                    
                                    # For wandb, use direct logging to handle out-of-order steps
                                    if config.get("report_to") == "wandb":
                                        import wandb
                                        # Ensure metrics are defined (fallback in case they weren't defined during init)
                                        if wandb.run is not None:
                                            for metric_name in metrics_for_logging.keys():
                                                try:
                                                    # Try to define metric if not already defined
                                                    # Use step_metric to allow independent step tracking
                                                    wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                                                except:
                                                    try:
                                                        # Fallback: define without step_metric
                                                        wandb.define_metric(metric_name, summary="last")
                                                    except:
                                                        # Metric might already be defined, that's OK
                                                        pass
                                        # Check current wandb step to see if we can log at the requested step
                                        # wandb.run.step tracks the last logged step
                                        current_wandb_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 0
                                        
                                        log_dict = dict(metrics_for_logging)
                                        log_dict["validation_step"] = step  # Always include checkpoint step as metric
                                        
                                        if step >= current_wandb_step:
                                            # Safe to log with step parameter - step is current or future
                                            wandb.log(log_dict, step=step)
                                        else:
                                            # Step is in the past - log without step to avoid data being ignored
                                            # The validation_step metric preserves which checkpoint this corresponds to
                                            wandb.log(log_dict)
                                            new_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 'unknown'
                                            logger.info(f"  Note: Logged at wandb step {new_step}, checkpoint step {step} preserved in 'validation_step' metric")
                                    else:
                                        # For tensorboard or other trackers, normal logging
                                        accelerator.log(metrics_for_logging, step=step)
                                    
                                    logger.info(f"✓ Metrics logged to {config.get('report_to', 'tracker')} at step {step}")
                                except Exception as e:
                                    logger.warning(f"Failed to log metrics: {e}")
                                    import traceback
                                    logger.warning(traceback.format_exc())
                            
                            # Mark as validated - ONLY on main process to ensure metrics are saved correctly
                            # Non-main processes have empty metrics dict, so they would overwrite with {}
                            if accelerator.is_main_process:
                                num_images = metrics.get("_num_images", 0) if metrics else 0
                                # Remove _num_images from metrics before saving (it's just for tracking)
                                metrics_for_manifest = {k: v for k, v in metrics.items() if k != "_num_images"} if metrics else {}
                                monitor.mark_validated(step, metrics_for_manifest, success=True, num_images=num_images)
                            else:
                                # Non-main processes just wait - main process will update manifest
                                logger.info(f"[Rank {accelerator.process_index}] Skipping manifest update (main process handles it)")
                            
                            logger.info(f"✓ Step {step} validated successfully")
                            last_validated_step = max(last_validated_step, int(step))
                            
                        except Exception as e:
                            logger.error(f"Failed to validate step {step}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            # Mark as failed - ONLY on main process
                            if accelerator.is_main_process:
                                monitor.mark_validated(step, {}, success=False, num_images=0)
                else:
                    if accelerator.is_local_main_process:
                        logger.info(f"No new step directories found. Waiting {args.check_interval}s...")
                
                # Wait before next check
                time.sleep(args.check_interval)
        
        # Normal monitoring loop for checkpoint validation, with explicit cross-GPU
        # coordination via a small sync file (no accelerator.wait_for_everyone()).
        monitor = CheckpointMonitor(
            output_dir=config["output_dir"],
            manifest_file=args.manifest_file,
        )

        # Use test_sync in test mode, train_sync in train mode, GT_sync in GT mode, validation_sync otherwise (backward compatible)
        if test_mode:
            sync_dir_name = "test_sync"
        elif train_mode:
            sync_dir_name = "train_sync"
        elif gt_mode:
            sync_dir_name = "GT_sync"
        else:
            sync_dir_name = "validation_sync"
        sync_dir = Path(config["output_dir"]) / sync_dir_name
        if accelerator.is_main_process:
            sync_dir.mkdir(parents=True, exist_ok=True)
            # IMPORTANT: remove any stale assignment file from previous runs so that
            # ranks don't pick up an old step (e.g., 5000) before rank 0 assigns
            # the current checkpoint (e.g., 2500).
            sync_file = sync_dir / "current_checkpoint.json"
            if sync_file.exists():
                try:
                    sync_file.unlink()
                    logger.info(f"Removed stale {sync_dir_name}/current_checkpoint.json at startup")
                except Exception as e:
                    logger.warning(f"Could not remove stale sync file {sync_file}: {e}")

        # Test mode: process specific checkpoints (continuous or one-time)
        if test_mode:
            test_continuous = args.test_continuous
            mode_str = "Continuous monitoring" if test_continuous else "One-time processing"
            logger.info("="*60)
            logger.info(f"TEST MODE: {mode_str} for specific checkpoints")
            logger.info(f"Checkpoints to validate: {checkpoint_numbers}")
            if test_continuous:
                logger.info(f"Check interval: {args.check_interval}s")
            logger.info(f"Manifest file: {args.manifest_file}")
            logger.info(f"Sync directory: {sync_dir_name}")
            logger.info("="*60)
            
            # Convert checkpoint_numbers to a set for fast lookup
            target_checkpoints = set(checkpoint_numbers)
            last_processed_step = -1
            
            # Helper function to process a single checkpoint
            def process_checkpoint(checkpoint_step, checkpoint_dir):
                """Process a single checkpoint - returns True if successful"""
                logger.info(f"\n{'='*60}")
                logger.info(f"TEST MODE: Validating checkpoint: {checkpoint_dir.name} (step {checkpoint_step})")
                logger.info(f"{'='*60}")
                
                # Mark as in progress
                if accelerator.is_main_process:
                    monitor.mark_in_progress(checkpoint_dir)
                    logger.info("✓ Marked checkpoint as in progress in manifest")
                
                # Publish assignment for all ranks
                if accelerator.is_main_process:
                    assignment = {
                        "step": int(checkpoint_step),
                        "checkpoint_dir": str(checkpoint_dir),
                        "timestamp": datetime.now().isoformat(),
                    }
                    try:
                        with open(sync_dir / "current_checkpoint.json", "w") as f:
                            json.dump(assignment, f)
                    except Exception as e:
                        logger.error(f"Failed to write checkpoint assignment file: {e}")
                
                # Wait for all ranks to see the assignment
                step, checkpoint_dir_str = _wait_for_new_checkpoint_task(
                    sync_dir=sync_dir,
                    last_step=-1,  # Always accept new assignment in test mode
                    check_interval=1,  # Check quickly in test mode
                    accelerator=accelerator,
                )
                
                if step is None or checkpoint_dir_str is None:
                    logger.error(f"Failed to get checkpoint assignment for step {checkpoint_step}")
                    if accelerator.is_main_process:
                        monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                    return False
                
                checkpoint_dir = Path(checkpoint_dir_str)
                
                try:
                    # Load checkpoint only if we're generating images (not loading from directory)
                    if not load_images_from_dir:
                        runner.load_checkpoint(checkpoint_dir)
                    else:
                        logger.info("Skipping checkpoint loading (loading pre-generated images from directory)")
                    
                    # Run validation for this checkpoint step
                    metrics = runner.run_validation(checkpoint_step)
                    
                    # Log metrics to wandb/tensorboard (main process only)
                    if metrics and accelerator.is_main_process:
                        try:
                            num_images = metrics.get("_num_images", 0)
                            metrics_for_logging = {k: v for k, v in metrics.items() if k != "_num_images"}
                            
                            if config.get("report_to") == "wandb":
                                import wandb
                                if wandb.run is not None:
                                    for metric_name in metrics_for_logging.keys():
                                        try:
                                            wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                                        except:
                                            try:
                                                wandb.define_metric(metric_name, summary="last")
                                            except:
                                                pass
                            
                            log_dict = dict(metrics_for_logging)
                            log_dict["test_step"] = checkpoint_step
                            
                            if config.get("report_to") == "wandb":
                                import wandb
                                if wandb.run is not None:
                                    wandb.log(log_dict, step=checkpoint_step)
                            else:
                                accelerator.log(metrics_for_logging, step=checkpoint_step)
                            
                            logger.info(f"✓ Metrics logged to {config.get('report_to', 'tracker')} at step {checkpoint_step}")
                        except Exception as e:
                            logger.warning(f"Failed to log metrics: {e}")
                            import traceback
                            logger.warning(traceback.format_exc())
                    
                    # Mark as validated - ONLY on main process
                    if accelerator.is_main_process:
                        num_images = metrics.get("_num_images", 0) if metrics else 0
                        metrics_for_manifest = {k: v for k, v in metrics.items() if k != "_num_images"} if metrics else {}
                        monitor.mark_validated(checkpoint_dir, metrics_for_manifest, success=True, num_images=num_images)
                    
                    logger.info(f"✓ TEST MODE: Checkpoint {checkpoint_dir.name} validated successfully")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to validate checkpoint {checkpoint_dir}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Mark as failed - ONLY on main process
                    if accelerator.is_main_process:
                        monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                    return False
            
            # One-time mode: process checkpoints once and exit
            if not test_continuous:
                logger.info("One-time mode: Processing specified checkpoints once...")
                for checkpoint_step in sorted(checkpoint_numbers):
                    checkpoint_dir = Path(config["output_dir"]) / f"checkpoint-{checkpoint_step}"
                    if not checkpoint_dir.exists():
                        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}, skipping")
                        if accelerator.is_main_process:
                            monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                        continue
                    
                    # Check if already processed
                    status = monitor.validated_checkpoints.get(f"checkpoint-{checkpoint_step}", {})
                    if status.get("status") == "completed":
                        logger.info(f"Checkpoint {checkpoint_dir.name} already validated, skipping")
                        continue
                    
                    process_checkpoint(checkpoint_step, checkpoint_dir)
                
                # Test mode complete
                logger.info("="*60)
                logger.info("TEST MODE: All checkpoints processed (one-time mode)")
                logger.info("="*60)
                return
            
            # Continuous monitoring mode
            logger.info("Continuous mode: Monitoring for checkpoints matching target list...")
            
            # First, process any existing checkpoints that match
            logger.info("Processing existing checkpoints that match target list...")
            for checkpoint_step in sorted(checkpoint_numbers):
                checkpoint_dir = Path(config["output_dir"]) / f"checkpoint-{checkpoint_step}"
                if checkpoint_dir.exists():
                    status = monitor.validated_checkpoints.get(f"checkpoint-{checkpoint_step}", {})
                    if status.get("status") not in ["completed", "in_progress"]:
                        logger.info(f"Found existing checkpoint: {checkpoint_dir.name}")
                    else:
                        logger.info(f"Checkpoint {checkpoint_dir.name} already processed, skipping")
            
            # Continuous monitoring loop
            while True:
                # Optional: stop once we've validated up to a specific step
                stop_at_step = config.get("stop_at_step")
                if stop_at_step is not None and last_processed_step >= stop_at_step:
                    if accelerator.is_local_main_process:
                        logger.info(f"Reached stop_at_step={stop_at_step} in test mode, exiting validation monitor.")
                    break
                
                # MAIN PROCESS: choose the next checkpoint to validate
                if accelerator.is_main_process:
                    new_checkpoints = monitor.get_new_checkpoints()
                    next_checkpoint_dir = None
                    next_step = None
                    
                    if new_checkpoints:
                        # Filter checkpoints to only those in our target list
                        def _step_from_dir(p: Path) -> int:
                            try:
                                return int(p.name.split("-")[1])
                            except Exception:
                                return 0
                        
                        sorted_checkpoints = sorted(new_checkpoints, key=_step_from_dir)
                        for ckpt in sorted_checkpoints:
                            step_val = _step_from_dir(ckpt)
                            # Only validate if step is in our target list
                            if step_val in target_checkpoints and step_val > last_processed_step:
                                next_checkpoint_dir = ckpt
                                next_step = step_val
                                break
                    
                    if next_checkpoint_dir is not None:
                        # Mark as in progress
                        monitor.mark_in_progress(next_checkpoint_dir)
                        logger.info(f"✓ Marked checkpoint {next_checkpoint_dir.name} as in progress")
                        
                        # Publish assignment for all ranks
                        assignment = {
                            "step": next_step,
                            "checkpoint_dir": str(next_checkpoint_dir),
                            "timestamp": datetime.now().isoformat(),
                        }
                        try:
                            with open(sync_dir / "current_checkpoint.json", "w") as f:
                                json.dump(assignment, f)
                        except Exception as e:
                            logger.error(f"Failed to write checkpoint assignment file: {e}")
                    else:
                        # No checkpoint to process, wait before next check
                        logger.info(f"No new matching checkpoints found. Waiting {args.check_interval}s before next check...")
                        time.sleep(args.check_interval)
                        continue
                
                # All ranks: wait for assignment
                step, checkpoint_dir_str = _wait_for_new_checkpoint_task(
                    sync_dir=sync_dir,
                    last_step=last_processed_step,
                    check_interval=args.check_interval,
                    accelerator=accelerator,
                )
                
                if step is None or checkpoint_dir_str is None:
                    # No assignment received, continue loop
                    continue
                
                checkpoint_dir = Path(checkpoint_dir_str)
                last_processed_step = step
                
                # Verify this checkpoint is in our target list
                if step not in target_checkpoints:
                    logger.warning(f"Checkpoint step {step} not in target list, skipping")
                    if accelerator.is_main_process:
                        monitor.mark_skipped(checkpoint_dir, "not_in_target_list")
                    continue
                
                # Process the checkpoint using the helper function
                process_checkpoint(step, checkpoint_dir)
                
                # Wait before next check
                time.sleep(args.check_interval)

        # Train mode: process specific checkpoints (continuous or one-time) - similar to test_mode
        elif train_mode:
            train_continuous = args.train_continuous
            mode_str = "Continuous monitoring" if train_continuous else "One-time processing"
            logger.info("="*60)
            logger.info(f"TRAIN MODE: {mode_str} for specific checkpoints")
            logger.info(f"Checkpoints to validate: {checkpoint_numbers}")
            if train_continuous:
                logger.info(f"Check interval: {args.check_interval}s")
            logger.info(f"Manifest file: {args.manifest_file}")
            logger.info(f"Sync directory: {sync_dir_name}")
            logger.info("="*60)
            
            # Convert checkpoint_numbers to a set for fast lookup
            target_checkpoints = set(checkpoint_numbers)
            last_processed_step = -1
            
            # Helper function to process a single checkpoint
            def process_checkpoint_train(checkpoint_step, checkpoint_dir):
                """Process a single checkpoint - returns True if successful"""
                logger.info(f"\n{'='*60}")
                logger.info(f"TRAIN MODE: Validating checkpoint: {checkpoint_dir.name} (step {checkpoint_step})")
                logger.info(f"{'='*60}")
                
                # Mark as in progress
                if accelerator.is_main_process:
                    monitor.mark_in_progress(checkpoint_dir)
                    logger.info("✓ Marked checkpoint as in progress in manifest")
                
                # Publish assignment for all ranks
                if accelerator.is_main_process:
                    assignment = {
                        "step": int(checkpoint_step),
                        "checkpoint_dir": str(checkpoint_dir),
                        "timestamp": datetime.now().isoformat(),
                    }
                    try:
                        with open(sync_dir / "current_checkpoint.json", "w") as f:
                            json.dump(assignment, f)
                    except Exception as e:
                        logger.error(f"Failed to write checkpoint assignment file: {e}")
                
                # Wait for all ranks to see the assignment
                step, checkpoint_dir_str = _wait_for_new_checkpoint_task(
                    sync_dir=sync_dir,
                    last_step=-1,  # Always accept new assignment in train mode
                    check_interval=1,  # Check quickly in train mode
                    accelerator=accelerator,
                )
                
                if step is None or checkpoint_dir_str is None:
                    logger.error(f"Failed to get checkpoint assignment for step {checkpoint_step}")
                    if accelerator.is_main_process:
                        monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                    return False
                
                checkpoint_dir = Path(checkpoint_dir_str)
                
                try:
                    # Load checkpoint only if we're generating images (not loading from directory)
                    if not load_images_from_dir:
                        runner.load_checkpoint(checkpoint_dir)
                    else:
                        logger.info("Skipping checkpoint loading (loading pre-generated images from directory)")
                    
                    # Run validation for this checkpoint step
                    metrics = runner.run_validation(checkpoint_step)
                    
                    # Log metrics to wandb/tensorboard (main process only)
                    if metrics and accelerator.is_main_process:
                        try:
                            num_images = metrics.get("_num_images", 0)
                            metrics_for_logging = {k: v for k, v in metrics.items() if k != "_num_images"}
                            
                            if config.get("report_to") == "wandb":
                                import wandb
                                if wandb.run is not None:
                                    for metric_name in metrics_for_logging.keys():
                                        try:
                                            wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                                        except:
                                            try:
                                                wandb.define_metric(metric_name, summary="last")
                                            except:
                                                pass
                            
                            log_dict = dict(metrics_for_logging)
                            log_dict["train_step"] = checkpoint_step
                            
                            if config.get("report_to") == "wandb":
                                import wandb
                                if wandb.run is not None:
                                    wandb.log(log_dict, step=checkpoint_step)
                            else:
                                accelerator.log(metrics_for_logging, step=checkpoint_step)
                            
                            logger.info(f"✓ Metrics logged to {config.get('report_to', 'tracker')} at step {checkpoint_step}")
                        except Exception as e:
                            logger.warning(f"Failed to log metrics: {e}")
                            import traceback
                            logger.warning(traceback.format_exc())
                    
                    # Mark as validated - ONLY on main process
                    if accelerator.is_main_process:
                        num_images = metrics.get("_num_images", 0) if metrics else 0
                        metrics_for_manifest = {k: v for k, v in metrics.items() if k != "_num_images"} if metrics else {}
                        monitor.mark_validated(checkpoint_dir, metrics_for_manifest, success=True, num_images=num_images)
                    
                    logger.info(f"✓ TRAIN MODE: Checkpoint {checkpoint_dir.name} validated successfully")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to validate checkpoint {checkpoint_dir}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Mark as failed - ONLY on main process
                    if accelerator.is_main_process:
                        monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                    return False
            
            # One-time mode: process checkpoints once and exit
            if not train_continuous:
                logger.info("One-time mode: Processing specified checkpoints once...")
                for checkpoint_step in sorted(checkpoint_numbers):
                    checkpoint_dir = Path(config["output_dir"]) / f"checkpoint-{checkpoint_step}"
                    if not checkpoint_dir.exists():
                        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}, skipping")
                        if accelerator.is_main_process:
                            monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                        continue
                    
                    # Check if already processed
                    status = monitor.validated_checkpoints.get(f"checkpoint-{checkpoint_step}", {})
                    if status.get("status") == "completed":
                        logger.info(f"Checkpoint {checkpoint_dir.name} already validated, skipping")
                        continue
                    
                    process_checkpoint_train(checkpoint_step, checkpoint_dir)
                
                # Train mode complete
                logger.info("="*60)
                logger.info("TRAIN MODE: All checkpoints processed (one-time mode)")
                logger.info("="*60)
                return
            
            # Continuous monitoring mode
            logger.info("Continuous mode: Monitoring for checkpoints matching target list...")
            
            # First, process any existing checkpoints that match
            logger.info("Processing existing checkpoints that match target list...")
            for checkpoint_step in sorted(checkpoint_numbers):
                checkpoint_dir = Path(config["output_dir"]) / f"checkpoint-{checkpoint_step}"
                if checkpoint_dir.exists():
                    status = monitor.validated_checkpoints.get(f"checkpoint-{checkpoint_step}", {})
                    if status.get("status") not in ["completed", "in_progress"]:
                        logger.info(f"Found existing checkpoint: {checkpoint_dir.name}")
                    else:
                        logger.info(f"Checkpoint {checkpoint_dir.name} already processed, skipping")
            
            # Continuous monitoring loop
            while True:
                # Optional: stop once we've validated up to a specific step
                stop_at_step = config.get("stop_at_step")
                if stop_at_step is not None and last_processed_step >= stop_at_step:
                    if accelerator.is_local_main_process:
                        logger.info(f"Reached stop_at_step={stop_at_step} in train mode, exiting validation monitor.")
                    break
                
                # MAIN PROCESS: choose the next checkpoint to validate
                if accelerator.is_main_process:
                    new_checkpoints = monitor.get_new_checkpoints()
                    next_checkpoint_dir = None
                    next_step = None
                    
                    if new_checkpoints:
                        # Filter checkpoints to only those in our target list
                        def _step_from_dir(p: Path) -> int:
                            try:
                                return int(p.name.split("-")[1])
                            except Exception:
                                return 0
                        
                        sorted_checkpoints = sorted(new_checkpoints, key=_step_from_dir)
                        for ckpt in sorted_checkpoints:
                            step_val = _step_from_dir(ckpt)
                            # Only validate if step is in our target list
                            if step_val in target_checkpoints and step_val > last_processed_step:
                                next_checkpoint_dir = ckpt
                                next_step = step_val
                                break
                    
                    if next_checkpoint_dir is not None:
                        # Mark as in progress
                        monitor.mark_in_progress(next_checkpoint_dir)
                        logger.info(f"✓ Marked checkpoint {next_checkpoint_dir.name} as in progress")
                        
                        # Publish assignment for all ranks
                        assignment = {
                            "step": next_step,
                            "checkpoint_dir": str(next_checkpoint_dir),
                            "timestamp": datetime.now().isoformat(),
                        }
                        try:
                            with open(sync_dir / "current_checkpoint.json", "w") as f:
                                json.dump(assignment, f)
                        except Exception as e:
                            logger.error(f"Failed to write checkpoint assignment file: {e}")
                    else:
                        # No checkpoint to process, wait before next check
                        logger.info(f"No new matching checkpoints found. Waiting {args.check_interval}s before next check...")
                        time.sleep(args.check_interval)
                        continue
                
                # All ranks: wait for assignment
                step, checkpoint_dir_str = _wait_for_new_checkpoint_task(
                    sync_dir=sync_dir,
                    last_step=last_processed_step,
                    check_interval=args.check_interval,
                    accelerator=accelerator,
                )
                
                if step is None or checkpoint_dir_str is None:
                    # No assignment received, continue loop
                    continue
                
                checkpoint_dir = Path(checkpoint_dir_str)
                last_processed_step = step
                
                # Verify this checkpoint is in our target list
                if step not in target_checkpoints:
                    logger.warning(f"Checkpoint step {step} not in target list, skipping")
                    if accelerator.is_main_process:
                        monitor.mark_skipped(checkpoint_dir, "not_in_target_list")
                    continue
                
                # Process the checkpoint using the helper function
                process_checkpoint_train(step, checkpoint_dir)
                
                # Wait before next check
                time.sleep(args.check_interval)

        # GT mode: process specific checkpoints (continuous or one-time) - similar to test_mode and train_mode
        elif gt_mode:
            gt_continuous = args.gt_continuous
            mode_str = "Continuous monitoring" if gt_continuous else "One-time processing"
            logger.info("="*60)
            logger.info(f"GT MODE: {mode_str} for specific checkpoints")
            logger.info(f"Checkpoints to validate: {checkpoint_numbers}")
            if gt_continuous:
                logger.info(f"Check interval: {args.check_interval}s")
            logger.info(f"Manifest file: {args.manifest_file}")
            logger.info(f"Sync directory: {sync_dir_name}")
            logger.info("="*60)
            
            # Convert checkpoint_numbers to a set for fast lookup
            target_checkpoints = set(checkpoint_numbers)
            last_processed_step = -1
            
            # Helper function to process a single checkpoint
            def process_checkpoint_gt(checkpoint_step, checkpoint_dir):
                """Process a single checkpoint - returns True if successful"""
                logger.info(f"\n{'='*60}")
                logger.info(f"GT MODE: Validating checkpoint: {checkpoint_dir.name} (step {checkpoint_step})")
                logger.info(f"{'='*60}")
                
                # Mark as in progress
                if accelerator.is_main_process:
                    monitor.mark_in_progress(checkpoint_dir)
                    logger.info("✓ Marked checkpoint as in progress in manifest")
                
                # Publish assignment for all ranks
                if accelerator.is_main_process:
                    assignment = {
                        "step": int(checkpoint_step),
                        "checkpoint_dir": str(checkpoint_dir),
                        "timestamp": datetime.now().isoformat(),
                    }
                    try:
                        with open(sync_dir / "current_checkpoint.json", "w") as f:
                            json.dump(assignment, f)
                    except Exception as e:
                        logger.error(f"Failed to write checkpoint assignment file: {e}")
                
                # Wait for all ranks to see the assignment
                step, checkpoint_dir_str = _wait_for_new_checkpoint_task(
                    sync_dir=sync_dir,
                    last_step=-1,  # Always accept new assignment in GT mode
                    check_interval=1,  # Check quickly in GT mode
                    accelerator=accelerator,
                )
                
                if step is None or checkpoint_dir_str is None:
                    logger.error(f"Failed to get checkpoint assignment for step {checkpoint_step}")
                    if accelerator.is_main_process:
                        monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                    return False
                
                checkpoint_dir = Path(checkpoint_dir_str)
                
                try:
                    # Load checkpoint only if we're generating images (not loading from directory)
                    if not load_images_from_dir:
                        runner.load_checkpoint(checkpoint_dir)
                    else:
                        logger.info("Skipping checkpoint loading (loading pre-generated images from directory)")
                    
                    # Run validation for this checkpoint step
                    metrics = runner.run_validation(checkpoint_step)
                    
                    # Log metrics to wandb/tensorboard (main process only)
                    if metrics and accelerator.is_main_process:
                        try:
                            num_images = metrics.get("_num_images", 0)
                            metrics_for_logging = {k: v for k, v in metrics.items() if k != "_num_images"}
                            
                            if config.get("report_to") == "wandb":
                                import wandb
                                if wandb.run is not None:
                                    for metric_name in metrics_for_logging.keys():
                                        try:
                                            wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                                        except:
                                            try:
                                                wandb.define_metric(metric_name, summary="last")
                                            except:
                                                pass
                            
                            log_dict = dict(metrics_for_logging)
                            log_dict["gt_step"] = checkpoint_step
                            
                            if config.get("report_to") == "wandb":
                                import wandb
                                if wandb.run is not None:
                                    wandb.log(log_dict, step=checkpoint_step)
                            else:
                                accelerator.log(metrics_for_logging, step=checkpoint_step)
                            
                            logger.info(f"✓ Metrics logged to {config.get('report_to', 'tracker')} at step {checkpoint_step}")
                        except Exception as e:
                            logger.warning(f"Failed to log metrics: {e}")
                            import traceback
                            logger.warning(traceback.format_exc())
                    
                    # Mark as validated - ONLY on main process
                    if accelerator.is_main_process:
                        num_images = metrics.get("_num_images", 0) if metrics else 0
                        metrics_for_manifest = {k: v for k, v in metrics.items() if k != "_num_images"} if metrics else {}
                        monitor.mark_validated(checkpoint_dir, metrics_for_manifest, success=True, num_images=num_images)
                    
                    logger.info(f"✓ GT MODE: Checkpoint {checkpoint_dir.name} validated successfully")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to validate checkpoint {checkpoint_dir}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Mark as failed - ONLY on main process
                    if accelerator.is_main_process:
                        monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                    return False
            
            # One-time mode: process checkpoints once and exit
            if not gt_continuous:
                logger.info("One-time mode: Processing specified checkpoints once...")
                for checkpoint_step in sorted(checkpoint_numbers):
                    checkpoint_dir = Path(config["output_dir"]) / f"checkpoint-{checkpoint_step}"
                    if not checkpoint_dir.exists():
                        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}, skipping")
                        if accelerator.is_main_process:
                            monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)
                        continue
                    
                    # Check if already processed
                    status = monitor.validated_checkpoints.get(f"checkpoint-{checkpoint_step}", {})
                    if status.get("status") == "completed":
                        logger.info(f"Checkpoint {checkpoint_dir.name} already validated, skipping")
                        continue
                    
                    process_checkpoint_gt(checkpoint_step, checkpoint_dir)
                
                # GT mode complete
                logger.info("="*60)
                logger.info("GT MODE: All checkpoints processed (one-time mode)")
                logger.info("="*60)
                return
            
            # Continuous monitoring mode
            logger.info("Continuous mode: Monitoring for checkpoints matching target list...")
            
            # First, process any existing checkpoints that match
            logger.info("Processing existing checkpoints that match target list...")
            for checkpoint_step in sorted(checkpoint_numbers):
                checkpoint_dir = Path(config["output_dir"]) / f"checkpoint-{checkpoint_step}"
                if checkpoint_dir.exists():
                    status = monitor.validated_checkpoints.get(f"checkpoint-{checkpoint_step}", {})
                    if status.get("status") not in ["completed", "in_progress"]:
                        logger.info(f"Found existing checkpoint: {checkpoint_dir.name}")
                    else:
                        logger.info(f"Checkpoint {checkpoint_dir.name} already processed, skipping")
            
            # Continuous monitoring loop
            while True:
                # Optional: stop once we've validated up to a specific step
                stop_at_step = config.get("stop_at_step")
                if stop_at_step is not None and last_processed_step >= stop_at_step:
                    if accelerator.is_local_main_process:
                        logger.info(f"Reached stop_at_step={stop_at_step} in GT mode, exiting validation monitor.")
                    break
                
                # MAIN PROCESS: choose the next checkpoint to validate
                if accelerator.is_main_process:
                    new_checkpoints = monitor.get_new_checkpoints()
                    next_checkpoint_dir = None
                    next_step = None
                    
                    if new_checkpoints:
                        # Filter checkpoints to only those in our target list
                        def _step_from_dir(p: Path) -> int:
                            try:
                                return int(p.name.split("-")[1])
                            except Exception:
                                return 0
                        
                        sorted_checkpoints = sorted(new_checkpoints, key=_step_from_dir)
                        for ckpt in sorted_checkpoints:
                            step_val = _step_from_dir(ckpt)
                            # Only validate if step is in our target list
                            if step_val in target_checkpoints and step_val > last_processed_step:
                                next_checkpoint_dir = ckpt
                                next_step = step_val
                                break
                    
                    if next_checkpoint_dir is not None:
                        # Mark as in progress
                        monitor.mark_in_progress(next_checkpoint_dir)
                        logger.info(f"✓ Marked checkpoint {next_checkpoint_dir.name} as in progress")
                        
                        # Publish assignment for all ranks
                        assignment = {
                            "step": next_step,
                            "checkpoint_dir": str(next_checkpoint_dir),
                            "timestamp": datetime.now().isoformat(),
                        }
                        try:
                            with open(sync_dir / "current_checkpoint.json", "w") as f:
                                json.dump(assignment, f)
                        except Exception as e:
                            logger.error(f"Failed to write checkpoint assignment file: {e}")
                    else:
                        # No checkpoint to process, wait before next check
                        logger.info(f"No new matching checkpoints found. Waiting {args.check_interval}s before next check...")
                        time.sleep(args.check_interval)
                        continue
                
                # All ranks: wait for assignment
                step, checkpoint_dir_str = _wait_for_new_checkpoint_task(
                    sync_dir=sync_dir,
                    last_step=last_processed_step,
                    check_interval=args.check_interval,
                    accelerator=accelerator,
                )
                
                if step is None or checkpoint_dir_str is None:
                    # No assignment received, continue loop
                    continue
                
                checkpoint_dir = Path(checkpoint_dir_str)
                last_processed_step = step
                
                # Verify this checkpoint is in our target list
                if step not in target_checkpoints:
                    logger.warning(f"Checkpoint step {step} not in target list, skipping")
                    if accelerator.is_main_process:
                        monitor.mark_skipped(checkpoint_dir, "not_in_target_list")
                    continue
                
                # Process the checkpoint using the helper function
                process_checkpoint_gt(step, checkpoint_dir)
                
                # Wait before next check
                time.sleep(args.check_interval)

        # Normal validation mode: monitoring loop
        last_processed_step = -1

        while True:
            # Optional: stop once we've validated up to a specific step
            stop_at_step = config.get("stop_at_step")
            if stop_at_step is not None and last_processed_step >= stop_at_step:
                if accelerator.is_local_main_process:
                    logger.info(f"Reached stop_at_step={stop_at_step} in checkpoint mode, exiting validation monitor.")
                break
            # MAIN PROCESS: choose the next checkpoint to validate and publish it.
            if accelerator.is_main_process:
                new_checkpoints = monitor.get_new_checkpoints()
                next_checkpoint_dir = None
                next_step = None

                if new_checkpoints:
                    # Ensure a deterministic, increasing order by step
                    def _step_from_dir(p: Path) -> int:
                        try:
                            return int(p.name.split("-")[1])
                        except Exception:
                            return 0

                    sorted_checkpoints = sorted(new_checkpoints, key=_step_from_dir)
                    for ckpt in sorted_checkpoints:
                        step_val = _step_from_dir(ckpt)
                        if not should_validate(step_val):
                            logger.info(f"Skipping checkpoint {ckpt.name} (step {step_val}) - not in validation schedule")
                            monitor.mark_skipped(ckpt, "not_in_schedule")
                            continue
                        if step_val > last_processed_step:
                            next_checkpoint_dir = ckpt
                            next_step = step_val
                            break

                if next_checkpoint_dir is not None:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Validating checkpoint: {next_checkpoint_dir.name} (step {next_step})")
                    logger.info(f"{'='*60}")

                    # Mark as in progress in manifest
                    monitor.mark_in_progress(next_checkpoint_dir)
                    logger.info("✓ Marked checkpoint as in progress in manifest")

                    # Publish assignment so all ranks use the SAME step/checkpoint_dir
                    assignment = {
                        "step": int(next_step),
                        "checkpoint_dir": str(next_checkpoint_dir),
                        "timestamp": datetime.now().isoformat(),
                    }
                    try:
                        with open(sync_dir / "current_checkpoint.json", "w") as f:
                            json.dump(assignment, f)
                    except Exception as e:
                        logger.error(f"Failed to write checkpoint assignment file: {e}")

            # ALL RANKS: wait for a new assignment newer than last_processed_step
            step, checkpoint_dir_str = _wait_for_new_checkpoint_task(
                sync_dir=sync_dir,
                last_step=last_processed_step,
                check_interval=args.check_interval,
                accelerator=accelerator,
            )

            if step is None or checkpoint_dir_str is None:
                if accelerator.is_local_main_process:
                    logger.info(f"No new checkpoints found. Waiting {args.check_interval}s...")
                time.sleep(args.check_interval)
                continue

            checkpoint_dir = Path(checkpoint_dir_str)

            # If requested, skip steps beyond stop_at_step (but still allow the
            # loop to terminate cleanly once last_processed_step >= stop_at_step)
            stop_at_step = config.get("stop_at_step")
            if stop_at_step is not None and step > stop_at_step:
                if accelerator.is_local_main_process:
                    logger.info(f"Skipping checkpoint step {step} > stop_at_step={stop_at_step}")
                last_processed_step = max(last_processed_step, int(step))
                continue

            try:
                # Load checkpoint (each rank loads its own copy)
                runner.load_checkpoint(checkpoint_dir)

                # Run validation for this checkpoint step
                metrics = runner.run_validation(step)

                # Log metrics to wandb/tensorboard (main process only)
                if metrics and accelerator.is_main_process:
                    try:
                        # Extract num_images before filtering metrics for logging
                        num_images = metrics.get("_num_images", 0)
                        # Filter out _num_images from metrics before logging (it's just for tracking)
                        metrics_for_logging = {k: v for k, v in metrics.items() if k != "_num_images"}

                        # For wandb, use direct logging to handle out-of-order steps
                        if config.get("report_to") == "wandb":
                            import wandb
                            # Ensure metrics are defined (fallback in case they weren't defined during init)
                            if wandb.run is not None:
                                for metric_name in metrics_for_logging.keys():
                                    try:
                                        # Try to define metric if not already defined
                                        # Use step_metric to allow independent step tracking
                                        wandb.define_metric(metric_name, step_metric="global_step", summary="last")
                                    except:
                                        try:
                                            # Fallback: define without step_metric
                                            wandb.define_metric(metric_name, summary="last")
                                        except:
                                            # Metric might already be defined, that's OK
                                            pass
                            # Check current wandb step to see if we can log at the requested step
                            # wandb.run.step tracks the last logged step
                            current_wandb_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 0

                            log_dict = dict(metrics_for_logging)
                            log_dict["validation_step"] = step  # Always include checkpoint step as metric

                            if step >= current_wandb_step:
                                # Safe to log with step parameter - step is current or future
                                wandb.log(log_dict, step=step)
                            else:
                                # Step is in the past - log without step to avoid data being ignored
                                # The validation_step metric preserves which checkpoint this corresponds to
                                wandb.log(log_dict)
                                new_step = wandb.run.step if (wandb.run and hasattr(wandb.run, 'step')) else 'unknown'
                                logger.info(f"  Note: Logged at wandb step {new_step}, checkpoint step {step} preserved in 'validation_step' metric")
                        else:
                            # For tensorboard or other trackers, normal logging
                            accelerator.log(metrics_for_logging, step=step)

                        logger.info(f"✓ Metrics logged to {config.get('report_to', 'tracker')} at step {step}")
                    except Exception as e:
                        logger.warning(f"Failed to log metrics: {e}")
                        import traceback
                        logger.warning(traceback.format_exc())

                # Mark as validated - ONLY on main process to ensure metrics are saved correctly
                # Non-main processes have empty metrics dict, so they would overwrite with {}
                if accelerator.is_main_process:
                    num_images = metrics.get("_num_images", 0) if metrics else 0
                    # Remove _num_images from metrics before saving (it's just for tracking)
                    metrics_for_manifest = {k: v for k, v in metrics.items() if k != "_num_images"} if metrics else {}
                    monitor.mark_validated(checkpoint_dir, metrics_for_manifest, success=True, num_images=num_images)
                else:
                    # Non-main processes just wait - main process will update manifest
                    logger.info(f"[Rank {accelerator.process_index}] Skipping manifest update (main process handles it)")

                logger.info(f"✓ Checkpoint {checkpoint_dir.name} validated successfully")

            except Exception as e:
                logger.error(f"Failed to validate checkpoint {checkpoint_dir}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Mark as failed - ONLY on main process
                if accelerator.is_main_process:
                    monitor.mark_validated(checkpoint_dir, {}, success=False, num_images=0)

            # Remember last successfully processed (or attempted) step so that we
            # only move forward in the sequence.
            last_processed_step = max(last_processed_step, int(step))

    except KeyboardInterrupt:
        logger.info("\nValidation monitoring stopped by user")
    except Exception as e:
        logger.error(f"Validation monitoring failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # End tracking
        if accelerator.is_main_process:
            try:
                accelerator.end_training()
                logger.info("✓ Tracking ended")
            except Exception as e:
                logger.warning(f"Failed to end tracking: {e}")


if __name__ == "__main__":
    main()
