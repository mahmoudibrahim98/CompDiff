#!/usr/bin/env python
"""
Generate synthetic dataset from WebDataset (WDS) using trained model.

This script generates synthetic images from a WDS dataset following the same
pipeline as validation (including HCN tokens, demographic tokens, prompt stripping).
Images are saved directly to disk and the script supports multi-GPU execution.

Usage:
    accelerate launch gen_source/generate_synthetic_dataset.py --config_file configs/your_config.yaml

Required config parameters:
    - pretrained_model_name_or_path: Path to pretrained model (or checkpoint)
    - generation_checkpoint_path: Path to checkpoint directory (REQUIRED - must contain trained model weights)
    - wds_dataset_path: Path to WDS dataset (directory with .tar files or single .tar file)
    - synthetic_output_dir: Output directory for generated images
    - resolution: Image resolution (default: 512)
    - guidance_scale: Classifier-free guidance scale (default: 7.5)
    - num_inference_steps: Number of diffusion steps (default: 50)
    - generation_batch_size: Batch size for generation (default: 1)
    - num_images_per_prompt: Number of images to generate per prompt (default: 1)
    - mixed_precision: "fp16", "bf16", or "no" (default: "bf16")
    
Optional config parameters:
    - use_hcn: Whether to use HCN (default: False)
    - use_demographic_encoder: Whether to use DemographicEncoder (default: False)
    - strip_demographics_in_validation: Strip demographics from prompts (default: True)
    - keep_age_in_prompt_validation: Keep age in prompt when stripping (default: False)
    - hcn_num_age_bins, hcn_num_sex, hcn_num_race: HCN configuration
    - demo_num_age_bins, demo_num_sex, demo_num_race: DemographicEncoder configuration
    - age_bins: Explicit age bins (optional, will be auto-generated if not provided)
    - max_samples: Maximum number of samples to generate (optional)
    - train_text_encoder: Whether text encoder was trained (default: False)

Output structure:
    synthetic_output_dir/
        gpu_0/
            synthetic_000000.png      # Generated synthetic image
            prompt_000000.txt        # Prompt text
            metadata_000000.json     # Metadata (demographics, labels, real_image_path reference)
            ...
        gpu_1/
            ...
"""

import os
import sys
import json
import logging
import torch
import numpy as np
import csv
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from typing import Optional, Dict, List
import argparse
import yaml
import pickle
import threading

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import save_image
try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    RepositoryNotFoundError = Exception

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dataset_wds import (
    RGFineTuningWebDataset,
    get_age_bins_from_num_bins,
    set_default_age_bins,
    extract_clinical_text,
)

logger = get_logger(__name__, log_level="INFO")


def load_config(file_path):
    """Load YAML config file."""
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # Map validation config fields to generation fields (for compatibility with training configs)
            if 'validation_guidance_scale' in config and 'guidance_scale' not in config:
                config['guidance_scale'] = config['validation_guidance_scale']
            if 'validation_num_inference_steps' in config and 'num_inference_steps' not in config:
                config['num_inference_steps'] = config['validation_num_inference_steps']
            if 'validation_num_images_per_prompt' in config and 'num_images_per_prompt' not in config:
                config['num_images_per_prompt'] = config['validation_num_images_per_prompt']
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None


class SyntheticDatasetGenerator:
    """Generates synthetic dataset from WDS following validation pipeline."""

    def __init__(self, accelerator: Accelerator, args: Dict, logger):
        self.accelerator = accelerator
        self.args = args
        self.logger = logger

        # Configuration flags
        # Support both generation-specific and validation-specific config names
        self.strip_demographics = args.get("strip_demographics_in_generation", 
                                           args.get("strip_demographics_in_validation", True))
        self.keep_age_in_prompt = args.get("keep_age_in_prompt", 
                                           args.get("keep_age_in_prompt_validation", False))
        self.use_hcn = args.get("use_hcn", False)
        self.use_demographic_encoder = args.get("use_demographic_encoder", False)
        self.use_hcn_timestep_injection = args.get("use_hcn_timestep_injection", False)
        
        # CSV tracking file handle (will be initialized in generate_dataset)
        self.csv_file = None
        self.csv_writer = None
        self.csv_lock = threading.Lock()

        # Models
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.noise_scheduler = None
        self.unet = None
        self.hcn = None
        self.demographic_encoder = None
        self.fg_classifier = None
        self.fg_preprocessing = None
        self.weight_dtype = None

        # Dataset
        self.dataset = None
        self.dataloader = None

        # Initialize models
        self._initialize_models()
        self._initialize_dataset()

    def _rank_prefix(self):
        return f"[Rank {self.accelerator.process_index}/{self.accelerator.num_processes}]"

    def _initialize_models(self):
        """Initialize all models (tokenizer, text_encoder, vae, scheduler, UNet, HCN, DemographicEncoder)."""
        logger.info(f"{self._rank_prefix()} Loading models...")

        # Tokenizer
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.args["pretrained_model_name_or_path"],
                subfolder="tokenizer",
                revision=self.args.get("revision"),
                use_auth_token=self.args.get("use_auth_token"),
            )
        except (RepositoryNotFoundError, Exception) as e:
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

        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args["pretrained_model_name_or_path"],
            subfolder="unet",
            revision=self.args.get("revision"),
            use_auth_token=self.args.get("use_auth_token"),
        )

        # HCN if enabled - matching validation monitor logic exactly
        self.hcn = None
        if self.use_hcn:
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
                        logger.info(f"✓ HCN with Ordinal Age Loss initialized (mode: {age_loss_mode})")
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
                        logger.info(f"✓ HCN architecture initialized")
            except Exception as e:
                logger.error(f"Failed to initialize HCN: {e}")
                raise

        # DemographicEncoder if enabled
        self.demographic_encoder = None
        if self.use_demographic_encoder:
            try:
                from demographic_encoder import DemographicEncoder
                self.demographic_encoder = DemographicEncoder(
                    num_age_bins=self.args.get("demo_num_age_bins", 5),
                    num_sex=self.args.get("demo_num_sex", 2),
                    num_race=self.args.get("demo_num_race", 4),
                    d_hidden=self.args.get("demo_d_hidden", 256),
                    d_output=self.args.get("demo_d_output", 1024),
                    mode=self.args.get("demo_mode", "single"),
                    classifier_depth=self.args.get("demo_classifier_depth", "shallow"),
                    aux_hidden_dim=self.args.get("demo_aux_hidden_dim", 512),
                    dropout=self.args.get("demo_dropout", 0.1),
                )
                if self.accelerator.is_local_main_process:
                    logger.info(f"✓ DemographicEncoder architecture initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DemographicEncoder: {e}")
                raise

        # Set weight dtype
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        # Move models to device
        self.text_encoder = self.text_encoder.to(self.accelerator.device)
        self.vae = self.vae.to(self.accelerator.device)
        self.unet = self.unet.to(self.accelerator.device)
        if self.hcn is not None:
            self.hcn = self.hcn.to(self.accelerator.device)
        if self.demographic_encoder is not None:
            self.demographic_encoder = self.demographic_encoder.to(self.accelerator.device)

        # Set to eval mode
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.unet.eval()
        self.unet.requires_grad_(False)
        if self.hcn is not None:
            self.hcn.eval()
            self.hcn.requires_grad_(False)
        if self.demographic_encoder is not None:
            self.demographic_encoder.eval()
            self.demographic_encoder.requires_grad_(False)

        # VAE always stays in fp32
        self.vae.to(dtype=torch.float32)
        self.text_encoder.to(dtype=self.weight_dtype)
        self.unet.to(dtype=self.weight_dtype)
        if self.hcn is not None:
            self.hcn.to(dtype=self.weight_dtype)
        if self.demographic_encoder is not None:
            self.demographic_encoder.to(dtype=self.weight_dtype)

        # Feedback guidance classifier if enabled
        use_feedback_guidance = self.args.get("use_feedback_guidance", False)
        if use_feedback_guidance:
            try:
                # from feedback_guidance import get_classifier_for_feedback  # Optional: uncomment if available
                
                fg_classifier_path = self.args.get("fg_classifier_path", None)
                fg_num_classes = self.args.get("fg_num_classes", 14)
                fg_from_scratch = self.args.get("fg_from_scratch", False)
                
                if self.accelerator.is_local_main_process:
                    logger.info(f"Loading feedback guidance classifier from: {fg_classifier_path or 'default'}")
                
                self.fg_classifier, self.fg_preprocessing = get_classifier_for_feedback(
                    classifier_path=fg_classifier_path,
                    num_classes=fg_num_classes,
                    device=str(self.accelerator.device),
                    from_scratch=fg_from_scratch,
                )
                
                if self.accelerator.is_local_main_process:
                    logger.info("✓ Feedback guidance classifier loaded")
            except Exception as e:
                logger.warning(f"Failed to load feedback guidance classifier: {e}")
                if self.accelerator.is_local_main_process:
                    logger.warning("Continuing without feedback guidance")
                use_feedback_guidance = False
                self.fg_classifier = None
                self.fg_preprocessing = None

        logger.info(f"{self._rank_prefix()} ✓ Models loaded")

    def _initialize_dataset(self):
        """Initialize WDS dataset and dataloader."""
        logger.info(f"{self._rank_prefix()} Loading WebDataset...")

        # Determine age_bins
        age_bins = self.args.get("age_bins", None)
        if age_bins is None:
            if self.use_hcn:
                num_age_bins = self.args.get("hcn_num_age_bins", 5)
            elif self.use_demographic_encoder:
                num_age_bins = self.args.get("demo_num_age_bins", 5)
            else:
                num_age_bins = None

            if num_age_bins is not None and num_age_bins > 0:
                age_bins = get_age_bins_from_num_bins(num_age_bins, max_age=100)
                if self.accelerator.is_local_main_process:
                    logger.info(f"Auto-generated age_bins from num_age_bins={num_age_bins}: {age_bins}")

        if age_bins is not None:
            set_default_age_bins(age_bins)
            if self.accelerator.is_local_main_process:
                logger.info(f"Using age_bins: {age_bins}")

        # Load WDS dataset
        wds_path = self.args.get("wds_dataset_path")
        if wds_path is None:
            raise ValueError("wds_dataset_path must be specified in config")

        # If wds_path is a directory, construct tar file paths
        import glob
        if os.path.isdir(wds_path):
            tar_files = sorted(glob.glob(os.path.join(wds_path, "*.tar")))
            if not tar_files:
                raise ValueError(f"No tar files found in WDS directory: {wds_path}")
            wds_urls = tar_files
        else:
            wds_urls = [wds_path] if isinstance(wds_path, str) else wds_path

        # Store WDS path for reference (normalize to directory if it's a single tar file)
        if os.path.isdir(wds_path):
            self.wds_dataset_path = wds_path
        else:
            # If it's a single tar file, use its directory
            self.wds_dataset_path = os.path.dirname(wds_path) if os.path.dirname(wds_path) else wds_path

        self.dataset = RGFineTuningWebDataset(
            url_list=wds_urls,
            tokenizer=self.tokenizer,
            use_hcn=self.use_hcn,
            use_demographic_encoder=self.use_demographic_encoder,
            include_text=True,  # Need full prompt text for generation
            age_bins=age_bins,
        )

        # Create dataloader
        batch_size = self.args.get("generation_batch_size", 1)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.get("num_workers", 4),
        )

        logger.info(f"{self._rank_prefix()} ✓ Dataset loaded")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint weights into models."""
        logger.info(f"{self._rank_prefix()} Loading checkpoint from {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path)
        
        # Validate checkpoint directory exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_path}\n"
                f"Expected format: outputs_summarized/VERSION/EXPERIMENT/checkpoint-STEP\n"
                f"Example: outputs_summarized/v0/0_train_baseline/checkpoint-10000"
            )
        
        if not checkpoint_path.is_dir():
            raise ValueError(f"Checkpoint path is not a directory: {checkpoint_path}")

        try:
            import safetensors.torch

            # Track what was actually loaded
            loaded_models = []
            
            # Load UNet
            if (checkpoint_path / "model.safetensors").exists():
                logger.info(f"{self._rank_prefix()} Loading UNet from model.safetensors...")
                state_dict = safetensors.torch.load_file(checkpoint_path / "model.safetensors")
                self.unet.load_state_dict(state_dict, strict=False)
                loaded_models.append("UNet")
            elif (checkpoint_path / "pytorch_model.bin").exists():
                logger.info(f"{self._rank_prefix()} Loading UNet from pytorch_model.bin...")
                state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=self.accelerator.device)
                self.unet.load_state_dict(state_dict, strict=False)
                loaded_models.append("UNet")
            else:
                raise FileNotFoundError(
                    f"No UNet checkpoint file found in {checkpoint_path}\n"
                    f"Expected: model.safetensors or pytorch_model.bin"
                )

            # Load text encoder if trained
            if self.args.get("train_text_encoder", False):
                if (checkpoint_path / "model_1.safetensors").exists():
                    logger.info(f"{self._rank_prefix()} Loading text encoder from model_1.safetensors...")
                    state_dict = safetensors.torch.load_file(checkpoint_path / "model_1.safetensors")
                    self.text_encoder.load_state_dict(state_dict, strict=False)
                    loaded_models.append("TextEncoder")
                elif (checkpoint_path / "pytorch_model_1.bin").exists():
                    logger.info(f"{self._rank_prefix()} Loading text encoder from pytorch_model_1.bin...")
                    state_dict = torch.load(checkpoint_path / "pytorch_model_1.bin", map_location=self.accelerator.device)
                    self.text_encoder.load_state_dict(state_dict, strict=False)
                    loaded_models.append("TextEncoder")
                else:
                    logger.warning(
                        f"{self._rank_prefix()} Text encoder was trained (train_text_encoder=True) but no checkpoint file found. "
                        f"Expected: model_1.safetensors or pytorch_model_1.bin in {checkpoint_path}"
                    )

            # Load HCN if enabled
            if self.hcn is not None:
                if (checkpoint_path / "model_2.safetensors").exists():
                    logger.info(f"{self._rank_prefix()} Loading HCN from model_2.safetensors...")
                    state_dict = safetensors.torch.load_file(checkpoint_path / "model_2.safetensors")
                    self.hcn.load_state_dict(state_dict, strict=False)
                    loaded_models.append("HCN")
                elif (checkpoint_path / "pytorch_model_2.bin").exists():
                    logger.info(f"{self._rank_prefix()} Loading HCN from pytorch_model_2.bin...")
                    state_dict = torch.load(checkpoint_path / "pytorch_model_2.bin", map_location=self.accelerator.device)
                    self.hcn.load_state_dict(state_dict, strict=False)
                    loaded_models.append("HCN")
                else:
                    raise FileNotFoundError(
                        f"HCN is enabled (use_hcn=True) but no HCN checkpoint file found in {checkpoint_path}\n"
                        f"Expected: model_2.safetensors or pytorch_model_2.bin\n"
                        f"This will cause incorrect generation - HCN will use random untrained weights!"
                    )

            # Load DemographicEncoder if enabled
            if self.demographic_encoder is not None:
                demo_encoder_path = checkpoint_path / "demographic_encoder"
                if demo_encoder_path.exists():
                    logger.info(f"{self._rank_prefix()} Loading DemographicEncoder from {demo_encoder_path}...")
                    try:
                        from demographic_encoder import DemographicEncoder
                        self.demographic_encoder = DemographicEncoder.from_pretrained(str(demo_encoder_path))
                        self.demographic_encoder = self.demographic_encoder.to(self.accelerator.device)
                        self.demographic_encoder.eval()
                        self.demographic_encoder.requires_grad_(False)
                        logger.info(f"{self._rank_prefix()} ✓ DemographicEncoder loaded from checkpoint")
                    except Exception as e:
                        logger.warning(f"Failed to load DemographicEncoder from {demo_encoder_path}: {e}")
                        # Fallback: try model_2 or model_3
                        if self.hcn is None:
                            if (checkpoint_path / "model_2.safetensors").exists():
                                state_dict = safetensors.torch.load_file(checkpoint_path / "model_2.safetensors")
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)
                            elif (checkpoint_path / "pytorch_model_2.bin").exists():
                                state_dict = torch.load(checkpoint_path / "pytorch_model_2.bin", map_location=self.accelerator.device)
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)
                        else:
                            if (checkpoint_path / "model_3.safetensors").exists():
                                state_dict = safetensors.torch.load_file(checkpoint_path / "model_3.safetensors")
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)
                            elif (checkpoint_path / "pytorch_model_3.bin").exists():
                                state_dict = torch.load(checkpoint_path / "pytorch_model_3.bin", map_location=self.accelerator.device)
                                self.demographic_encoder.load_state_dict(state_dict, strict=False)

            # Verify at least UNet was loaded
            if not loaded_models:
                raise RuntimeError(
                    f"No checkpoint files were loaded from {checkpoint_path}\n"
                    f"This means models will use pretrained/random weights instead of trained weights!\n"
                    f"Please verify the checkpoint path is correct."
                )
            
            logger.info(f"{self._rank_prefix()} ✓ Checkpoint loaded successfully")
            logger.info(f"{self._rank_prefix()}   Loaded models: {', '.join(loaded_models)}")

        except Exception as e:
            logger.error(f"{self._rank_prefix()} Failed to load checkpoint: {e}")
            raise

        # Set to eval mode
        self.unet.eval()
        self.text_encoder.eval()
        if self.hcn is not None:
            self.hcn.eval()
        if self.demographic_encoder is not None:
            self.demographic_encoder.eval()

    def _add_hcn_conditioning(self, text_embeddings, metadata_list):
        """Add HCN conditioning to text embeddings.
        
        Matches validation monitor logic for V9 continuous age, V8 ordinal, and standard HCN.
        """
        if self.hcn is None:
            return text_embeddings, None

        # Check if using V9 (continuous age) - check both config and HCN type
        age_indices = []
        sex_indices = []
        race_indices = []

        for metadata in metadata_list:
            has_demographics = "sex_idx" in metadata and "race_idx" in metadata
            has_age = "age_idx" in metadata
            
            if has_demographics and has_age:
                sex_idx = metadata["sex_idx"]
                race_idx = metadata["race_idx"]
                
                sex_indices.append(sex_idx)
                race_indices.append(race_idx)
                
                if "age_idx" in metadata:
                    age_idx = metadata["age_idx"]
                    age_indices.append(age_idx)

        if sex_indices and race_indices:
            sex_indices = torch.stack(sex_indices).squeeze().to(self.accelerator.device)
            race_indices = torch.stack(race_indices).squeeze().to(self.accelerator.device)
            
            if sex_indices.dim() == 0:
                sex_indices = sex_indices.unsqueeze(0)
            if race_indices.dim() == 0:
                race_indices = race_indices.unsqueeze(0)

            self.hcn.eval()
            
            # HCN returns 5 values: ctx, mu, logsigma, aux_logits, time_emb
            # Call with appropriate arguments
            if age_indices:
                # V7/V8: Check if V8 Ordinal (uses positional args) or V7/V8 (uses keyword args)
                age_indices = torch.stack(age_indices).squeeze().to(self.accelerator.device)
                if age_indices.dim() == 0:
                    age_indices = age_indices.unsqueeze(0)
                
                # V10: Check if age encoding is enabled
                encode_age = getattr(self.hcn, 'encode_age', True)
                age_indices_for_hcn = age_indices if encode_age else None
                
                # Check if V8 Ordinal by checking if forward signature expects (age, sex, race) positional
                # V8 Ordinal: forward(age_idx, sex_idx, race_idx)
                # V7/V8: forward(sex_idx, race_idx, age_idx=None)
                try:
                    from hcn_v8_ordinal import HierarchicalConditionerV8Ordinal
                    if isinstance(self.hcn, HierarchicalConditionerV8Ordinal):
                        # V8 Ordinal: positional arguments (age, sex, race)
                        # Note: V8 Ordinal doesn't support encode_age=False, so always pass age
                        hcn_ctx, _, _, _, time_emb = self.hcn(age_indices, sex_indices, race_indices)
                    else:
                        # V7/V8: keyword arguments (sex, race, age)
                        hcn_ctx, _, _, _, time_emb = self.hcn(
                            sex_idx=sex_indices,
                            race_idx=race_indices,
                            age_idx=age_indices_for_hcn,
                        )
                except ImportError:
                    # V8 Ordinal not available, use standard V7/V8
                    hcn_ctx, _, _, _, time_emb = self.hcn(
                        sex_idx=sex_indices,
                        race_idx=race_indices,
                        age_idx=age_indices_for_hcn,
                    )
            else:
                # No age data available
                hcn_ctx, _, _, _, time_emb = self.hcn(
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

    def _add_demographic_encoder_conditioning(self, text_embeddings, metadata_list):
        """Add DemographicEncoder conditioning to text embeddings."""
        if self.demographic_encoder is None:
            return text_embeddings

        age_indices = []
        sex_indices = []
        race_indices = []

        for metadata in metadata_list:
            if "age_idx" in metadata and "sex_idx" in metadata and "race_idx" in metadata:
                age_idx = metadata["age_idx"]
                sex_idx = metadata["sex_idx"]
                race_idx = metadata["race_idx"]

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

            self.demographic_encoder.eval()
            # Returns tokens: [B, 1, d] if mode='single', [B, 3, d] if mode='separate'
            demo_tokens, _ = self.demographic_encoder(age_indices, sex_indices, race_indices)

            text_embeddings = torch.cat([text_embeddings, demo_tokens], dim=1)

        return text_embeddings

    def _generate_image_batch(self, batch_images_list, all_prompts_data, batch_start_idx: int):
        """Generate a batch of images following validation pipeline.
        
        Args:
            batch_images_list: List of (prompt_idx, image_idx) tuples indicating which images to generate
            all_prompts_data: List of dicts with keys: "prompt", "metadata"
            batch_start_idx: Starting index for seed generation
        """
        local_batch_size = len(batch_images_list)

        # Extract prompts and metadata
        prompts = []
        metadata_list = []
        for (prompt_idx, image_idx) in batch_images_list:
            prompt_data = all_prompts_data[prompt_idx]
            prompts.append(prompt_data["prompt"])
            metadata_list.append(prompt_data["metadata"])

        # Strip demographics from prompts if requested
        if self.strip_demographics:
            prompts = [extract_clinical_text(prompt, keep_age=self.keep_age_in_prompt) for prompt in prompts]
            if self.accelerator.is_local_main_process and batch_start_idx == 0:
                self.logger.info(
                    f"✓ Stripped demographics from prompts "
                    f"(strip_demographics_in_validation=True, keep_age_in_prompt_validation={self.keep_age_in_prompt})"
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
        prompt_embeds = self.text_encoder(input_ids=text_input_ids, return_dict=False)
        text_embeddings = prompt_embeds[0]

        # Add HCN conditioning if available
        hcn_time_emb = None
        if self.hcn is not None:
            text_embeddings, hcn_time_emb = self._add_hcn_conditioning(text_embeddings, metadata_list)

        # Add DemographicEncoder conditioning if available
        if self.demographic_encoder is not None:
            text_embeddings = self._add_demographic_encoder_conditioning(text_embeddings, metadata_list)

        # Create unconditional embeddings
        uncond_inputs = self.tokenizer(
            [""] * local_batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.accelerator.device)
        uncond_embeds = self.text_encoder(input_ids=uncond_input_ids, return_dict=False)
        uncond_embeddings = uncond_embeds[0]

        # Ensure unconditional embeddings match conditional ones
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
            self.unet.config.in_channels,
            self.args["resolution"] // 8,
            self.args["resolution"] // 8,
        )

        # Generate latents with unique seeds
        # Use prompt_idx and image_idx to ensure uniqueness across all prompts and image variations
        generators = []
        for i, (prompt_idx, image_idx) in enumerate(batch_images_list):
            # Create unique seed based on prompt_idx, image_idx, and GPU index
            seed = self.accelerator.process_index * 1000000 + prompt_idx * 1000 + image_idx
            generators.append(
                torch.Generator(device=self.accelerator.device).manual_seed(seed)
            )
        latents = torch.stack([
            torch.randn(
                (1, self.unet.config.in_channels, self.args["resolution"] // 8, self.args["resolution"] // 8),
                generator=gen,
                device=self.accelerator.device,
                dtype=text_embeddings.dtype,
            )[0] for gen in generators
        ])

        init_noise_sigma = getattr(self.noise_scheduler, 'init_noise_sigma', 1.0)
        latents = latents * init_noise_sigma

        # Diffusion loop
        num_inference_steps = self.args.get("num_inference_steps", 50)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        self.noise_scheduler.timesteps = timesteps

        # Feedback guidance parameters
        use_feedback_guidance = self.args.get("use_feedback_guidance", False)
        fg_scale = self.args.get("fg_scale", 1.0)
        fg_criterion = self.args.get("fg_criterion", "loss")
        fg_guidance_freq = self.args.get("fg_guidance_freq", 1)  # Apply feedback every N steps
        fg_target_class_idx = self.args.get("fg_target_class_idx", None)
        fg_target_labels = self.args.get("fg_target_labels", None)  # [B, num_classes] tensor

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            timestep_tensor = t.expand(latent_model_input.shape[0]) if t.ndim == 0 else t.repeat(latent_model_input.shape[0] // t.shape[0])

            # V6 timestep injection: add HCN time embedding to UNet's internal timestep embedding
            # Only use if enabled in config
            if hcn_time_emb is not None and self.use_hcn_timestep_injection:
                zero_time_emb = torch.zeros_like(hcn_time_emb)
                combined_time_emb = torch.cat([zero_time_emb, hcn_time_emb], dim=0)

                from train_loop import TimestepInjectionContext
                with TimestepInjectionContext(self.unet, combined_time_emb):
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep_tensor,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
            else:
                noise_pred = self.unet(
                    latent_model_input,
                    timestep_tensor,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_scale = self.args.get("guidance_scale", 7.5)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply feedback guidance if enabled
            if use_feedback_guidance and self.fg_classifier is not None and i % fg_guidance_freq == 0:
                try:
                    # from feedback_guidance import compute_feedback_gradients  # Optional: uncomment if available
                    
                    # Use conditional embeddings for feedback (not unconditional)
                    txt_embd_for_guidance = encoder_hidden_states.chunk(2)[1] if encoder_hidden_states.shape[0] > local_batch_size else encoder_hidden_states
                    
                    # Compute feedback gradients
                    fg_grads = compute_feedback_gradients(
                        latents=latents,
                        timestep=t,
                        noise_scheduler=self.noise_scheduler,
                        unet=self.unet,
                        encoder_hidden_states=txt_embd_for_guidance,
                        vae=self.vae,
                        classifier=self.fg_classifier,
                        preprocessing=self.fg_preprocessing,
                        fg_criterion=fg_criterion,
                        fg_scale=fg_scale,
                        target_class_idx=fg_target_class_idx,
                        target_labels=fg_target_labels,
                    )
                    
                    # Apply feedback gradients to latents
                    latents = latents + fg_grads
                except Exception as e:
                    if self.accelerator.is_local_main_process:
                        self.logger.warning(f"Feedback guidance failed at step {i}: {e}. Continuing without feedback.")

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = 1 / 0.18215 * latents
        latents_for_decode = latents.to(dtype=torch.float32)
        images = self.vae.decode(latents_for_decode).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        # Extract real image references (paths) from prompt data
        # The reference path is already stored in all_prompts_data during collection
        real_image_refs = []
        for i, (prompt_idx, image_idx) in enumerate(batch_images_list):
            prompt_data = all_prompts_data[prompt_idx]
            real_image_ref = prompt_data.get("real_image_ref", None)
            real_image_refs.append(real_image_ref)

        return images, prompts, metadata_list, batch_images_list, real_image_refs

    def _save_images_batch(self, images, prompts, metadata_list, batch_images_list, real_image_refs, output_dir: Path, local_idx: int):
        """Save images directly to disk and write to CSV.
        
        Args:
            images: Tensor of synthetic images [B, C, H, W]
            prompts: List of prompt strings
            metadata_list: List of metadata dicts (extracted from batches)
            batch_images_list: List of (prompt_idx, image_idx) tuples for naming
            real_image_refs: List of real image path references (or None if not available)
            output_dir: Output directory
            local_idx: Local index for this GPU (starts from 0 for each GPU)
        """
        gpu_idx = self.accelerator.process_index
        gpu_dir = output_dir / f"gpu_{gpu_idx}"
        gpu_dir.mkdir(parents=True, exist_ok=True)

        local_batch_size = images.shape[0]
        csv_rows = []

        for i in range(local_batch_size):
            prompt_idx, image_idx = batch_images_list[i]
            
            # Save image (use local index per GPU)
            save_idx = local_idx + i
            image_filename = f"synthetic_{save_idx:06d}.png"
            image_path = gpu_dir / image_filename
            save_image(images[i], image_path)

            # Get real image reference path (directory/path to original image in WDS)
            real_image_ref = real_image_refs[i] if i < len(real_image_refs) else None

            # Save prompt
            prompt_path = gpu_dir / f"prompt_{save_idx:06d}.txt"
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(str(prompts[i]))

            # Save metadata (demographics, labels, etc.)
            metadata = metadata_list[i]
            save_metadata = {
                "prompt": prompts[i],
                "prompt_idx": prompt_idx,  # Original prompt index in dataset
                "image_idx": image_idx,     # Which image variation (0, 1, 2, ...)
                "synthetic_image": image_filename,
                "real_image_path": real_image_ref,  # Path/directory reference to original real image in WDS
            }

            # Extract all metadata fields
            age_idx = None
            sex_idx = None
            race_idx = None
            age = None
            age_continuous = None
            disease_labels = None
            
            if "age_idx" in metadata:
                age_idx_val = metadata["age_idx"]
                age_idx = int(age_idx_val.cpu().item()) if isinstance(age_idx_val, torch.Tensor) else int(age_idx_val)
                save_metadata["age_idx"] = age_idx
            if "sex_idx" in metadata:
                sex_idx_val = metadata["sex_idx"]
                sex_idx = int(sex_idx_val.cpu().item()) if isinstance(sex_idx_val, torch.Tensor) else int(sex_idx_val)
                save_metadata["sex_idx"] = sex_idx
            if "race_idx" in metadata:
                race_idx_val = metadata["race_idx"]
                race_idx = int(race_idx_val.cpu().item()) if isinstance(race_idx_val, torch.Tensor) else int(race_idx_val)
                save_metadata["race_idx"] = race_idx
            if "age" in metadata:
                age_val = metadata["age"]
                age = float(age_val.cpu().item()) if isinstance(age_val, torch.Tensor) else float(age_val)
                save_metadata["age"] = age
            if "age_continuous" in metadata:
                age_cont_val = metadata["age_continuous"]
                age_continuous = float(age_cont_val.cpu().item()) if isinstance(age_cont_val, torch.Tensor) else float(age_cont_val)
                save_metadata["age_continuous"] = age_continuous
            if "disease_labels" in metadata:
                disease_labels_val = metadata["disease_labels"]
                disease_labels = disease_labels_val.cpu().tolist() if isinstance(disease_labels_val, torch.Tensor) else disease_labels_val
                save_metadata["disease_labels"] = disease_labels

            metadata_path = gpu_dir / f"metadata_{save_idx:06d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(save_metadata, f, indent=2)
            
            # Prepare CSV row
            csv_row = {
                "gpu_idx": gpu_idx,
                "local_idx": save_idx,
                "prompt_idx": prompt_idx,
                "image_idx": image_idx,
                "synthetic_image_path": str(image_path),
                "synthetic_image_filename": image_filename,
                "real_image_path": real_image_ref if real_image_ref else "",
                "prompt": prompts[i],
                "age_idx": age_idx if age_idx is not None else "",
                "sex_idx": sex_idx if sex_idx is not None else "",
                "race_idx": race_idx if race_idx is not None else "",
                "age": age if age is not None else "",
                "age_continuous": age_continuous if age_continuous is not None else "",
                "disease_labels": json.dumps(disease_labels) if disease_labels is not None else "",
                "timestamp": datetime.now().isoformat(),
            }
            csv_rows.append(csv_row)
        
        # Write to CSV (thread-safe)
        if self.csv_writer is not None:
            with self.csv_lock:
                for row in csv_rows:
                    self.csv_writer.writerow(row)
                self.csv_file.flush()  # Flush after each batch for safety

    def _initialize_csv_tracking(self, output_dir: Path):
        """Initialize CSV tracking file for this GPU."""
        gpu_idx = self.accelerator.process_index
        gpu_dir = output_dir / f"gpu_{gpu_idx}"
        gpu_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = gpu_dir / "generation_log.csv"
        
        # Define CSV fields
        csv_fields = [
            "gpu_idx", "local_idx", "prompt_idx", "image_idx",
            "synthetic_image_path", "synthetic_image_filename", "real_image_path",
            "prompt", "age_idx", "sex_idx", "race_idx", "age", "age_continuous",
            "disease_labels", "timestamp"
        ]
        
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=csv_fields)
        self.csv_writer.writeheader()
        
        logger.info(f"{self._rank_prefix()} CSV tracking initialized: {csv_path}")
        
    def _close_csv_tracking(self):
        """Close CSV tracking file."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def generate_dataset(self, checkpoint_path: str, output_dir: str = None, max_samples: Optional[int] = None):
        """Generate synthetic dataset from WDS.
        
        Args:
            checkpoint_path: Path to checkpoint directory (REQUIRED - must contain trained model weights)
            output_dir: Output directory for synthetic images
            max_samples: Maximum number of samples to generate (optional)
        """
        if checkpoint_path == "":
            raise ValueError(
                "checkpoint_path cannot be empty. Please provide generation_checkpoint_path in config file "
                "or --checkpoint_path via command line."
            )
        
        # Load checkpoint (required for using trained model)
        self.load_checkpoint(checkpoint_path)

        if output_dir is None:
            output_dir = self.args.get("synthetic_output_dir", "synthetic_dataset")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV tracking
        self._initialize_csv_tracking(output_dir)

        logger.info(f"{self._rank_prefix()} Starting dataset generation...")
        logger.info(f"{self._rank_prefix()} Output directory: {output_dir}")

        # Get number of images per prompt
        num_images_per_prompt = self.args.get("num_images_per_prompt", 1)
        generation_batch_size = self.args.get("generation_batch_size", 1)

        # Memory optimization: Log warning for large datasets
        # Each prompt_data entry uses ~1-5KB, so 1M samples = ~1-5GB RAM
        estimated_samples = max_samples if max_samples else 1000000  # Assume 1M if unknown
        estimated_memory_mb = estimated_samples * 3 / 1024  # ~3KB per entry average
        if estimated_memory_mb > 5000:  # More than 5GB
            logger.warning(
                f"⚠️ Large dataset detected! Estimated {estimated_samples} samples will use ~{estimated_memory_mb:.0f}MB RAM. "
                f"Consider using --max_samples to limit, or ensure sufficient system memory."
            )
        
        # Collect all prompts first (extract only necessary metadata, not entire batches)
        all_prompts_data = []
        logger.info(f"{self._rank_prefix()} Collecting prompts from dataset...")
        
        for batch in self.dataloader:
            batch_size = batch["pixel_values"].shape[0]
            for i in range(batch_size):
                # Handle different prompt formats
                if "text" in batch:
                    prompt = batch["text"][i] if isinstance(batch["text"], list) else batch["text"]
                elif "prompt" in batch:
                    prompt = batch["prompt"][i] if isinstance(batch["prompt"], list) else batch["prompt"]
                else:
                    # Fallback: try to get from input_ids (decode)
                    if "input_ids" in batch:
                        input_ids = batch["input_ids"][i] if batch["input_ids"].dim() > 1 else batch["input_ids"]
                        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    else:
                        raise ValueError("No prompt text found in batch. Dataset must include 'text' or 'prompt' field.")
                
                # Extract __key__ if available (for real image reference)
                real_image_ref = None
                if "__key__" in batch:
                    if isinstance(batch["__key__"], (list, tuple)):
                        key = batch["__key__"][i] if i < len(batch["__key__"]) else None
                    elif isinstance(batch["__key__"], torch.Tensor):
                        key = batch["__key__"][i].item() if batch["__key__"].dim() > 0 else batch["__key__"].item()
                    else:
                        key = batch["__key__"]
                    
                    if key is not None:
                        # Construct reference path: {wds_dataset_path}/{key}
                        real_image_ref = os.path.join(self.wds_dataset_path, str(key))
                
                # Extract only necessary metadata fields (not entire batch to save memory)
                # Move to CPU to free GPU memory and ensure consistent tensor types
                metadata = {}
                if "age_idx" in batch:
                    age_idx = batch["age_idx"][i] if batch["age_idx"].dim() > 0 else batch["age_idx"]
                    if isinstance(age_idx, torch.Tensor):
                        metadata["age_idx"] = age_idx.clone().detach().cpu()
                    else:
                        metadata["age_idx"] = torch.tensor(age_idx, dtype=torch.long)
                if "sex_idx" in batch:
                    sex_idx = batch["sex_idx"][i] if batch["sex_idx"].dim() > 0 else batch["sex_idx"]
                    if isinstance(sex_idx, torch.Tensor):
                        metadata["sex_idx"] = sex_idx.clone().detach().cpu()
                    else:
                        metadata["sex_idx"] = torch.tensor(sex_idx, dtype=torch.long)
                if "race_idx" in batch:
                    race_idx = batch["race_idx"][i] if batch["race_idx"].dim() > 0 else batch["race_idx"]
                    if isinstance(race_idx, torch.Tensor):
                        metadata["race_idx"] = race_idx.clone().detach().cpu()
                    else:
                        metadata["race_idx"] = torch.tensor(race_idx, dtype=torch.long)
                if "age" in batch:
                    age = batch["age"][i] if batch["age"].dim() > 0 else batch["age"]
                    if isinstance(age, torch.Tensor):
                        metadata["age"] = age.clone().detach().cpu()
                    else:
                        metadata["age"] = torch.tensor(age, dtype=torch.float32)
                # V9: Support age_continuous for continuous age encoding
                if "age_continuous" in batch:
                    age_cont = batch["age_continuous"][i] if batch["age_continuous"].dim() > 0 else batch["age_continuous"]
                    if isinstance(age_cont, torch.Tensor):
                        metadata["age_continuous"] = age_cont.clone().detach().cpu()
                    else:
                        metadata["age_continuous"] = torch.tensor(age_cont, dtype=torch.float32)
                if "disease_labels" in batch:
                    disease_labels = batch["disease_labels"][i] if batch["disease_labels"].dim() > 0 else batch["disease_labels"]
                    if isinstance(disease_labels, torch.Tensor):
                        metadata["disease_labels"] = disease_labels.clone().detach().cpu()
                    else:
                        metadata["disease_labels"] = disease_labels
                
                all_prompts_data.append({
                    "prompt": prompt,
                    "metadata": metadata,  # Store only extracted metadata, not entire batch
                    "real_image_ref": real_image_ref,  # Store reference path here
                })

                if max_samples is not None and len(all_prompts_data) >= max_samples:
                    break
            
            if max_samples is not None and len(all_prompts_data) >= max_samples:
                break

        if self.accelerator.is_local_main_process:
            logger.info(f"Collected {len(all_prompts_data)} prompts from dataset")
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

        # Track local index for this GPU (starts from 0 for each GPU)
        local_idx = 0
        samples_processed = 0

        with torch.no_grad():
            progress_bar = tqdm(
                total=len(images_to_generate),
                desc=f"Generating (GPU {gpu_idx})",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch_start in range(0, len(images_to_generate), generation_batch_size):
                batch_end = min(batch_start + generation_batch_size, len(images_to_generate))
                batch_images = images_to_generate[batch_start:batch_end]

                # Generate images
                images, prompts, metadata_list, batch_images_list, real_image_refs = self._generate_image_batch(
                    batch_images, all_prompts_data, batch_start_idx=batch_start
                )

                # Save images directly to disk (using local index per GPU)
                self._save_images_batch(
                    images, prompts, metadata_list, batch_images_list, real_image_refs, output_dir, local_idx
                )

                local_batch_size = images.shape[0]
                local_idx += local_batch_size
                samples_processed += local_batch_size

                progress_bar.update(local_batch_size)
                progress_bar.set_postfix({"samples": samples_processed})

                # Clear cache periodically
                if batch_start % (generation_batch_size * 10) == 0:
                    torch.cuda.empty_cache()

            progress_bar.close()

        # NOTE: DO NOT use wait_for_everyone() here - it causes NCCL timeouts
        # Each GPU saves independently to disk, so no synchronization is needed
        # The validation code follows the same pattern (no wait_for_everyone at end)
        
        # Close CSV tracking file
        self._close_csv_tracking()
        
        logger.info(f"{self._rank_prefix()} Generation complete!")
        logger.info(f"{self._rank_prefix()}   Output directory: {output_dir}")
        logger.info(f"{self._rank_prefix()}   Images saved in gpu_{self.accelerator.process_index}/ subdirectory")
        logger.info(f"{self._rank_prefix()}   Generated {samples_processed} images")
        logger.info(f"{self._rank_prefix()}   CSV log saved: {output_dir}/gpu_{self.accelerator.process_index}/generation_log.csv")
        
        if self.accelerator.is_main_process:
            logger.info(f"✓ All GPUs finished generation")
            logger.info(f"  Total images: {samples_processed * self.accelerator.num_processes} (across {self.accelerator.num_processes} GPUs)")
            logger.info(f"  CSV logs saved in each gpu_*/generation_log.csv - combine with: cat gpu_*/generation_log.csv > all_generations.csv")


def merge_csv_files(output_dir: str, output_filename: str = "all_generations.csv"):
    """Merge all GPU CSV files into a single file.
    
    Args:
        output_dir: Output directory containing gpu_* subdirectories
        output_filename: Name of merged output file
    """
    output_dir = Path(output_dir)
    csv_files = sorted(output_dir.glob("gpu_*/generation_log.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {output_dir}")
        return None
    
    merged_path = output_dir / output_filename
    
    with open(merged_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = None
        for i, csv_file in enumerate(csv_files):
            with open(csv_file, 'r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                if writer is None:
                    # Write header from first file
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    
    print(f"✓ Merged {len(csv_files)} CSV files into {merged_path}")
    return merged_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from WDS")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint directory (optional, overrides generation_checkpoint_path from config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for synthetic images (optional, can be in config)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to generate (optional)")
    parser.add_argument("--merge_csv", action="store_true", help="Merge all GPU CSV files into single file after generation (run on main process only)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_file)
    if config is None:
        raise ValueError(f"Failed to load config from {args.config_file}")

    # Get checkpoint path: command line arg takes precedence, then config
    checkpoint_path = None
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    elif "generation_checkpoint_path" in config:
        checkpoint_path = config["generation_checkpoint_path"]
    elif "checkpoint_path" in config:  # Backward compatibility
        checkpoint_path = config["checkpoint_path"]
    
    # Require checkpoint path
    if checkpoint_path is None or checkpoint_path == "":
        raise ValueError(
            "checkpoint_path is REQUIRED. Please provide one of:\n"
            "  1. generation_checkpoint_path in config file, OR\n"
            "  2. --checkpoint_path via command line argument"
        )
    
    # Store in config for use in generate_dataset
    config["generation_checkpoint_path"] = checkpoint_path
    if args.output_dir is not None:
        config["synthetic_output_dir"] = args.output_dir
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if not torch.cuda.is_available():
        logger.error("CUDA is not available! This script requires GPU.")
        raise RuntimeError("CUDA not available")

    logger.info(f"✓ Accelerator initialized on {accelerator.device}")

    # Initialize generator
    generator = SyntheticDatasetGenerator(
        accelerator=accelerator,
        args=config,
        logger=logger,
    )

    # Generate dataset
    checkpoint_path = config.get("generation_checkpoint_path")
    output_dir = config.get("synthetic_output_dir", "synthetic_dataset")
    max_samples = config.get("max_samples")

    try:
        generator.generate_dataset(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            max_samples=max_samples,
        )
    finally:
        # Ensure CSV file is closed even on error
        generator._close_csv_tracking()
    
    # Merge CSV files (only on main process to avoid race conditions)
    if args.merge_csv and accelerator.is_main_process:
        import time
        # Wait a bit for other processes to finish writing
        time.sleep(5)
        merge_csv_files(output_dir)


if __name__ == "__main__":
    main()

