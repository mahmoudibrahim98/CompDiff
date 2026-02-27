from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import yaml
import argparse


@dataclass
class BaseConfig:
    """All the configurable parameters for the training script."""

    # Path to pretrained model or model identifier from huggingface.co/models.
    pretrained_model_name_or_path: str = None
    # Revision of pretrained model identifier from huggingface.co/models.
    revision: str = None
    # Path to text_encoder model or model identifier from huggingface.co/models.
    pretrained_text_encoder_name_or_path: str = None
    # Hugging face authentication token
    use_auth_token: str = None
    embedding_method: str = "last_hidden_state"
    use_attention_mask: bool = False
    # Cache directory when loading models
    cache_dir: str = None
    # Whether to initialize the unet with random weights
    random_unet: bool = False
    enforce_tokenizer_max_sentence_length: int = None

    # Whether to load dataset from wds instead of image/prompt directories
    use_wds_dataset: bool = False
    url_root: str = None
    # Path to WDS dataset for test/generation (directory with .tar files or single .tar file)
    wds_dataset_path: str = None
    # A folder containing the training data of images.
    image_dir: str = None
    image_type: str = "pt"
    # A folder containing the training data of prompts.
    prompt_dir: str = None
    # A file filtering based on file names.
    data_filter_file: str = None
    data_filter_split_token: str = "\n"
    loss_weights_file: str = None
    loss_weights_split_token: str = "\n"

    inference_prompt_file: str = None
    inference_prompt_split_token: str = "\n"
    inference_prompt_number_per_prompt: int = 4
    inference_prompt_output_file: str = None

    # save only the weights that were modified instead of the entire pipeline
    save_only_modified_weights: bool = False
    do_not_save_weights: bool = False
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: str = None

    # Seed number for reproducible training
    seed: int = 10
    # The resolution for input images, all the images in the train/validation
    # dataset will be resized to this
    resolution: int = 512
    # Whether to center crop images before resizing to resolution
    center_crop: bool = False

    # Whether to train the text encoder
    train_text_encoder: bool = False
    # Batch size (per device) for the training dataloader.
    train_batch_size: int = 4
    # For debugging purposes or quicker training, truncate the number
    # of training examples to this value if set
    max_train_samples: int = None
    num_train_epochs: int = 100
    # Total number of training steps to perform.
    # If provided, overrides num_train_epochs.
    max_train_steps: int = None
    # Maximum number of steps to actually train (when training should end).
    # If provided, training will stop at this step even if max_train_steps is higher.
    # This does NOT affect the learning rate schedule, which is still based on max_train_steps.
    # If None, training will continue until max_train_steps.
    max_actual_train_steps: int = None
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1
    # Whether or not to use gradient checkpointing to save memory
    # at the expense of slower backward pass.
    gradient_checkpointing: bool = False

    # Initial learning rate (after the potential warmup period) to use.
    learning_rate: float = 5e-06
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    scale_lr: bool = False
    # Choose between ["linear", "cosine", "cosine_with_restarts",
    # "polynomial", "constant", "constant_with_warmup"]
    lr_scheduler: str = "constant"
    # Number of steps for the warmup in the lr scheduler.
    lr_warmup_steps: int = 500
    # Whether or not to use 8-bit Adam from bitsandbytes.
    use_8bit_adam: bool = False
    # Whether to use EMA model for the unet.
    use_ema: bool = False

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-02
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0

    logging_dir: str = "logs"
    report_to: str = "wandb"

    # Choose between fp16 and bf16 (requires Nvidia Ampere GPU)
    mixed_precision: str = "no"
    # For distributed training: local_rank
    local_rank: int = 1

    # Save a checkpoint of the training state every X updates.
    # Checkpoints can be used for resuming training via `--resume_from_checkpoint`.
    checkpointing_steps: int = 500
    # Max number of checkpoints to store.
    checkpoints_total_limit: int = None
    # Whether training should be resumed from a previous checkpoint. Use a path saved by
    # `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
    resume_from_checkpoint: str = None

    # ====== HCN (Hierarchical Conditioner Network) Parameters ======
    # Whether to use HCN for compositional demographic embeddings
    use_hcn: bool = False
    # Note: use_hcn_v8 and use_hcn_v7 are deprecated - HCN is now the default
    # These flags are kept for backward compatibility but are no longer needed
    use_hcn_v8: bool = False  # Deprecated: no longer needed, kept for compatibility
    use_hcn_v7: bool = False  # Deprecated: no longer needed, kept for compatibility
    # Hidden dimension for HCN embeddings
    hcn_d_node: int = 256
    # Output dimension for HCN (should match text encoder output, e.g., 1024 for SD 2.1)
    hcn_d_ctx: int = 1024
    # Dropout probability in HCN MLPs
    hcn_dropout: float = 0.1
    # Whether to use uncertainty quantification in HCN
    hcn_use_uncertainty: bool = True
    # Number of age bins for categorization
    hcn_num_age_bins: int = 5
    # Age bin thresholds (list of age thresholds). If None, uses default [18, 40, 60, 80] for 5 bins.
    # For N bins, provide N-1 thresholds. Example: [10, 20, 30, 40, 50, 60, 70, 80, 90] creates 10 bins.
    # If None, will be auto-generated from hcn_num_age_bins if needed.
    age_bins: List[int] = None
    # Number of sex categories (typically 2: M/F)
    hcn_num_sex: int = 2
    # Number of race/ethnicity categories
    hcn_num_race: int = 4
    # Weight for KL divergence loss (uncertainty regularization)
    hcn_kl_weight: float = 0.001
    # Number of steps to anneal KL weight from 0 to target value
    hcn_kl_anneal_steps: int = 10000
    # Weight for compositional consistency loss
    hcn_comp_weight: float = 0.01
    # Weight for auxiliary loss (overall multiplier)
    hcn_aux_weight: float = 0.1
    # Per-attribute weights for auxiliary loss (used by HCN V7)
    # These allow different importance for age, sex, and race classification
    hcn_aux_weight_age: float = 1.0
    hcn_aux_weight_sex: float = 1.0
    hcn_aux_weight_race: float = 1.0
    # Hidden dimension for auxiliary classifiers (used by HCN V7)
    hcn_aux_hidden_dim: int = 512
    # Whether to encode age in HCN hierarchy (V10: if False, only sex × race composition)
    hcn_encode_age: bool = True
    # Age loss mode: 'ce' (cross-entropy), 'ordinal' (recommended), 'soft_ce', or 'corn'
    hcn_age_loss_mode: str = 'ce'
    # Gaussian sigma for soft_ce mode (only used when hcn_age_loss_mode='soft_ce')
    hcn_soft_ce_sigma: float = 0.75
    
    # ====== V6: Timestep Embedding Injection ======
    # When True, HCN outputs an embedding that gets ADDED to UNet's timestep embedding
    use_hcn_timestep_injection: bool = False
    # Timestep embedding dimension (1280 for SD 2.1, 4 * block_out_channels[0])
    hcn_d_time_emb: int = 1280
    # Disable FiLM mode (mutually exclusive with timestep injection)
    use_hcn_film: bool = False

    # ====== Demographic Encoder (V4) Parameters ======
    # Whether to use DemographicEncoder (lightweight embeddings-based conditioning)
    use_demographic_encoder: bool = False
    # Hidden dimension for demographic embeddings
    demo_d_hidden: int = 256
    # Output dimension (should match text encoder output, e.g., 1024 for SD 2.1)
    demo_d_output: int = 1024
    # Mode for demographic encoder: 'single' (fused token) or 'separate' (3 tokens)
    demo_mode: str = 'single'
    # Classifier depth: 'shallow' (single linear) or 'deep' (MLP like HCN V7/V8)
    demo_classifier_depth: str = 'shallow'
    # Hidden dimension for deep auxiliary classifiers (only used when classifier_depth='deep')
    demo_aux_hidden_dim: int = 512
    # Dropout probability in demographic encoder MLP (deprecated, but used for deep classifiers)
    demo_dropout: float = 0.1
    # Number of age bins for categorization
    demo_num_age_bins: int = 5
    # Note: age_bins (defined above) is shared between HCN and DemographicEncoder
    # Number of sex categories (typically 2: M/F)
    demo_num_sex: int = 2
    # Number of race/ethnicity categories
    demo_num_race: int = 4
    # Weight for auxiliary demographic classification losses
    demo_aux_weight: float = 1.0
    # Whether to use demographic dropout strategy (50% of batches remove demographics from text)
    demo_use_dropout: bool = False
    # Dropout probability for removing demographics from text prompt
    demo_text_dropout_prob: float = 0.5
    # Step at which to start demographic dropout (allows warm-up period)
    demo_dropout_start_step: int = 0
    # Whether to strip demographics from training prompts before tokenization.
    # This option works independently of use_hcn or use_demographic_encoder modes.
    # When True, demographics will be stripped regardless of the mode settings.
    strip_demographics: bool = False
    # Whether to keep age in the prompt even when demographics are stripped.
    # When True and strip_demographics is True, age will be preserved in the prompt.
    # Default is False (age is stripped along with other demographics).
    keep_age_in_prompt: bool = False
    # Path to pretrained demographic encoder (optional)
    demographic_encoder_pretrained_path: str = None
    
    # ====== FairDiffusion Parameters ======
    use_fairdiffusion: bool = False
    fairdiffusion_input_perturbation: float = 0.0
    fairdiffusion_time_window: int = 250
    fairdiffusion_exploitation_rate: float = 0.7
    fairdiffusion_sigma_init: float = 1.0
    fairdiffusion_sigma_min: float = 0.0
    fairdiffusion_sigma_max: float = 1.0
    fairdiffusion_min_instance_weight: float = 0.1
    fairdiffusion_ucb_beta: float = 0.1
    fairdiffusion_attribute_fields: List[str] = field(
        default_factory=lambda: ["race_idx", "sex_idx", "age_idx"]
    )
    fairdiffusion_attribute_cardinalities: Dict[str, int] = field(default_factory=dict)

    # ====== FairCLIP Regularizer (Option 1: fairness loss via frozen FairCLIP) ======
    # When True, add a Sinkhorn-based fairness loss so image–text similarity is balanced across demographic groups.
    use_fairclip_regularizer: bool = False
    # Path to FairCLIP checkpoint (.pth with 'model_state_dict' from FairCLIP training).
    fairclip_checkpoint_path: str = None
    # Attribute to balance: race_idx | sex_idx (must exist in batch).
    fairclip_attribute: str = "race_idx"
    # Weight for the FairCLIP fairness loss (e.g. 1e-4).
    fairclip_loss_weight: float = 1e-4
    # Sinkhorn blur for geomloss (e.g. 1e-4).
    fairclip_sinkhorn_blur: float = 1e-4
    # FairCLIP model arch (must match checkpoint): vit-b16 | vit-l14
    fairclip_model_arch: str = "vit-b16"

    # ====== Validation Parameters ======
    # Whether to run validation during training
    run_validation: bool = False
    # Run validation every N steps
    validation_steps: int = 2500
    # Optional offsets applied to each validation interval (e.g., [-1500, 0])
    validation_schedule_offsets: Optional[List[int]] = None
    # Optional minimum global step before applying schedule
    validation_schedule_min_step: Optional[int] = None
    # Number of validation samples to use (-1 for all)
    num_validation_samples: int = 100
    # Batch size for validation generation (number of images generated in parallel)
    val_batch_size: int = 1
    # Path to validation CSV file
    validation_csv: str = None
    # Path to real validation images directory
    validation_images_dir: str = None
    # Path to sex model checkpoint for validation
    validation_sex_model_path: str = None
    # Number of images to generate per validation prompt
    validation_num_images_per_prompt: int = 4
    # Guidance scale for validation generation
    validation_guidance_scale: float = 7.5
    # Number of inference steps for validation
    validation_num_inference_steps: int = 50
    # Whether to save validation images
    validation_save_images: bool = False
    # Batch size for loading and processing validation images when computing metrics (to avoid OOM)
    validation_metrics_batch_size: int = 32
    # Whether to strip demographics from validation prompts before tokenization.
    # This option works independently of use_hcn or use_demographic_encoder modes.
    # When True, demographics will be stripped regardless of the mode settings.
    strip_demographics_in_validation: bool = True
    # Whether to keep age in the prompt even when demographics are stripped in validation.
    # When True and strip_demographics_in_validation is True, age will be preserved in the prompt.
    # Default is False (age is stripped along with other demographics).
    keep_age_in_prompt_validation: bool = False
    # Whether to compute subgroup-specific metrics (FID, MS-SSIM, BioViL per sex/race/age and intersectional subgroups)
    # This can significantly increase validation time, so it's disabled by default
    compute_subgroup_metrics: bool = False
    # Whether to compute disease AUROC during validation
    compute_disease_auroc: bool = False
    # Whether to compute fundus disease AUROC (glaucoma/CDR) during validation
    compute_fundus_disease_auroc: bool = False
    # Whether to compute demographic prediction metrics during validation
    compute_demographic_prediction: bool = False
    # Whether to compute BioViL similarity during validation
    compute_biovil_similarity: bool = False
    # Whether to compute FID with RadImageNet during validation
    compute_fid_radimagenet: bool = False
    # Whether to compute FairDiffusion metrics (FID, MIFID, IS) during validation
    compute_fairdiffusion_metrics: bool = False
    # Demographic groups for FairDiffusion groupwise metrics (e.g. [" Asian", " Black", " Female", " Male"])
    demographic_groups: List[str] = field(default_factory=list)
    # Path to test directory (used for test mode validation)
    test_dir: str = None
    # Path to test CSV file
    test_csv: str = None
    # Path to test images directory
    test_images_dir: str = None
    # Path to training directory (used for generation_train mode)
    train_dir: str = None
    # Path to GT (Ground Truth) dataset directory (used for GT generation mode)
    GT_dir: str = None
    # Alternative name for GT directory (for compatibility)
    gt_dir: str = None
    # Path to GT WDS dataset for synthetic generation (used by generate_synthetic_dataset.py)
    gt_wds_dataset_path: str = None
    # Number of images to generate per prompt (for generation/inference)
    num_images_per_prompt: int = None
    # Batch size for generation (number of images generated in parallel)
    generation_batch_size: int = None
    # Path to checkpoint for generation (used by generation scripts, not training)
    generation_checkpoint_path: str = None

    def get_config(self):
        return self.__dict__


def load_config(file_path):
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None


def get_args_from_config():
    parser = argparse.ArgumentParser(
        description="Training script to fine-tune unet of the stable diffusion model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        required=True,
        help="Experiment config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config_file)
    my_args = BaseConfig(**config)
    return my_args
