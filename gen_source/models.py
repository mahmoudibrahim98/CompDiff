import itertools
import os
from typing import Iterable, Optional

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    CLIPTextConfig,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPFeatureExtractor,
)
from huggingface_hub.errors import RepositoryNotFoundError

from diffusers.utils.import_utils import is_xformers_available


##########################################################
def load_models(args, accelerator, logger):

    kwargs_from_pretrained = {}
    if args.cache_dir is not None:
        kwargs_from_pretrained["cache_dir"] = args.cache_dir
        kwargs_from_pretrained["revision"] = args.revision
    # Hugging Face token for gated models (e.g. stabilityai/stable-diffusion-2-1-base)
    hf_token = getattr(args, "use_auth_token", None) or os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs_from_pretrained["token"] = hf_token

    # Text encoder name to use (pretrained if specified, or default same as model name)
    text_encoder_name = (
        args.pretrained_text_encoder_name_or_path
        if args.pretrained_text_encoder_name_or_path is not None
        else args.pretrained_model_name_or_path
    )

    # Known Hub IDs for Stable Diffusion (use CLIP + subfolders text_encoder/, tokenizer/)
    _SD_HUB_IDS = [
        "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2",
        "stabilityai/stable-diffusion-2-base",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-1-base",
    ]
    # Local folder with same layout as Hub (e.g. saved via save_sd21_cache_as_local.py)
    _is_local_sd_path = (
        isinstance(text_encoder_name, str)
        and os.path.isdir(text_encoder_name)
        and os.path.isfile(os.path.join(text_encoder_name, "text_encoder", "config.json"))
    )
    _use_sd_subfolders = text_encoder_name in _SD_HUB_IDS or _is_local_sd_path

    # When using a local folder, load only from disk (no Hub calls)
    _model_path = getattr(args, "pretrained_model_name_or_path", None)
    if (
        isinstance(_model_path, str)
        and os.path.isdir(_model_path)
        and os.path.isfile(os.path.join(_model_path, "text_encoder", "config.json"))
    ):
        kwargs_from_pretrained["local_files_only"] = True

    if _use_sd_subfolders:
        text_encoder_config_class = CLIPTextConfig
        tokenizer_class = CLIPTokenizer
        text_encoder_model_class = CLIPTextModel
        freeze_pooler = False
        tokenizer_subfolder_to_use = "tokenizer"
        model_subfolder_to_use = "text_encoder"
    else:
        text_encoder_config_class = AutoConfig
        tokenizer_class = AutoTokenizer
        text_encoder_model_class = AutoModel
        freeze_pooler = True
        tokenizer_subfolder_to_use = None
        model_subfolder_to_use = None

    # Get text encoder config
    if model_subfolder_to_use is not None:
        text_encoder_config = text_encoder_config_class.from_pretrained(
            text_encoder_name,
            subfolder=model_subfolder_to_use,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )
    else:
        text_encoder_config = text_encoder_config_class.from_pretrained(
            text_encoder_name,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )

    # CHECK: which case is this?
    if args.embedding_method != "last_hidden_state" and accelerator.is_main_process:
        logger.info(
            "Careful, there might be compatibilities of your selected embedding method with the stable diffusion pipeline"
        )
    # CHECK: which case is this?
    if args.embedding_method[:-1] == "hidden_state_numbered_from_the_end_":
        assert args.embedding_method[-1].isnumeric()
        text_encoder_config.output_hidden_states = True

    # Load the text encoder
    if model_subfolder_to_use is not None:
        text_encoder = text_encoder_model_class.from_pretrained(
            text_encoder_name,
            config=text_encoder_config,
            subfolder=model_subfolder_to_use,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )
    else:
        text_encoder = text_encoder_model_class.from_pretrained(
            text_encoder_name,
            config=text_encoder_config,
            trust_remote_code=True,
            **kwargs_from_pretrained,
        )

    # Load the tokenizer
    # Handle potential 404 error when transformers tries to check for chat templates
    tokenizer_model_max_length = (
        args.enforce_tokenizer_max_sentence_length
        if args.enforce_tokenizer_max_sentence_length is not None
        else (
            text_encoder_config.max_position_embeddings
            if hasattr(text_encoder_config, "max_position_embeddings")
            else None
        )
    )
    try:
        if tokenizer_subfolder_to_use is not None:
            tokenizer = tokenizer_class.from_pretrained(
                text_encoder_name,
                model_max_length=tokenizer_model_max_length,
                subfolder=tokenizer_subfolder_to_use,
                trust_remote_code=True,
                **kwargs_from_pretrained,
            )
        else:
            tokenizer = tokenizer_class.from_pretrained(
                text_encoder_name,
                model_max_length=tokenizer_model_max_length,
                trust_remote_code=True,
                **kwargs_from_pretrained,
            )
    except (RepositoryNotFoundError, Exception) as e:
        # If there's an error (e.g., 404 for chat templates), try with local_files_only
        if isinstance(e, RepositoryNotFoundError) or "404" in str(e) or "RepositoryNotFoundError" in str(type(e)):
            logger.warning(f"Got repository error when loading tokenizer, trying with local cache: {e}")
            try:
                if tokenizer_subfolder_to_use is not None:
                    tokenizer = tokenizer_class.from_pretrained(
                        text_encoder_name,
                        model_max_length=tokenizer_model_max_length,
                        subfolder=tokenizer_subfolder_to_use,
                        trust_remote_code=True,
                        local_files_only=True,
                        **kwargs_from_pretrained,
                    )
                else:
                    tokenizer = tokenizer_class.from_pretrained(
                        text_encoder_name,
                        model_max_length=tokenizer_model_max_length,
                        trust_remote_code=True,
                        local_files_only=True,
                        **kwargs_from_pretrained,
                    )
            except Exception as e2:
                # If that fails, try without subfolder (fallback)
                logger.warning(f"Local files only failed, trying alternative loading: {e2}")
                tokenizer = tokenizer_class.from_pretrained(
                    text_encoder_name,
                    model_max_length=tokenizer_model_max_length,
                    trust_remote_code=True,
                    **kwargs_from_pretrained,
                )
        else:
            raise

    # Load the vae
    if args.image_type == "pt":
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            **kwargs_from_pretrained,
        )
    else:
        vae = None

    # Get the unet config
    # CHECK: which cases is this block used in?
    unet_config_changed = False
    if text_encoder_name != args.pretrained_model_name_or_path:
        # In this case, we need to check that things match
        unet_config = UNet2DConditionModel.load_config(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            **kwargs_from_pretrained,
        )
        if text_encoder_config.hidden_size != unet_config["cross_attention_dim"]:
            unet_config_changed = True

            if accelerator.is_main_process:
                logger.info(
                    f"different hidden size {text_encoder_config.hidden_size} {unet_config['cross_attention_dim']}"
                )
                logger.info(
                    "Unet config will be updated to match the text encoder hidden size"
                )

            unet_config["cross_attention_dim"] = text_encoder_config.hidden_size

        if unet_config_changed:
            if not args.random_unet:
                if accelerator.is_main_process:
                    logger.info(
                        "you did not choose to start unet randomly, rectifying!"
                    )
                args.random_unet = True
    else:
        unet_config = None

    # Load the unet
    if args.random_unet:
        if accelerator.is_main_process:
            logger.info("Initializing unet randomly")

        if unet_config_changed:
            if accelerator.is_main_process:
                logger.info("init unet from config file")
            unet = UNet2DConditionModel.from_config(
                unet_config,
                **kwargs_from_pretrained,
            )
        else:
            if accelerator.is_main_process:
                logger.info("init unet from online")
            unet = UNet2DConditionModel.from_config(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                **kwargs_from_pretrained,
            )
    else:
        assert not unet_config_changed
        if accelerator.is_main_process:
            logger.info("Initializing unet with saved weights")
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            **kwargs_from_pretrained,
        )

    # Load the noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        **kwargs_from_pretrained,
    )

    return (
        text_encoder,
        tokenizer,
        vae,
        unet,
        noise_scheduler,
        text_encoder_name,
        freeze_pooler,
        unet_config_changed,
        unet_config,
        kwargs_from_pretrained,
    )


##########################################################
# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            (
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
            )
            for p in self.shadow_params
        ]


##########################################################
def load_hcn(args, logger):
    """
    Load and initialize HCN (Hierarchical Conditioner Network) if enabled.

    Args:
        args: Training arguments/config
        logger: Logger instance

    Returns:
        hcn: HierarchicalConditioner or HierarchicalConditionerV8 instance or None if not enabled
    """
    if not args.use_hcn:
        logger.info("HCN disabled (use_hcn=False)")
        return None

    # CompDiff-2: typed compositional conditioner (checked FIRST — supersedes
    # all HCN variants when enabled)
    if getattr(args, 'use_compdiff2', False):
        logger.info("Loading CompDiff-2 Typed Compositional Conditioner")
        from compdiff2 import load_compdiff2
        return load_compdiff2(args, logger)

    # Check if using V9 continuous age mode
    use_continuous_age = getattr(args, 'use_continuous_age', False)
    
    if use_continuous_age:
        # V9: Continuous age encoding
        logger.info("Loading HCN V9 with Continuous Age Encoding")
        try:
            from hcn_v9_continuous_age import HierarchicalConditionerV9, load_hcn_v9
        except ImportError:
            raise ImportError(
                "Could not import HCN V9 module. Make sure hcn_v9_continuous_age.py is in the same directory."
            )
        return load_hcn_v9(args, logger)
        
    # Check if we should use HCN V7 (hcn_v7.py) instead of the default (hcn.py)
    use_hcn_v7 = getattr(args, 'use_hcn_v7', False)
    
    if use_hcn_v7:
        # Check if using ordinal age loss (new V8 variant)
        use_ordinal_age = getattr(args, 'hcn_age_loss_mode', 'ce') != 'ce'
        
        if use_ordinal_age:
            logger.info("Initializing HCN V8 with Ordinal Age Loss")
            from hcn_v8_ordinal import HierarchicalConditionerV8Ordinal, load_hcn_v8_ordinal
            hcn = load_hcn_v8_ordinal(args, logger)
        else:
            logger.info("Initializing HCN V8 (standard CE)")
            from hcn_v7 import HierarchicalConditionerV8, load_hcn_v8
            hcn = load_hcn_v8(args, logger)
        
        return hcn
    else:
        logger.info("Initializing Hierarchical Conditioner Network (HCN from hcn.py)")

    try:
        from hcn import HierarchicalConditioner
    except ImportError:
        raise ImportError(
            "Could not import HCN module. Make sure hcn.py is in the same directory."
        )

    # Determine if auxiliary loss should be enabled based on hcn_aux_weight
    use_aux_loss = getattr(args, 'hcn_aux_weight', 0.0) > 0.0
    
    # Determine if FiLM output mode should be used (V5)
    use_film_output = getattr(args, 'use_hcn_film', False)
    
    # Determine if timestep injection should be used (V6)
    use_timestep_injection = getattr(args, 'use_hcn_timestep_injection', False)
    d_time_emb = getattr(args, 'hcn_d_time_emb', 1280)
    
    hcn = HierarchicalConditioner(
        num_age_bins=args.hcn_num_age_bins,
        num_sex=args.hcn_num_sex,
        num_race=args.hcn_num_race,
        d_node=args.hcn_d_node,
        d_ctx=args.hcn_d_ctx,
        dropout=args.hcn_dropout,
        use_uncertainty=args.hcn_use_uncertainty,
        use_aux_loss=use_aux_loss,
        use_film_output=use_film_output,
        use_timestep_injection=use_timestep_injection,
        d_time_emb=d_time_emb,
    )

    num_params = sum(p.numel() for p in hcn.parameters())
    logger.info(f"HCN initialized with {num_params:,} parameters")
    logger.info(f"  - Age bins: {args.hcn_num_age_bins}")
    logger.info(f"  - Sex categories: {args.hcn_num_sex}")
    logger.info(f"  - Race categories: {args.hcn_num_race}")
    logger.info(f"  - Node dimension: {args.hcn_d_node}")
    logger.info(f"  - Context dimension: {args.hcn_d_ctx}")
    logger.info(f"  - Uncertainty: {args.hcn_use_uncertainty}")
    logger.info(f"  - Auxiliary loss: {use_aux_loss} (hcn_aux_weight={getattr(args, 'hcn_aux_weight', 0.0)})")
    logger.info(f"  - FiLM output mode (V5): {use_film_output}")
    logger.info(f"  - Timestep injection (V6): {use_timestep_injection}")

    return hcn


##########################################################
def load_film_components(args, unet, logger):
    """
    Load FiLM adapter and create wrapped UNet if FiLM mode is enabled.
    
    Args:
        args: Config arguments
        unet: The base UNet model
        logger: Logger instance
        
    Returns:
        Tuple of (film_adapter, wrapped_unet) or (None, unet) if FiLM disabled
    """
    if not getattr(args, 'use_hcn_film', False):
        logger.info("FiLM conditioning disabled (use_hcn_film=False)")
        return None, unet
    
    try:
        from film import FiLMAdapter, FiLMUNetWrapper, get_unet_block_channels
    except ImportError:
        raise ImportError(
            "Could not import FiLM module. Make sure film.py is in the same directory."
        )
    
    logger.info("=" * 60)
    logger.info("Initializing FiLM Conditioning (V5)")
    logger.info("=" * 60)
    
    # Get UNet block channels
    block_channels = get_unet_block_channels(unet)
    logger.info(f"  UNet has {len(block_channels)} ResNet blocks")
    logger.info(f"  Block channels: {block_channels[:5]}... (showing first 5)")
    
    # Create FiLM adapter
    d_input = getattr(args, 'hcn_d_node', 256)
    d_hidden = getattr(args, 'film_d_hidden', 512)
    
    film_adapter = FiLMAdapter(
        d_input=d_input,
        block_channels=block_channels,
        d_hidden=d_hidden,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in film_adapter.parameters())
    logger.info(f"  FiLM adapter parameters: {num_params:,}")
    
    # Create wrapped UNet
    film_scale = getattr(args, 'film_scale', 1.0)
    film_blocks = getattr(args, 'film_blocks', 'all')
    
    wrapped_unet = FiLMUNetWrapper(
        unet=unet,
        film_adapter=film_adapter,
        film_scale=film_scale,
        film_blocks=film_blocks,
    )
    
    logger.info(f"  FiLM scale: {film_scale}")
    logger.info(f"  FiLM blocks: {film_blocks}")
    logger.info("=" * 60)
    
    return film_adapter, wrapped_unet
