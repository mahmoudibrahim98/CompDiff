import itertools, os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import shutil
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import warnings
import json
from accelerate import PartialState


class TimestepInjectionContext:
    """
    Context manager for V6 timestep embedding injection.
    
    Temporarily hooks into the UNet's time_embedding layer to add
    demographic conditioning to the timestep embedding.
    
    Usage:
        with TimestepInjectionContext(unet, hcn_time_emb):
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)
    """
    
    def __init__(self, unet, hcn_time_emb: torch.Tensor):
        """
        Args:
            unet: The UNet model (can be DDP wrapped)
            hcn_time_emb: [B, d_time_emb] embedding to add to timestep
        """
        self.unet = unet
        self.hcn_time_emb = hcn_time_emb
        self.hook_handle = None
        
    def _hook_fn(self, module, input, output):
        """Hook that adds HCN embedding to timestep embedding output."""
        # output is [B, d_time_emb] from time_embedding layer
        return output + self.hcn_time_emb
    
    def __enter__(self):
        # Get the unwrapped UNet (in case of DDP)
        unet_unwrapped = self.unet
        if hasattr(self.unet, 'module'):
            unet_unwrapped = self.unet.module
        
        # Register hook on time_embedding layer
        if hasattr(unet_unwrapped, 'time_embedding'):
            self.hook_handle = unet_unwrapped.time_embedding.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove the hook
        if self.hook_handle is not None:
            self.hook_handle.remove()
        return False


# Validation has been moved to a separate script: run_validation_monitor.py
# The run_validation_pass function has been removed from here.
def train_loop(
    logger,
    args,
    initial_global_step,
    first_epoch,
    accelerator,
    train_dataloader,
    unet,
    text_encoder,
    vae,
    noise_scheduler,
    weight_dtype,
    optimizer,
    lr_scheduler,
    ema_unet,
    hcn=None,  # Hierarchical Conditioner Network for compositional demographics
    demographic_encoder=None,  # V4: Lightweight demographic encoder
    fair_controller=None,
    film_adapter=None,  # V5: FiLM adapter for demographic conditioning
):
    # Only show the progress bar once on each machine.
    # Use max_actual_train_steps if provided, otherwise use max_train_steps
    max_steps_for_progress = args.max_actual_train_steps if args.max_actual_train_steps is not None else args.max_train_steps
    progress_bar = tqdm(
        range(0, max_steps_for_progress),
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = initial_global_step
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info("Epoch {}, global step {}".format(epoch, global_step))

        if args.train_text_encoder:
            text_encoder.train()

        # Set HCN to training mode
        if hcn is not None:
            hcn.train()
        
        # Set DemographicEncoder to training mode
        if demographic_encoder is not None:
            demographic_encoder.train()
        
        # Set FiLM adapter to training mode (V5)
        if film_adapter is not None:
            film_adapter.train()

        for step, batch in enumerate(train_dataloader):
            logger.info("*** batch {} ***".format(batch["pixel_values"].shape))
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.image_type == "pt":
                    # VAE is frozen, so set to eval mode to save memory (no dropout, no batch norm updates)
                    vae.eval()
                    with torch.no_grad():  # No gradients needed for VAE encoding
                        # VAE runs in fp32 for numerical stability, accepts fp32 input
                        latents = vae.encode(
                            batch["pixel_values"].to(dtype=torch.float32)
                        ).latent_dist.sample()
                        # Legacy fudge factor from original dreambooth code
                        latents = latents * 0.18215  # vae.config.scaling_factor
                    # Cast latents to weight_dtype for UNet (this is the only dtype conversion needed)
                    latents = latents.to(dtype=weight_dtype)
                elif args.image_type == "parameters":
                    latents = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    raise Exception("not supported")
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                noise_for_latents = noise
                if (
                    args.use_fairdiffusion
                    and args.fairdiffusion_input_perturbation > 0
                ):
                    noise_for_latents = noise + args.fairdiffusion_input_perturbation * torch.randn_like(
                        noise
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise_for_latents, timesteps
                )

                # Get the text embedding for conditioning
                # Note: For HCN mode, demographics are already stripped at dataset level (v1 behavior)
                # For FairDiffusion/DemographicEncoder modes, full prompt is used
                text_input_ids = batch["input_ids"]
                if args.use_attention_mask:
                    attention_mask = batch["attention_mask"]
                else:
                    attention_mask = None

                prompt_embeds = text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                encoder_hidden_states = prompt_embeds[0]  # [B, 77, d_ctx]

                # === HCN: Hierarchical Conditioning ===
                kl_loss = None
                comp_loss = None
                aux_loss = None
                aux_metrics = None  # Store metrics for logging (used by HCN V7)
                hcn_ctx_norm = None
                hcn_dropout_mask = None
                film_gammas = None
                film_betas = None
                hcn_time_emb = None  # V6: timestep embedding injection
                use_hcn_film = getattr(args, 'use_hcn_film', False)
                use_hcn_timestep_injection = getattr(args, 'use_hcn_timestep_injection', False)
                

                use_compdiff2 = getattr(args, 'use_compdiff2', False)

                if hcn is not None:
                    # Get HCN demographic context
                    # V9: Support both continuous and categorical age
                    use_continuous_age = getattr(args, 'use_continuous_age', False)

                    if use_compdiff2:
                        # CompDiff-2: typed conditioner (age continuous, in composer)
                        hcn_ctx, mu, logsigma, aux_logits, time_emb = hcn(
                            sex_idx=batch["sex_idx"],
                            race_idx=batch["race_idx"],
                            age_continuous=batch["age_continuous"],
                        )
                    elif use_continuous_age:
                        hcn_ctx, mu, logsigma, aux_logits, time_emb = hcn(
                            sex_idx=batch["sex_idx"],
                            race_idx=batch["race_idx"],
                            age_continuous=batch["age_continuous"],
                        )
                    else:
                        # V10: Support optional age encoding
                        encode_age = getattr(hcn, 'encode_age', True)
                        age_idx = batch.get("age_idx") if encode_age else None
                        hcn_ctx, mu, logsigma, aux_logits, time_emb = hcn(
                            sex_idx=batch["sex_idx"],
                            race_idx=batch["race_idx"],
                            age_idx=age_idx,
                        )
                    if use_compdiff2:
                        # === CompDiff-2 Dual-Route Mode ===
                        # Route A: concatenate the T conditioner tokens to the
                        # text context (works for T=1 or multi-token).
                        # Route B: if the conditioner emitted a timestep
                        # modulation vector (zero-init), inject it via the
                        # existing TimestepInjectionContext at the unet call.
                        # NOTE: conditioning dropout is handled INSIDE the
                        # CompDiff-2 module (learned null embeddings), so the
                        # legacy demo_use_dropout token-zeroing is skipped.
                        encoder_hidden_states = torch.cat(
                            [encoder_hidden_states, hcn_ctx], dim=1
                        )  # [B, 77+T, d_ctx]
                        hcn_ctx_norm = hcn_ctx.norm(dim=-1).mean()
                        if time_emb is not None:
                            hcn_time_emb = time_emb

                    elif use_hcn_timestep_injection and time_emb is not None:
                        # === V6 Timestep Injection Mode ===
                        # time_emb is [B, d_time_emb] - gets added to UNet's timestep embedding
                        
                        # Apply null conditioning (dropout) if enabled
                        if getattr(args, 'demo_use_dropout', False) and global_step >= getattr(args, 'demo_dropout_start_step', 0):
                            demo_dropout_prob = getattr(args, 'demo_text_dropout_prob', 0.5)
                            hcn_dropout_mask = torch.rand(bsz, device=time_emb.device) > demo_dropout_prob
                            time_emb = time_emb * hcn_dropout_mask.view(-1, 1).float()
                        
                        hcn_time_emb = time_emb
                        hcn_ctx_norm = time_emb.norm(dim=-1).mean()
                        
                        # Note: In V6 mode, we do NOT concatenate to encoder_hidden_states
                        # The demographic info is injected via timestep embedding
                        
                    elif use_hcn_film and film_adapter is not None:
                        # === V5 FiLM Mode ===
                        # hcn_ctx is [B, d_node] - use it to generate FiLM parameters
                        
                        # Apply null conditioning (dropout) for FiLM
                        if getattr(args, 'demo_use_dropout', False) and global_step >= getattr(args, 'demo_dropout_start_step', 0):
                            demo_dropout_prob = getattr(args, 'demo_text_dropout_prob', 0.5)
                            hcn_dropout_mask = torch.rand(bsz, device=hcn_ctx.device) > demo_dropout_prob
                            # Zero out HCN hidden state for dropped samples
                            hcn_ctx = hcn_ctx * hcn_dropout_mask.view(-1, 1).float()
                        
                        # Generate FiLM parameters from HCN hidden state
                        # Unwrap film_adapter from DDP if needed
                        film_adapter_unwrapped = accelerator.unwrap_model(film_adapter)
                        film_gammas, film_betas = film_adapter_unwrapped(hcn_ctx)
                        
                        # Set FiLM parameters on the UNet wrapper
                        # The wrapper will apply them during forward pass
                        unet_unwrapped = accelerator.unwrap_model(unet)
                        if hasattr(unet_unwrapped, 'set_film_parameters'):
                            unet_unwrapped.set_film_parameters(film_gammas, film_betas)
                        
                        hcn_ctx_norm = hcn_ctx.norm(dim=-1).mean()
                        
                        # Note: In FiLM mode, we do NOT concatenate to encoder_hidden_states
                        # The demographic info is injected via FiLM modulation
                    else:
                        # === V1 Token Mode (backward compatible) ===
                        # hcn_ctx is [B, 1, d_ctx] - concatenate to encoder_hidden_states
                        
                        # Apply null conditioning (demographic dropout) if enabled
                        if getattr(args, 'demo_use_dropout', False) and global_step >= getattr(args, 'demo_dropout_start_step', 0):
                            demo_dropout_prob = getattr(args, 'demo_text_dropout_prob', 0.5)
                            hcn_dropout_mask = torch.rand(bsz, device=hcn_ctx.device) > demo_dropout_prob
                            hcn_ctx = hcn_ctx * hcn_dropout_mask.view(-1, 1, 1).float()

                        # Concatenate text and demographic contexts
                        encoder_hidden_states = torch.cat(
                            [encoder_hidden_states, hcn_ctx], dim=1
                        )  # [B, 78, d_ctx] = 77 text tokens + 1 demographic token

                        hcn_ctx_norm = hcn_ctx.norm(dim=-1).mean()

                    # Compute KL divergence loss (uncertainty regularization)
                    # KL(N(mu, sigma) || N(0, 1))
                    kl_loss = -0.5 * torch.sum(
                        1 + 2 * logsigma - mu ** 2 - torch.exp(2 * logsigma),
                        dim=-1
                    ).mean()

                    # Compute compositional consistency loss
                    # Unwrap hcn from DDP if needed to access custom methods
                    hcn_unwrapped = accelerator.unwrap_model(hcn)

                    if use_compdiff2:
                        comp_loss = hcn_unwrapped.compute_compositional_loss(
                            sex_idx=batch["sex_idx"],
                            race_idx=batch["race_idx"],
                            age_continuous=batch["age_continuous"],
                        )
                    elif use_continuous_age:
                        comp_loss = hcn_unwrapped.compute_compositional_loss(
                            sex_idx=batch["sex_idx"],
                            race_idx=batch["race_idx"],
                            age_continuous=batch["age_continuous"],
                        )
                    else:
                        comp_loss = hcn_unwrapped.compute_compositional_loss(
                            age_idx=batch["age_idx"],
                            sex_idx=batch["sex_idx"],
                            race_idx=batch["race_idx"],
                        )
                    # Auxiliary demographic classification losses
                    # Only compute if aux_logits are available (i.e., use_aux_loss=True)
                    if aux_logits is not None:
                        use_continuous_age = getattr(args, 'use_continuous_age', False)

                        if use_compdiff2:
                            # CompDiff-2: CE(sex) + CE(race) + regression(age) + CE(joint)
                            from compdiff2 import compute_aux_loss_cd2
                            aux_loss, aux_metrics = compute_aux_loss_cd2(
                                aux_logits,
                                sex_idx=batch["sex_idx"],
                                race_idx=batch["race_idx"],
                                age_continuous=batch["age_continuous"],
                                age_idx=batch.get("age_idx"),
                                max_age=float(getattr(args, 'max_age', 100)),
                                sex_weight=getattr(args, 'hcn_aux_weight_sex', 1.0),
                                race_weight=getattr(args, 'hcn_aux_weight_race', 1.0),
                                age_weight=getattr(args, 'hcn_aux_weight_age', 1.0),
                                joint_weight=getattr(args, 'cd2_aux_weight_joint', 0.5),
                                age_loss_scale=getattr(args, 'cd2_age_loss_scale', 10.0),
                                num_sex=getattr(args, 'hcn_num_sex', 2),
                                num_race=getattr(args, 'hcn_num_race', 4),
                            )
                        elif use_continuous_age:
                            # V9: Continuous age mode
                            from hcn_v9_continuous_age import compute_aux_loss
                            aux_loss, aux_metrics = compute_aux_loss(
                                aux_logits,
                                sex_idx=batch["sex_idx"],
                                race_idx=batch["race_idx"],
                                age_continuous=batch["age_continuous"],
                                use_continuous_age=True,
                                num_age_bins=getattr(args, 'hcn_num_age_bins', 5),
                                age_weight=getattr(args, 'hcn_aux_weight_age', 1.0),
                                sex_weight=getattr(args, 'hcn_aux_weight_sex', 1.0),
                                race_weight=getattr(args, 'hcn_aux_weight_race', 1.0),
                            )
                        elif getattr(args, 'use_hcn_v7', False):
                            # V7/V8: Check if using ordinal age loss
                            age_loss_mode = getattr(args, 'hcn_age_loss_mode', 'ce')
                            
                            if age_loss_mode != 'ce':
                                # V8 Ordinal: Use ordinal-aware compute_aux_loss
                                from hcn_v8_ordinal import compute_aux_loss
                                aux_loss, aux_metrics = compute_aux_loss(
                                    aux_logits,
                                    age_idx=batch["age_idx"],
                                    sex_idx=batch["sex_idx"],
                                    race_idx=batch["race_idx"],
                                    use_continuous_age=False,
                                    age_loss_mode=age_loss_mode,
                                    soft_ce_sigma=getattr(args, 'hcn_soft_ce_sigma', 1.0),
                                    num_age_bins=getattr(args, 'hcn_num_age_bins', 5),
                                    age_weight=getattr(args, 'hcn_aux_weight_age', 1.0),
                                    sex_weight=getattr(args, 'hcn_aux_weight_sex', 1.0),
                                    race_weight=getattr(args, 'hcn_aux_weight_race', 1.0),
                                )
                            else:
                                # V7/V8 Standard: Use hcn_v7's compute_aux_loss (supports optional age)
                                from hcn_v7 import compute_aux_loss
                                encode_age = getattr(hcn, 'encode_age', True) if hcn is not None else True
                                age_idx_for_loss = batch.get("age_idx") if encode_age else None
                                aux_loss, aux_metrics = compute_aux_loss(
                                    aux_logits,
                                    sex_idx=batch["sex_idx"],
                                    race_idx=batch["race_idx"],
                                    age_idx=age_idx_for_loss,
                                    age_weight=getattr(args, 'hcn_aux_weight_age', 1.0),
                                    sex_weight=getattr(args, 'hcn_aux_weight_sex', 1.0),
                                    race_weight=getattr(args, 'hcn_aux_weight_race', 1.0),
                                )
                        else:
                            # Original HCN
                            age_ce = F.cross_entropy(aux_logits["age"], batch["age_idx"])
                            sex_ce = F.cross_entropy(aux_logits["sex"], batch["sex_idx"])
                            race_ce = F.cross_entropy(aux_logits["race"], batch["race_idx"])
                            aux_loss = (age_ce + sex_ce + race_ce) / 3.0
                            aux_metrics = None
                    # if aux_logits is not None:
                    #     use_hcn_v7 = getattr(args, 'use_hcn_v7', False)
                        
                    #     if use_hcn_v7:
                    #         # Check if using ordinal age loss
                    #         age_loss_mode = getattr(args, 'hcn_age_loss_mode', 'ce')
                            
                    #         age_weight = getattr(args, 'hcn_aux_weight_age', 1.0)
                    #         sex_weight = getattr(args, 'hcn_aux_weight_sex', 1.0)
                    #         race_weight = getattr(args, 'hcn_aux_weight_race', 1.0)
                            
                    #         if age_loss_mode != 'ce':
                    #             # Use ordinal-aware compute_aux_loss from hcn_v8_ordinal.py
                    #             from hcn_v8_ordinal import compute_aux_loss
                    #             aux_loss, aux_metrics = compute_aux_loss(
                    #                 aux_logits,
                    #                 batch["age_idx"],
                    #                 batch["sex_idx"],
                    #                 batch["race_idx"],
                    #                 age_loss_mode=age_loss_mode,
                    #                 soft_ce_sigma=getattr(args, 'hcn_soft_ce_sigma', 0.75),
                    #                 num_age_bins=getattr(args, 'hcn_num_age_bins', 5),
                    #                 age_weight=age_weight,
                    #                 sex_weight=sex_weight,
                    #                 race_weight=race_weight,
                    #             )
                    #         else:
                    #             # Use standard compute_aux_loss from hcn_v7.py
                    #             from hcn_v7 import compute_aux_loss
                    #             aux_loss, aux_metrics = compute_aux_loss(
                    #                 aux_logits,
                    #                 batch["age_idx"],
                    #                 batch["sex_idx"],
                    #                 batch["race_idx"],
                    #                 age_weight=age_weight,
                    #                 sex_weight=sex_weight,
                    #                 race_weight=race_weight,
                    #             )
                    #     else:
                    #         # Original HCN (hcn.py)
                    #         age_ce = F.cross_entropy(aux_logits["age"], batch["age_idx"])
                    #         sex_ce = F.cross_entropy(aux_logits["sex"], batch["sex_idx"])
                    #         race_ce = F.cross_entropy(aux_logits["race"], batch["race_idx"])
                    #         aux_loss = (age_ce + sex_ce + race_ce) / 3.0
                    # else:
                    #     aux_loss = None
                # === V0.5: Demographic Encoder Conditioning ===
                demo_aux_loss = None
                demo_ctx_norm = None
                demo_dropout_mask = None
                if demographic_encoder is not None:
                    # Get demographic encoder output
                    # Returns tokens: [B, 1, d_ctx] if mode='single', [B, 3, d_ctx] if mode='separate'
                    demo_tokens, demo_aux_logits = demographic_encoder(
                        batch["age_idx"],
                        batch["sex_idx"],
                        batch["race_idx"],
                    )
                    
                    # Apply null conditioning (demographic dropout) if enabled
                    # This forces the model to learn to use demographic tokens by contrasting
                    # cases with and without them, similar to classifier-free guidance
                    if getattr(args, 'demo_use_dropout', False) and global_step >= getattr(args, 'demo_dropout_start_step', 0):
                        demo_dropout_prob = getattr(args, 'demo_text_dropout_prob', 0.5)
                        # Create random mask: True = keep demographic token, False = zero it out
                        demo_dropout_mask = torch.rand(bsz, device=demo_tokens.device) > demo_dropout_prob
                        # Zero out demographic tokens for samples where mask is False
                        # Handle both single [B, 1, d] and separate [B, 3, d] modes
                        if demo_tokens.shape[1] == 1:
                            demo_tokens = demo_tokens * demo_dropout_mask.view(-1, 1, 1).float()
                        else:
                            # For separate mode, can drop all 3 tokens together or individually
                            # Current implementation: drop all 3 together per sample
                            demo_tokens = demo_tokens * demo_dropout_mask.view(-1, 1, 1).float()
                    
                    # Concatenate text and demographic contexts
                    # Note: If HCN was also used, this would be [B, 77+N+1, d_ctx]
                    # But typically only one of HCN or DemographicEncoder is active
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, demo_tokens], dim=1
                    )  # [B, 77+N, d_ctx] where N=1 (single) or N=3 (separate)
                    
                    demo_ctx_norm = demo_tokens.norm(dim=-1).mean()
                    
                    # Compute auxiliary demographic classification losses (strong supervision)
                    # These are now computed on individual embeddings (not fused tokens)
                    age_ce = F.cross_entropy(demo_aux_logits["age"], batch["age_idx"])
                    sex_ce = F.cross_entropy(demo_aux_logits["sex"], batch["sex_idx"])
                    race_ce = F.cross_entropy(demo_aux_logits["race"], batch["race_idx"])
                    demo_aux_loss = (age_ce + sex_ce + race_ce) / 3.0

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    print(
                        "are you sure ? --> noise_scheduler.config.prediction_type == v_prediction"
                    )
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual
                # V6: Use timestep injection context if enabled
                if hcn_time_emb is not None:
                    with TimestepInjectionContext(unet, hcn_time_emb):
                        noise_pred = unet(
                            noisy_latents, timesteps, encoder_hidden_states
                        ).sample
                else:
                    noise_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                # Compute instance loss
                per_sample_loss = F.mse_loss(
                    noise_pred.float(), target.float(), reduction="none"
                ).mean([1, 2, 3])
                per_sample_loss = per_sample_loss.to(weight_dtype)
                base_loss_weights = batch["loss_weights"].to(dtype=weight_dtype)

                fairness_logs = {}
                fairness_weights = torch.ones_like(per_sample_loss)
                if fair_controller is not None:
                    attribute_tensors = []
                    for field in fair_controller.attribute_fields:
                        if field not in batch:
                            raise KeyError(
                                f"Batch is missing required FairDiffusion field '{field}'."
                            )
                        attribute_tensors.append(
                            batch[field].to(accelerator.device, dtype=torch.long)
                        )
                    attributes = torch.stack(attribute_tensors, dim=1)
                    fairness_result = fair_controller.apply(
                        per_sample_loss.detach(),
                        attributes.detach(),
                        global_step,
                    )
                    fairness_weights = fairness_result.instance_weights.to(
                        device=per_sample_loss.device, dtype=weight_dtype
                    )
                    fairness_logs = fairness_result.logs

                combined_weights = base_loss_weights * fairness_weights
                combined_weights = combined_weights / (
                    combined_weights.sum() + 1e-12
                )
                loss = (per_sample_loss * combined_weights).sum()

                # === Add HCN losses ===
                if kl_loss is not None:
                    # Anneal KL weight from 0 to target value over training
                    kl_weight = min(1.0, global_step / args.hcn_kl_anneal_steps) * args.hcn_kl_weight
                    loss = loss + kl_weight * kl_loss

                if comp_loss is not None:
                    loss = loss + args.hcn_comp_weight * comp_loss

                # Add aux_loss only if it was computed (i.e., use_aux_loss=True)
                if aux_loss is not None:
                    loss = loss + args.hcn_aux_weight * aux_loss
                
                # === Add DemographicEncoder losses ===
                if demo_aux_loss is not None and args.demo_aux_weight > 0:
                    loss = loss + args.demo_aux_weight * demo_aux_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    # Add HCN parameters to gradient clipping
                    if hcn is not None:
                        params_to_clip = itertools.chain(params_to_clip, hcn.parameters())
                    
                    # Add DemographicEncoder parameters to gradient clipping
                    if demographic_encoder is not None:
                        params_to_clip = itertools.chain(params_to_clip, demographic_encoder.parameters())
                    
                    # Add FiLM adapter parameters to gradient clipping (V5)
                    if film_adapter is not None:
                        params_to_clip = itertools.chain(params_to_clip, film_adapter.parameters())

                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                # Save state checkpoint
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            # Convert to int in case it was loaded as string from YAML
                            try:
                                checkpoints_total_limit = int(args.checkpoints_total_limit)
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid checkpoints_total_limit value: {args.checkpoints_total_limit}. Skipping checkpoint cleanup.")
                                checkpoints_total_limit = None
                            
                            # Only proceed with cleanup if we have a valid limit
                            if checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )
                                
                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints) - checkpoints_total_limit + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            args.output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Note: Validation has been moved to a separate script (run_validation_monitor.py)
                # It monitors this directory and automatically validates new checkpoints

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            if fairness_logs:
                logs.update(fairness_logs)
            # Add HCN losses to logging
            if kl_loss is not None:
                logs["kl_loss"] = kl_loss.detach().item()
                logs["kl_weight"] = kl_weight
            if comp_loss is not None:
                logs["comp_loss"] = comp_loss.detach().item()
            if aux_loss is not None:
                logs["aux_loss"] = aux_loss.detach().item()
                # Add accuracy metrics if available (from HCN V7 compute_aux_loss)
                if aux_metrics is not None:
                    logs.update(aux_metrics)  # Adds: aux_loss_age, aux_loss_sex, aux_loss_race, aux_acc_age, aux_acc_sex, aux_acc_race
            if hcn_ctx_norm is not None:
                logs["hcn_ctx_norm"] = hcn_ctx_norm.detach().item()
            
            # Add DemographicEncoder losses to logging
            if demo_aux_loss is not None:
                logs["demo_aux_loss"] = demo_aux_loss.detach().item()
            if demo_ctx_norm is not None:
                logs["demo_ctx_norm"] = demo_ctx_norm.detach().item()
            
            # Log demographic dropout statistics if enabled
            if getattr(args, 'demo_use_dropout', False) and global_step >= getattr(args, 'demo_dropout_start_step', 0):
                if hcn_dropout_mask is not None:
                    dropout_rate = (1.0 - hcn_dropout_mask.float().mean()).item()
                    logs["hcn_dropout_rate"] = dropout_rate
                if demo_dropout_mask is not None:
                    dropout_rate = (1.0 - demo_dropout_mask.float().mean()).item()
                    logs["demo_dropout_rate"] = dropout_rate
            
            # Log FiLM mode statistics (V5)
            if getattr(args, 'use_hcn_film', False) and film_adapter is not None:
                logs["film_mode"] = 1.0  # Indicator that FiLM mode is active
            
            # Log V6 timestep injection mode
            if getattr(args, 'use_hcn_timestep_injection', False) and hcn_time_emb is not None:
                logs["timestep_injection"] = 1.0  # Indicator that V6 mode is active

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Check if we should stop training
            # Use max_actual_train_steps if provided, otherwise use max_train_steps
            stop_step = args.max_actual_train_steps if args.max_actual_train_steps is not None else args.max_train_steps
            if global_step >= stop_step:
                break
    logger.info("Training finished")
    return (
        logger,
        args,
        accelerator,
        train_dataloader,
        unet,
        text_encoder,
        vae,
        noise_scheduler,
        weight_dtype,
        optimizer,
        lr_scheduler,
        ema_unet,
        progress_bar,
    )
