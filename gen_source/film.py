"""
Feature-wise Linear Modulation (FiLM) for Demographic Conditioning in Diffusion Models

This module implements FiLM-based demographic conditioning that directly modulates
UNet feature maps, bypassing the cross-attention mechanism that ignores concatenated tokens.

Key insight: The UNet's cross-attention is pre-trained to use CLIP embeddings at specific
positions. Concatenating a new HCN token at position 78 is invisible to these attention
patterns. FiLM directly modulates features via multiplication and addition, which cannot
be ignored.

Architecture:
    demographics → HCN → FiLM Adapter → (γ, β) per UNet block → Feature modulation

References:
    - FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)
    - DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)

Authors: RoentGen V5 Team
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from diffusers import UNet2DConditionModel


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation to features: out = γ * x + β
    
    Args:
        num_features: Number of feature channels to modulate
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
    
    def forward(
        self, 
        x: torch.Tensor, 
        gamma: torch.Tensor, 
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            x: Input features [B, C, H, W] or [B, C]
            gamma: Scale parameters [B, C] (will be reshaped as needed)
            beta: Shift parameters [B, C] (will be reshaped as needed)
            
        Returns:
            Modulated features with same shape as input
        """
        # Reshape gamma/beta for broadcasting
        if x.dim() == 4:
            # [B, C, H, W] - typical conv features
            gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
            beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        elif x.dim() == 3:
            # [B, L, C] - typical transformer features
            gamma = gamma.unsqueeze(1)  # [B, 1, C]
            beta = beta.unsqueeze(1)    # [B, 1, C]
        # For [B, C], no reshape needed
        
        return gamma * x + beta


class FiLMAdapter(nn.Module):
    """
    Adapter that converts HCN hidden state to FiLM parameters for multiple UNet blocks.
    
    Takes the compositional demographic embedding from HCN and produces per-block
    (gamma, beta) parameters for FiLM modulation.
    
    Args:
        d_input: Input dimension from HCN (d_node, typically 256)
        block_channels: List of channel dimensions for each UNet block to modulate
        d_hidden: Hidden dimension for the adapter MLP (default: 512)
        
    Example:
        For SD 2.1 UNet, typical block_channels might be:
        [320, 320, 640, 640, 1280, 1280, 1280, 1280, 640, 640, 320, 320]
    """
    
    def __init__(
        self,
        d_input: int,
        block_channels: List[int],
        d_hidden: int = 512,
    ):
        super().__init__()
        
        self.d_input = d_input
        self.block_channels = block_channels
        self.num_blocks = len(block_channels)
        
        # Shared encoder that processes HCN embedding
        self.shared_encoder = nn.Sequential(
            nn.LayerNorm(d_input),
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )
        
        # Per-block heads that output (gamma, beta)
        # Initialize gamma to 1 and beta to 0 (identity transform initially)
        self.gamma_heads = nn.ModuleList()
        self.beta_heads = nn.ModuleList()
        
        for channels in block_channels:
            gamma_head = nn.Linear(d_hidden, channels)
            beta_head = nn.Linear(d_hidden, channels)
            
            # Initialize for identity: gamma=1, beta=0
            nn.init.zeros_(gamma_head.weight)
            nn.init.ones_(gamma_head.bias)
            nn.init.zeros_(beta_head.weight)
            nn.init.zeros_(beta_head.bias)
            
            self.gamma_heads.append(gamma_head)
            self.beta_heads.append(beta_head)
    
    def forward(
        self, 
        hcn_hidden: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert HCN hidden state to per-block FiLM parameters.
        
        Args:
            hcn_hidden: HCN compositional embedding [B, d_input]
            
        Returns:
            gammas: List of [B, C_i] tensors for each block
            betas: List of [B, C_i] tensors for each block
        """
        # Encode through shared MLP
        h = self.shared_encoder(hcn_hidden)  # [B, d_hidden]
        
        # Generate per-block parameters
        gammas = [head(h) for head in self.gamma_heads]
        betas = [head(h) for head in self.beta_heads]
        
        return gammas, betas


class FiLMUNetWrapper(nn.Module):
    """
    Wrapper around UNet2DConditionModel that applies FiLM modulation at ResNet blocks.
    
    This wrapper intercepts the forward pass and applies demographic conditioning
    through FiLM layers, which directly modulate feature maps. This bypasses the
    cross-attention mechanism that ignores concatenated tokens.
    
    Args:
        unet: Pre-trained UNet2DConditionModel
        film_adapter: FiLMAdapter that produces (gamma, beta) from demographics
        film_scale: Scaling factor for FiLM modulation (0=identity, 1=full)
        film_blocks: Which blocks to apply FiLM to ('down', 'mid', 'up', 'all')
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        film_adapter: FiLMAdapter,
        film_scale: float = 1.0,
        film_blocks: str = 'all',
    ):
        super().__init__()
        
        self.unet = unet
        self.film_adapter = film_adapter
        self.film_scale = film_scale
        self.film_blocks = film_blocks
        
        # Get the channel dimensions from UNet config
        self.down_block_channels = list(unet.config.block_out_channels)
        self.up_block_channels = list(reversed(unet.config.block_out_channels))
        
        # Store FiLM parameters during forward pass
        self._film_gammas: Optional[List[torch.Tensor]] = None
        self._film_betas: Optional[List[torch.Tensor]] = None
        
        # Register hooks on ResNet blocks
        self._hooks = []
        self._register_hooks()
    
    def _get_block_indices(self) -> List[int]:
        """Get indices of blocks to apply FiLM based on film_blocks setting."""
        num_down = len(self.down_block_channels)
        num_up = len(self.up_block_channels)
        
        if self.film_blocks == 'all':
            return list(range(num_down + 1 + num_up))  # down + mid + up
        elif self.film_blocks == 'down':
            return list(range(num_down))
        elif self.film_blocks == 'mid':
            return [num_down]
        elif self.film_blocks == 'up':
            return list(range(num_down + 1, num_down + 1 + num_up))
        else:
            raise ValueError(f"Unknown film_blocks setting: {self.film_blocks}")
    
    def _make_hook(self, block_idx: int):
        """Create a hook function for a specific block."""
        def hook(module, input, output):
            if self._film_gammas is None or self._film_betas is None:
                return output
            
            # Apply FiLM modulation
            gamma = self._film_gammas[block_idx]
            beta = self._film_betas[block_idx]
            
            # Scale the modulation (allows gradual introduction)
            if self.film_scale != 1.0:
                gamma = 1.0 + self.film_scale * (gamma - 1.0)
                beta = self.film_scale * beta
            
            # Handle different output types
            if isinstance(output, tuple):
                # ResNet blocks often return (hidden_states, ...)
                hidden_states = output[0]
                
                # Apply FiLM
                gamma_view = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
                beta_view = beta.view(beta.size(0), beta.size(1), 1, 1)
                modulated = gamma_view * hidden_states + beta_view
                
                return (modulated,) + output[1:]
            else:
                # Direct tensor output
                gamma_view = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
                beta_view = beta.view(beta.size(0), beta.size(1), 1, 1)
                return gamma_view * output + beta_view
        
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on UNet ResNet blocks."""
        block_idx = 0
        
        # Down blocks
        for i, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, 'resnets'):
                for resnet in down_block.resnets:
                    if block_idx < len(self.film_adapter.block_channels):
                        hook = resnet.register_forward_hook(self._make_hook(block_idx))
                        self._hooks.append(hook)
                        block_idx += 1
        
        # Mid block
        if hasattr(self.unet, 'mid_block') and self.unet.mid_block is not None:
            if hasattr(self.unet.mid_block, 'resnets'):
                for resnet in self.unet.mid_block.resnets:
                    if block_idx < len(self.film_adapter.block_channels):
                        hook = resnet.register_forward_hook(self._make_hook(block_idx))
                        self._hooks.append(hook)
                        block_idx += 1
        
        # Up blocks
        for i, up_block in enumerate(self.unet.up_blocks):
            if hasattr(up_block, 'resnets'):
                for resnet in up_block.resnets:
                    if block_idx < len(self.film_adapter.block_channels):
                        hook = resnet.register_forward_hook(self._make_hook(block_idx))
                        self._hooks.append(hook)
                        block_idx += 1
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def set_film_parameters(
        self, 
        gammas: List[torch.Tensor], 
        betas: List[torch.Tensor]
    ):
        """
        Set FiLM parameters for the next forward pass.
        
        Args:
            gammas: List of scale tensors [B, C_i] for each block
            betas: List of shift tensors [B, C_i] for each block
        """
        self._film_gammas = gammas
        self._film_betas = betas
    
    def clear_film_parameters(self):
        """Clear FiLM parameters (disable modulation)."""
        self._film_gammas = None
        self._film_betas = None
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs
    ):
        """
        Forward pass with FiLM modulation.
        
        Note: FiLM parameters should be set via set_film_parameters() before calling.
        """
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )
    
    # Delegate attribute access to the wrapped UNet
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)
    
    def parameters(self, recurse: bool = True):
        """Return parameters from both UNet and FiLM adapter."""
        yield from self.unet.parameters(recurse)
        yield from self.film_adapter.parameters(recurse)
    
    def train(self, mode: bool = True):
        """Set both UNet and FiLM adapter to training mode."""
        self.unet.train(mode)
        self.film_adapter.train(mode)
        return self
    
    def eval(self):
        """Set both UNet and FiLM adapter to eval mode."""
        return self.train(False)


def get_unet_block_channels(unet: UNet2DConditionModel) -> List[int]:
    """
    Extract the channel dimensions for each ResNet block in the UNet.
    
    This is needed to configure the FiLMAdapter with correct output dimensions.
    
    Args:
        unet: The UNet model
        
    Returns:
        List of channel dimensions for each ResNet block (down + mid + up)
    """
    channels = []
    
    # Down blocks
    for i, down_block in enumerate(unet.down_blocks):
        if hasattr(down_block, 'resnets'):
            for _ in down_block.resnets:
                channels.append(unet.config.block_out_channels[i])
    
    # Mid block
    if hasattr(unet, 'mid_block') and unet.mid_block is not None:
        if hasattr(unet.mid_block, 'resnets'):
            for _ in unet.mid_block.resnets:
                channels.append(unet.config.block_out_channels[-1])
    
    # Up blocks
    reversed_channels = list(reversed(unet.config.block_out_channels))
    for i, up_block in enumerate(unet.up_blocks):
        if hasattr(up_block, 'resnets'):
            for _ in up_block.resnets:
                channels.append(reversed_channels[i])
    
    return channels


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


def test_film():
    """Quick test of FiLM module."""
    print("Testing FiLM module...")
    
    # Test FiLMLayer
    film_layer = FiLMLayer(64)
    x = torch.randn(2, 64, 32, 32)
    gamma = torch.ones(2, 64)
    beta = torch.zeros(2, 64)
    out = film_layer(x, gamma, beta)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert torch.allclose(out, x), "Identity transform failed"
    print("✓ FiLMLayer test passed")
    
    # Test FiLMAdapter
    block_channels = [320, 320, 640, 640, 1280, 1280]
    adapter = FiLMAdapter(d_input=256, block_channels=block_channels)
    hcn_hidden = torch.randn(2, 256)
    gammas, betas = adapter(hcn_hidden)
    assert len(gammas) == len(block_channels)
    assert len(betas) == len(block_channels)
    for i, (g, b, c) in enumerate(zip(gammas, betas, block_channels)):
        assert g.shape == (2, c), f"Gamma {i}: expected (2, {c}), got {g.shape}"
        assert b.shape == (2, c), f"Beta {i}: expected (2, {c}), got {b.shape}"
    print("✓ FiLMAdapter test passed")
    
    print("✓ All FiLM tests passed!")


if __name__ == "__main__":
    test_film()

