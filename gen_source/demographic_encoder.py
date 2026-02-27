"""
Demographic Encoder for V0.5 + Demographics

A simplified demographic encoder that represents demographics (age, sex, race) as learned
embeddings rather than text tokens. This avoids the token budget limitation of
CLIP and provides a clean separation between clinical and demographic conditioning.

Key fix from V4 (following HCN V7/V8 approach):
- Aux loss on OUTPUT TOKENS (after fusion/projection), not on embeddings
- This forces fusion/projection layers to preserve demographic information
- Prevents "collapsed" tokens that discard demographics

Architecture:
    Single mode: (age_idx, sex_idx, race_idx) → Embeddings → Fusion → TOKEN → aux_classifiers
    Separate mode: (age_idx, sex_idx, race_idx) → Embeddings → Separate Proj → TOKENS → aux_classifiers
    
Auxiliary supervision:
    Applied to TOKENS (d_output dimension), not embeddings (d_hidden dimension)
    This ensures fusion/projection MUST preserve demographics (like HCN V7/V8)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class DemographicEncoder(nn.Module):
    """
    Simplified demographic encoder for V0.5 + demographics.
    
    Key fix from V4 (following HCN V7/V8):
    - Aux loss on OUTPUT TOKENS (after fusion/projection), not on embeddings
    - This forces fusion/projection to preserve demographic information
    - Prevents "collapsed" tokens that discard demographics
    
    Args:
        num_age_bins: Number of age categories (default: 5)
        num_sex: Number of sex categories (default: 2)
        num_race: Number of race categories (default: 4)
        d_hidden: Hidden dimension for embeddings (default: 256)
        d_output: Output dimension matching UNet cross_attention_dim (default: 1024)
        mode: 'single' (fused token) or 'separate' (3 tokens) (default: 'single')
        classifier_depth: 'shallow' (single linear) or 'deep' (MLP with hidden layer) (default: 'shallow')
        aux_hidden_dim: Hidden dimension for deep classifiers (default: 512)
        dropout: Dropout probability for deep classifiers (default: 0.1)
        
    Input:
        age_idx: [B] Long tensor of age bin indices (0 to num_age_bins-1)
        sex_idx: [B] Long tensor of sex indices (0 to num_sex-1)
        race_idx: [B] Long tensor of race indices (0 to num_race-1)
        
    Output:
        tokens: [B, 1, d_output] if mode='single', [B, 3, d_output] if mode='separate'
        aux_logits: Dict with 'age', 'sex', 'race' logits (FROM TOKENS, not embeddings)
    """
    
    def __init__(
        self,
        num_age_bins: int = 5,
        num_sex: int = 2,
        num_race: int = 4,
        d_hidden: int = 256,
        d_output: int = 1024,
        mode: str = 'single',  # 'single' or 'separate'
        classifier_depth: str = 'shallow',  # 'shallow' or 'deep'
        aux_hidden_dim: int = 512,  # For deep classifiers
        dropout: float = 0.1,  # For deep classifiers
    ):
        super().__init__()
        
        self.mode = mode
        self.classifier_depth = classifier_depth
        
        # Store config for saving/loading
        self.config = {
            'num_age_bins': num_age_bins,
            'num_sex': num_sex,
            'num_race': num_race,
            'd_hidden': d_hidden,
            'd_output': d_output,
            'mode': mode,
            'classifier_depth': classifier_depth,
            'aux_hidden_dim': aux_hidden_dim,
            'dropout': dropout,
        }
        
        self.num_age_bins = num_age_bins
        self.num_sex = num_sex
        self.num_race = num_race
        self.d_hidden = d_hidden
        self.d_output = d_output
        
        # === Categorical Embeddings ===
        # These are learned from scratch (not from CLIP)
        self.emb_age = nn.Embedding(num_age_bins, d_hidden)
        self.emb_sex = nn.Embedding(num_sex, d_hidden)
        self.emb_race = nn.Embedding(num_race, d_hidden)
        
        if mode == 'single':
            # Fuse into single token
            self.fusion = nn.Sequential(
                nn.Linear(3 * d_hidden, d_hidden * 2),
                nn.LayerNorm(d_hidden * 2),
                nn.GELU(),
                nn.Linear(d_hidden * 2, d_output),
            )
        else:
            # Separate projections
            self.proj_age = nn.Linear(d_hidden, d_output)
            self.proj_sex = nn.Linear(d_hidden, d_output)
            self.proj_race = nn.Linear(d_hidden, d_output)
        
        # === V0.5 Fix: Aux classifiers on OUTPUT TOKENS (d_output), not embeddings (d_hidden) ===
        # This forces fusion/projection to preserve demographics (following HCN V7/V8 approach)
        # Can be shallow (single linear) or deep (MLP like HCN V7/V8)
        if classifier_depth == 'deep':
            # Deep classifiers: LayerNorm → Linear → SiLU → Dropout → Linear
            # Input is d_output (token dimension), not d_hidden (embedding dimension)
            self.cls_age = nn.Sequential(
                nn.LayerNorm(d_output),
                nn.Linear(d_output, aux_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_dim, num_age_bins),
            )
            self.cls_sex = nn.Sequential(
                nn.LayerNorm(d_output),
                nn.Linear(d_output, aux_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_dim, num_sex),
            )
            self.cls_race = nn.Sequential(
                nn.LayerNorm(d_output),
                nn.Linear(d_output, aux_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_dim, num_race),
            )
        else:
            # Shallow classifiers: single linear layer
            # Input is d_output (token dimension), not d_hidden (embedding dimension)
            self.cls_age = nn.Linear(d_output, num_age_bins)
            self.cls_sex = nn.Linear(d_output, num_sex)
            self.cls_race = nn.Linear(d_output, num_race)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings and linear layers with small values."""
        # Initialize embeddings with small values
        nn.init.normal_(self.emb_age.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_sex.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_race.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the demographic encoder.
        
        Args:
            age_idx: [B] Long tensor of age bin indices
            sex_idx: [B] Long tensor of sex indices  
            race_idx: [B] Long tensor of race indices
            
        Returns:
            tokens: [B, 1, d_output] if mode='single', [B, 3, d_output] if mode='separate'
            aux_logits: Dict with 'age', 'sex', 'race' logits (FROM TOKENS, not embeddings)
        """
        # Get embeddings
        e_age = self.emb_age(age_idx)    # [B, d_hidden]
        e_sex = self.emb_sex(sex_idx)    # [B, d_hidden]
        e_race = self.emb_race(race_idx) # [B, d_hidden]
        
        # Output token(s)
        if self.mode == 'single':
            h = torch.cat([e_age, e_sex, e_race], dim=-1)
            token = self.fusion(h)  # [B, d_output] - fused token
            tokens = token.unsqueeze(1)  # [B, 1, d_output]
            
            # === V0.5 Fix: Aux logits FROM TOKEN (not from embeddings!) ===
            # This forces fusion to preserve demographics (following HCN V7/V8)
            # The fused token must contain all demographic information
            aux_logits = {
                'age': self.cls_age(token),   # [B, num_age_bins]
                'sex': self.cls_sex(token),   # [B, num_sex]
                'race': self.cls_race(token), # [B, num_race]
            }
        else:
            # Separate mode: 3 separate tokens
            token_age = self.proj_age(e_age)  # [B, d_output]
            token_sex = self.proj_sex(e_sex)  # [B, d_output]
            token_race = self.proj_race(e_race)  # [B, d_output]
            tokens = torch.stack([token_age, token_sex, token_race], dim=1)  # [B, 3, d_output]
            
            # === V0.5 Fix: Aux logits FROM TOKENS (not from embeddings!) ===
            # Each token corresponds to one demographic attribute
            # This forces each projection to preserve its demographic information
            aux_logits = {
                'age': self.cls_age(token_age),   # [B, num_age_bins]
                'sex': self.cls_sex(token_sex),   # [B, num_sex]
                'race': self.cls_race(token_race), # [B, num_race]
            }
        
        return tokens, aux_logits
    
    def save_pretrained(self, save_directory: str):
        """Save the demographic encoder config and weights."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save weights
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)
        
        print(f"DemographicEncoder saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load a demographic encoder from saved config and weights."""
        import os
        import json
        
        # Load config
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        weights_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"DemographicEncoder loaded from {load_directory}")
        return model


def load_demographic_encoder(args, logger) -> Optional[DemographicEncoder]:
    """
    Load or create a DemographicEncoder based on config.
    
    Args:
        args: Config object with demographic encoder settings
        logger: Logger for status messages
        
    Returns:
        DemographicEncoder if args.use_demographic_encoder is True, else None
    """
    if not getattr(args, 'use_demographic_encoder', False):
        logger.info("DemographicEncoder disabled (use_demographic_encoder=False)")
        return None
    
    logger.info("=" * 60)
    logger.info("Loading DemographicEncoder (V0.5)")
    logger.info("=" * 60)
    
    # Get config parameters
    num_age_bins = getattr(args, 'demo_num_age_bins', 5)
    num_sex = getattr(args, 'demo_num_sex', 2)
    num_race = getattr(args, 'demo_num_race', 4)
    d_hidden = getattr(args, 'demo_d_hidden', 256)
    d_output = getattr(args, 'demo_d_output', 1024)
    mode = getattr(args, 'demo_mode', 'single')  # 'single' or 'separate'
    classifier_depth = getattr(args, 'demo_classifier_depth', 'shallow')  # 'shallow' or 'deep'
    aux_hidden_dim = getattr(args, 'demo_aux_hidden_dim', 512)  # For deep classifiers
    dropout = getattr(args, 'demo_dropout', 0.1)  # For deep classifiers (deprecated but kept for compatibility)
    
    # Create encoder
    demo_encoder = DemographicEncoder(
        num_age_bins=num_age_bins,
        num_sex=num_sex,
        num_race=num_race,
        d_hidden=d_hidden,
        d_output=d_output,
        mode=mode,
        classifier_depth=classifier_depth,
        aux_hidden_dim=aux_hidden_dim,
        dropout=dropout,
    )
    
    logger.info(f"  Num age bins: {num_age_bins}")
    logger.info(f"  Num sex categories: {num_sex}")
    logger.info(f"  Num race categories: {num_race}")
    logger.info(f"  Hidden dim: {d_hidden}")
    logger.info(f"  Output dim: {d_output}")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Classifier depth: {classifier_depth}")
    if classifier_depth == 'deep':
        logger.info(f"  Aux hidden dim: {aux_hidden_dim}")
    
    # Count parameters
    num_params = sum(p.numel() for p in demo_encoder.parameters())
    logger.info(f"  Total parameters: {num_params:,}")
    
    # Optional: Load from pretrained checkpoint
    pretrained_path = getattr(args, 'demographic_encoder_pretrained_path', None)
    if pretrained_path:
        logger.info(f"  Loading from pretrained: {pretrained_path}")
        demo_encoder = DemographicEncoder.from_pretrained(pretrained_path)
    
    logger.info("=" * 60)
    
    return demo_encoder







