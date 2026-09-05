"""
Hierarchical Conditioner Network V9: Continuous Age Encoding

Key improvement over V7/V8:
- Age uses sinusoidal encoding (like timestep embeddings in diffusion)
- Ordinal structure is BUILT-IN, not learned
- Similar ages → similar embeddings automatically
- Better generalization for rare age groups

Backward Compatibility:
- Set use_continuous_age=False to use categorical bins (V7/V8 behavior)
- Can load V7/V8 checkpoints when use_continuous_age=False
- Sex and race remain categorical (no ordinal structure needed)

Authors: RoentGen V9 Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import json
import os
import math


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm and SiLU activation."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousAgeEncoder(nn.Module):
    """
    Encode continuous age (0-100) using sinusoidal position encoding.
    
    Similar to timestep encoding in diffusion models:
    - Similar ages → similar embeddings (automatic ordinal structure)
    - Smooth interpolation between any ages
    - No need to learn ordinal relationships
    
    Args:
        d_out: Output embedding dimension
        max_age: Maximum age for normalization (default: 100)
        learnable_mlp: Whether to apply learnable MLP after sinusoidal encoding
    """
    
    def __init__(self, d_out: int, max_age: float = 100.0, learnable_mlp: bool = True):
        super().__init__()
        self.d_out = d_out
        self.max_age = max_age
        self.learnable_mlp = learnable_mlp
        
        if learnable_mlp:
            # MLP to transform sinusoidal encoding
            # This allows the model to learn task-specific age representations
            self.mlp = nn.Sequential(
                nn.Linear(d_out, d_out * 2),
                nn.SiLU(),
                nn.Linear(d_out * 2, d_out),
            )
        
        # Precompute frequency bands (not learnable)
        half_dim = d_out // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        self.register_buffer(
            'freq_bands', 
            torch.exp(torch.arange(half_dim) * -emb_scale)
        )
    
    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age: [B] tensor of ages in years (0-100 range)
            
        Returns:
            [B, d_out] age embeddings with built-in ordinal structure
        """
        # Normalize age to [0, 1] range
        age_norm = age.float() / self.max_age  # [B]
        
        # Sinusoidal encoding: [B, half_dim]
        # Each frequency captures different "scale" of age variation
        angles = age_norm.unsqueeze(-1) * self.freq_bands.unsqueeze(0) * math.pi
        
        # Concatenate sin and cos: [B, d_out]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        # Apply learnable MLP if enabled
        if self.learnable_mlp:
            emb = self.mlp(emb)
        
        return emb


class AgeRegressionHead(nn.Module):
    """
    Regression head for predicting continuous age from token.
    
    Used for auxiliary loss when use_continuous_age=True.
    """
    
    def __init__(self, d_input: int, hidden_dim: int = 512, dropout: float = 0.1, max_age: float = 100.0):
        super().__init__()
        self.max_age = max_age
        
        self.net = nn.Sequential(
            nn.LayerNorm(d_input),
            nn.Linear(d_input, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Single output: predicted age
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_input] token embeddings
            
        Returns:
            [B] predicted ages (0-100 range)
        """
        # Predict normalized age, then scale to actual years
        raw = self.net(x).squeeze(-1)  # [B]
        # Sigmoid to bound output, then scale to [0, max_age]
        age = torch.sigmoid(raw) * self.max_age
        return age


class HierarchicalConditionerV9(nn.Module):
    """
    V9: Hierarchical Conditioning Network with Continuous Age Encoding
    
    Key features:
    - Continuous age encoding (sinusoidal) for ordinal structure
    - Backward compatible with V7/V8 categorical mode
    - Sex and race remain categorical
    
    Backward Compatibility:
        use_continuous_age=False → Identical to V7/V8, can load old checkpoints
        use_continuous_age=True  → New continuous age mode
    
    Args:
        num_age_bins: Number of age bins (only used if use_continuous_age=False)
        num_sex: Number of sex categories (typically 2)
        num_race: Number of race categories
        d_node: Hidden dimension for embeddings (default: 256)
        d_ctx: Output dimension matching UNet cross_attention_dim (default: 1024)
        dropout: Dropout probability (default: 0.1)
        use_uncertainty: Whether to use variational sampling
        use_aux_loss: Whether to include auxiliary classifiers
        aux_hidden_dim: Hidden dimension for auxiliary heads (default: 512)
        use_continuous_age: Whether to use continuous age encoding (default: False for backward compat)
        max_age: Maximum age for continuous encoding (default: 100)
    """
    
    def __init__(
        self,
        num_age_bins: int = 5,
        num_sex: int = 2,
        num_race: int = 4,
        d_node: int = 256,
        d_ctx: int = 1024,
        dropout: float = 0.1,
        use_uncertainty: bool = True,
        use_aux_loss: bool = True,
        aux_hidden_dim: int = 512,
        use_continuous_age: bool = False,  # Default False for backward compatibility
        max_age: float = 100.0,
    ):
        super().__init__()

        # Store config for saving/loading
        self.config = {
            'num_age_bins': num_age_bins,
            'num_sex': num_sex,
            'num_race': num_race,
            'd_node': d_node,
            'd_ctx': d_ctx,
            'dropout': dropout,
            'use_uncertainty': use_uncertainty,
            'use_aux_loss': use_aux_loss,
            'aux_hidden_dim': aux_hidden_dim,
            'use_continuous_age': use_continuous_age,
            'max_age': max_age,
        }

        self.num_age = num_age_bins
        self.num_sex = num_sex
        self.num_race = num_race
        self.d_node = d_node
        self.d_ctx = d_ctx
        self.use_uncertainty = use_uncertainty
        self.use_aux_loss = use_aux_loss
        self.use_continuous_age = use_continuous_age
        self.max_age = max_age

        # === Age Embedding (V9: supports both categorical and continuous) ===
        if use_continuous_age:
            self.emb_age = ContinuousAgeEncoder(d_node, max_age=max_age)
        else:
            # Backward compatible: categorical embedding like V7/V8
            self.emb_age = nn.Embedding(num_age_bins, d_node)
        
        # === Sex and Race: Always categorical ===
        self.emb_sex = nn.Embedding(num_sex, d_node)
        self.emb_race = nn.Embedding(num_race, d_node)

        # === Parent composers (pairwise compositions) ===
        self.compose_age_sex = MLP(2 * d_node, 2 * d_node, d_node, dropout)
        self.compose_age_race = MLP(2 * d_node, 2 * d_node, d_node, dropout)
        self.compose_sex_race = MLP(2 * d_node, 2 * d_node, d_node, dropout)

        # === Child composer (triple composition) ===
        self.compose_all = MLP(3 * d_node, 2 * d_node, d_node, dropout)

        # === Uncertainty heads ===
        if use_uncertainty:
            self.mu_head = nn.Linear(d_node, d_node)
            self.logsigma_head = nn.Linear(d_node, d_node)

        # === Project to UNet cross-attention dimension ===
        self.proj_ctx = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_ctx),
        )

        # === Auxiliary heads (on output token) ===
        if use_aux_loss:
            if use_continuous_age:
                # Regression head for continuous age
                self.age_head = AgeRegressionHead(
                    d_input=d_ctx,
                    hidden_dim=aux_hidden_dim,
                    dropout=dropout,
                    max_age=max_age,
                )
            else:
                # Classification head for categorical age (V7/V8 style)
                self.age_classifier = nn.Sequential(
                    nn.LayerNorm(d_ctx),
                    nn.Linear(d_ctx, aux_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(aux_hidden_dim, num_age_bins),
                )
            
            # Sex and race: always classification
            self.sex_classifier = nn.Sequential(
                nn.LayerNorm(d_ctx),
                nn.Linear(d_ctx, aux_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_dim, num_sex),
            )
            self.race_classifier = nn.Sequential(
                nn.LayerNorm(d_ctx),
                nn.Linear(d_ctx, aux_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_dim, num_race),
            )
        else:
            self.age_head = None
            self.age_classifier = None
            self.sex_classifier = None
            self.race_classifier = None

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small normal distribution."""
        # Only init categorical embeddings (continuous encoder has its own init)
        if not self.use_continuous_age:
            nn.init.normal_(self.emb_age.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_sex.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_race.weight, mean=0.0, std=0.02)

        if self.use_uncertainty:
            nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(self.logsigma_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.logsigma_head.bias, -1.0)

    def forward(
        self,
        age_idx: Optional[torch.Tensor] = None,
        sex_idx: torch.Tensor = None,
        race_idx: torch.Tensor = None,
        age_continuous: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], None]:
        """
        Forward pass through hierarchical conditioning network.
        
        Args:
            age_idx: [B] Long tensor of age bin indices (used if use_continuous_age=False)
            sex_idx: [B] Long tensor of sex indices
            race_idx: [B] Long tensor of race indices
            age_continuous: [B] Float tensor of ages in years (used if use_continuous_age=True)

        Returns:
            ctx: [B, 1, d_ctx] - Demographic context token
            mu: [B, d_node] - Mean of variational distribution
            logsigma: [B, d_node] - Log std of variational distribution
            aux_logits: Dict with age/sex/race predictions
            time_emb: None (compatibility placeholder)
        """
        # === Get age embedding based on mode ===
        if self.use_continuous_age:
            if age_continuous is None:
                raise ValueError("age_continuous required when use_continuous_age=True")
            e_age = self.emb_age(age_continuous)  # [B, d_node]
        else:
            if age_idx is None:
                raise ValueError("age_idx required when use_continuous_age=False")
            e_age = self.emb_age(age_idx)  # [B, d_node]
        
        # === Sex and race embeddings (always categorical) ===
        e_sex = self.emb_sex(sex_idx)    # [B, d_node]
        e_race = self.emb_race(race_idx)  # [B, d_node]

        # === Level 2: Parent compositions ===
        h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], dim=-1))
        h_age_race = self.compose_age_race(torch.cat([e_age, e_race], dim=-1))
        h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], dim=-1))

        # === Level 3: Child composition ===
        h_child = self.compose_all(
            torch.cat([h_age_sex, h_age_race, h_sex_race], dim=-1)
        )

        # === Uncertainty quantification ===
        if self.use_uncertainty:
            mu = self.mu_head(h_child)
            logsigma = torch.clamp(self.logsigma_head(h_child), min=-5.0, max=1.0)
            if self.training:
                z = mu + torch.exp(logsigma) * torch.randn_like(mu)
            else:
                z = mu
        else:
            mu = h_child
            logsigma = torch.zeros_like(h_child)
            z = h_child

        # === Project to context token ===
        ctx = self.proj_ctx(z).unsqueeze(1)  # [B, 1, d_ctx]

        # === Auxiliary predictions (from output token) ===
        aux_logits = None
        if self.use_aux_loss:
            token = ctx.squeeze(1)  # [B, d_ctx]
            
            if self.use_continuous_age:
                # Regression for age
                aux_logits = {
                    "age": self.age_head(token),  # [B] continuous age prediction
                    "sex": self.sex_classifier(token),   # [B, num_sex]
                    "race": self.race_classifier(token), # [B, num_race]
                }
            else:
                # Classification for age (V7/V8 style)
                aux_logits = {
                    "age": self.age_classifier(token),   # [B, num_age_bins]
                    "sex": self.sex_classifier(token),   # [B, num_sex]
                    "race": self.race_classifier(token), # [B, num_race]
                }

        return ctx, mu, logsigma, aux_logits, None

    def compute_compositional_loss(
        self,
        age_idx: Optional[torch.Tensor] = None,
        sex_idx: torch.Tensor = None,
        race_idx: torch.Tensor = None,
        age_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute compositional consistency loss."""
        # Get age embedding
        if self.use_continuous_age:
            e_age = self.emb_age(age_continuous)
        else:
            e_age = self.emb_age(age_idx)
        
        e_sex = self.emb_sex(sex_idx)
        e_race = self.emb_race(race_idx)

        h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], -1))
        h_age_race = self.compose_age_race(torch.cat([e_age, e_race], -1))
        h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], -1))
        h_child = self.compose_all(torch.cat([h_age_sex, h_age_race, h_sex_race], -1))

        h_additive = e_age + e_sex + e_race
        cos_sim = F.cosine_similarity(h_child, h_additive, dim=-1)
        return (1 - cos_sim).mean()

    def get_uncertainty(
        self,
        age_idx: Optional[torch.Tensor] = None,
        sex_idx: torch.Tensor = None,
        race_idx: torch.Tensor = None,
        age_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get uncertainty (sigma) for given demographic groups."""
        _, _, logsigma, _, _ = self.forward(
            age_idx=age_idx,
            sex_idx=sex_idx,
            race_idx=race_idx,
            age_continuous=age_continuous,
        )
        return torch.exp(logsigma).mean(dim=-1)

    def save_pretrained(self, save_dir: str):
        """Save HCN model and config."""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        print(f"HCN V9 saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = "cpu"):
        """Load HCN model from saved checkpoint."""
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"HCN V9 loaded from {save_dir}")
        return model


# =============================================================================
# Loss Functions
# =============================================================================

def compute_aux_loss(
    aux_logits: Dict[str, torch.Tensor],
    age_idx: Optional[torch.Tensor] = None,
    sex_idx: torch.Tensor = None,
    race_idx: torch.Tensor = None,
    age_continuous: Optional[torch.Tensor] = None,
    use_continuous_age: bool = False,
    num_age_bins: int = 5,
    age_weight: float = 1.0,
    sex_weight: float = 1.0,
    race_weight: float = 1.0,
    # Legacy parameters for backward compatibility (ignored if use_continuous_age=True)
    age_loss_mode: str = 'ce',
    soft_ce_sigma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute auxiliary loss with support for both continuous and categorical age.
    
    Args:
        aux_logits: Dict with 'age', 'sex', 'race' predictions
        age_idx: [B] age bin indices (for categorical mode)
        sex_idx: [B] sex indices
        race_idx: [B] race indices
        age_continuous: [B] continuous ages in years (for continuous mode)
        use_continuous_age: Whether using continuous age encoding
        num_age_bins: Number of age bins (for categorical mode)
        *_weight: Loss weights for each attribute
        
    Returns:
        total_loss: Weighted sum of losses
        metrics: Dict with individual losses and accuracies
    """
    # === Age loss ===
    if use_continuous_age:
        # Regression loss (MSE)
        age_pred = aux_logits["age"]  # [B] predicted ages
        age_loss = F.mse_loss(age_pred, age_continuous)
        
        # Metrics
        with torch.no_grad():
            age_mae = (age_pred - age_continuous).abs().mean()
            age_rmse = torch.sqrt(F.mse_loss(age_pred, age_continuous))
            # Compute bin accuracy for comparison with categorical models
            pred_bins = age_to_bins(age_pred, num_age_bins)
            true_bins = age_to_bins(age_continuous, num_age_bins)
            age_bin_acc = (pred_bins == true_bins).float().mean()
    else:
        # Classification loss (CE or soft_ce)
        age_logits = aux_logits["age"]  # [B, num_bins]
        
        if age_loss_mode == 'soft_ce':
            age_loss = compute_soft_ce_loss(age_logits, age_idx, num_age_bins, soft_ce_sigma)
        else:
            age_loss = F.cross_entropy(age_logits, age_idx)
        
        # Metrics
        with torch.no_grad():
            age_pred_bins = age_logits.argmax(dim=-1)
            age_bin_acc = (age_pred_bins == age_idx).float().mean()
            age_mae = (age_pred_bins.float() - age_idx.float()).abs().mean()
            # Approximate RMSE in years (using bin centers)
            bin_centers = get_bin_centers(num_age_bins)
            pred_years = bin_centers[age_pred_bins]
            true_years = bin_centers[age_idx]
            age_rmse = torch.sqrt(F.mse_loss(pred_years, true_years))
    
    # === Sex and race losses (always classification) ===
    sex_loss = F.cross_entropy(aux_logits["sex"], sex_idx)
    race_loss = F.cross_entropy(aux_logits["race"], race_idx)
    
    # === Weighted total ===
    total_loss = (
        age_weight * age_loss +
        sex_weight * sex_loss +
        race_weight * race_loss
    ) / (age_weight + sex_weight + race_weight)
    
    # === Compute all metrics ===
    with torch.no_grad():
        sex_acc = (aux_logits["sex"].argmax(-1) == sex_idx).float().mean()
        race_acc = (aux_logits["race"].argmax(-1) == race_idx).float().mean()
    
    metrics = {
        "aux_loss_age": age_loss.item(),
        "aux_loss_sex": sex_loss.item(),
        "aux_loss_race": race_loss.item(),
        "aux_acc_sex": sex_acc.item(),
        "aux_acc_race": race_acc.item(),
        "aux_acc_age_bin": age_bin_acc.item(),  # Bin accuracy (comparable across modes)
        "aux_mae_age": age_mae.item(),          # MAE in bins (categorical) or years (continuous)
        "aux_rmse_age": age_rmse.item(),        # RMSE in years
    }
    
    return total_loss, metrics


def compute_soft_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    sigma: float = 1.0
) -> torch.Tensor:
    """Soft label cross-entropy with Gaussian smoothing (for categorical age)."""
    device = logits.device
    bins = torch.arange(num_classes, device=device).float()
    targets_float = targets.float().unsqueeze(-1)
    soft_targets = torch.exp(-0.5 * ((bins - targets_float) / sigma) ** 2)
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    return loss


def age_to_bins(age: torch.Tensor, num_bins: int = 5, max_age: float = 100.0) -> torch.Tensor:
    """Convert continuous age to bin indices."""
    bin_size = max_age / num_bins
    bins = (age / bin_size).long().clamp(0, num_bins - 1)
    return bins


def get_bin_centers(num_bins: int = 5, max_age: float = 100.0) -> torch.Tensor:
    """Get center age of each bin for RMSE calculation."""
    bin_size = max_age / num_bins
    centers = torch.arange(num_bins).float() * bin_size + bin_size / 2
    return centers


# =============================================================================
# Loader function
# =============================================================================

def load_hcn_v9(args, logger):
    """
    Load and initialize HCN V9.
    
    Config options:
        args.use_continuous_age: True for continuous age encoding (default: False)
        args.max_age: Maximum age for continuous encoding (default: 100)
        
    Backward compatible: set use_continuous_age=False to get V7/V8 behavior.
    """
    if not getattr(args, 'use_hcn', False):
        logger.info("HCN disabled (use_hcn=False)")
        return None
    
    use_continuous_age = getattr(args, 'use_continuous_age', False)
    use_aux_loss = getattr(args, 'hcn_aux_weight', 0.0) > 0.0
    
    logger.info("=" * 60)
    if use_continuous_age:
        logger.info("Initializing HCN V9 with Continuous Age Encoding")
    else:
        logger.info("Initializing HCN V9 (Categorical Age Mode - V7/V8 Compatible)")
    logger.info("=" * 60)
    
    hcn = HierarchicalConditionerV9(
        num_age_bins=getattr(args, 'hcn_num_age_bins', 5),
        num_sex=getattr(args, 'hcn_num_sex', 2),
        num_race=getattr(args, 'hcn_num_race', 4),
        d_node=getattr(args, 'hcn_d_node', 256),
        d_ctx=getattr(args, 'hcn_d_ctx', 1024),
        dropout=getattr(args, 'hcn_dropout', 0.1),
        use_uncertainty=getattr(args, 'hcn_use_uncertainty', True),
        use_aux_loss=use_aux_loss,
        aux_hidden_dim=getattr(args, 'hcn_aux_hidden_dim', 512),
        use_continuous_age=use_continuous_age,
        max_age=getattr(args, 'max_age', 100.0),
    )
    
    num_params = sum(p.numel() for p in hcn.parameters())
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Continuous age: {use_continuous_age}")
    if not use_continuous_age:
        logger.info(f"  Age bins: {hcn.num_age}")
    logger.info(f"  Sex categories: {hcn.num_sex}")
    logger.info(f"  Race categories: {hcn.num_race}")
    logger.info(f"  Node dimension: {hcn.d_node}")
    logger.info(f"  Context dimension: {hcn.d_ctx}")
    logger.info(f"  Uncertainty: {hcn.use_uncertainty}")
    logger.info(f"  Auxiliary loss: {use_aux_loss}")
    logger.info("=" * 60)
    
    return hcn


# =============================================================================
# Tests
# =============================================================================

def test_continuous_age_encoder():
    """Test that similar ages produce similar embeddings."""
    print("Testing ContinuousAgeEncoder...")
    print("=" * 60)
    
    encoder = ContinuousAgeEncoder(d_out=256)
    
    # Test ages
    ages = torch.tensor([20.0, 21.0, 22.0, 50.0, 80.0, 81.0])
    embeddings = encoder(ages)
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute pairwise cosine similarities
    emb_norm = F.normalize(embeddings, dim=-1)
    sim_matrix = emb_norm @ emb_norm.T
    
    print("\nCosine similarity matrix:")
    print("Ages:    20    21    22    50    80    81")
    for i, age in enumerate([20, 21, 22, 50, 80, 81]):
        row = "  ".join([f"{sim_matrix[i, j]:.2f}" for j in range(6)])
        print(f"  {age}:  {row}")
    
    # Check that adjacent ages are similar
    assert sim_matrix[0, 1] > 0.9, "Ages 20 and 21 should be very similar"
    assert sim_matrix[4, 5] > 0.9, "Ages 80 and 81 should be very similar"
    assert sim_matrix[0, 4] < sim_matrix[0, 1], "Ages 20 and 80 should be less similar than 20 and 21"
    
    print("\n✓ Adjacent ages have similar embeddings")
    print("✓ Distant ages have different embeddings")
    print("=" * 60)


def test_hcn_v9_continuous():
    """Test HCN V9 in continuous age mode."""
    print("\nTesting HCN V9 (Continuous Age Mode)...")
    print("=" * 60)
    
    hcn = HierarchicalConditionerV9(
        num_age_bins=5,  # Still needed for bin accuracy calculation
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=True,
        use_continuous_age=True,  # Enable continuous age!
    )
    
    batch_size = 8
    age_continuous = torch.rand(batch_size) * 100  # Random ages 0-100
    sex = torch.randint(0, 2, (batch_size,))
    race = torch.randint(0, 4, (batch_size,))
    
    # Forward pass
    ctx, mu, logsigma, aux_logits, _ = hcn(
        sex_idx=sex,
        race_idx=race,
        age_continuous=age_continuous,
    )
    
    print(f"Context shape: {ctx.shape}")
    print(f"Age prediction shape: {aux_logits['age'].shape}")  # Should be [B], not [B, 5]
    print(f"Sex logits shape: {aux_logits['sex'].shape}")
    print(f"Race logits shape: {aux_logits['race'].shape}")
    
    assert ctx.shape == (batch_size, 1, 1024)
    assert aux_logits["age"].shape == (batch_size,), "Age should be regression output"
    assert aux_logits["sex"].shape == (batch_size, 2)
    assert aux_logits["race"].shape == (batch_size, 4)
    
    # Test loss computation
    loss, metrics = compute_aux_loss(
        aux_logits,
        sex_idx=sex,
        race_idx=race,
        age_continuous=age_continuous,
        use_continuous_age=True,
    )
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Age RMSE: {metrics['aux_rmse_age']:.2f} years")
    print(f"Age MAE: {metrics['aux_mae_age']:.2f} years")
    print(f"Age bin accuracy: {metrics['aux_acc_age_bin']:.2%}")
    
    print("\n✓ HCN V9 continuous age mode works!")
    print("=" * 60)


def test_hcn_v9_categorical():
    """Test HCN V9 in categorical mode (backward compatibility)."""
    print("\nTesting HCN V9 (Categorical Age Mode - V7/V8 Compatible)...")
    print("=" * 60)
    
    hcn = HierarchicalConditionerV9(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=True,
        use_continuous_age=False,  # Categorical mode
    )
    
    batch_size = 8
    age_idx = torch.randint(0, 5, (batch_size,))
    sex = torch.randint(0, 2, (batch_size,))
    race = torch.randint(0, 4, (batch_size,))
    
    # Forward pass (same as V7/V8)
    ctx, mu, logsigma, aux_logits, _ = hcn(
        age_idx=age_idx,
        sex_idx=sex,
        race_idx=race,
    )
    
    print(f"Context shape: {ctx.shape}")
    print(f"Age logits shape: {aux_logits['age'].shape}")  # Should be [B, 5]
    
    assert ctx.shape == (batch_size, 1, 1024)
    assert aux_logits["age"].shape == (batch_size, 5), "Age should be classification output"
    
    # Test loss computation (same interface as V7/V8)
    loss, metrics = compute_aux_loss(
        aux_logits,
        age_idx=age_idx,
        sex_idx=sex,
        race_idx=race,
        use_continuous_age=False,
    )
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Age bin accuracy: {metrics['aux_acc_age_bin']:.2%}")
    
    print("\n✓ HCN V9 categorical mode (backward compatible) works!")
    print("=" * 60)


def test_gradient_flow():
    """Test that gradients flow correctly through continuous age encoder."""
    print("\nTesting gradient flow...")
    print("=" * 60)
    
    hcn = HierarchicalConditionerV9(
        use_continuous_age=True,
        use_aux_loss=True,
    )
    
    age_continuous = torch.tensor([25.0, 45.0, 65.0, 85.0], requires_grad=False)
    sex = torch.randint(0, 2, (4,))
    race = torch.randint(0, 4, (4,))
    
    # Forward
    ctx, _, _, aux_logits, _ = hcn(
        sex_idx=sex,
        race_idx=race,
        age_continuous=age_continuous,
    )
    
    # Backward through age prediction
    loss, _ = compute_aux_loss(
        aux_logits,
        sex_idx=sex,
        race_idx=race,
        age_continuous=age_continuous,
        use_continuous_age=True,
    )
    
    hcn.zero_grad()
    loss.backward()
    
    # Check gradients exist
    age_encoder_grad = hcn.emb_age.mlp[0].weight.grad
    proj_ctx_grad = hcn.proj_ctx[1].weight.grad
    
    assert age_encoder_grad is not None, "Age encoder should have gradients"
    assert proj_ctx_grad is not None, "Projection should have gradients"
    assert age_encoder_grad.abs().sum() > 0, "Age encoder gradients should be non-zero"
    
    print(f"Age encoder gradient norm: {age_encoder_grad.norm().item():.6f}")
    print(f"Projection gradient norm: {proj_ctx_grad.norm().item():.6f}")
    print("\n✓ Gradients flow correctly!")
    print("=" * 60)


if __name__ == "__main__":
    test_continuous_age_encoder()
    test_hcn_v9_continuous()
    test_hcn_v9_categorical()
    test_gradient_flow()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)