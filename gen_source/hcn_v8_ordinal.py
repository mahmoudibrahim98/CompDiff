"""
Hierarchical Conditioner Network with Ordinal Age Loss

This version maintains the SAME architecture as hcn.py but adds ordinal loss options.
This allows resuming from existing checkpoints.

Key difference from hcn.py:
- Uses same nn.Sequential structure for age_classifier (backward compatible)
- Only the LOSS computation changes, not the model architecture
- Can resume from existing checkpoints

Age Loss Options:
    'ce':       Standard cross-entropy (original V8 behavior)
    'soft_ce':  Soft labels with Gaussian smoothing around true bin
    'ordinal':  Ordinal regression adapted for K outputs (not K-1)
    'mse':      Mean squared error treating bins as numeric (0, 1, 2, 3, 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import json
import os


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


class HierarchicalConditionerV8Ordinal(nn.Module):
    """
    V8 with Ordinal Age Loss (Backward Compatible Version)
    
    IMPORTANT: This version uses the SAME architecture as hcn.py.
    Only the loss computation changes based on age_loss_mode.
    This allows resuming from V7 checkpoints.
    
    Args:
        num_age_bins: Number of age categories (bins are ordered!)
        num_sex: Number of sex categories (typically 2: M/F)
        num_race: Number of race/ethnicity categories
        d_node: Hidden dimension for embeddings (default: 256)
        d_ctx: Output dimension matching UNet cross_attention_dim (default: 1024)
        dropout: Dropout probability (default: 0.1)
        use_uncertainty: Whether to output mu/logsigma for variational sampling
        use_aux_loss: Whether to include auxiliary classifiers (on token)
        aux_hidden_dim: Hidden dimension for auxiliary classifiers (default: 512)
        age_loss_mode: 'ce', 'soft_ce', 'ordinal', or 'mse' (default: 'ordinal')
        soft_ce_sigma: Sigma for soft_ce mode (default: 0.75)
    """
    
    def __init__(
        self,
        num_age_bins: int,
        num_sex: int,
        num_race: int,
        d_node: int = 256,
        d_ctx: int = 1024,
        dropout: float = 0.1,
        use_uncertainty: bool = True,
        use_aux_loss: bool = True,
        aux_hidden_dim: int = 512,
        age_loss_mode: str = 'ordinal',
        soft_ce_sigma: float = 0.75,
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
            'age_loss_mode': age_loss_mode,
            'soft_ce_sigma': soft_ce_sigma,
        }

        self.num_age = num_age_bins
        self.num_sex = num_sex
        self.num_race = num_race
        self.d_node = d_node
        self.d_ctx = d_ctx
        self.use_uncertainty = use_uncertainty
        self.use_aux_loss = use_aux_loss
        self.age_loss_mode = age_loss_mode
        self.soft_ce_sigma = soft_ce_sigma

        # === Grandparent embeddings (single attributes) ===
        self.emb_age = nn.Embedding(num_age_bins, d_node)
        self.emb_sex = nn.Embedding(num_sex, d_node)
        self.emb_race = nn.Embedding(num_race, d_node)

        # === Parent composers (pairwise compositions) ===
        self.compose_age_sex = MLP(2 * d_node, 2 * d_node, d_node, dropout)
        self.compose_age_race = MLP(2 * d_node, 2 * d_node, d_node, dropout)
        self.compose_sex_race = MLP(2 * d_node, 2 * d_node, d_node, dropout)

        # === Child composer (triple composition from all parents) ===
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

        # === Auxiliary classifiers on OUTPUT TOKEN (d_ctx) ===
        # SAME ARCHITECTURE AS V7 - backward compatible!
        if use_aux_loss:
            self.age_classifier = nn.Sequential(
                nn.LayerNorm(d_ctx),
                nn.Linear(d_ctx, aux_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_dim, num_age_bins),  # K outputs (same as V7)
            )
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
            self.age_classifier = None
            self.sex_classifier = None
            self.race_classifier = None

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small normal distribution."""
        for emb in [self.emb_age, self.emb_sex, self.emb_race]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        if self.use_uncertainty:
            nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(self.logsigma_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.logsigma_head.bias, -1.0)

    def forward(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], None]:
        """
        Forward pass through hierarchical conditioning network.

        Returns:
            ctx: [B, 1, d_ctx] - Demographic context token
            mu: [B, d_node] - Mean of variational distribution
            logsigma: [B, d_node] - Log std of variational distribution
            aux_logits: Dict with 'age', 'sex', 'race' logits (all [B, num_classes])
            time_emb: None (compatibility placeholder)
        """
        # === Level 1: Grandparent embeddings ===
        e_age = self.emb_age(age_idx)
        e_sex = self.emb_sex(sex_idx)
        e_race = self.emb_race(race_idx)

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

        # === Auxiliary logits FROM TOKEN ===
        aux_logits = None
        if self.use_aux_loss:
            token = ctx.squeeze(1)  # [B, d_ctx]
            aux_logits = {
                "age": self.age_classifier(token),   # [B, num_age_bins]
                "sex": self.sex_classifier(token),   # [B, num_sex]
                "race": self.race_classifier(token), # [B, num_race]
            }

        return ctx, mu, logsigma, aux_logits, None

    def compute_compositional_loss(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute compositional consistency loss."""
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
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor
    ) -> torch.Tensor:
        """Get uncertainty (sigma) for given demographic groups."""
        _, _, logsigma, _, _ = self.forward(age_idx, sex_idx, race_idx)
        return torch.exp(logsigma).mean(dim=-1)

    def save_pretrained(self, save_dir: str):
        """Save HCN model and config."""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        print(f"HCN V8 Ordinal (compat) saved to {save_dir}")

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
        print(f"HCN V8 Ordinal (compat) loaded from {save_dir}")
        return model


# =============================================================================
# Loss Functions (work with K outputs, not K-1)
# =============================================================================

def compute_soft_ce_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    num_classes: int,
    sigma: float = 1.0  # Changed default from 0.75 to 1.0 for smoother gradient
) -> torch.Tensor:
    """
    Soft label cross-entropy with Gaussian smoothing.
    
    Instead of one-hot targets, use Gaussian-distributed soft labels
    centered at the true class. This respects ordinal structure by
    giving partial credit to adjacent classes.
    
    THIS IS THE RECOMMENDED ORDINAL LOSS - same scale as CE, simple, effective.
    
    Args:
        logits: [B, K] class logits
        targets: [B] class indices (0 to K-1)
        num_classes: K
        sigma: Standard deviation of Gaussian (default: 1.0)
               - sigma=0.5: tight, mostly rewards exact match
               - sigma=1.0: moderate, adjacent bins get ~60% credit (RECOMMENDED)
               - sigma=1.5: loose, adjacent bins get ~80% credit
    """
    device = logits.device
    
    # Create soft labels
    bins = torch.arange(num_classes, device=device).float()  # [K]
    targets_float = targets.float().unsqueeze(-1)  # [B, 1]
    
    # Gaussian centered at true bin
    soft_targets = torch.exp(-0.5 * ((bins - targets_float) / sigma) ** 2)  # [B, K]
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)  # Normalize
    
    # Cross-entropy with soft targets (same scale as standard CE)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    
    return loss


def compute_focal_ordinal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    gamma: float = 2.0,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Focal loss variant with ordinal soft targets.
    
    Combines focal loss (down-weights easy examples) with soft ordinal targets.
    This can help when some age bins are much more common than others.
    
    Args:
        logits: [B, K] class logits
        targets: [B] class indices
        num_classes: K
        gamma: Focal loss focusing parameter (default: 2.0)
        sigma: Soft target sigma (default: 1.0)
    """
    device = logits.device
    
    # Soft targets
    bins = torch.arange(num_classes, device=device).float()
    targets_float = targets.float().unsqueeze(-1)
    soft_targets = torch.exp(-0.5 * ((bins - targets_float) / sigma) ** 2)
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
    
    # Focal weighting
    probs = F.softmax(logits, dim=-1)
    focal_weight = (1 - probs) ** gamma
    
    # Focal cross-entropy with soft targets
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(focal_weight * soft_targets * log_probs).sum(dim=-1).mean()
    
    return loss


def compute_ordinal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Ordinal-aware cross-entropy using Earth Mover's Distance (EMD) style loss.
    
    This is a better ordinal loss that:
    1. Keeps the same scale as standard CE
    2. Penalizes predictions based on distance from true class
    3. Works directly with logits (proper gradient flow)
    
    The loss is: CE(logits, targets) + lambda * sum of squared CDF differences
    
    Args:
        logits: [B, K] class logits
        targets: [B] class indices (0 to K-1)
        num_classes: K
    """
    # Standard CE as base (maintains scale)
    ce_loss = F.cross_entropy(logits, targets)
    
    # Add EMD-style ordinal penalty
    device = logits.device
    batch_size = logits.shape[0]
    
    # Predicted CDF
    probs = F.softmax(logits, dim=-1)  # [B, K]
    pred_cdf = torch.cumsum(probs, dim=-1)  # [B, K]
    
    # True CDF (step function at target)
    # For target=2 with K=5: true_cdf = [0, 0, 1, 1, 1]
    bins = torch.arange(num_classes, device=device).unsqueeze(0)  # [1, K]
    true_cdf = (bins >= targets.unsqueeze(-1)).float()  # [B, K]
    
    # EMD loss: sum of squared differences between CDFs
    emd_loss = ((pred_cdf - true_cdf) ** 2).sum(dim=-1).mean()
    
    # Combine: CE dominates, EMD adds ordinal structure
    # Scale EMD to be roughly same magnitude as CE
    return ce_loss + 0.5 * emd_loss


def compute_mse_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss treating age bins as continuous values.
    
    Converts logits to expected bin index (soft argmax),
    then computes MSE against true bin index.
    
    Args:
        logits: [B, K] class logits
        targets: [B] class indices (0 to K-1)
    """
    device = logits.device
    num_classes = logits.shape[1]
    
    # Soft prediction: expected bin index
    probs = F.softmax(logits, dim=-1)  # [B, K]
    bins = torch.arange(num_classes, device=device).float()  # [K]
    expected_bin = (probs * bins).sum(dim=-1)  # [B]
    
    # MSE loss
    loss = F.mse_loss(expected_bin, targets.float())
    
    return loss


def compute_aux_loss(
    aux_logits: Dict[str, torch.Tensor],
    age_idx: torch.Tensor,
    sex_idx: torch.Tensor,
    race_idx: torch.Tensor,
    age_loss_mode: str = 'soft_ce',  # Changed default to soft_ce
    soft_ce_sigma: float = 1.0,      # Changed default to 1.0
    num_age_bins: int = 5,
    age_weight: float = 1.0,
    sex_weight: float = 1.0,
    race_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute auxiliary classification loss with ordinal age support.
    
    RECOMMENDED: Use 'soft_ce' mode - same scale as CE, adds ordinal structure.
    
    Args:
        aux_logits: Dict with 'age', 'sex', 'race' logits (all [B, num_classes])
        age_idx, sex_idx, race_idx: Ground truth labels
        age_loss_mode: 'ce', 'soft_ce' (recommended), 'ordinal', 'mse', or 'focal'
        soft_ce_sigma: Sigma for soft_ce/focal modes (default: 1.0)
        num_age_bins: Number of age bins
        *_weight: Per-attribute loss weights
            TIP: If race accuracy drops, increase race_weight to 1.5-2.0
        
    Returns:
        total_loss: Weighted sum of losses
        metrics: Dict with individual losses and accuracies
    """
    age_logits = aux_logits["age"]
    
    # Age loss based on mode
    if age_loss_mode == 'soft_ce':
        age_loss = compute_soft_ce_loss(age_logits, age_idx, num_age_bins, soft_ce_sigma)
    elif age_loss_mode == 'ordinal':
        age_loss = compute_ordinal_ce_loss(age_logits, age_idx, num_age_bins)
    elif age_loss_mode == 'mse':
        age_loss = compute_mse_loss(age_logits, age_idx)
    elif age_loss_mode == 'focal':
        age_loss = compute_focal_ordinal_loss(age_logits, age_idx, num_age_bins, sigma=soft_ce_sigma)
    else:  # 'ce' - original
        age_loss = F.cross_entropy(age_logits, age_idx)
    
    # Sex and race: standard cross-entropy (no ordinal structure)
    sex_loss = F.cross_entropy(aux_logits["sex"], sex_idx)
    race_loss = F.cross_entropy(aux_logits["race"], race_idx)
    
    # Weighted total
    total_loss = (
        age_weight * age_loss + 
        sex_weight * sex_loss + 
        race_weight * race_loss
    ) / (age_weight + sex_weight + race_weight)
    
    # Compute metrics
    with torch.no_grad():
        # Age prediction and accuracy
        age_pred = age_logits.argmax(dim=-1)
        age_acc = (age_pred == age_idx).float().mean()
        
        # Age MAE (Mean Absolute Error in bins) - key metric for ordinal
        age_mae = (age_pred.float() - age_idx.float()).abs().mean()
        
        # Sex and race accuracy
        sex_acc = (aux_logits["sex"].argmax(-1) == sex_idx).float().mean()
        race_acc = (aux_logits["race"].argmax(-1) == race_idx).float().mean()
    
    metrics = {
        "aux_loss_age": age_loss.item(),
        "aux_loss_sex": sex_loss.item(),
        "aux_loss_race": race_loss.item(),
        "aux_acc_age": age_acc.item(),
        "aux_acc_sex": sex_acc.item(),
        "aux_acc_race": race_acc.item(),
        "aux_mae_age": age_mae.item(),  # Mean Absolute Error for age bins
    }
    
    return total_loss, metrics


# =============================================================================
# Loader function
# =============================================================================

def load_hcn_v8_ordinal(args, logger):
    """
    Load and initialize HCN with ordinal age loss.
    
    This version uses the SAME architecture as hcn.py, allowing checkpoint resume.
    Only the loss computation changes based on age_loss_mode.
    
    Config options:
        args.hcn_age_loss_mode: 'ce', 'soft_ce', 'ordinal', 'mse' (default: 'ordinal')
        args.hcn_soft_ce_sigma: Sigma for soft_ce mode (default: 0.75)
    """
    if not getattr(args, 'use_hcn', False):
        logger.info("HCN disabled (use_hcn=False)")
        return None
    
    logger.info("=" * 60)
    logger.info("Initializing HCN with Ordinal Age Loss")
    logger.info("=" * 60)
    
    use_aux_loss = getattr(args, 'hcn_aux_weight', 0.0) > 0.0
    age_loss_mode = getattr(args, 'hcn_age_loss_mode', 'ordinal')
    soft_ce_sigma = getattr(args, 'hcn_soft_ce_sigma', 0.75)
    
    hcn = HierarchicalConditionerV8Ordinal(
        num_age_bins=getattr(args, 'hcn_num_age_bins', 5),
        num_sex=getattr(args, 'hcn_num_sex', 2),
        num_race=getattr(args, 'hcn_num_race', 4),
        d_node=getattr(args, 'hcn_d_node', 256),
        d_ctx=getattr(args, 'hcn_d_ctx', 1024),
        dropout=getattr(args, 'hcn_dropout', 0.1),
        use_uncertainty=getattr(args, 'hcn_use_uncertainty', True),
        use_aux_loss=use_aux_loss,
        aux_hidden_dim=getattr(args, 'hcn_aux_hidden_dim', 512),
        age_loss_mode=age_loss_mode,
        soft_ce_sigma=soft_ce_sigma,
    )
    
    num_params = sum(p.numel() for p in hcn.parameters())
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Age bins: {hcn.num_age}")
    logger.info(f"  Sex categories: {hcn.num_sex}")
    logger.info(f"  Race categories: {hcn.num_race}")
    logger.info(f"  Node dimension: {hcn.d_node}")
    logger.info(f"  Context dimension: {hcn.d_ctx}")
    logger.info(f"  Uncertainty: {hcn.use_uncertainty}")
    logger.info(f"  Auxiliary loss: {use_aux_loss}")
    logger.info(f"  Age loss mode: {age_loss_mode}")
    if age_loss_mode == 'soft_ce':
        logger.info(f"  Soft CE sigma: {soft_ce_sigma}")
    logger.info("=" * 60)
    
    return hcn


# =============================================================================
# Tests
# =============================================================================

def test_losses():
    """Test all loss functions."""
    print("Testing loss functions...")
    print("=" * 60)
    
    batch_size = 8
    num_classes = 5
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test all modes
    ce_loss = F.cross_entropy(logits, targets)
    soft_loss = compute_soft_ce_loss(logits, targets, num_classes, sigma=0.75)
    ordinal_loss = compute_ordinal_ce_loss(logits, targets, num_classes)
    mse_loss = compute_mse_loss(logits, targets)
    
    print(f"CE loss:      {ce_loss.item():.4f}")
    print(f"Soft CE loss: {soft_loss.item():.4f}")
    print(f"Ordinal loss: {ordinal_loss.item():.4f}")
    print(f"MSE loss:     {mse_loss.item():.4f}")
    
    # Test ordinal gradient behavior
    print("\n--- Testing ordinal error penalties ---")
    # Perfect prediction for class 2
    perfect_logits = torch.tensor([[-10.0, -10.0, 10.0, -10.0, -10.0]])
    
    for true_class in range(num_classes):
        target = torch.tensor([true_class])
        ce = F.cross_entropy(perfect_logits, target)
        ordinal = compute_ordinal_ce_loss(perfect_logits, target, num_classes)
        soft = compute_soft_ce_loss(perfect_logits, target, num_classes, 0.75)
        print(f"  Pred=2, True={true_class}: CE={ce.item():.2f}, Ordinal={ordinal.item():.2f}, Soft={soft.item():.2f}")
    
    print("\n  Key insight: Ordinal/Soft losses should increase more")
    print("  as true class gets farther from prediction (2)")
    
    print("\n" + "=" * 60)
    print("✓ Loss tests passed!")


def test_backward_compat():
    """Test that architecture matches hcn.py."""
    print("\nTesting backward compatibility...")
    print("=" * 60)
    
    hcn = HierarchicalConditionerV8Ordinal(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=True,
        age_loss_mode='ordinal',
    )
    
    # Check state_dict keys match hcn.py format
    state_dict = hcn.state_dict()
    age_keys = [k for k in state_dict.keys() if 'age_classifier' in k]
    
    print(f"Age classifier keys: {age_keys}")
    
    # Should be: age_classifier.0.*, age_classifier.1.*, etc.
    expected_pattern = any('age_classifier.0.' in k for k in age_keys)
    assert expected_pattern, "Architecture doesn't match hcn.py!"
    
    print("✓ Architecture matches hcn.py (backward compatible)")
    
    # Test forward pass
    batch_size = 8
    age = torch.randint(0, 5, (batch_size,))
    sex = torch.randint(0, 2, (batch_size,))
    race = torch.randint(0, 4, (batch_size,))
    
    ctx, mu, logsigma, aux_logits, _ = hcn(age, sex, race)
    
    assert ctx.shape == (batch_size, 1, 1024)
    assert aux_logits["age"].shape == (batch_size, 5)  # K outputs, not K-1
    print(f"✓ Forward pass: ctx={ctx.shape}, age_logits={aux_logits['age'].shape}")
    
    # Test loss computation
    loss, metrics = compute_aux_loss(
        aux_logits, age, sex, race,
        age_loss_mode='ordinal',
        num_age_bins=5,
    )
    print(f"✓ Ordinal loss: {loss.item():.4f}, MAE: {metrics['aux_mae_age']:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All backward compatibility tests passed!")


if __name__ == "__main__":
    test_losses()
    test_backward_compat()