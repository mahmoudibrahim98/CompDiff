"""
Hierarchical Conditioner Network: Auxiliary Loss on Output Token

Key insight: auxiliary classifiers are applied to the OUTPUT TOKEN (after proj_ctx)
rather than to mu. This forces the projection to preserve demographic information.

    h_child → mu_head → mu → proj_ctx → token → UNet
                                          ↓
                                  aux_classifiers ✓ (supervised here)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import json
import os


class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm and SiLU activation.

    Args:
        d_in: Input dimension
        d_hidden: Hidden layer dimension
        d_out: Output dimension
        dropout: Dropout probability (default: 0.1)
    """
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


class HierarchicalConditioner(nn.Module):
    """
    Hierarchical Conditioning Network with Auxiliary Loss on Output Token
    
    Auxiliary classifiers are applied to the OUTPUT TOKEN (after proj_ctx)
    rather than to mu. This forces the projection to preserve demographic information.
    
    Architecture:
        Grandparents (single attributes) → Parents (pairwise) → Child (triple)
        → mu/logsigma → sample z → proj_ctx → TOKEN → aux_classifiers
                                                ↓
                                              UNet

    Args:
        num_age_bins: Number of age categories
        num_sex: Number of sex categories (typically 2: M/F)
        num_race: Number of race/ethnicity categories
        d_node: Hidden dimension for embeddings (default: 256)
        d_ctx: Output dimension matching UNet cross_attention_dim (default: 1024)
        dropout: Dropout probability (default: 0.1)
        use_uncertainty: Whether to output mu/logsigma for variational sampling
        use_aux_loss: Whether to include auxiliary classifiers (on token)
        aux_hidden_dim: Hidden dimension for auxiliary classifiers (default: 512)
        encode_age: Whether to include age in the hierarchy (default: True)
                     If False, only sex × race composition is used

    Input:
        age_idx: [B] Long tensor of age bin indices (0 to num_age_bins-1), optional if encode_age=False
        sex_idx: [B] Long tensor of sex indices (0 to num_sex-1)
        race_idx: [B] Long tensor of race indices (0 to num_race-1)

    Output:
        ctx: [B, 1, d_ctx] - Demographic context token to concatenate with text
        mu: [B, d_node] - Mean of variational distribution
        logsigma: [B, d_node] - Log std of variational distribution
        aux_logits: Dict with 'age' (if encode_age), 'sex', 'race' logits (from TOKEN, not mu)
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
        aux_hidden_dim: int = 512,  # V8: Hidden dim for token classifiers
        encode_age: bool = True,  # V10: Optionally exclude age
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
            'encode_age': encode_age,
        }

        self.num_age = num_age_bins
        self.num_sex = num_sex
        self.num_race = num_race
        self.d_node = d_node
        self.d_ctx = d_ctx
        self.use_uncertainty = use_uncertainty
        self.use_aux_loss = use_aux_loss
        self.encode_age = encode_age

        # === Grandparent embeddings (single attributes) ===
        self.emb_sex = nn.Embedding(num_sex, d_node)
        self.emb_race = nn.Embedding(num_race, d_node)
        
        if encode_age:
            self.emb_age = nn.Embedding(num_age_bins, d_node)
        else:
            self.emb_age = None

        # === Parent composers (pairwise compositions) ===
        if encode_age:
            # Full hierarchy: age × sex × race
            self.compose_age_sex = MLP(
                d_in=2 * d_node,
                d_hidden=2 * d_node,
                d_out=d_node,
                dropout=dropout
            )
            self.compose_age_race = MLP(
                d_in=2 * d_node,
                d_hidden=2 * d_node,
                d_out=d_node,
                dropout=dropout
            )
            self.compose_sex_race = MLP(
                d_in=2 * d_node,
                d_hidden=2 * d_node,
                d_out=d_node,
                dropout=dropout
            )
            # === Child composer (triple composition from all parents) ===
            self.compose_all = MLP(
                d_in=3 * d_node,
                d_hidden=2 * d_node,
                d_out=d_node,
                dropout=dropout
            )
        else:
            # Simplified: sex × race only
            self.compose_age_sex = None
            self.compose_age_race = None
            self.compose_sex_race = MLP(
                d_in=2 * d_node,
                d_hidden=2 * d_node,
                d_out=d_node,
                dropout=dropout
            )
            self.compose_all = None

        # === Uncertainty heads (for rare group detection) ===
        if use_uncertainty:
            self.mu_head = nn.Linear(d_node, d_node)
            self.logsigma_head = nn.Linear(d_node, d_node)

        # === Project to UNet cross-attention dimension ===
        self.proj_ctx = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_ctx),
        )

        # === Auxiliary classifiers on OUTPUT TOKEN (d_ctx), not mu (d_node) ===
        # This forces proj_ctx to preserve demographics
        if use_aux_loss:
            # More expressive classifiers since we're working from d_ctx
            if encode_age:
                self.age_classifier = nn.Sequential(
                    nn.LayerNorm(d_ctx),
                    nn.Linear(d_ctx, aux_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(aux_hidden_dim, num_age_bins),
                )
            else:
                self.age_classifier = None
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
        embeddings = [self.emb_sex, self.emb_race]
        if self.emb_age is not None:
            embeddings.append(self.emb_age)
        for emb in embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # Initialize uncertainty heads conservatively
        if self.use_uncertainty:
            nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(self.logsigma_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.logsigma_head.bias, -1.0)  # Start with low variance

    def forward(
        self,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
        age_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through hierarchical conditioning network.

        Args:
            sex_idx: [B] Long tensor of sex indices
            race_idx: [B] Long tensor of race indices
            age_idx: [B] Long tensor of age bin indices (optional if encode_age=False)

        Returns:
            ctx: [B, 1, d_ctx] - Demographic context to concatenate with text
            mu: [B, d_node] - Mean of variational distribution
            logsigma: [B, d_node] - Log std of variational distribution
            aux_logits: Dict with 'age' (if encode_age), 'sex', 'race' logits (FROM TOKEN)
            time_emb: None - V8 does not support timestep injection
        """
        # === Level 1: Grandparent embeddings (single attributes) ===
        e_sex = self.emb_sex(sex_idx)    # [B, d_node]
        e_race = self.emb_race(race_idx) # [B, d_node]
        
        if self.encode_age and age_idx is not None:
            e_age = self.emb_age(age_idx)    # [B, d_node]
            # === Level 2: Parent compositions (pairwise) ===
            h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], dim=-1))
            h_age_race = self.compose_age_race(torch.cat([e_age, e_race], dim=-1))
            h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], dim=-1))
            # === Level 3: Child composition (from all parents) ===
            h_child = self.compose_all(
                torch.cat([h_age_sex, h_age_race, h_sex_race], dim=-1)
            )
        else:
            # === Simplified: sex × race only ===
            h_child = self.compose_sex_race(torch.cat([e_sex, e_race], dim=-1))

        # === Uncertainty quantification (variational) ===
        if self.use_uncertainty:
            mu = self.mu_head(h_child)
            logsigma = torch.clamp(
                self.logsigma_head(h_child),
                min=-5.0,  # Minimum variance (stable training)
                max=1.0    # Maximum variance (prevent explosion)
            )

            # Sample during training (reparameterization trick)
            # Use mean during inference (deterministic)
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

        # === Auxiliary logits FROM TOKEN (not from mu!) ===
        # This forces proj_ctx to preserve demographics
        aux_logits = None
        if self.use_aux_loss:
            token = ctx.squeeze(1)  # [B, d_ctx]
            aux_logits = {
                "sex": self.sex_classifier(token),
                "race": self.race_classifier(token),
            }
            if self.encode_age and self.age_classifier is not None:
                aux_logits["age"] = self.age_classifier(token)

        # Timestep injection not supported in this version, return None for compatibility
        time_emb = None

        return ctx, mu, logsigma, aux_logits, time_emb

    def compute_compositional_loss(
        self,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
        age_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute compositional consistency loss.

        Enforces that the hierarchical composition is consistent with
        simple additive composition of grandparent embeddings.
        """
        # Get grandparent embeddings
        e_sex = self.emb_sex(sex_idx)
        e_race = self.emb_race(race_idx)
        
        if self.encode_age and age_idx is not None:
            e_age = self.emb_age(age_idx)
            # Hierarchical composition
            h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], -1))
            h_age_race = self.compose_age_race(torch.cat([e_age, e_race], -1))
            h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], -1))
            h_child = self.compose_all(torch.cat([h_age_sex, h_age_race, h_sex_race], -1))
            # Simple additive baseline
            h_additive = e_age + e_sex + e_race
        else:
            # Simplified: sex × race only
            h_child = self.compose_sex_race(torch.cat([e_sex, e_race], -1))
            # Simple additive baseline
            h_additive = e_sex + e_race

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(h_child, h_additive, dim=-1)
        loss_comp = (1 - cos_sim).mean()

        return loss_comp

    def get_uncertainty(
        self,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
        age_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get uncertainty (sigma) for given demographic groups.
        Useful for detecting which groups the model is uncertain about.
        """
        _, _, logsigma, _ = self.forward(sex_idx, race_idx, age_idx)
        sigma = torch.exp(logsigma).mean(dim=-1)
        return sigma

    def save_pretrained(self, save_dir: str):
        """Save HCN model and config."""
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        # Save weights
        weights_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

        print(f"HCN saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = "cpu"):
        """Load HCN model from saved checkpoint."""
        # Load config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model
        model = cls(**config)

        # Load weights
        weights_path = os.path.join(save_dir, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        print(f"HCN loaded from {save_dir}")
        return model


def compute_aux_loss(
    aux_logits: Dict[str, torch.Tensor],
    sex_idx: torch.Tensor,
    race_idx: torch.Tensor,
    age_idx: Optional[torch.Tensor] = None,
    age_weight: float = 1.0,
    sex_weight: float = 1.0,
    race_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute auxiliary classification loss from token logits.
    
    Args:
        aux_logits: Dict with 'age' (optional), 'sex', 'race' logits
        sex_idx, race_idx: Ground truth labels
        age_idx: Ground truth age labels (optional if age not encoded)
        *_weight: Per-attribute loss weights
        
    Returns:
        total_loss: Weighted sum of CE losses
        metrics: Dict with individual losses and accuracies
    """
    losses = []
    weights = []
    metrics = {}
    
    sex_ce = F.cross_entropy(aux_logits["sex"], sex_idx)
    race_ce = F.cross_entropy(aux_logits["race"], race_idx)
    losses.append(sex_ce)
    weights.append(sex_weight)
    losses.append(race_ce)
    weights.append(race_weight)
    
    # Compute accuracies for logging
    with torch.no_grad():
        sex_acc = (aux_logits["sex"].argmax(-1) == sex_idx).float().mean()
        race_acc = (aux_logits["race"].argmax(-1) == race_idx).float().mean()
    
    metrics["aux_loss_sex"] = sex_ce.item()
    metrics["aux_loss_race"] = race_ce.item()
    metrics["aux_acc_sex"] = sex_acc.item()
    metrics["aux_acc_race"] = race_acc.item()
    
    # Age loss (if age is encoded)
    if "age" in aux_logits and age_idx is not None:
        age_ce = F.cross_entropy(aux_logits["age"], age_idx)
        losses.append(age_ce)
        weights.append(age_weight)
        with torch.no_grad():
            age_acc = (aux_logits["age"].argmax(-1) == age_idx).float().mean()
        metrics["aux_loss_age"] = age_ce.item()
        metrics["aux_acc_age"] = age_acc.item()
    
    # Weighted average
    total_loss = sum(w * l for w, l in zip(weights, losses)) / sum(weights)
    
    return total_loss, metrics


def load_hcn(args, logger):
    """
    Load and initialize HCN.
    
    Args:
        args: Training arguments/config
        logger: Logger instance
        
    Returns:
        hcn: HierarchicalConditioner instance or None
    """
    if not getattr(args, 'use_hcn', False):
        logger.info("HCN disabled (use_hcn=False)")
        return None
    
    logger.info("=" * 60)
    logger.info("Initializing HCN (Auxiliary Loss on Token)")
    logger.info("=" * 60)
    
    # Determine aux_loss setting
    use_aux_loss = getattr(args, 'hcn_aux_weight', 0.0) > 0.0
    aux_hidden_dim = getattr(args, 'hcn_aux_hidden_dim', 512)
    encode_age = getattr(args, 'hcn_encode_age', True)  # V10: Optionally exclude age
    
    hcn = HierarchicalConditioner(
        num_age_bins=getattr(args, 'hcn_num_age_bins', 5),
        num_sex=getattr(args, 'hcn_num_sex', 2),
        num_race=getattr(args, 'hcn_num_race', 4),
        d_node=getattr(args, 'hcn_d_node', 256),
        d_ctx=getattr(args, 'hcn_d_ctx', 1024),
        dropout=getattr(args, 'hcn_dropout', 0.1),
        use_uncertainty=getattr(args, 'hcn_use_uncertainty', True),
        use_aux_loss=use_aux_loss,
        aux_hidden_dim=aux_hidden_dim,
        encode_age=encode_age,
    )
    
    num_params = sum(p.numel() for p in hcn.parameters())
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Encode age: {encode_age}")
    if encode_age:
        logger.info(f"  Age bins: {hcn.num_age}")
    logger.info(f"  Sex categories: {hcn.num_sex}")
    logger.info(f"  Race categories: {hcn.num_race}")
    logger.info(f"  Node dimension: {hcn.d_node}")
    logger.info(f"  Context dimension: {hcn.d_ctx}")
    logger.info(f"  Uncertainty: {hcn.use_uncertainty}")
    logger.info(f"  Auxiliary loss (on token): {use_aux_loss}")
    if use_aux_loss:
        logger.info(f"  Auxiliary hidden dimension: {aux_hidden_dim}")
    logger.info("=" * 60)
    
    return hcn


# =============================================================================
# Training loop integration example
# =============================================================================

def train_step(
    hcn: HierarchicalConditioner,
    batch: Dict[str, torch.Tensor],
    encoder_hidden_states: torch.Tensor,
    args,
    global_step: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
    """
    Training step - get HCN token and compute losses.
    
    Returns:
        encoder_hidden_states: Modified with HCN token concatenated
        kl_loss: KL divergence loss
        comp_loss: Compositional consistency loss  
        aux_loss: Auxiliary classification loss (on token)
        logs: Dict of metrics to log
    """
    # Get HCN outputs
    age_idx = batch.get("age_idx") if hcn.encode_age else None
    hcn_ctx, mu, logsigma, aux_logits, _ = hcn(
        sex_idx=batch["sex_idx"],
        race_idx=batch["race_idx"],
        age_idx=age_idx,
    )
    
    # Concatenate HCN token to text embeddings
    encoder_hidden_states = torch.cat(
        [encoder_hidden_states, hcn_ctx], dim=1
    )  # [B, 78, d_ctx]
    
    # Compute KL loss
    kl_loss = -0.5 * torch.sum(
        1 + 2 * logsigma - mu ** 2 - torch.exp(2 * logsigma),
        dim=-1
    ).mean()
    
    # Compute compositional loss (need unwrapped model for custom methods)
    comp_loss = hcn.compute_compositional_loss(
        sex_idx=batch["sex_idx"],
        race_idx=batch["race_idx"],
        age_idx=age_idx,
    )
    
    # Compute auxiliary loss (on token)
    aux_loss = None
    logs = {
        "hcn_ctx_norm": hcn_ctx.norm(dim=-1).mean().item(),
        "kl_loss": kl_loss.item(),
        "comp_loss": comp_loss.item(),
    }
    
    if aux_logits is not None:
        aux_loss, aux_metrics = compute_aux_loss(
            aux_logits,
            sex_idx=batch["sex_idx"],
            race_idx=batch["race_idx"],
            age_idx=age_idx,
        )
        logs.update(aux_metrics)
        logs["aux_loss"] = aux_loss.item()
    
    return encoder_hidden_states, kl_loss, comp_loss, aux_loss, logs


# =============================================================================
# Tests
# =============================================================================

def test_hcn():
    """Test HCN module."""
    print("Testing HCN...")
    print("=" * 60)

    # Create model
    hcn = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=True,
        aux_hidden_dim=512,
    )

    batch_size = 8
    age = torch.randint(0, 5, (batch_size,))
    sex = torch.randint(0, 2, (batch_size,))
    race = torch.randint(0, 4, (batch_size,))

    # Test forward pass with age
    hcn.train()
    ctx, mu, logsigma, aux_logits, time_emb = hcn(sex_idx=sex, race_idx=race, age_idx=age)

    assert ctx.shape == (batch_size, 1, 1024), f"Expected (8, 1, 1024), got {ctx.shape}"
    assert mu.shape == (batch_size, 256), f"Expected (8, 256), got {mu.shape}"
    assert logsigma.shape == (batch_size, 256), f"Expected (8, 256), got {logsigma.shape}"
    assert aux_logits is not None, "aux_logits should not be None"
    assert time_emb is None, "time_emb should be None"
    print(f"✓ Forward pass (with age): ctx shape = {ctx.shape}")

    # Test that aux_logits have correct shapes
    assert aux_logits["age"].shape == (batch_size, 5), f"Age logits wrong shape"
    assert aux_logits["sex"].shape == (batch_size, 2), f"Sex logits wrong shape"
    assert aux_logits["race"].shape == (batch_size, 4), f"Race logits wrong shape"
    print(f"✓ Aux logits shapes correct")

    # Test auxiliary loss computation
    aux_loss, metrics = compute_aux_loss(aux_logits, sex_idx=sex, race_idx=race, age_idx=age)
    assert aux_loss.ndim == 0, "Aux loss should be scalar"
    print(f"✓ Aux loss: {aux_loss.item():.4f}")
    print(f"  Age acc: {metrics['aux_acc_age']:.2%}")
    print(f"  Sex acc: {metrics['aux_acc_sex']:.2%}")
    print(f"  Race acc: {metrics['aux_acc_race']:.2%}")

    # Test compositional loss
    comp_loss = hcn.compute_compositional_loss(sex_idx=sex, race_idx=race, age_idx=age)
    assert comp_loss.ndim == 0, "Compositional loss should be scalar"
    print(f"✓ Compositional loss: {comp_loss.item():.4f}")

    # Test uncertainty
    sigma = hcn.get_uncertainty(sex_idx=sex, race_idx=race, age_idx=age)
    assert sigma.shape == (batch_size,), f"Expected ({batch_size},), got {sigma.shape}"
    print(f"✓ Uncertainty: mean sigma = {sigma.mean().item():.4f}")

    # Test gradient flow through aux classifiers to proj_ctx
    print("\n--- Testing gradient flow ---")
    hcn.zero_grad()
    ctx, _, _, aux_logits, _ = hcn(sex_idx=sex, race_idx=race, age_idx=age)
    aux_loss, _ = compute_aux_loss(aux_logits, sex_idx=sex, race_idx=race, age_idx=age)
    aux_loss.backward()
    
    # Check that proj_ctx gets gradients from aux_loss
    proj_ctx_grad = hcn.proj_ctx[1].weight.grad
    assert proj_ctx_grad is not None, "proj_ctx should have gradients!"
    assert proj_ctx_grad.abs().sum() > 0, "proj_ctx gradients should be non-zero!"
    print(f"✓ proj_ctx gradient norm: {proj_ctx_grad.norm().item():.6f}")
    print("  This confirms aux_loss flows back through proj_ctx!")

    # Test save/load
    print("\n--- Testing save/load ---")
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    try:
        hcn.save_pretrained(temp_dir)
        hcn_loaded = HierarchicalConditioner.from_pretrained(temp_dir)
        ctx_loaded, _, _, aux_logits_loaded, _ = hcn_loaded(sex_idx=sex, race_idx=race, age_idx=age)
        assert aux_logits_loaded is not None
        print(f"✓ Save/load successful")
    finally:
        shutil.rmtree(temp_dir)
    
    # Test without age encoding
    print("\n--- Testing without age encoding ---")
    hcn_no_age = HierarchicalConditionerV8(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=True,
        aux_hidden_dim=512,
        encode_age=False,
    )
    hcn_no_age.train()
    ctx_no_age, mu_no_age, logsigma_no_age, aux_logits_no_age, _ = hcn_no_age(
        sex_idx=sex, race_idx=race, age_idx=None
    )
    assert ctx_no_age.shape == (batch_size, 1, 1024), f"Expected (8, 1, 1024), got {ctx_no_age.shape}"
    assert "age" not in aux_logits_no_age, "Age should not be in aux_logits when encode_age=False"
    assert "sex" in aux_logits_no_age, "Sex should be in aux_logits"
    assert "race" in aux_logits_no_age, "Race should be in aux_logits"
    print(f"✓ Forward pass (without age): ctx shape = {ctx_no_age.shape}")
    
    # Test auxiliary loss without age
    aux_loss_no_age, metrics_no_age = compute_aux_loss(
        aux_logits_no_age, sex_idx=sex, race_idx=race, age_idx=None
    )
    assert aux_loss_no_age.ndim == 0, "Aux loss should be scalar"
    assert "aux_acc_age" not in metrics_no_age, "Age metrics should not exist"
    print(f"✓ Aux loss (no age): {aux_loss_no_age.item():.4f}")
    print(f"  Sex acc: {metrics_no_age['aux_acc_sex']:.2%}")
    print(f"  Race acc: {metrics_no_age['aux_acc_race']:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY DESIGN:")
    print("  aux_classifiers(token)  - proj_ctx MUST preserve demographics")
    print("=" * 60)
    print(f"✓ All tests passed!")
    print(f"✓ Total parameters: {sum(p.numel() for p in hcn.parameters()):,}")


if __name__ == "__main__":
    test_hcn()