"""
Hierarchical Conditioner Network (HCN) for Compositional Demographic Embeddings

Project context (RoentGen-v2):
    We condition a chest X-ray diffusion model on clinical text + demographics (age, sex, race).
    Putting demographics in the text prompt uses up CLIP's 77-token budget and hurts FID; removing
    them improves FID but then the UNet has no demographic signal. HCN turns (age_idx, sex_idx,
    race_idx) into a single embedding that can be injected as a token (V1), via FiLM (V5), or
    via timestep embedding (V6)—so we can keep demographics out of the prompt and still control them.

This module implements a hierarchical approach to learning demographic representations
for fairness in medical image generation. It addresses the data scarcity problem at
demographic intersections by leveraging compositional structure.

Architecture:
    Grandparents (single attributes) → Parents (pairwise) → Child (triple) → Context

Key Features:
    - Compositional learning: Leverages abundant single-attribute data
    - Uncertainty quantification: Knows when it's uncertain about rare groups
    - Hierarchical composition: Age×Sex, Age×Race, Sex×Race → Age×Sex×Race

Output modes (see forward() and config use_hcn_film / use_hcn_timestep_injection):
    - Token: ctx [B, 1, d_ctx] concatenated with text embeddings (V1)
    - FiLM:  ctx [B, d_node] passed to FiLM adapter for per-block modulation (V5)
    - V6:    time_emb [B, d_time_emb] added to UNet timestep embedding; ctx is None

See PROGRESS_LOG.md "HCN: Full Context" for version history and design rationale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union


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
    Hierarchical Conditioning Network for compositional demographic embeddings.

    Learns demographic representations at three levels:
    - Grandparents: Single attributes (age, sex, race)
    - Parents: Pairwise compositions (age×sex, age×race, sex×race)
    - Child: Triple composition (age×sex×race)

    The hierarchical structure allows the model to leverage abundant data from
    single attributes and pairs to improve representations of rare intersections.

    Args:
        num_age_bins: Number of age categories
        num_sex: Number of sex categories (typically 2: M/F)
        num_race: Number of race/ethnicity categories
        d_node: Hidden dimension for embeddings (default: 256)
        d_ctx: Output dimension matching UNet cross_attention_dim (default: 1024)
        dropout: Dropout probability (default: 0.1)
        use_uncertainty: Whether to output mu/logsigma for variational sampling

    Input:
        age_idx: [B] Long tensor of age bin indices (0 to num_age_bins-1)
        sex_idx: [B] Long tensor of sex indices (0 to num_sex-1)
        race_idx: [B] Long tensor of race indices (0 to num_race-1)

    Output:
        ctx: [B, 1, d_ctx] - Demographic context token to concatenate with text
        mu: [B, d_node] - Mean of variational distribution
        logsigma: [B, d_node] - Log std of variational distribution
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
        use_aux_loss: bool = False,
        use_film_output: bool = False,  # V5: Output hidden state for FiLM instead of token
        use_timestep_injection: bool = False,  # V6: Output embedding to add to timestep
        d_time_emb: int = 1280,  # V6: Timestep embedding dimension (1280 for SD 2.1)
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
            'use_film_output': use_film_output,
            'use_timestep_injection': use_timestep_injection,
            'd_time_emb': d_time_emb,
        }

        self.num_age = num_age_bins
        self.num_sex = num_sex
        self.num_race = num_race
        self.d_node = d_node
        self.d_ctx = d_ctx
        self.d_time_emb = d_time_emb
        self.use_uncertainty = use_uncertainty
        self.use_aux_loss = use_aux_loss
        self.use_film_output = use_film_output
        self.use_timestep_injection = use_timestep_injection

        # === Grandparent embeddings (single attributes) ===
        self.emb_age = nn.Embedding(num_age_bins, d_node)
        self.emb_sex = nn.Embedding(num_sex, d_node)
        self.emb_race = nn.Embedding(num_race, d_node)

        # === Parent composers (pairwise compositions) ===
        # These learn how age and sex interact, age and race interact, etc.
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
            d_in=3 * d_node,  # Three parent embeddings
            d_hidden=2 * d_node,
            d_out=d_node,
            dropout=dropout
        )

        # === Uncertainty heads (for rare group detection) ===
        if use_uncertainty:
            self.mu_head = nn.Linear(d_node, d_node)
            self.logsigma_head = nn.Linear(d_node, d_node)

        # === Project to UNet cross-attention dimension ===
        # Only create proj_ctx when NOT in FiLM/timestep mode (to avoid unused parameters in DDP)
        if not use_film_output and not use_timestep_injection:
            self.proj_ctx = nn.Sequential(
                nn.LayerNorm(d_node),
                nn.Linear(d_node, d_ctx),
            )
        else:
            self.proj_ctx = None
        
        # === V6: Project to timestep embedding dimension ===
        # This embedding gets added to the UNet's timestep embedding
        if use_timestep_injection:
            self.proj_time = nn.Sequential(
                nn.LayerNorm(d_node),
                nn.Linear(d_node, d_time_emb),
                nn.SiLU(),
                nn.Linear(d_time_emb, d_time_emb),
            )
            # Initialize to output near-zero (don't disrupt initial timestep embedding)
            nn.init.zeros_(self.proj_time[-1].weight)
            nn.init.zeros_(self.proj_time[-1].bias)
        else:
            self.proj_time = None

        # === Auxiliary demographic classifiers (for diagnostic losses) ===
        # Only include if use_aux_loss is True (matches v1 behavior when aux_weight=0)
        if use_aux_loss:
            self.age_classifier = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, num_age_bins),
        )
            self.sex_classifier = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, num_sex),
        )
            self.race_classifier = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, num_race),
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

        # Initialize uncertainty heads conservatively
        if self.use_uncertainty:
            nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(self.logsigma_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.logsigma_head.bias, -1.0)  # Start with low variance

    def forward(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through hierarchical conditioning network.

        Args:
            age_idx: [B] Long tensor of age bin indices
            sex_idx: [B] Long tensor of sex indices
            race_idx: [B] Long tensor of race indices

        Returns:
            ctx: [B, 1, d_ctx] - Demographic context to concatenate with text (token mode)
                 OR [B, d_node] - Hidden state for FiLM adapter (FiLM mode)
                 OR None - When using timestep injection (V6 mode)
            mu: [B, d_node] - Mean of variational distribution
            logsigma: [B, d_node] - Log std of variational distribution
            aux_logits: Optional dict with 'age', 'sex', 'race' logits
            time_emb: [B, d_time_emb] - Timestep embedding to add to UNet (V6 mode)
                      OR None - When not using timestep injection
        """
        # === Level 1: Grandparent embeddings (single attributes) ===
        e_age = self.emb_age(age_idx)    # [B, d_node]
        e_sex = self.emb_sex(sex_idx)    # [B, d_node]
        e_race = self.emb_race(race_idx) # [B, d_node]

        # === Level 2: Parent compositions (pairwise) ===
        h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], dim=-1))
        h_age_race = self.compose_age_race(torch.cat([e_age, e_race], dim=-1))
        h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], dim=-1))

        # === Level 3: Child composition (from all parents) ===
        h_child = self.compose_all(
            torch.cat([h_age_sex, h_age_race, h_sex_race], dim=-1)
        )

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

        # === Output depends on mode ===
        time_emb = None  # V6: timestep embedding injection
        
        if self.use_timestep_injection:
            # V6 mode: return timestep embedding to add to UNet's time_emb
            # Shape: [B, d_time_emb]
            time_emb = self.proj_time(z)
            ctx = None  # No context token in V6 mode
        elif self.use_film_output:
            # FiLM mode (V5): return hidden state for FiLM adapter
            # Shape: [B, d_node] - will be processed by FiLMAdapter
            ctx = z  # Hidden state for FiLM adapter
        else:
            # Token mode (V1 behavior): project to context and add sequence axis
            ctx = self.proj_ctx(z).unsqueeze(1)  # [B, 1, d_ctx]

        # === Auxiliary logits (use mu which is deterministic at inference) ===
        # Only compute if use_aux_loss is True (matches v1 behavior when aux_weight=0)
        if self.use_aux_loss:
            aux_logits = {
                "age": self.age_classifier(mu),
                "sex": self.sex_classifier(mu),
                "race": self.race_classifier(mu),
        }
        else:
            aux_logits = None

        return ctx, mu, logsigma, aux_logits, time_emb

    def compute_compositional_loss(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute compositional consistency loss.

        Enforces that the hierarchical composition is consistent with
        simple additive composition of grandparent embeddings.

        Intuition: E(age, sex, race) should be predictable from
        E(age) + E(sex) + E(race) in direction (not necessarily magnitude).

        Args:
            age_idx: [B] Age indices
            sex_idx: [B] Sex indices
            race_idx: [B] Race indices

        Returns:
            loss_comp: Scalar compositional loss (cosine-based)
        """
        # Get grandparent embeddings
        e_age = self.emb_age(age_idx)
        e_sex = self.emb_sex(sex_idx)
        e_race = self.emb_race(race_idx)

        # Hierarchical composition (normal forward pass)
        h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], -1))
        h_age_race = self.compose_age_race(torch.cat([e_age, e_race], -1))
        h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], -1))
        h_child = self.compose_all(torch.cat([h_age_sex, h_age_race, h_sex_race], -1))

        # Simple additive baseline (perfect compositionality would match this)
        h_additive = e_age + e_sex + e_race

        # Cosine similarity (direction alignment, not magnitude)
        # Loss = 1 - cos_sim, so 0 when perfectly aligned, 2 when opposite
        cos_sim = F.cosine_similarity(h_child, h_additive, dim=-1)
        loss_comp = (1 - cos_sim).mean()

        return loss_comp

    def get_uncertainty(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Get uncertainty (sigma) for given demographic groups.
        Useful for detecting which groups the model is uncertain about.

        Returns:
            sigma: [B] - Standard deviation for each sample
        """
        _, _, logsigma = self.forward(age_idx, sex_idx, race_idx)
        sigma = torch.exp(logsigma).mean(dim=-1)  # Average across dimensions
        return sigma

    def save_pretrained(self, save_dir: str):
        """Save HCN model and config."""
        import os
        import json

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
        import os
        import json

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


def test_hcn():
    """Quick test of HCN module."""
    print("Testing HCN module...")

    # Test HCN with aux_loss enabled
    print("  Testing HCN with aux_loss=True...")
    hcn_with_aux = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=True,
    )

    batch_size = 8
    age = torch.randint(0, 5, (batch_size,))
    sex = torch.randint(0, 2, (batch_size,))
    race = torch.randint(0, 4, (batch_size,))

    hcn_with_aux.train()
    ctx, mu, logsigma, aux_logits = hcn_with_aux(age, sex, race)

    assert ctx.shape == (batch_size, 1, 1024), f"Expected (8, 1, 1024), got {ctx.shape}"
    assert mu.shape == (batch_size, 256), f"Expected (8, 256), got {mu.shape}"
    assert logsigma.shape == (batch_size, 256), f"Expected (8, 256), got {logsigma.shape}"
    assert aux_logits is not None, "aux_logits should not be None when use_aux_loss=True"
    assert all(k in aux_logits for k in ("age", "sex", "race"))
    
    # Test HCN with aux_loss disabled (v1 behavior)
    print("  Testing HCN with aux_loss=False...")
    hcn_no_aux = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=False,
    )

    hcn_no_aux.train()
    ctx, mu, logsigma, aux_logits = hcn_no_aux(age, sex, race)

    assert ctx.shape == (batch_size, 1, 1024), f"Expected (8, 1, 1024), got {ctx.shape}"
    assert mu.shape == (batch_size, 256), f"Expected (8, 256), got {mu.shape}"
    assert logsigma.shape == (batch_size, 256), f"Expected (8, 256), got {logsigma.shape}"
    assert aux_logits is None, "aux_logits should be None when use_aux_loss=False"
    
    # Test HCN with FiLM output mode (V5)
    print("  Testing HCN with use_film_output=True (V5)...")
    hcn_film = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
        use_aux_loss=True,
        use_film_output=True,
    )

    hcn_film.train()
    ctx_film, mu_film, logsigma_film, aux_logits_film = hcn_film(age, sex, race)

    # In FiLM mode, ctx should be [B, d_node] instead of [B, 1, d_ctx]
    assert ctx_film.shape == (batch_size, 256), f"Expected (8, 256) in FiLM mode, got {ctx_film.shape}"
    assert mu_film.shape == (batch_size, 256), f"Expected (8, 256), got {mu_film.shape}"
    assert logsigma_film.shape == (batch_size, 256), f"Expected (8, 256), got {logsigma_film.shape}"
    assert aux_logits_film is not None, "aux_logits should not be None when use_aux_loss=True"
    print(f"✓ FiLM mode: ctx shape = {ctx_film.shape}")
    
    # Use hcn_with_aux for remaining tests
    hcn = hcn_with_aux

    print(f"✓ Forward pass: ctx shape = {ctx.shape}")

    # Test compositional loss
    comp_loss = hcn.compute_compositional_loss(age, sex, race)
    assert comp_loss.ndim == 0, "Compositional loss should be scalar"
    print(f"✓ Compositional loss: {comp_loss.item():.4f}")

    # Test uncertainty
    sigma = hcn.get_uncertainty(age, sex, race)
    assert sigma.shape == (batch_size,), f"Expected ({batch_size},), got {sigma.shape}"
    print(f"✓ Uncertainty: mean sigma = {sigma.mean().item():.4f}")

    # Test save/load
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    try:
        hcn_with_aux.save_pretrained(temp_dir)
        hcn_loaded = HierarchicalConditioner.from_pretrained(temp_dir)
        ctx_loaded, _, _, aux_logits_loaded = hcn_loaded(age, sex, race)
        assert aux_logits_loaded is not None, "Loaded model should have aux_logits when use_aux_loss=True"
        print(f"✓ Save/load successful")
    finally:
        shutil.rmtree(temp_dir)

    print(f"✓ All tests passed!")
    print(f"✓ Total parameters: {sum(p.numel() for p in hcn.parameters()):,}")


if __name__ == "__main__":
    test_hcn()
