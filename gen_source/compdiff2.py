"""
CompDiff-2: Typed Compositional Conditioner (staged design).

Design rationale (from the CompDiff-1 ablation evidence):
- CompDiff-1 routed age through the text prompt because age-in-HCN failed
  (age RMSE 15.3 vs no-info floor 17.4): a nominal embedding table + CE aux
  loss gives age no ordinal geometry, and a single joint token loses the
  cross-attention competition against 77 clinical tokens.
- CompDiff-2 fixes the type error instead of patching around it:
  every attribute enters the composer in its native type, and the composed
  signal reaches the UNet through channels matched to how each attribute
  expresses in the image.

Components (each independently switchable for the staged runs 2a-2e):
  * Typed encoders: sex/race = nn.Embedding (nominal);
    age = continuous years -> sinusoidal/Fourier features -> MLP (ordinal).
  * Composer: 'hierarchical' (CompDiff-1 pairwise-MLP topology, typed inputs),
    'transformer' (2-layer encoder over attribute tokens + [CLS]), or
    'flat' (parameter/depth-matched NON-factorized MLP control; review item 3).
  * Output: single fused token (T=1) or multi-token
    (t_age, t_sex, t_race, t_cls [, registers]) concatenated to text context.
  * Route B: [CLS] latent -> ZERO-INIT linear -> added to the UNet timestep
    embedding (global modulation channel; starts as exact identity).
  * Aux supervision ON THE OUTPUT TOKENS the UNet sees (CompDiff-1 V8 lesson):
    CE(sex), CE(race), REGRESSION(age, normalized years), CE(joint cell).
  * Per-attribute conditioning dropout to learned null embeddings
    (marginal training / partial conditioning / CFG-style guidance).

Stage map:
  2a: typed age in composer, hierarchical, single token, no Route B
  2b: + multi-token output with per-token aux heads
  2c: + Route B (zero-init timestep modulation)
  2d: + transformer composer
  2e: + per-attribute dropout (full design)

Interface contract (drop-in for the load_hcn slot):
  forward(sex_idx, race_idx, age_continuous=None, age_idx=None)
    -> (ctx [B,T,d_ctx], mu [B,d_node], logsigma [B,d_node],
        aux_logits dict|None, time_emb [B,d_time_emb]|None)
"""

import math
import json
import os
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """LayerNorm -> Linear -> SiLU -> Dropout -> Linear (same block as CompDiff-1)."""

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


def sinusoidal_age_features(age_years: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Sinusoidal features of age in years — the same encoding family the UNet
    uses for the diffusion timestep, giving age smooth ordinal geometry by
    construction (nearby ages -> nearby features).

    Args:
        age_years: [B] float tensor of ages in years
        dim: feature dimension (must be even)
    Returns:
        [B, dim] float tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=age_years.device)
        / half
    )
    args = age_years.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]


class CompDiff2Conditioner(nn.Module):
    """
    Typed compositional demographic conditioner (CompDiff-2).

    Args:
        num_sex, num_race: category counts for the nominal attributes
        num_age_bins: bin count kept ONLY for the joint-cell aux head and
            monitoring (age itself is continuous inside the conditioner)
        d_node: composer latent dimension
        d_ctx: UNet cross-attention dimension (1024 for SD 2.1)
        d_time_emb: UNet timestep-embedding dimension (1280 for SD 2.1)
        max_age: normalization constant for the age regression target
        age_freq_dim: sinusoidal feature dimension for the age encoder
        composer: 'hierarchical' | 'transformer'
        multi_token: single fused token (False) vs per-attribute tokens (True)
        route_b: emit a zero-init timestep-embedding modulation vector
        num_registers: extra unsupervised register tokens (multi_token only)
        attr_dropout_prob: per-sample per-attribute prob of replacing an
            attribute with its learned null embedding (training only)
        full_dropout_prob: per-sample prob of dropping ALL attributes at once
            (training only; CFG-style unconditional demographic branch)
        use_uncertainty: variational latent on the composed representation
        use_aux_loss: build aux heads on the output tokens
        aux_hidden_dim: hidden dim of aux heads
        dropout: dropout inside MLPs / transformer
        transformer_layers, transformer_heads: composer size ('transformer')
        flat_hidden: hidden width of the 'flat' composer blocks. 664 matches the
            hierarchical multi-token composer's parameter count within 0.1%
            (2,898,632 vs 2,896,640) at d_node=256 -- see composer_num_params().
    """

    def __init__(
        self,
        num_sex: int = 2,
        num_race: int = 4,
        num_age_bins: int = 5,
        d_node: int = 256,
        d_ctx: int = 1024,
        d_time_emb: int = 1280,
        max_age: float = 100.0,
        age_freq_dim: int = 128,
        composer: str = "hierarchical",
        multi_token: bool = False,
        route_b: bool = False,
        num_registers: int = 0,
        attr_dropout_prob: float = 0.0,
        full_dropout_prob: float = 0.0,
        use_uncertainty: bool = True,
        use_aux_loss: bool = True,
        aux_hidden_dim: int = 512,
        dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        flat_hidden: int = 664,
    ):
        super().__init__()
        assert composer in ("hierarchical", "transformer", "flat"), f"Unknown composer: {composer}"
        assert age_freq_dim % 2 == 0, "age_freq_dim must be even"

        self.config = {
            "num_sex": num_sex,
            "num_race": num_race,
            "num_age_bins": num_age_bins,
            "d_node": d_node,
            "d_ctx": d_ctx,
            "d_time_emb": d_time_emb,
            "max_age": max_age,
            "age_freq_dim": age_freq_dim,
            "composer": composer,
            "multi_token": multi_token,
            "route_b": route_b,
            "num_registers": num_registers,
            "attr_dropout_prob": attr_dropout_prob,
            "full_dropout_prob": full_dropout_prob,
            "use_uncertainty": use_uncertainty,
            "use_aux_loss": use_aux_loss,
            "aux_hidden_dim": aux_hidden_dim,
            "dropout": dropout,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "flat_hidden": flat_hidden,
        }

        self.num_sex = num_sex
        self.num_race = num_race
        self.num_age_bins = num_age_bins
        self.d_node = d_node
        self.d_ctx = d_ctx
        self.d_time_emb = d_time_emb
        self.max_age = float(max_age)
        self.age_freq_dim = age_freq_dim
        self.composer_type = composer
        self.multi_token = multi_token
        self.route_b = route_b
        self.num_registers = num_registers if multi_token else 0
        self.attr_dropout_prob = attr_dropout_prob
        self.full_dropout_prob = full_dropout_prob
        self.use_uncertainty = use_uncertainty
        self.use_aux_loss = use_aux_loss
        # Age is always inside the composer for CompDiff-2 (that is the point);
        # kept as an attribute for pipeline code that introspects it.
        self.encode_age = True

        # === Typed attribute encoders ===
        self.emb_sex = nn.Embedding(num_sex, d_node)
        self.emb_race = nn.Embedding(num_race, d_node)
        self.age_encoder = nn.Sequential(
            nn.Linear(age_freq_dim, d_node),
            nn.SiLU(),
            nn.Linear(d_node, d_node),
        )

        # Learned null embeddings ("attribute unspecified") for dropout and
        # partial conditioning at inference.
        self.null_age = nn.Parameter(torch.zeros(d_node))
        self.null_sex = nn.Parameter(torch.zeros(d_node))
        self.null_race = nn.Parameter(torch.zeros(d_node))

        # === Composer ===
        if composer == "hierarchical":
            # CompDiff-1 topology with typed inputs (stage 2a-2c)
            self.compose_age_sex = MLP(2 * d_node, 2 * d_node, d_node, dropout)
            self.compose_age_race = MLP(2 * d_node, 2 * d_node, d_node, dropout)
            self.compose_sex_race = MLP(2 * d_node, 2 * d_node, d_node, dropout)
            self.compose_all = MLP(3 * d_node, 2 * d_node, d_node, dropout)
            if multi_token:
                # Contextualize each attribute against the composed child so
                # attribute tokens are "attribute-in-context" representations.
                self.ctx_age = MLP(2 * d_node, 2 * d_node, d_node, dropout)
                self.ctx_sex = MLP(2 * d_node, 2 * d_node, d_node, dropout)
                self.ctx_race = MLP(2 * d_node, 2 * d_node, d_node, dropout)
        elif composer == "flat":
            # Parameter/depth-matched NON-compositional control (review item 3).
            # Same three-stage MLP pipeline as 'hierarchical' (pair-level ->
            # compose_all -> per-attribute contextualization), same block type
            # (LN -> Linear -> SiLU -> Dropout -> Linear), same depth (6 linear
            # layers to the attribute tokens, 4 to h_demo), but every stage is a
            # single MLP over the FULL concatenation: no pairwise factorization,
            # no per-attribute routing. Widths are tuned (flat_hidden) so the
            # composer parameter count matches 'hierarchical' within ~0.1%.
            #   stage 1: [e_age, e_sex, e_race] (3d) -> H -> 3d      (~ 3 pair MLPs)
            #   stage 2: 3d -> H -> d  = h_demo                      (~ compose_all)
            #   stage 3: [e_age, e_sex, e_race, h_demo] (4d) -> H -> 3d,
            #            split into (c_age, c_sex, c_race)           (~ ctx_age/sex/race)
            H = int(flat_hidden)
            self.flat_stage1 = MLP(3 * d_node, H, 3 * d_node, dropout)
            self.flat_stage2 = MLP(3 * d_node, H, d_node, dropout)
            if multi_token:
                self.flat_stage3 = MLP(4 * d_node, H, 3 * d_node, dropout)
        else:
            # Transformer composer (stage 2d): [t_age, t_sex, t_race, CLS, regs]
            self.cls_token = nn.Parameter(torch.zeros(d_node))
            num_slots = 4 + self.num_registers
            self.type_emb = nn.Parameter(torch.zeros(num_slots, d_node))
            if self.num_registers > 0:
                self.register_tokens = nn.Parameter(torch.zeros(self.num_registers, d_node))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_node,
                nhead=transformer_heads,
                dim_feedforward=2 * d_node,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.composer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # === Variational latent on the composed representation ===
        if use_uncertainty:
            self.mu_head = nn.Linear(d_node, d_node)
            self.logsigma_head = nn.Linear(d_node, d_node)

        # === Projections to cross-attention space ===
        def make_proj():
            return nn.Sequential(nn.LayerNorm(d_node), nn.Linear(d_node, d_ctx))

        self.proj_cls = make_proj()
        if multi_token:
            self.proj_age = make_proj()
            self.proj_sex = make_proj()
            self.proj_race = make_proj()
            if self.num_registers > 0:
                self.proj_reg = make_proj()

        # === Route B: zero-init projection into the timestep embedding ===
        if route_b:
            self.time_proj = nn.Linear(d_node, d_time_emb)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

        # === Aux heads ON OUTPUT TOKENS (post-projection, d_ctx) ===
        if use_aux_loss:
            def make_head(d_out):
                return nn.Sequential(
                    nn.LayerNorm(d_ctx),
                    nn.Linear(d_ctx, aux_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(aux_hidden_dim, d_out),
                )

            self.sex_classifier = make_head(num_sex)
            self.race_classifier = make_head(num_race)
            self.age_regressor = make_head(1)
            self.joint_classifier = make_head(num_age_bins * num_sex * num_race)

        self._init_weights()

    # ------------------------------------------------------------------
    @property
    def num_output_tokens(self) -> int:
        if not self.multi_token:
            return 1
        return 4 + self.num_registers  # age, sex, race, cls (+ registers)

    def composer_num_params(self) -> int:
        """Parameter count of the COMPOSER ONLY (everything between the typed
        attribute embeddings and the variational/projection heads). Used to
        parameter-match the 'flat' control against 'hierarchical'."""
        if self.composer_type == "hierarchical":
            mods = [self.compose_age_sex, self.compose_age_race, self.compose_sex_race, self.compose_all]
            if self.multi_token:
                mods += [self.ctx_age, self.ctx_sex, self.ctx_race]
        elif self.composer_type == "flat":
            mods = [self.flat_stage1, self.flat_stage2]
            if self.multi_token:
                mods.append(self.flat_stage3)
        else:
            mods = [self.composer]
            extra = self.cls_token.numel() + self.type_emb.numel()
            if self.num_registers > 0:
                extra += self.register_tokens.numel()
            return sum(p.numel() for m in mods for p in m.parameters()) + extra
        return sum(p.numel() for m in mods for p in m.parameters())

    def composer_depth(self) -> int:
        """Number of nn.Linear layers on the longest input->output path of the composer."""
        if self.composer_type == "hierarchical":
            return 6 if self.multi_token else 4
        if self.composer_type == "flat":
            return 6 if self.multi_token else 4
        return 4 * self.config["transformer_layers"]  # per layer: attn in_proj, out_proj, FFN x2

    def _init_weights(self):
        for emb in (self.emb_sex, self.emb_race):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for p in (self.null_age, self.null_sex, self.null_race):
            nn.init.normal_(p, mean=0.0, std=0.02)
        if self.composer_type == "transformer":
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            nn.init.normal_(self.type_emb, mean=0.0, std=0.02)
            if self.num_registers > 0:
                nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        if self.use_uncertainty:
            nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(self.logsigma_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.logsigma_head.bias, -1.0)

    # ------------------------------------------------------------------
    def _encode_attributes(
        self,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
        age_continuous: Optional[torch.Tensor],
        apply_dropout: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Typed grandparent embeddings, with optional null-dropout."""
        e_sex = self.emb_sex(sex_idx)
        e_race = self.emb_race(race_idx)
        B = e_sex.shape[0]

        if age_continuous is not None:
            feats = sinusoidal_age_features(age_continuous, self.age_freq_dim)
            e_age = self.age_encoder(feats.to(dtype=e_sex.dtype))
        else:
            # Partial conditioning: age unspecified
            e_age = self.null_age.unsqueeze(0).expand(B, -1).to(dtype=e_sex.dtype)

        if apply_dropout and self.training and (self.attr_dropout_prob > 0 or self.full_dropout_prob > 0):
            device = e_sex.device
            full = torch.rand(B, device=device) < self.full_dropout_prob
            for name, null in (("age", self.null_age), ("sex", self.null_sex), ("race", self.null_race)):
                drop = (torch.rand(B, device=device) < self.attr_dropout_prob) | full
                mask = drop.unsqueeze(-1).to(dtype=e_sex.dtype)
                null_row = null.unsqueeze(0).to(dtype=e_sex.dtype)
                if name == "age":
                    e_age = e_age * (1 - mask) + null_row * mask
                elif name == "sex":
                    e_sex = e_sex * (1 - mask) + null_row * mask
                else:
                    e_race = e_race * (1 - mask) + null_row * mask

        return e_age, e_sex, e_race

    @torch.no_grad()
    def forward_unconditional(self, batch_size: int, device=None, dtype=None):
        """Tokens (+ Route B vector) for the model's TRAINED 'demographics
        unspecified' state: all three attributes at their learned null
        embeddings.

        Only meaningful for models trained with attribute dropout (stage 2e);
        for the others the null embeddings never received gradient and this is
        not a trained state. Deterministic (z = mu, no sampling), so it is safe
        as the unconditional branch of classifier-free guidance.

        Add-only: does not touch forward() semantics.
        """
        device = device or self.null_sex.device
        dtype = dtype or self.null_sex.dtype
        exp = lambda p: p.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        e_age, e_sex, e_race = exp(self.null_age), exp(self.null_sex), exp(self.null_race)

        h_demo, attr_ctx = self._compose(e_age, e_sex, e_race)
        z = self.mu_head(h_demo) if self.use_uncertainty else h_demo

        t_cls = self.proj_cls(z)
        if self.multi_token:
            tokens = [
                self.proj_age(attr_ctx["age"]),
                self.proj_sex(attr_ctx["sex"]),
                self.proj_race(attr_ctx["race"]),
                t_cls,
            ]
            if self.num_registers > 0:
                if self.composer_type == "transformer":
                    regs = attr_ctx["registers"]
                else:
                    regs = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                tokens.extend(self.proj_reg(regs[:, r]) for r in range(self.num_registers))
            ctx = torch.stack(tokens, dim=1)
        else:
            ctx = t_cls.unsqueeze(1)

        time_emb = self.time_proj(z) if self.route_b else None
        return ctx, time_emb

    def _compose(
        self,
        e_age: torch.Tensor,
        e_sex: torch.Tensor,
        e_race: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Run the composer.

        Returns:
            h_demo: [B, d_node] composed representation
            attr_ctx: dict of contextualized per-attribute states [B, d_node]
                      (None when multi_token=False)
        """
        if self.composer_type == "hierarchical":
            h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], dim=-1))
            h_age_race = self.compose_age_race(torch.cat([e_age, e_race], dim=-1))
            h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], dim=-1))
            h_demo = self.compose_all(torch.cat([h_age_sex, h_age_race, h_sex_race], dim=-1))
            attr_ctx = None
            if self.multi_token:
                attr_ctx = {
                    "age": self.ctx_age(torch.cat([e_age, h_demo], dim=-1)),
                    "sex": self.ctx_sex(torch.cat([e_sex, h_demo], dim=-1)),
                    "race": self.ctx_race(torch.cat([e_race, h_demo], dim=-1)),
                }
            return h_demo, attr_ctx
        elif self.composer_type == "flat":
            x = torch.cat([e_age, e_sex, e_race], dim=-1)
            h1 = self.flat_stage1(x)
            h_demo = self.flat_stage2(h1)
            attr_ctx = None
            if self.multi_token:
                c = self.flat_stage3(torch.cat([x, h_demo], dim=-1))
                c_age, c_sex, c_race = torch.split(c, self.d_node, dim=-1)
                attr_ctx = {"age": c_age, "sex": c_sex, "race": c_race}
            return h_demo, attr_ctx
        else:
            B = e_age.shape[0]
            seq = [e_age, e_sex, e_race, self.cls_token.unsqueeze(0).expand(B, -1)]
            if self.num_registers > 0:
                for r in range(self.num_registers):
                    seq.append(self.register_tokens[r].unsqueeze(0).expand(B, -1))
            x = torch.stack(seq, dim=1)  # [B, S, d_node]
            x = x + self.type_emb.unsqueeze(0)
            out = self.composer(x)
            h_demo = out[:, 3]  # CLS position
            attr_ctx = None
            if self.multi_token:
                attr_ctx = {"age": out[:, 0], "sex": out[:, 1], "race": out[:, 2]}
                if self.num_registers > 0:
                    attr_ctx["registers"] = out[:, 4:]
            return h_demo, attr_ctx

    # ------------------------------------------------------------------
    def forward(
        self,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
        age_continuous: Optional[torch.Tensor] = None,
        age_idx: Optional[torch.Tensor] = None,  # accepted for interface compat; unused
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        e_age, e_sex, e_race = self._encode_attributes(
            sex_idx, race_idx, age_continuous, apply_dropout=True
        )
        h_demo, attr_ctx = self._compose(e_age, e_sex, e_race)

        # DDP: the learned null embeddings are only consumed on dropout /
        # partial-conditioning batches; tie them into the graph with a
        # zero-scaled anchor so every parameter produces a gradient on every
        # step (otherwise DDP's reducer errors with "parameters that were not
        # used in producing loss" — observed as SLURM job 122136, indices 0-2).
        h_demo = h_demo + 0.0 * (self.null_age + self.null_sex + self.null_race).sum()

        # Variational latent on the composed (CLS) representation only
        if self.use_uncertainty:
            mu = self.mu_head(h_demo)
            logsigma = torch.clamp(self.logsigma_head(h_demo), min=-5.0, max=1.0)
            if self.training:
                z = mu + torch.exp(logsigma) * torch.randn_like(mu)
            else:
                z = mu
        else:
            mu = h_demo
            logsigma = torch.zeros_like(h_demo)
            z = h_demo

        # Output tokens for cross-attention (Route A)
        t_cls = self.proj_cls(z)  # [B, d_ctx]
        if self.multi_token:
            tokens = [
                self.proj_age(attr_ctx["age"]),
                self.proj_sex(attr_ctx["sex"]),
                self.proj_race(attr_ctx["race"]),
                t_cls,
            ]
            if self.num_registers > 0:
                if self.composer_type == "transformer":
                    regs = attr_ctx["registers"]  # [B, R, d_node]
                else:
                    B = t_cls.shape[0]
                    regs = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
                tokens.extend([self.proj_reg(regs[:, r]) for r in range(self.num_registers)])
            ctx = torch.stack(tokens, dim=1)  # [B, T, d_ctx]
        else:
            ctx = t_cls.unsqueeze(1)  # [B, 1, d_ctx]

        # Route B: global modulation vector for the timestep embedding
        time_emb = self.time_proj(z) if self.route_b else None

        # Aux logits from the tokens the UNet actually sees
        aux_logits = None
        if self.use_aux_loss:
            if self.multi_token:
                tok_age, tok_sex, tok_race = ctx[:, 0], ctx[:, 1], ctx[:, 2]
                tok_joint = ctx[:, 3]
            else:
                tok_age = tok_sex = tok_race = tok_joint = ctx[:, 0]
            aux_logits = {
                "sex": self.sex_classifier(tok_sex),
                "race": self.race_classifier(tok_race),
                "age_pred": self.age_regressor(tok_age).squeeze(-1),  # normalized age
                "joint": self.joint_classifier(tok_joint),
            }

        return ctx, mu, logsigma, aux_logits, time_emb

    # ------------------------------------------------------------------
    def compute_compositional_loss(
        self,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
        age_continuous: Optional[torch.Tensor] = None,
        age_idx: Optional[torch.Tensor] = None,  # interface compat; unused
    ) -> torch.Tensor:
        """Soft additive anchor: cos(h_demo, e_age + e_sex + e_race)."""
        e_age, e_sex, e_race = self._encode_attributes(
            sex_idx, race_idx, age_continuous, apply_dropout=False
        )
        h_demo, _ = self._compose(e_age, e_sex, e_race)
        h_additive = e_age + e_sex + e_race
        cos_sim = F.cosine_similarity(h_demo, h_additive, dim=-1)
        return (1 - cos_sim).mean()

    def get_uncertainty(
        self,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
        age_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, logsigma, _, _ = self.forward(sex_idx, race_idx, age_continuous=age_continuous)
        return torch.exp(logsigma).mean(dim=-1)

    # ------------------------------------------------------------------
    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        print(f"CompDiff-2 conditioner saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = "cpu"):
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"CompDiff-2 conditioner loaded from {save_dir}")
        return model


# ======================================================================
# Loss
# ======================================================================

def compute_aux_loss_cd2(
    aux_logits: Dict[str, torch.Tensor],
    sex_idx: torch.Tensor,
    race_idx: torch.Tensor,
    age_continuous: torch.Tensor,
    age_idx: Optional[torch.Tensor] = None,
    max_age: float = 100.0,
    sex_weight: float = 1.0,
    race_weight: float = 1.0,
    age_weight: float = 1.0,
    joint_weight: float = 0.5,
    age_loss_scale: float = 10.0,
    num_sex: int = 2,
    num_race: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Aux loss for CompDiff-2 output tokens.

    - sex / race: cross-entropy (nominal)
    - age: SmoothL1 on normalized age (ordinal/continuous), scaled by
      age_loss_scale so its magnitude is commensurate with the CE terms
    - joint: cross-entropy over the binned age x sex x race cell (interaction
      supervision on the CLS token); skipped if age_idx is None

    Returns:
        total_loss: weighted average of the individual losses
        metrics: individual losses + accuracies + age MAE in years
    """
    losses, weights = [], []
    metrics: Dict[str, float] = {}

    sex_ce = F.cross_entropy(aux_logits["sex"], sex_idx)
    race_ce = F.cross_entropy(aux_logits["race"], race_idx)
    losses += [sex_ce, race_ce]
    weights += [sex_weight, race_weight]

    age_target = (age_continuous.float() / max_age).clamp(0.0, 1.0)
    age_loss = F.smooth_l1_loss(aux_logits["age_pred"].float(), age_target) * age_loss_scale
    losses.append(age_loss)
    weights.append(age_weight)

    with torch.no_grad():
        metrics["aux_acc_sex"] = (aux_logits["sex"].argmax(-1) == sex_idx).float().mean().item()
        metrics["aux_acc_race"] = (aux_logits["race"].argmax(-1) == race_idx).float().mean().item()
        metrics["aux_age_mae_years"] = (
            (aux_logits["age_pred"].float() - age_target).abs().mean().item() * max_age
        )
    metrics["aux_loss_sex"] = sex_ce.item()
    metrics["aux_loss_race"] = race_ce.item()
    metrics["aux_loss_age"] = age_loss.item()

    if age_idx is not None and "joint" in aux_logits:
        joint_target = age_idx * (num_sex * num_race) + sex_idx * num_race + race_idx
        joint_ce = F.cross_entropy(aux_logits["joint"], joint_target)
        losses.append(joint_ce)
        weights.append(joint_weight)
        with torch.no_grad():
            metrics["aux_acc_joint"] = (
                (aux_logits["joint"].argmax(-1) == joint_target).float().mean().item()
            )
        metrics["aux_loss_joint"] = joint_ce.item()

    total = sum(w * l for w, l in zip(weights, losses)) / sum(weights)
    return total, metrics


# ======================================================================
# Config-driven builder (mirrors load_hcn_v8 in hcn_v7.py)
# ======================================================================

def load_compdiff2(args, logger):
    """Build a CompDiff2Conditioner from the training config namespace."""
    use_aux_loss = getattr(args, "hcn_aux_weight", 0.0) > 0.0

    logger.info("=" * 60)
    logger.info("Initializing CompDiff-2 Typed Compositional Conditioner")
    logger.info("=" * 60)

    model = CompDiff2Conditioner(
        num_sex=getattr(args, "hcn_num_sex", 2),
        num_race=getattr(args, "hcn_num_race", 4),
        num_age_bins=getattr(args, "hcn_num_age_bins", 5),
        d_node=getattr(args, "hcn_d_node", 256),
        d_ctx=getattr(args, "hcn_d_ctx", 1024),
        d_time_emb=getattr(args, "hcn_d_time_emb", 1280),
        max_age=getattr(args, "max_age", 100),
        age_freq_dim=getattr(args, "cd2_age_freq_dim", 128),
        composer=getattr(args, "cd2_composer", "hierarchical"),
        multi_token=getattr(args, "cd2_multi_token", False),
        route_b=getattr(args, "cd2_route_b", False),
        num_registers=getattr(args, "cd2_num_registers", 0),
        attr_dropout_prob=getattr(args, "cd2_attr_dropout_prob", 0.0),
        full_dropout_prob=getattr(args, "cd2_full_dropout_prob", 0.0),
        use_uncertainty=getattr(args, "hcn_use_uncertainty", True),
        use_aux_loss=use_aux_loss,
        aux_hidden_dim=getattr(args, "hcn_aux_hidden_dim", 512),
        dropout=getattr(args, "hcn_dropout", 0.1),
        transformer_layers=getattr(args, "cd2_transformer_layers", 2),
        transformer_heads=getattr(args, "cd2_transformer_heads", 4),
        flat_hidden=getattr(args, "cd2_flat_hidden", 664),
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Composer: {model.composer_type} "
                f"(composer-only params: {model.composer_num_params():,}, depth: {model.composer_depth()} linears)")
    logger.info(f"  Output tokens: {model.num_output_tokens} (multi_token={model.multi_token})")
    logger.info(f"  Route B (timestep modulation): {model.route_b}")
    logger.info(f"  Registers: {model.num_registers}")
    logger.info(f"  Attr dropout: {model.attr_dropout_prob} | full dropout: {model.full_dropout_prob}")
    logger.info(f"  Uncertainty: {model.use_uncertainty} | aux heads: {model.use_aux_loss}")
    logger.info("=" * 60)
    return model


# ======================================================================
# Self-tests (CPU, no data needed):  python compdiff2.py
# ======================================================================

def _stage_kwargs(stage: str) -> dict:
    common = dict(num_sex=2, num_race=4, num_age_bins=5, d_node=256, d_ctx=1024)
    stages = {
        "2a": dict(composer="hierarchical", multi_token=False, route_b=False),
        "2b": dict(composer="hierarchical", multi_token=True, route_b=False),
        "2c": dict(composer="hierarchical", multi_token=True, route_b=True),
        "2d": dict(composer="transformer", multi_token=True, route_b=True),
        "2e": dict(composer="transformer", multi_token=True, route_b=True,
                   attr_dropout_prob=0.1, full_dropout_prob=0.1),
        # Review item 3: parameter/depth-matched flat control against 2b
        "2b_flat": dict(composer="flat", multi_token=True, route_b=False),
    }
    return {**common, **stages[stage]}


def test_compdiff2():
    torch.manual_seed(0)
    B = 8
    sex = torch.randint(0, 2, (B,))
    race = torch.randint(0, 4, (B,))
    age_years = torch.tensor([23.0, 35.0, 47.0, 55.0, 63.0, 71.0, 84.0, 91.0])
    age_bin = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])

    for stage in ("2a", "2b", "2c", "2d", "2e", "2b_flat"):
        print(f"\n=== Stage {stage} ===")
        model = CompDiff2Conditioner(**_stage_kwargs(stage))
        model.train()
        ctx, mu, logsigma, aux, time_emb = model(sex, race, age_continuous=age_years)

        T = model.num_output_tokens
        assert ctx.shape == (B, T, 1024), f"{stage}: ctx {ctx.shape}, expected (B,{T},1024)"
        assert mu.shape == (B, 256) and logsigma.shape == (B, 256)
        assert set(aux.keys()) == {"sex", "race", "age_pred", "joint"}
        assert aux["age_pred"].shape == (B,)
        assert aux["joint"].shape == (B, 5 * 2 * 4)
        if model.route_b:
            assert time_emb is not None and time_emb.shape == (B, 1280)
            # zero-init: exact identity at start
            assert time_emb.abs().max().item() == 0.0, f"{stage}: Route B must start at zero"
        else:
            assert time_emb is None
        print(f"  forward OK: T={T}, route_b={model.route_b}")

        # Training-like loss + gradient flow. DDP requires EVERY parameter to
        # produce a gradient on every step (grad may be zero, but must exist) —
        # this check reproduces the reducer condition that killed job 122136.
        model.zero_grad()
        ctx, mu, logsigma, aux, time_emb = model(sex, race, age_continuous=age_years)
        aux_loss, metrics = compute_aux_loss_cd2(
            aux, sex, race, age_continuous=age_years, age_idx=age_bin
        )
        kl = -0.5 * torch.sum(1 + 2 * logsigma - mu ** 2 - torch.exp(2 * logsigma), -1).mean()
        loss = aux_loss + 0.005 * kl + 1e-3 * ctx.sum()  # ctx term: diffusion-loss proxy
        if time_emb is not None:
            loss = loss + 1e-3 * time_emb.sum()  # Route B participates via the UNet
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"{stage}: params without grad (DDP would crash): {missing}"
        assert model.proj_cls[1].weight.grad.abs().sum() > 0, f"{stage}: no grad into proj_cls"
        comp = model.compute_compositional_loss(sex, race, age_continuous=age_years)
        assert comp.ndim == 0
        print(f"  training-like loss OK ({loss.item():.4f}), all {sum(1 for _ in model.parameters())} "
              f"params received grads, comp loss OK ({comp.item():.4f})")

        # Eval determinism
        model.eval()
        with torch.no_grad():
            c1, *_ = model(sex, race, age_continuous=age_years)
            c2, *_ = model(sex, race, age_continuous=age_years)
        assert torch.allclose(c1, c2), f"{stage}: eval forward not deterministic"

        # Partial conditioning: age unspecified must work
        with torch.no_grad():
            c3, *_ = model(sex, race, age_continuous=None)
        assert c3.shape == (B, T, 1024)
        print("  eval determinism + partial conditioning OK")

        # Ordinal geometry: age-token trajectory must be smooth in age
        with torch.no_grad():
            ages = torch.tensor([50.0, 51.0, 90.0])
            feats = model.age_encoder(sinusoidal_age_features(ages, model.age_freq_dim))
            d_near = (feats[0] - feats[1]).norm().item()
            d_far = (feats[0] - feats[2]).norm().item()
        assert d_near < d_far, f"{stage}: age encoder not ordinal (d(50,51)={d_near:.3f} vs d(50,90)={d_far:.3f})"
        print(f"  ordinal geometry OK: d(50,51)={d_near:.3f} < d(50,90)={d_far:.3f}")

        # Save/load roundtrip
        import tempfile, shutil
        tmp = tempfile.mkdtemp()
        try:
            model.save_pretrained(tmp)
            loaded = CompDiff2Conditioner.from_pretrained(tmp)
            with torch.no_grad():
                c_orig, *_ = model(sex, race, age_continuous=age_years)
                c_load, *_ = loaded(sex, race, age_continuous=age_years)
            assert torch.allclose(c_orig, c_load, atol=1e-6), f"{stage}: save/load mismatch"
        finally:
            shutil.rmtree(tmp)
        print("  save/load OK")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,} (composer-only: {model.composer_num_params():,}, "
              f"depth {model.composer_depth()} linears)")

    # Parameter matching of the flat control against 2b (review item 3)
    m_h = CompDiff2Conditioner(**_stage_kwargs("2b"))
    m_f = CompDiff2Conditioner(**_stage_kwargs("2b_flat"))
    ph, pf = m_h.composer_num_params(), m_f.composer_num_params()
    th, tf = sum(p.numel() for p in m_h.parameters()), sum(p.numel() for p in m_f.parameters())
    rel = abs(pf - ph) / ph
    print(f"\nComposer params  hierarchical(2b)={ph:,}  flat={pf:,}  rel diff={rel*100:.3f}%")
    print(f"Total params     hierarchical(2b)={th:,}  flat={tf:,}  rel diff={abs(tf-th)/th*100:.3f}%")
    assert rel < 0.02, f"flat control not parameter-matched (rel diff {rel:.4f} >= 2%)"
    assert m_h.composer_depth() == m_f.composer_depth(), "flat control depth mismatch"
    # Everything outside the composer must be identical
    outside_h = {n: p.shape for n, p in m_h.named_parameters() if not n.startswith(("compose_", "ctx_"))}
    outside_f = {n: p.shape for n, p in m_f.named_parameters() if not n.startswith("flat_")}
    assert outside_h == outside_f, "flat control differs from 2b outside the composer"
    print("Flat control: parameter-matched (<2%), depth-matched, identical outside the composer OK")

    # Dropout statistics sanity (stage 2e)
    torch.manual_seed(1)
    model = CompDiff2Conditioner(**_stage_kwargs("2e"))
    model.train()
    big_sex = torch.zeros(2000, dtype=torch.long)
    big_race = torch.zeros(2000, dtype=torch.long)
    big_age = torch.full((2000,), 60.0)
    e_age, e_sex, e_race = model._encode_attributes(big_sex, big_race, big_age, apply_dropout=True)
    frac_null_sex = (e_sex - model.null_sex.unsqueeze(0)).norm(dim=-1).lt(1e-6).float().mean().item()
    expected = 0.1 + 0.1 - 0.1 * 0.1  # attr OR full
    assert abs(frac_null_sex - expected) < 0.05, f"dropout rate {frac_null_sex:.3f} != ~{expected:.3f}"
    print(f"\nDropout statistics OK: null-sex fraction {frac_null_sex:.3f} (expected ~{expected:.3f})")

    print("\n" + "=" * 60)
    print("All CompDiff-2 self-tests passed (stages 2a-2e + 2b_flat).")
    print("=" * 60)


if __name__ == "__main__":
    test_compdiff2()
