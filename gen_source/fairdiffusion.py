import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist

try:
    from botorch.acquisition import UpperConfidenceBound  # type: ignore
    from botorch.fit import fit_gpytorch_mll  # type: ignore
    from botorch.models import SingleTaskGP  # type: ignore
    from botorch.optim import optimize_acqf  # type: ignore
    from gpytorch.mlls import ExactMarginalLogLikelihood  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "FairDiffusion integration requires `botorch` and `gpytorch`. "
        "Please install them (e.g. `pip install botorch gpytorch`)."
    ) from exc


@dataclass
class FairnessResult:
    """Container for per-step FairDiffusion statistics."""

    instance_weights: torch.Tensor
    logs: Dict[str, float]


class FairDiffusionController:
    """
    Implements the FairDiffusion adaptive re-weighting procedure described in
    FairDiffusion for Stable Diffusion fine-tuning.
    """

    def __init__(self, args, accelerator, logger):
        self.args = args
        self.accelerator = accelerator
        self.logger = logger
        self.device = accelerator.device

        self.attribute_fields = (
            list(args.fairdiffusion_attribute_fields)
            if args.fairdiffusion_attribute_fields
            else ["race_idx", "sex_idx", "age_idx"]
        )
        self.attribute_cardinalities = self._infer_attribute_cardinalities(args)
        self.total_groups = sum(self.attribute_cardinalities)

        self.input_perturbation = float(args.fairdiffusion_input_perturbation)
        self.time_window = max(1, int(args.fairdiffusion_time_window))
        self.exploitation_rate = float(args.fairdiffusion_exploitation_rate)
        self.sigma_bounds = (
            float(args.fairdiffusion_sigma_min),
            float(args.fairdiffusion_sigma_max),
        )
        self.min_instance_weight = float(args.fairdiffusion_min_instance_weight)
        self.bo_beta = float(args.fairdiffusion_ucb_beta)

        self.sigma = torch.full(
            (self.total_groups,),
            float(args.fairdiffusion_sigma_init),
            device=self.device,
            dtype=torch.float32,
        )

        self.history_steps: List[int] = []
        self.history_sigmas: List[torch.Tensor] = []
        self.history_acq_values: List[float] = []
        self.history_loss_gaps: List[float] = []

        self._records: List[Dict[str, float]] = []
        self.history_path = Path(args.output_dir) / "fairdiffusion_history.json"

        if accelerator.is_main_process:
            logger.info(
                "✓ FairDiffusion controller initialized "
                f"(attributes={self.attribute_fields}, total_groups={self.total_groups})"
            )

    def _infer_attribute_cardinalities(self, args) -> List[int]:
        default_mapping = {
            "age_idx": getattr(args, "hcn_num_age_bins", 1),
            "sex_idx": getattr(args, "hcn_num_sex", 2),
            "race_idx": getattr(args, "hcn_num_race", 4),
        }
        cardinalities = []
        for field in self.attribute_fields:
            if args.fairdiffusion_attribute_cardinalities and field in args.fairdiffusion_attribute_cardinalities:
                cardinalities.append(int(args.fairdiffusion_attribute_cardinalities[field]))
            else:
                cardinalities.append(int(default_mapping.get(field, 1)))
        return cardinalities

    def sample_instance_weights(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Sample per-instance perturbations conditioned on demographic attributes.
        """
        attributes = attributes.to(self.device).long()
        weights = torch.ones(attributes.size(0), device=self.device, dtype=torch.float32)
        offset = 0
        for idx, size in enumerate(self.attribute_cardinalities):
            selected = attributes[:, idx].clamp(min=0, max=size - 1)
            sigma_slice = self.sigma[offset : offset + size]
            perturbation = sigma_slice[selected] * torch.randn_like(weights)
            weights = weights + perturbation
            offset += size

        weights = torch.clamp(weights, min=self.min_instance_weight)
        return weights

    def apply(
        self,
        per_sample_loss: torch.Tensor,
        attributes: torch.Tensor,
        step: int,
    ) -> FairnessResult:
        """
        Apply FairDiffusion weighting to the provided per-sample loss tensor.
        """
        instance_weights = self.sample_instance_weights(attributes)
        logs: Dict[str, float] = {
            "fair_sigma_mean": self.sigma.mean().item(),
            "fair_sigma_std": self.sigma.std().item(),
        }

        if self.accelerator.is_main_process:
            gap = self._compute_loss_gap(per_sample_loss.detach(), attributes.detach())
            logs["fair_loss_gap"] = gap
            self.history_steps.append(step)
            self.history_sigmas.append(self.sigma.detach().cpu().clone())
            self.history_loss_gaps.append(gap)

            if step > 0 and (step % self.time_window) == 0:
                logs["fair_sigma_update"] = float(self._optimize_sigma())
            else:
                logs["fair_sigma_update"] = 0.0
        else:
            logs["fair_loss_gap"] = 0.0
            logs["fair_sigma_update"] = 0.0

        self._apply_exploration_noise()
        self._sync_sigma()

        logs["fair_sigma_mean"] = self.sigma.mean().item()
        logs["fair_sigma_std"] = self.sigma.std().item()

        if self.accelerator.is_main_process:
            self._records.append(
                {
                    "step": step,
                    "sigma_mean": logs["fair_sigma_mean"],
                    "sigma_std": logs["fair_sigma_std"],
                    "loss_gap": logs["fair_loss_gap"],
                }
            )

        return FairnessResult(instance_weights=instance_weights, logs=logs)

    def _compute_loss_gap(
        self,
        per_sample_loss: torch.Tensor,
        attributes: torch.Tensor,
    ) -> float:
        per_sample_loss = per_sample_loss.to(torch.float32)
        attributes = attributes.to(torch.long)
        gap_total = 0.0
        for idx, size in enumerate(self.attribute_cardinalities):
            attr_values = attributes[:, idx]
            group_means: List[torch.Tensor] = []
            for group in range(size):
                mask = attr_values == group
                if mask.any():
                    group_means.append(per_sample_loss[mask].mean())
            if len(group_means) >= 2:
                stacked = torch.stack(group_means)
                gap_total += (stacked.max() - stacked.min()).item()
        return float(gap_total)

    def _optimize_sigma(self) -> bool:
        if not self.history_sigmas:
            return False

        x = torch.stack([sigma.to(torch.float64) for sigma in self.history_sigmas])
        y = -torch.tensor(self.history_loss_gaps, dtype=torch.float64)[:, None]

        try:
            gp = SingleTaskGP(x, y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            bounds = torch.stack(
                [
                    torch.full_like(self.sigma, self.sigma_bounds[0], dtype=torch.float64),
                    torch.full_like(self.sigma, self.sigma_bounds[1], dtype=torch.float64),
                ]
            )
            acquisition = UpperConfidenceBound(gp, beta=self.bo_beta)
            candidate, acq_value = optimize_acqf(
                acquisition,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=32,
            )
            self.sigma = candidate.squeeze(0).to(device=self.device, dtype=torch.float32)
            self.history_acq_values.append(acq_value.item())
            if self.logger is not None:
                self.logger.info(
                    f"FairDiffusion BO update: sigma_mean={self.sigma.mean().item():.4f}, "
                    f"acq={acq_value.item():.4f}"
                )
        except RuntimeError as exc:
            if self.logger is not None:
                self.logger.warning(f"FairDiffusion BO failed: {exc}")
            self.history_sigmas.clear()
            self.history_loss_gaps.clear()
            return False

        self.history_sigmas.clear()
        self.history_loss_gaps.clear()
        return True

    def _apply_exploration_noise(self):
        random_component = torch.rand_like(self.sigma)
        self.sigma = (
            self.sigma * self.exploitation_rate
            + random_component * (1.0 - self.exploitation_rate)
        )
        self.sigma = torch.clamp(self.sigma, self.sigma_bounds[0], self.sigma_bounds[1])

    def _sync_sigma(self):
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.sigma, src=0)

    def finalize(self):
        if not self.accelerator.is_main_process:
            return
        try:
            with open(self.history_path, "w", encoding="utf-8") as fp:
                json.dump(self._records, fp, indent=2)
            if self.logger is not None:
                self.logger.info(f"FairDiffusion history saved to {self.history_path}")
        except Exception as exc:  # pragma: no cover - filesystem guard
            if self.logger is not None:
                self.logger.warning(f"Could not save FairDiffusion history: {exc}")

