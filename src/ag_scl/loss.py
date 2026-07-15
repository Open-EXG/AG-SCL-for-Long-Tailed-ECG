"""Angular Gaussian scoring, adaptive logit adjustment, and AG-SCL loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import LossConfig
from .statistics import OnlineClassMomentEstimator


class AdaptiveLogitAdjustment(nn.Module):
    """Bounded learnable label-state prior adjustment (ALA)."""

    def __init__(
        self,
        state_priors: tuple[float, float],
        lower_bound: float = 0.5,
        upper_bound: float = 3.0,
    ) -> None:
        super().__init__()
        priors = torch.tensor(state_priors, dtype=torch.float32)
        priors = priors / priors.sum()
        self.register_buffer("log_priors", priors.log().view(1, 2))
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        target = (1.0 - self.lower_bound) / (self.upper_bound - self.lower_bound)
        initial_gamma = math.log(target / (1 - target))
        self.gamma = nn.Parameter(torch.full((1, 2), initial_gamma))

    @property
    def scales(self) -> Tensor:
        return self.lower_bound + (self.upper_bound - self.lower_bound) * torch.sigmoid(
            self.gamma
        )

    def adjust(self, logits: Tensor) -> Tensor:
        return logits + self.scales * self.log_priors

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return functional.cross_entropy(self.adjust(logits), targets)


class AngularGaussianScorer(nn.Module):
    """MGF-derived Angular Gaussian class-state scoring with online moments."""

    def __init__(self, embedding_dim: int, temperature: float, shrinkage_tau: float) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.history = OnlineClassMomentEstimator(embedding_dim, 2, shrinkage_tau)
        self.current = OnlineClassMomentEstimator(embedding_dim, 2, shrinkage_tau)

    def begin_epoch(self) -> None:
        self.history.copy_from(self.current)
        self.current.reset()

    def forward(self, features: Tensor, labels: Tensor | None = None) -> Tensor:
        if labels is not None:
            detached = features.detach()
            self.history.update(detached, labels)
            self.current.update(detached, labels)
        # The second augmented view updates the registered moment buffers before
        # backward. Clone the scoring snapshot so the first view's autograd graph
        # never references storage that is subsequently updated in place.
        mean = self.history.mean.detach().clone()
        covariance = self.history.covariance.detach().clone()
        alignment = features @ mean.transpose(0, 1) / self.temperature
        uncertainty = torch.einsum("nd,cdk,nk->nc", features, covariance, features)
        return alignment + uncertainty / (2 * self.temperature**2)


@dataclass(frozen=True)
class AGSCLLabelLossOutput:
    classification: Tensor
    contrastive: Tensor


class AGSCLLabelLoss(nn.Module):
    """Classification and two-view Angular Gaussian loss for one binary label."""

    def __init__(self, state_priors: tuple[float, float], config: LossConfig) -> None:
        super().__init__()
        self.scorer = AngularGaussianScorer(
            config.embedding_dim,
            config.contrastive_temperature,
            config.shrinkage_tau,
        )
        self.ala = AdaptiveLogitAdjustment(
            state_priors,
            config.ala_lower_bound,
            config.ala_upper_bound,
        )

    def begin_epoch(self) -> None:
        self.scorer.begin_epoch()

    def forward(
        self,
        raw_logits: Tensor,
        projected_views: Tensor,
        labels: Tensor,
    ) -> AGSCLLabelLossOutput:
        if projected_views.shape[0] != 2:
            raise ValueError("projected_views must contain exactly two augmented views.")
        contrastive_losses = []
        for view in projected_views:
            scores = self.scorer(view, labels)
            contrastive_losses.append(self.ala(scores, labels))
        return AGSCLLabelLossOutput(
            classification=self.ala(raw_logits, labels),
            contrastive=torch.stack(contrastive_losses).mean(),
        )


@dataclass(frozen=True)
class AGSCLLossOutput:
    total: Tensor
    classification: Tensor
    contrastive: Tensor


class AGSCLLoss(nn.Module):
    """Multi-label AG-SCL objective with one state model and ALA module per label."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.contrastive_weight = float(config.contrastive_weight)
        self.label_losses = nn.ModuleList(
            AGSCLLabelLoss(priors, config) for priors in config.label_state_priors
        )

    def begin_epoch(self) -> None:
        for label_loss in self.label_losses:
            label_loss.begin_epoch()

    def forward(
        self,
        raw_logits: Tensor,
        projected_views: Tensor,
        labels: Tensor,
    ) -> AGSCLLossOutput:
        if raw_logits.ndim != 3 or raw_logits.shape[-1] != 2:
            raise ValueError("raw_logits must have shape [batch, num_labels, 2].")
        if projected_views.ndim != 4 or projected_views.shape[0] != 2:
            raise ValueError(
                "projected_views must have shape [2, batch, num_labels, embedding_dim]."
            )
        if labels.shape != raw_logits.shape[:2]:
            raise ValueError("labels must match the batch and label dimensions of raw_logits.")
        if len(self.label_losses) != raw_logits.shape[1]:
            raise ValueError("The number of configured label losses does not match raw_logits.")

        classification = raw_logits.new_zeros(())
        contrastive = raw_logits.new_zeros(())
        for index, label_loss in enumerate(self.label_losses):
            output = label_loss(
                raw_logits[:, index],
                projected_views[:, :, index],
                labels[:, index],
            )
            classification = classification + output.classification
            contrastive = contrastive + output.contrastive
        total = classification + self.contrastive_weight * contrastive
        return AGSCLLossOutput(total, classification, contrastive)
