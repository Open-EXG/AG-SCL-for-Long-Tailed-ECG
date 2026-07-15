"""Online class-state moments and count-based full-covariance shrinkage."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class OnlineClassMomentEstimator(nn.Module):
    """Track class means, second moments, counts, and shrunk covariances."""

    def __init__(self, embedding_dim: int, num_states: int = 2, shrinkage_tau: float = 800.0) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_states = int(num_states)
        self.shrinkage_tau = float(shrinkage_tau)
        self.register_buffer("mean", torch.empty(num_states, embedding_dim))
        self.register_buffer("count", torch.zeros(num_states))
        self.register_buffer("covariance", torch.empty(num_states, embedding_dim, embedding_dim))
        self.register_buffer("second_moment", torch.empty(num_states, embedding_dim, embedding_dim))
        self.register_buffer("identity", torch.eye(embedding_dim).repeat(num_states, 1, 1))
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.mean.copy_(functional.normalize(torch.randn_like(self.mean), dim=1))
        self.count.zero_()
        self.covariance.copy_(self.identity)
        self.second_moment.copy_(self.identity)

    @torch.no_grad()
    def copy_from(self, other: "OnlineClassMomentEstimator") -> None:
        self.mean.copy_(other.mean)
        self.count.copy_(other.count)
        self.covariance.copy_(other.covariance)
        self.second_moment.copy_(other.second_moment)

    @torch.no_grad()
    def update(self, features: Tensor, labels: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.embedding_dim:
            raise ValueError("features must have shape [batch, embedding_dim].")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape [batch].")
        if torch.any((labels < 0) | (labels >= self.num_states)):
            raise ValueError("labels contain an invalid class state.")

        batch_size = features.shape[0]
        one_hot = torch.zeros(
            batch_size,
            self.num_states,
            device=features.device,
            dtype=features.dtype,
        )
        one_hot.scatter_(1, labels[:, None], 1)
        batch_counts = one_hot.sum(dim=0)
        membership = one_hot.transpose(0, 1)
        batch_means = membership @ features / (batch_counts[:, None] + 1e-8)
        weights = batch_counts / (self.count + batch_counts + 1e-8)
        self.mean.copy_(
            self.mean * (1 - weights[:, None]) + batch_means * weights[:, None]
        )

        outer_products = features[:, :, None] @ features[:, None, :]
        batch_second = (membership @ outer_products.reshape(batch_size, -1)).reshape(
            self.num_states, self.embedding_dim, self.embedding_dim
        )
        batch_second = batch_second / (batch_counts[:, None, None] + 1e-8)
        self.second_moment.copy_(
            self.second_moment * (1 - weights[:, None, None])
            + batch_second * weights[:, None, None]
        )

        raw_covariance = self.second_moment - self.mean[:, :, None] @ self.mean[:, None, :]
        updated_counts = self.count + batch_counts
        shrinkage = torch.exp(-updated_counts / self.shrinkage_tau)[:, None, None]
        self.covariance.copy_(
            (1 - shrinkage) * raw_covariance + shrinkage * self.identity
        )
        self.count.add_(batch_counts)


class CountBasedCovarianceShrinkage(nn.Module):
    """Standalone form of the shrinkage equation used by AG-SCL."""

    def __init__(self, tau: float = 800.0) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive.")
        self.tau = float(tau)

    def forward(self, covariance: Tensor, counts: Tensor) -> Tensor:
        identity = torch.eye(
            covariance.shape[-1], device=covariance.device, dtype=covariance.dtype
        ).expand_as(covariance)
        weight = torch.exp(-counts / self.tau)[..., None, None]
        return (1 - weight) * covariance + weight * identity

