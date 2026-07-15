"""Label-specific binary classification and projection heads."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class NormalizedLinear(nn.Module):
    """Cosine-normalized linear classifier used in the reported experiments."""

    def __init__(self, input_features: int, output_features: int, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_features, output_features))
        self.scale = float(scale)
        with torch.no_grad():
            self.weight.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.scale * functional.normalize(inputs, dim=1).mm(
            functional.normalize(self.weight, dim=0)
        )


class LabelSpecificHeads(nn.Module):
    """Independent binary classifier and normalized projection head per label."""

    def __init__(
        self,
        feature_dim: int,
        projection_dim: int,
        num_labels: int,
        classifier_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.classifiers = nn.ModuleList(
            NormalizedLinear(feature_dim, 2, classifier_scale) for _ in range(num_labels)
        )
        self.projectors = nn.ModuleList(
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, projection_dim),
            )
            for _ in range(num_labels)
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        logits = torch.stack([head(features) for head in self.classifiers], dim=1)
        projections = torch.stack(
            [functional.normalize(head(features), dim=1) for head in self.projectors],
            dim=1,
        )
        return logits, projections

    def classify(self, features: Tensor) -> Tensor:
        return torch.stack([head(features) for head in self.classifiers], dim=1)

