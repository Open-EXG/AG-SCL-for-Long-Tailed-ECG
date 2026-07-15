"""Top-level AG-SCL training and inference interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .augmentation import TailAwareMultiViewAugmenter
from .config import AGSCLConfig
from .loss import AGSCLLoss
from .model import AGSCLModel


@dataclass(frozen=True)
class AGSCLTrainingOutput:
    loss: Tensor
    classification_loss: Tensor
    contrastive_loss: Tensor
    raw_logits: Tensor


class AGSCL(nn.Module):
    """Complete AG-SCL core method without a dataset-specific training loop."""

    def __init__(self, config: AGSCLConfig | None = None) -> None:
        super().__init__()
        self.config = config or AGSCLConfig()
        self.config.validate()
        self.augmenter = TailAwareMultiViewAugmenter(self.config)
        self.model = AGSCLModel(self.config)
        self.criterion = AGSCLLoss(self.config.loss)

    def _validate_ecg(self, ecg: Tensor) -> None:
        expected = (
            self.config.data.input_channels,
            self.config.data.signal_length,
        )
        if ecg.ndim != 3 or tuple(ecg.shape[1:]) != expected:
            raise ValueError(
                f"ecg must have shape [batch, {expected[0]}, {expected[1]}]."
            )
        if not torch.is_floating_point(ecg):
            raise TypeError("ecg must use a floating-point dtype.")

    def _validate_labels(self, labels: Tensor, batch_size: int, device: torch.device) -> None:
        expected = (batch_size, self.config.data.num_labels)
        if tuple(labels.shape) != expected:
            raise ValueError(f"labels must have shape {expected}.")
        if labels.device != device:
            raise ValueError("ecg and labels must be on the same device.")
        if torch.any((labels != 0) & (labels != 1)):
            raise ValueError("labels must contain only binary values 0 and 1.")

    def begin_epoch(self) -> None:
        """Roll current online moments into the scoring history and reset the epoch accumulator."""
        self.criterion.begin_epoch()

    def training_step(self, ecg: Tensor, labels: Tensor) -> AGSCLTrainingOutput:
        self._validate_ecg(ecg)
        self._validate_labels(labels, ecg.shape[0], ecg.device)
        labels = labels.long()
        views = self.augmenter(ecg, labels)
        raw_logits, projected_views = self.model.forward_views(views)
        losses = self.criterion(raw_logits, projected_views, labels)
        return AGSCLTrainingOutput(
            loss=losses.total,
            classification_loss=losses.classification,
            contrastive_loss=losses.contrastive,
            raw_logits=raw_logits,
        )

    def forward(self, ecg: Tensor) -> Tensor:
        """Return raw binary logits for the unmodified ECG view."""
        self._validate_ecg(ecg)
        return self.model(self.augmenter.original_view(ecg))

