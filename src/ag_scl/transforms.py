"""Tail-aware time- and frequency-domain ECG transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


def positive_record_mask(labels: Tensor) -> Tensor:
    """Return the records that contain at least one positive binary label."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, num_labels].")
    return labels.bool().any(dim=1)


class TailAwareTransform(nn.Module, ABC):
    """Base class for transforms applied only to selected positive records."""

    domain: str

    def __init__(self, probability: float) -> None:
        super().__init__()
        if probability < 0 or probability > 1:
            raise ValueError("probability must be between zero and one.")
        self.probability = float(probability)

    def application_mask(self, labels: Tensor, device: torch.device) -> Tensor:
        eligible = positive_record_mask(labels).to(device=device)
        sampled = torch.rand(labels.shape[0], device=device) < self.probability
        return eligible & sampled

    @abstractmethod
    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError


class SignalNegation(TailAwareTransform):
    """Multiply selected ECG records by -1."""

    domain = "time"

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        mask = self.application_mask(labels, inputs.device)
        factors = (1.0 - 2.0 * mask.to(inputs.dtype)).view(-1, 1, 1, 1)
        return inputs * factors


class GlobalAmplitudeScaling(TailAwareTransform):
    """Apply one independently sampled global gain to each selected record."""

    domain = "time"

    def __init__(self, probability: float, std: float = 0.2) -> None:
        super().__init__(probability)
        self.std = float(std)

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        mask = self.application_mask(labels, inputs.device)
        factors = torch.ones((inputs.shape[0], 1, 1, 1), device=inputs.device, dtype=inputs.dtype)
        count = int(mask.sum().item())
        if count:
            factors[mask] = torch.normal(
                mean=1.0,
                std=self.std,
                size=(count, 1, 1, 1),
                device=inputs.device,
                dtype=inputs.dtype,
            )
        return inputs * factors


class AdditiveJitter(TailAwareTransform):
    """Add independent Gaussian jitter with a fixed normalized-signal scale."""

    domain = "time"

    def __init__(self, probability: float, std: float = 0.01) -> None:
        super().__init__(probability)
        self.std = float(std)

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        mask = self.application_mask(labels, inputs.device).view(-1, 1, 1, 1)
        noise = torch.randn_like(inputs) * self.std
        return inputs + noise * mask.to(inputs.dtype)


class FrequencyMasking(TailAwareTransform):
    """Zero an independently sampled contiguous frequency interval per record."""

    domain = "frequency"

    def __init__(self, probability: float, mask_ratio: float = 0.3) -> None:
        super().__init__(probability)
        if mask_ratio <= 0 or mask_ratio >= 1:
            raise ValueError("mask_ratio must be between zero and one.")
        self.mask_ratio = float(mask_ratio)

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        batch, _, _, spectrum_length = inputs.shape
        selected = self.application_mask(labels, inputs.device)
        max_width = int(spectrum_length * self.mask_ratio)
        if max_width <= 1:
            raise ValueError("The configured frequency mask is too narrow for this input.")

        widths = torch.randint(1, max_width + 1, (batch, 1), device=inputs.device)
        starts = torch.randint(0, spectrum_length - max_width, (batch, 1), device=inputs.device)
        indices = torch.arange(spectrum_length, device=inputs.device).unsqueeze(0)
        mask = (indices >= starts) & (indices < starts + widths)
        mask &= selected.unsqueeze(1)
        return inputs.masked_fill(mask[:, None, None, :], 0)


class BandConstrainedPhaseJitter(TailAwareTransform):
    """Perturb normalized FFT phase with frequency-dependent limits."""

    domain = "frequency"
    bands = (
        (0.5, 7.0, 5.0 / 180.0),
        (7.0, 25.0, 2.0 / 180.0),
        (25.0, 50.0, 10.0 / 180.0),
    )

    def __init__(self, probability: float, sampling_rate: float = 100.0) -> None:
        super().__init__(probability)
        self.sampling_rate = float(sampling_rate)

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        batch, _, _, spectrum_length = inputs.shape
        selected = self.application_mask(labels, inputs.device)
        offsets = torch.zeros((batch, spectrum_length), device=inputs.device, dtype=inputs.dtype)
        frequencies = torch.fft.fftfreq(
            spectrum_length, d=1.0 / self.sampling_rate, device=inputs.device
        ).abs()

        count = int(selected.sum().item())
        if count:
            selected_offsets = offsets[selected]
            for low, high, limit in self.bands:
                band = (frequencies >= low) & (frequencies < high)
                if band.any():
                    delta = torch.empty((count, 1), device=inputs.device, dtype=inputs.dtype).uniform_(
                        -limit, limit
                    )
                    selected_offsets[:, band] = delta
            offsets[selected] = selected_offsets

        output = inputs.clone()
        phases = output[:, 1::2]
        output[:, 1::2] = (phases + offsets[:, None, None, :] + 1.0).remainder(2.0) - 1.0
        return output


class BandConstrainedMagnitudeScaling(TailAwareTransform):
    """Scale FFT magnitude outside and inside the QRS-dominant band."""

    domain = "frequency"
    bands = (
        (0.5, 7.0, 0.90, 1.10),
        (7.0, 25.0, 0.97, 1.03),
        (25.0, 50.0, 0.60, 1.60),
    )

    def __init__(self, probability: float, sampling_rate: float = 100.0) -> None:
        super().__init__(probability)
        self.sampling_rate = float(sampling_rate)

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        batch, _, _, spectrum_length = inputs.shape
        selected = self.application_mask(labels, inputs.device)
        scales = torch.ones((batch, spectrum_length), device=inputs.device, dtype=inputs.dtype)
        frequencies = torch.fft.fftfreq(
            spectrum_length, d=1.0 / self.sampling_rate, device=inputs.device
        ).abs()

        count = int(selected.sum().item())
        if count:
            selected_scales = scales[selected]
            for low, high, minimum, maximum in self.bands:
                band = (frequencies >= low) & (frequencies < high)
                if band.any():
                    values = torch.empty((count, 1), device=inputs.device, dtype=inputs.dtype).uniform_(
                        minimum, maximum
                    )
                    selected_scales[:, band] = values
            scales[selected] = selected_scales

        output = inputs.clone()
        output[:, 0::2] *= scales[:, None, None, :]
        return output

