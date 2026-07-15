"""Window-wise spectral conversion and AG-SCL multi-view augmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import AGSCLConfig
from .transforms import (
    AdditiveJitter,
    BandConstrainedMagnitudeScaling,
    BandConstrainedPhaseJitter,
    FrequencyMasking,
    GlobalAmplitudeScaling,
    SignalNegation,
    TailAwareTransform,
)


def window_signal(ecg: Tensor, num_windows: int) -> Tensor:
    """Reshape `[B, C, L]` ECG signals into `[B, C, windows, samples]`."""
    if ecg.ndim != 3:
        raise ValueError("ecg must have shape [batch, channels, signal_length].")
    if ecg.shape[-1] % num_windows:
        raise ValueError("The ECG signal length must be divisible by num_windows.")
    return ecg.reshape(ecg.shape[0], ecg.shape[1], num_windows, ecg.shape[-1] // num_windows)


def fft_magnitude_phase(windowed_ecg: Tensor) -> Tensor:
    """Convert windowed ECG to interleaved magnitude and normalized phase channels."""
    if windowed_ecg.ndim != 4:
        raise ValueError("windowed_ecg must have shape [batch, channels, windows, samples].")
    spectrum = torch.fft.fft(windowed_ecg, dim=-1)
    magnitude_phase = torch.stack((spectrum.abs(), spectrum.angle() / torch.pi), dim=-1)
    magnitude_phase = magnitude_phase.permute(0, 1, 4, 2, 3)
    batch, channels, components, windows, samples = magnitude_phase.shape
    return magnitude_phase.reshape(batch, channels * components, windows, samples)


class TailAwareMultiViewAugmenter(nn.Module):
    """Create one original and two independently sampled tail-aware ECG views."""

    def __init__(self, config: AGSCLConfig) -> None:
        super().__init__()
        config.validate()
        self.num_windows = config.data.num_windows
        probabilities = config.augmentation.transforms
        sampling_rate = config.data.sampling_rate
        self.transforms = nn.ModuleList(
            [
                SignalNegation(probabilities.signal_negation),
                GlobalAmplitudeScaling(
                    probabilities.global_amplitude_scaling,
                    probabilities.amplitude_scaling_std,
                ),
                AdditiveJitter(probabilities.additive_jitter, probabilities.jitter_std),
                FrequencyMasking(
                    probabilities.frequency_masking,
                    probabilities.frequency_mask_ratio,
                ),
                BandConstrainedPhaseJitter(
                    probabilities.band_constrained_phase_jitter,
                    sampling_rate,
                ),
                BandConstrainedMagnitudeScaling(
                    probabilities.band_constrained_magnitude_scaling,
                    sampling_rate,
                ),
            ]
        )
        self.num_augmented_views = config.augmentation.num_augmented_views

    def original_view(self, ecg: Tensor) -> Tensor:
        return fft_magnitude_phase(window_signal(ecg, self.num_windows))

    def _sample_view(self, windowed_ecg: Tensor, labels: Tensor) -> Tensor:
        transform: TailAwareTransform = self.transforms[
            int(torch.randint(len(self.transforms), (1,), device=windowed_ecg.device).item())
        ]
        if transform.domain == "time":
            return fft_magnitude_phase(transform(windowed_ecg, labels))
        return transform(fft_magnitude_phase(windowed_ecg), labels)

    def forward(self, ecg: Tensor, labels: Tensor) -> Tensor:
        windowed = window_signal(ecg, self.num_windows)
        views = [fft_magnitude_phase(windowed)]
        views.extend(self._sample_view(windowed, labels) for _ in range(self.num_augmented_views))
        return torch.stack(views, dim=0)

