"""Typed configuration for the public AG-SCL implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def _pair(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class DataConfig:
    num_labels: int = 6
    signal_length: int = 1000
    num_windows: int = 50
    sampling_rate: float = 100.0
    input_channels: int = 1

    @property
    def window_length(self) -> int:
        return self.signal_length // self.num_windows

    def validate(self) -> None:
        if self.num_labels < 1:
            raise ValueError("data.num_labels must be positive.")
        if self.signal_length < 1 or self.num_windows < 1:
            raise ValueError("Signal length and number of windows must be positive.")
        if self.signal_length % self.num_windows != 0:
            raise ValueError("data.signal_length must be divisible by data.num_windows.")
        if self.input_channels != 1:
            raise ValueError("This public release supports one ECG channel only.")


@dataclass(frozen=True)
class ModelConfig:
    patch_size: tuple[int, int] = (1, 10)
    window_size: tuple[int, int] = (3, 3)
    embed_dim: int = 64
    depths: tuple[int, ...] = (2, 2, 4)
    num_heads: int = 4
    feature_dim: int = 256
    projection_dim: int = 256
    dropout: float = 0.2
    attention_dropout: float = 0.2
    drop_path_rate: float = 0.1
    classifier_scale: float = 30.0

    def validate(self) -> None:
        if not self.depths or any(depth < 1 for depth in self.depths):
            raise ValueError("model.depths must contain positive integers.")
        if self.embed_dim < 1 or self.feature_dim < 1 or self.projection_dim < 1:
            raise ValueError("Model dimensions must be positive.")
        for stage in range(len(self.depths)):
            if (self.embed_dim * (2**stage)) % self.num_heads != 0:
                raise ValueError("Every Swin stage dimension must be divisible by model.num_heads.")


@dataclass(frozen=True)
class LossConfig:
    embedding_dim: int = 256
    contrastive_weight: float = 1.0
    contrastive_temperature: float = 0.2
    shrinkage_tau: float = 800.0
    ala_lower_bound: float = 0.5
    ala_upper_bound: float = 3.0
    label_state_priors: tuple[tuple[float, float], ...] = (
        (0.92807009, 0.07192991),
        (0.96210835, 0.03789165),
        (0.97077848, 0.02922152),
        (0.99, 0.01),
        (0.99, 0.01),
        (0.98651314, 0.01348686),
    )

    def validate(self, num_labels: int, projection_dim: int) -> None:
        if self.embedding_dim != projection_dim:
            raise ValueError("loss.embedding_dim must equal model.projection_dim.")
        if len(self.label_state_priors) != num_labels:
            raise ValueError("loss.label_state_priors must have one pair per label.")
        for prior in self.label_state_priors:
            if len(prior) != 2 or any(value <= 0 for value in prior):
                raise ValueError("Every label-state prior must contain two positive values.")
            if abs(sum(prior) - 1.0) > 1e-6:
                raise ValueError("Every label-state prior pair must sum to one.")
        if self.contrastive_temperature <= 0 or self.shrinkage_tau <= 0:
            raise ValueError("Loss temperatures must be positive.")
        if not self.ala_lower_bound < 1.0 < self.ala_upper_bound:
            raise ValueError("ALA bounds must contain the initial scale 1.0.")


@dataclass(frozen=True)
class TransformConfig:
    signal_negation: float = 0.1
    global_amplitude_scaling: float = 0.5
    additive_jitter: float = 0.5
    frequency_masking: float = 0.5
    band_constrained_phase_jitter: float = 0.5
    band_constrained_magnitude_scaling: float = 0.5
    amplitude_scaling_std: float = 0.2
    jitter_std: float = 0.01
    frequency_mask_ratio: float = 0.3

    def validate(self) -> None:
        probabilities = (
            self.signal_negation,
            self.global_amplitude_scaling,
            self.additive_jitter,
            self.frequency_masking,
            self.band_constrained_phase_jitter,
            self.band_constrained_magnitude_scaling,
        )
        if any(probability < 0 or probability > 1 for probability in probabilities):
            raise ValueError("Augmentation probabilities must be between zero and one.")


@dataclass(frozen=True)
class AugmentationConfig:
    eligibility: str = "any_positive_label"
    num_augmented_views: int = 2
    transform_selection: str = "uniform_per_view_batch"
    transforms: TransformConfig = field(default_factory=TransformConfig)

    def validate(self) -> None:
        if self.eligibility != "any_positive_label":
            raise ValueError("Only eligibility='any_positive_label' is supported.")
        if self.num_augmented_views != 2:
            raise ValueError("AG-SCL requires exactly two augmented views.")
        if self.transform_selection != "uniform_per_view_batch":
            raise ValueError("Only uniform_per_view_batch transform selection is supported.")
        self.transforms.validate()


@dataclass(frozen=True)
class AGSCLConfig:
    method: str = "ag_scl"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def validate(self) -> None:
        if self.method != "ag_scl":
            raise ValueError("method must be 'ag_scl'.")
        self.data.validate()
        self.model.validate()
        self.loss.validate(self.data.num_labels, self.model.projection_dim)
        self.augmentation.validate()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AGSCLConfig":
        data = DataConfig(**raw.get("data", {}))

        model_raw = dict(raw.get("model", {}))
        if "patch_size" in model_raw:
            model_raw["patch_size"] = _pair(model_raw["patch_size"], "model.patch_size")
        if "window_size" in model_raw:
            model_raw["window_size"] = _pair(model_raw["window_size"], "model.window_size")
        if "depths" in model_raw:
            model_raw["depths"] = tuple(int(value) for value in model_raw["depths"])
        model = ModelConfig(**model_raw)

        loss_raw = dict(raw.get("loss", {}))
        if "label_state_priors" in loss_raw:
            loss_raw["label_state_priors"] = tuple(
                (float(pair[0]), float(pair[1])) for pair in loss_raw["label_state_priors"]
            )
        loss = LossConfig(**loss_raw)

        augmentation_raw = dict(raw.get("augmentation", {}))
        transforms = TransformConfig(**augmentation_raw.pop("transforms", {}))
        augmentation = AugmentationConfig(transforms=transforms, **augmentation_raw)

        config = cls(
            method=str(raw.get("method", "ag_scl")),
            data=data,
            model=model,
            loss=loss,
            augmentation=augmentation,
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AGSCLConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("The configuration root must be a mapping.")
        return cls.from_dict(raw)

