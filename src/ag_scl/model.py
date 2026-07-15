"""Single-lead spectro-temporal Swin encoder and label-specific heads."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import AGSCLConfig, ModelConfig
from .heads import LabelSpecificHeads
from .swin import PatchEmbedding, SwinStage, padded_image_size


class SingleModalityAttentionFusion(nn.Module):
    """Attention aggregation retained from the experimental single-modality path."""

    def __init__(self, feature_dim: int, num_heads: int, attention_dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch, intervals, sensors, channels = inputs.shape
        flattened = inputs.reshape(batch * intervals, sensors, channels)
        normalized = self.norm(flattened)
        query = normalized.mean(dim=1, keepdim=True)
        fused, _ = self.attention(query, normalized, normalized, need_weights=False)
        return fused.reshape(batch, intervals, channels)


class SwinECGEncoder(nn.Module):
    """Frequency-domain Swin encoder for `[B, 2, windows, spectrum]` inputs."""

    def __init__(self, data_config, model_config: ModelConfig) -> None:
        super().__init__()
        self.num_windows = data_config.num_windows
        self.spectrum_length = data_config.window_length
        self.config = model_config
        self.image_size = (self.num_windows, self.spectrum_length)
        self.padded_size = padded_image_size(
            self.image_size,
            model_config.window_size,
            model_config.patch_size,
            len(model_config.depths),
        )
        self.patch_embedding = PatchEmbedding(
            self.padded_size,
            model_config.patch_size,
            input_channels=2,
            embedding_dimension=model_config.embed_dim,
        )

        initial_resolution = self.patch_embedding.resolution
        total_blocks = sum(model_config.depths)
        drop_paths = torch.linspace(0, model_config.drop_path_rate, total_blocks).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for index, depth in enumerate(model_config.depths):
            dimension = model_config.embed_dim * (2**index)
            resolution = (
                initial_resolution[0] // (2**index),
                initial_resolution[1] // (2**index),
            )
            self.stages.append(
                SwinStage(
                    dimension=dimension,
                    input_resolution=resolution,
                    depth=depth,
                    num_heads=model_config.num_heads,
                    window_size=model_config.window_size,
                    dropout=model_config.dropout,
                    attention_dropout=model_config.attention_dropout,
                    drop_path=drop_paths[offset : offset + depth],
                    downsample=index < len(model_config.depths) - 1,
                )
            )
            offset += depth

        final_dimension = model_config.embed_dim * (2 ** (len(model_config.depths) - 1))
        final_resolution = (
            initial_resolution[0] // (2 ** (len(model_config.depths) - 1)),
            initial_resolution[1] // (2 ** (len(model_config.depths) - 1)),
        )
        self.feature_projection = nn.Linear(
            final_resolution[0] * final_resolution[1] * final_dimension,
            model_config.feature_dim,
        )
        self.fusion = SingleModalityAttentionFusion(
            model_config.feature_dim,
            model_config.num_heads,
            model_config.attention_dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 2:
            raise ValueError("Spectral inputs must have shape [batch, 2, windows, spectrum].")
        if tuple(inputs.shape[-2:]) != self.image_size:
            raise ValueError("Spectral input dimensions do not match the configured ECG representation.")
        padding = (0, self.padded_size[1] - self.image_size[1], 0, self.padded_size[0] - self.image_size[0])
        encoded = self.patch_embedding(functional.pad(inputs, padding))
        for stage in self.stages:
            encoded = stage(encoded)
        features = self.feature_projection(encoded.reshape(encoded.shape[0], -1))
        return self.fusion(features[:, None, None, :]).flatten(start_dim=1)


class AGSCLModel(nn.Module):
    """Shared encoder with label-specific classification and projection heads."""

    def __init__(self, config: AGSCLConfig) -> None:
        super().__init__()
        self.encoder = SwinECGEncoder(config.data, config.model)
        self.heads = LabelSpecificHeads(
            feature_dim=config.model.feature_dim,
            projection_dim=config.model.projection_dim,
            num_labels=config.data.num_labels,
            classifier_scale=config.model.classifier_scale,
        )

    def forward_views(self, views: Tensor) -> tuple[Tensor, Tensor]:
        if views.ndim != 5 or views.shape[0] != 3:
            raise ValueError("views must have shape [3, batch, 2, windows, spectrum].")
        num_views, batch = views.shape[:2]
        features = self.encoder(views.flatten(0, 1))
        logits, projections = self.heads(features)
        logits = logits.view(num_views, batch, *logits.shape[1:])
        projections = projections.view(num_views, batch, *projections.shape[1:])
        return logits[0], projections[1:]

    def forward(self, spectral_input: Tensor) -> Tensor:
        return self.heads.classify(self.encoder(spectral_input))

