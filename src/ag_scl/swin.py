from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def padded_image_size(
    image_size: tuple[int, int],
    window_size: tuple[int, int],
    patch_size: tuple[int, int],
    num_stages: int,
) -> tuple[int, int]:
    scale = 2 ** (num_stages - 1)
    multiples = (
        window_size[0] * patch_size[0] * scale,
        window_size[1] * patch_size[1] * scale,
    )
    return tuple(math.ceil(max(size, multiple) / multiple) * multiple for size, multiple in zip(image_size, multiples))


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep_probability = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep_probability)
        return inputs * mask / keep_probability


class MLP(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


def window_partition(inputs: Tensor, window_size: tuple[int, int]) -> Tensor:
    batch, height, width, channels = inputs.shape
    window_height, window_width = window_size
    inputs = inputs.view(
        batch,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
        channels,
    )
    return inputs.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_height, window_width, channels
    )


def window_reverse(
    windows: Tensor, window_size: tuple[int, int], height: int, width: int
) -> Tensor:
    window_height, window_width = window_size
    batch = int(windows.shape[0] / (height * width / window_height / window_width))
    windows = windows.view(
        batch,
        height // window_height,
        width // window_width,
        window_height,
        window_width,
        -1,
    )
    return windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        window_size: tuple[int, int],
        num_heads: int,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dimension // num_heads) ** -0.5

        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size[0]),
                torch.arange(window_size[1]),
                indexing="ij",
            )
        )
        coordinates = torch.flatten(coordinates, 1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size[0] - 1
        relative[:, :, 1] += window_size[1] - 1
        relative[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative.sum(-1))

        self.qkv = nn.Linear(dimension, dimension * 3, bias=True)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dimension, dimension)
        self.projection_dropout = nn.Dropout(projection_dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(tokens, tokens, -1)
        attention = attention + relative_bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.num_heads, tokens, tokens)

        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection_dropout(self.projection(output))


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        adjusted_window = list(window_size)
        adjusted_shift = list(shift_size)
        for axis in range(2):
            if input_resolution[axis] <= adjusted_window[axis]:
                adjusted_window[axis] = input_resolution[axis]
                adjusted_shift[axis] = 0
            if not 0 <= adjusted_shift[axis] < adjusted_window[axis]:
                raise ValueError("shift_size must be smaller than window_size.")
        self.window_size = tuple(adjusted_window)
        self.shift_size = tuple(adjusted_shift)

        self.norm1 = nn.LayerNorm(dimension)
        self.attention = WindowAttention(
            dimension,
            self.window_size,
            num_heads,
            attention_dropout,
            dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = MLP(dimension, int(dimension * mlp_ratio), dropout)
        self.register_buffer("attention_mask", self._build_attention_mask())

    def _build_attention_mask(self) -> Tensor | None:
        if min(self.shift_size) == 0:
            return None
        height, width = self.input_resolution
        window_height, window_width = self.window_size
        shift_height, shift_width = self.shift_size
        image_mask = torch.zeros((1, height, width, 1))
        height_slices = (
            slice(0, -window_height),
            slice(-window_height, -shift_height),
            slice(-shift_height, None),
        )
        width_slices = (
            slice(0, -window_width),
            slice(-window_width, -shift_width),
            slice(-shift_width, None),
        )
        counter = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[:, height_slice, width_slice] = counter
                counter += 1
        windows = window_partition(image_mask, self.window_size).view(
            -1, window_height * window_width
        )
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, inputs: Tensor) -> Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = inputs.shape
        if tokens != height * width:
            raise ValueError("Swin input token count does not match its configured resolution.")

        shortcut = inputs
        inputs = self.norm1(inputs).view(batch, height, width, channels)
        if min(self.shift_size) > 0:
            shifted = torch.roll(
                inputs,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            shifted = inputs

        windows = window_partition(shifted, self.window_size).view(
            -1, self.window_size[0] * self.window_size[1], channels
        )
        attended = self.attention(windows, self.attention_mask).view(
            -1, self.window_size[0], self.window_size[1], channels
        )
        shifted = window_reverse(attended, self.window_size, height, width)
        if min(self.shift_size) > 0:
            shifted = torch.roll(
                shifted,
                shifts=(self.shift_size[0], self.shift_size[1]),
                dims=(1, 2),
            )
        output = shifted.view(batch, height * width, channels)
        output = shortcut + self.drop_path(output)
        return output + self.drop_path(self.mlp(self.norm2(output)))


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dimension: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dimension = dimension
        self.norm = nn.LayerNorm(4 * dimension)
        self.reduction = nn.Linear(4 * dimension, 2 * dimension, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = inputs.shape
        if tokens != height * width or height % 2 or width % 2:
            raise ValueError("Patch merging requires an even two-dimensional token grid.")
        inputs = inputs.view(batch, height, width, channels)
        merged = torch.cat(
            (
                inputs[:, 0::2, 0::2],
                inputs[:, 1::2, 0::2],
                inputs[:, 0::2, 1::2],
                inputs[:, 1::2, 1::2],
            ),
            dim=-1,
        ).view(batch, -1, 4 * channels)
        return self.reduction(self.norm(merged))


class SwinStage(nn.Module):
    def __init__(
        self,
        dimension: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: tuple[int, int],
        dropout: float,
        attention_dropout: float,
        drop_path: Sequence[float],
        downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(
                dimension=dimension,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if index % 2 == 0 else (window_size[0] // 2, window_size[1] // 2),
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=float(drop_path[index]),
            )
            for index in range(depth)
        )
        self.downsample = PatchMerging(input_resolution, dimension) if downsample else None

    def forward(self, inputs: Tensor) -> Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return self.downsample(inputs) if self.downsample is not None else inputs


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        input_channels: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.resolution = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.projection = nn.Conv2d(
            input_channels,
            embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embedding_dimension)

    def forward(self, inputs: Tensor) -> Tensor:
        if tuple(inputs.shape[-2:]) != self.image_size:
            raise ValueError("Spectral input size does not match the configured padded image size.")
        return self.norm(self.projection(inputs).flatten(2).transpose(1, 2))

