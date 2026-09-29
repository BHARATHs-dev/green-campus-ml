"""
Satellite CNN architecture for Above-Ground Biomass Density (AGBD) regression.

Supports:
1. Research-oriented architecture (Proposed):
   15-channel Sentinel-2 feature input
   -> Initial convolution (Conv2D 15->32, BatchNorm, ReLU)
   -> Residual CNN blocks:
      - ResidualBlock 32 (32->32, stride 1)
      - ResidualBlock 64 (32->64, stride 2)
      - ResidualBlock 128 (64->128, stride 2)
   -> Spatial Attention Module (CBAM-style lightweight spatial attention)
   -> Adaptive Global Average Pooling
   -> Flatten
   -> Dropout
   -> Linear (128 -> hidden)
   -> ReLU
   -> Dropout
   -> Linear (hidden -> 1)
   -> AGBD (Mg/ha)

2. Baseline architecture:
   Conv32 -> Conv64 -> Conv128 -> AdaptiveAvgPool -> FC256 -> FC1 -> AGBD

Target: AGBD in Mg/ha
Carbon: AGBD * 0.47 (Mg C/ha)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Reusable Residual Block for satellite feature extraction.
    Consists of two 3x3 convolutions with BatchNorm, ReLU activation,
    and a shortcut projection if spatial dimensions or channel counts change.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    """
    Lightweight Spatial Attention Module.
    Aggregates channel information via average and max pooling,
    then applies a 7x7 convolution, batch norm, and sigmoid activation
    to produce a spatial attention map that weights canopy features.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        scale = self.bn(scale)
        scale = self.sigmoid(scale)
        return x * scale


class ResidualAttentionSatelliteCNN(nn.Module):
    """
    Research-oriented Satellite CNN architecture with Residual Blocks
    and Spatial Attention for Above-Ground Biomass Density (AGBD) regression.

    Accepts:
        [batch_size, 15, height, width]
    Returns:
        [batch_size, 1] (AGBD continuous value in Mg/ha)
    """

    def __init__(
        self,
        in_channels: int = 15,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        self.in_channels = in_channels
        self.hidden = hidden
        self.dropout_rate = dropout

        # Initial convolution: 15 -> 32
        self.init_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Residual CNN blocks
        self.res1 = ResidualBlock(32, 32, stride=1)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 128, stride=2)

        # Spatial Attention Module
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Adaptive Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten & Dropout
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=dropout)

        # Fully Connected Regression Head
        self.fc1 = nn.Linear(128, hidden)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial Conv -> BN -> ReLU
        x = self.relu(self.init_bn(self.init_conv(x)))

        # Residual Blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # Spatial Attention
        x = self.spatial_attention(x)

        # Global Average Pooling & Flatten
        x = self.gap(x)
        x = self.flatten(x)

        # Regression Head
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class BaselineSatelliteCNN(nn.Module):
    """
    Baseline Satellite CNN architecture for AGBD regression.
    Original simple feed-forward CNN (Conv32 -> Conv64 -> Conv128 -> GAP -> FC256 -> FC1).
    Kept for benchmarking, ablation, and backward compatibility.
    """

    def __init__(self, in_channels: int = 15, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        self.in_channels = in_channels
        self.hidden = hidden
        self.dropout_rate = dropout

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # Regression head
        self.fc1 = nn.Linear(128, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Alias SatelliteCNN to the proposed architecture
SatelliteCNN = ResidualAttentionSatelliteCNN


def build_satellite_cnn(
    architecture: Optional[str] = "Residual CNN + Spatial Attention",
    in_channels: int = 15,
    hidden: int = 256,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to construct the requested satellite CNN architecture.

    Supported architectures:
      - "Residual CNN + Spatial Attention" (Proposed research model)
      - "SatelliteCNN" / "Baseline" / legacy architecture string
    """
    arch = (architecture or "").strip()
    is_baseline = (
        "baseline" in arch.lower()
        or arch.startswith("SatelliteCNN(Conv32")
    )

    if is_baseline:
        return BaselineSatelliteCNN(
            in_channels=in_channels, hidden=hidden, dropout=dropout, **kwargs
        )
    return ResidualAttentionSatelliteCNN(
        in_channels=in_channels, hidden=hidden, dropout=dropout, **kwargs
    )


def load_satellite_checkpoint(
    checkpoint_path: Union[str, Path],
    metadata: Optional[Union[Dict[str, Any], str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Reconstruct and load a trained satellite CNN model from checkpoint and metadata.
    Automatically determines whether to instantiate ResidualAttentionSatelliteCNN
    or BaselineSatelliteCNN based on metadata['architecture'].
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    meta: Dict[str, Any] = {}
    if isinstance(metadata, (str, Path)):
        meta_path = Path(metadata)
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
    elif isinstance(metadata, dict):
        meta = metadata
    else:
        sibling_meta = checkpoint_path.with_name(f"{checkpoint_path.stem}_metadata.json")
        if sibling_meta.exists():
            with open(sibling_meta, "r") as f:
                meta = json.load(f)

    in_channels = meta.get("input_channels", meta.get("in_channels", 15))
    arch = meta.get("architecture", "Residual CNN + Spatial Attention")

    model = build_satellite_cnn(architecture=arch, in_channels=in_channels)
    map_location = device if device is not None else torch.device("cpu")
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    model.eval()
    return model, meta
