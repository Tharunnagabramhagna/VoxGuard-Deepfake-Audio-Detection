"""2D Convolutional Neural Network for Log-Mel Spectrogram Deepfake Audio Detection.

Architecture:
─────────────
Input: Log-Mel spectrogram tensor of shape (batch, 1, n_mels=64, time_frames=401).

Layer Breakdown:
  - Block 1: Conv2d(1 -> 32, k=3, p=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2, 2)
    Shape: (B, 32, 32, 200)
  - Block 2: Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2, 2)
    Shape: (B, 64, 16, 100)
  - Block 3: Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(2, 2)
    Shape: (B, 128, 8, 50)
  - Block 4: Conv2d(128 -> 256, k=3, p=1) -> BatchNorm2d(256) -> ReLU -> MaxPool2d(2, 2)
    Shape: (B, 256, 4, 25)

Global Aggregation & Classifier Head:
  - AdaptiveAvgPool2d((1, 1)) -> Shape: (B, 256, 1, 1) -> Flatten -> (B, 256)
  - Linear(256 -> dense_dim=128) -> ReLU -> Dropout(p=0.3)
  - Linear(128 -> num_classes=2) -> Binary prediction logits: [logit_bonafide, logit_spoof]

Predictions:
  - Softmax(logits, dim=1) -> [P(bonafide), P(spoof)]
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard 2D Convolutional block: Conv2d + BatchNorm2d + ReLU + MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class SpectrogramCNN(nn.Module):
    """2D Convolutional Neural Network for Log-Mel Spectrogram deepfake audio detection."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: Optional[List[int]] = None,
        dense_dim: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        """Initialize SpectrogramCNN.

        Args:
            in_channels: Number of input audio channels (default: 1 for mono).
            conv_channels: Channel depths for conv blocks (default: [32, 64, 128, 256]).
            dense_dim: Hidden dimension of fully connected projection head.
            dropout: Dropout probability in classification head.
            num_classes: Number of target classes (2: 0=bonafide, 1=spoof).
        """
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.dense_dim = dense_dim
        self.dropout_rate = dropout
        self.num_classes = num_classes

        # Build convolutional backbone
        blocks = []
        curr_in = in_channels
        for c_out in conv_channels:
            blocks.append(ConvBlock(in_channels=curr_in, out_channels=c_out))
            curr_in = c_out
        self.backbone = nn.Sequential(*blocks)

        # Global spatial aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification projection head
        final_conv_dim = conv_channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_conv_dim, dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Spectrogram tensor of shape (batch, in_channels, n_mels, time_frames).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Ensure 4D tensor (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Feature maps extraction
        features = self.backbone(x)  # Shape: (B, conv_channels[-1], H', W')

        # Global spatial pooling
        pooled = self.global_pool(features)  # Shape: (B, conv_channels[-1], 1, 1)
        flat = torch.flatten(pooled, start_dim=1)  # Shape: (B, conv_channels[-1])

        # Prediction logits
        logits = self.classifier(flat)  # Shape: (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated probabilities across classes.

        Args:
            x: Spectrogram tensor.

        Returns:
            Probability tensor of shape (batch, 2) where col 0 is P(bonafide) and col 1 is P(spoof).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "model_type": "SpectrogramCNN",
            "in_channels": self.in_channels,
            "conv_channels": self.conv_channels,
            "dense_dim": self.dense_dim,
            "dropout": self.dropout_rate,
            "num_classes": self.num_classes,
        }

