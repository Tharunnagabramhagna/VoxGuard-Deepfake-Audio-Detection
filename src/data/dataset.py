"""PyTorch Dataset definitions for VoxGuard deep learning models."""

from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Dataset for 2D Log-Mel spectrogram tensors and classification labels."""

    def __init__(
        self,
        spectrograms: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
    ):
        """Initialize dataset.

        Args:
            spectrograms: Tensor or array of shape (N, 1, n_mels, time_frames) or (N, n_mels, time_frames).
            labels: 1D array or tensor of integer labels {0: bonafide, 1: spoof}.
        """
        if isinstance(spectrograms, np.ndarray):
            spectrograms = torch.from_numpy(spectrograms)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        self.spectrograms = spectrograms.to(dtype=torch.float32)
        if self.spectrograms.dim() == 3:
            # Add channel dimension: (N, 1, H, W)
            self.spectrograms = self.spectrograms.unsqueeze(1)

        self.labels = labels.to(dtype=torch.int64)

        if len(self.spectrograms) != len(self.labels):
            raise ValueError(
                f"Mismatch: {len(self.spectrograms)} spectrograms vs {len(self.labels)} labels."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectrograms[idx], self.labels[idx]

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for cross-entropy loss.

        Formula:
            w_c = total_samples / (num_classes * count_c)

        Returns:
            Tensor of shape (2,) with weights for class 0 (bonafide) and class 1 (spoof).
        """
        counts = torch.bincount(self.labels, minlength=2).float()
        total = float(len(self.labels))
        num_classes = 2.0
        weights = total / (num_classes * torch.clamp(counts, min=1.0))
        return weights.to(dtype=torch.float32)

