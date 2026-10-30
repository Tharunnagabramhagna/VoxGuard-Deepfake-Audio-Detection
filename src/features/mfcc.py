"""Mel-Frequency Cepstral Coefficients (MFCC) feature extraction for VoxGuard.

Extracts 40 MFCC coefficients from fixed-duration 16 kHz audio clips and computes
documented temporal summary statistics (mean, standard deviation, and optional deltas).

Summary Statistics Architecture (documented):
────────────────────────────────────────────────────────────────────────
1. Frame Extraction:
   - 40 MFCC coefficients extracted per frame via DCT-II applied to Log-Mel bands.
   - For 4.0s (64,000 samples) at 16 kHz with hop=160: 401 temporal frames.
   - Frame matrix shape: (40, 401).

2. Temporal Summary Statistics:
   - Statistic 1: Mean across time (40 features)
     μ_k = (1 / T) * Σ_{t=1}^T c_{k, t}
   - Statistic 2: Standard Deviation across time (40 features)
     σ_k = sqrt((1 / T) * Σ_{t=1}^T (c_{k, t} - μ_k)^2)

3. Optional First Derivatives (Deltas):
   - When include_deltas=True:
     - Delta Mean across time (40 features)
     - Delta Standard Deviation across time (40 features)

4. Output Dimensionality:
   - Static only (default): 40 (mean) + 40 (std) = 80 features.
   - Static + Deltas: 40 (mean) + 40 (std) + 40 (delta_mean) + 40 (delta_std) = 160 features.

All values are guaranteed to be finite float32 numbers.
────────────────────────────────────────────────────────────────────────
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.fftpack import dct
import torch

from src.features.mel_spectrogram import extract_log_mel_spectrogram


def compute_deltas(features: np.ndarray, width: int = 5) -> np.ndarray:
    """Compute first-order temporal derivatives (velocity) along time frames.

    Uses standard speech analysis linear regression formula:
        d_t = (Σ_{n=1}^N n * (c_{t+n} - c_{t-n})) / (2 * Σ_{n=1}^N n^2)
    where N = (width - 1) // 2.

    Args:
        features: Array of shape (n_features, time_frames).
        width: Analysis window width (odd integer >= 3, default: 5).

    Returns:
        Delta features array of identical shape (n_features, time_frames).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D array (n_features, time_frames), got {features.ndim}D")

    n_features, time_frames = features.shape
    if time_frames < width:
        # Fall back to simple differences for very short audio
        diff = np.diff(features, axis=1)
        pad = np.zeros((n_features, 1), dtype=features.dtype)
        return np.hstack([diff, pad])

    half = (width - 1) // 2
    denom = 2.0 * sum(n ** 2 for n in range(1, half + 1))
    padded = np.pad(features, ((0, 0), (half, half)), mode="edge")

    deltas = np.zeros_like(features)
    for n in range(1, half + 1):
        deltas += n * (padded[:, half + n : half + n + time_frames] -
                       padded[:, half - n : half - n + time_frames])

    deltas /= denom
    return deltas.astype(features.dtype)


def extract_mfcc_frames(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 64,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Extract raw frame-by-frame MFCC matrix from a waveform.

    Applies Discrete Cosine Transform (DCT Type-II with orthonormal basis)
    to the Log-Mel filterbank energies.

    Args:
        waveform: 1D audio waveform array or tensor.
        sample_rate: Sampling rate in Hz.
        n_mfcc: Number of cepstral coefficients to extract (default: 40).
        n_mels: Number of Mel filterbank bands (default: 64, must be >= n_mfcc).
        n_fft: FFT window size.
        win_length: Window size in samples.
        hop_length: Hop size in samples.
        f_min: Minimum frequency in Hz.
        f_max: Maximum frequency in Hz.

    Returns:
        NumPy array of shape (n_mfcc, time_frames) with dtype float32.
    """
    n_mels = max(n_mels, n_mfcc)

    # Extract Log-Mel spectrogram: shape (1, n_mels, time_frames)
    log_mel_tensor = extract_log_mel_spectrogram(
        waveform=waveform,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        add_channel_dim=False,
    )

    log_mel = log_mel_tensor.cpu().numpy()  # Shape: (n_mels, time_frames)

    # DCT-II along Mel bands (axis 0) with orthonormal normalization
    mfcc_all = dct(log_mel, type=2, axis=0, norm="ortho")

    # Retain the requested number of cepstral coefficients (0 through n_mfcc - 1)
    mfcc = mfcc_all[:n_mfcc, :]

    # Strict finite check
    if not np.isfinite(mfcc).all():
        mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=50.0, neginf=-50.0)

    return mfcc.astype(np.float32)


def extract_mfcc_summary(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 64,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    include_deltas: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Extract a 1D summary feature vector from a 16 kHz audio waveform.

    Computes temporal summary statistics across all 40 coefficients:
      - mean: 40 features
      - std:  40 features
      - (optional) delta_mean: 40 features
      - (optional) delta_std:  40 features

    Args:
        waveform: 1D audio waveform.
        sample_rate: Sampling rate in Hz.
        n_mfcc: Number of MFCC coefficients (default: 40).
        n_mels: Number of Mel bands used for intermediate representation (default: 64).
        n_fft: FFT window size.
        win_length: Window size in samples.
        hop_length: Hop size in samples.
        f_min: Minimum frequency in Hz.
        f_max: Maximum frequency in Hz.
        include_deltas: Whether to append first derivative (delta) statistics.

    Returns:
        Tuple of (feature_vector, metadata_dict):
          - feature_vector: 1D float32 array of shape (80,) or (160,).
          - metadata_dict: Dictionary with feature names and dimensions.
    """
    mfcc = extract_mfcc_frames(
        waveform=waveform,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
    )

    # 1. Static summary statistics
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)

    components = [means, stds]
    feature_names = [f"mfcc_{i}_mean" for i in range(n_mfcc)] + [
        f"mfcc_{i}_std" for i in range(n_mfcc)
    ]

    # 2. Optional delta statistics
    if include_deltas:
        deltas = compute_deltas(mfcc)
        d_means = np.mean(deltas, axis=1)
        d_stds = np.std(deltas, axis=1)
        components.extend([d_means, d_stds])
        feature_names += [f"mfcc_{i}_delta_mean" for i in range(n_mfcc)] + [
            f"mfcc_{i}_delta_std" for i in range(n_mfcc)
        ]

    feature_vector = np.concatenate(components).astype(np.float32)

    # Strict finite verification
    if not np.isfinite(feature_vector).all():
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=50.0, neginf=-50.0)

    meta = {
        "n_mfcc": n_mfcc,
        "n_features": len(feature_vector),
        "include_deltas": include_deltas,
        "feature_names": feature_names,
        "summary_statistics": ["mean", "std"] + (["delta_mean", "delta_std"] if include_deltas else []),
    }

    return feature_vector, meta

