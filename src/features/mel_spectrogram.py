"""Log-Mel spectrogram feature extraction for VoxGuard.

Extracts calibrated 2D Log-Mel spectrograms from preprocessed 16 kHz mono waveforms.
Produces consistent, finite PyTorch tensors suitable for 2D convolutional networks.

Acoustic Parameters (documented):
────────────────────────────────────────────────────────────────────────
- Sample rate: 16,000 Hz (standard ASVspoof 2019 sampling rate)
- Mel bins: Configurable (default 64 bins, spanning 20 Hz to 8,000 Hz)
- FFT size: 1024 points (frequency resolution: 16000 / 1024 ≈ 15.625 Hz/bin)
- Window length: 400 samples (25 ms analysis window with Hann taper)
- Hop length: 160 samples (10 ms frame shift)
- Target duration: 4.0 seconds (64,000 samples) → 401 time frames
- Energy power: 2.0 (power spectrogram: |STFT|^2)
- Compression: 10 * log10(power + log_offset)
- Output shape: (1, n_mels, time_frames) = (1, 64, 401)
────────────────────────────────────────────────────────────────────────
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a triangular Mel-scale filterbank matrix.

    Converts linear frequency bins [0, n_fft // 2] to Mel scale.
    Mel conversion uses standard O'Shaughnessy formula:
        m = 2595 * log10(1 + f / 700)
        f = 700 * (10^(m / 2595) - 1)

    Args:
        sample_rate: Audio sampling rate in Hz.
        n_fft: FFT window length.
        n_mels: Number of Mel frequency bands (initially 64).
        f_min: Minimum frequency in Hz.
        f_max: Maximum frequency in Hz (typically Nyquist = sample_rate / 2).
        device: PyTorch device on which to construct the filterbank.

    Returns:
        Tensor of shape (n_mels, n_fft // 2 + 1) with area-normalized triangular weights.
    """
    n_freqs = n_fft // 2 + 1
    # Nyquist bound
    f_max = min(f_max, sample_rate / 2.0)

    # Convert Hz to Mel
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)

    # Linearly spaced points in Mel scale (n_mels + 2 points for n_mels triangular filters)
    m_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    # Convert back to Hz
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    # Map Hz to FFT bin indices
    bins = np.floor((n_fft + 1) * f_pts / sample_rate).astype(np.int64)
    bins = np.clip(bins, 0, n_freqs - 1)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bins[m - 1]
        f_center = bins[m]
        f_right = bins[m + 1]

        # Up-slope
        if f_center > f_left:
            weights[m - 1, f_left:f_center] = (
                np.arange(f_left, f_center) - f_left
            ) / (f_center - f_left)
        # Down-slope
        if f_right > f_center:
            weights[m - 1, f_center:f_right] = (
                f_right - np.arange(f_center, f_right)
            ) / (f_right - f_center)

        # Peak at center
        weights[m - 1, f_center] = 1.0

        # Area normalization: divide by band width to normalize energy
        enorm = 2.0 / (f_pts[m + 1] - f_pts[m - 1] + 1e-8)
        weights[m - 1] *= enorm

    filterbank = torch.from_numpy(weights).to(device=device)
    return filterbank


def extract_log_mel_spectrogram(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    power: float = 2.0,
    log_offset: float = 1e-6,
    add_channel_dim: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Extract a Log-Mel spectrogram tensor from a 1D audio waveform.

    Steps:
        1. Convert waveform to PyTorch float32 tensor.
        2. Compute Short-Time Fourier Transform (STFT) with Hann window.
        3. Compute power spectrum (|STFT|^power).
        4. Apply triangular Mel filterbank matrix multiplication.
        5. Logarithmic compression: 10 * log10(MelPower + log_offset).
        6. Verify all output values are finite (no NaN, no Inf).

    Args:
        waveform: 1D audio array or tensor of shape (samples,).
        sample_rate: Sampling rate in Hz (expected 16,000 Hz).
        n_mels: Number of Mel frequency bins (default: 64).
        n_fft: FFT window size (default: 1024).
        win_length: Window size in samples (default: 400 = 25 ms at 16 kHz).
        hop_length: Hop size in samples (default: 160 = 10 ms at 16 kHz).
        f_min: Minimum frequency (default: 20 Hz).
        f_max: Maximum frequency (default: 8000 Hz).
        power: Power exponent (default: 2.0 for power spectrogram).
        log_offset: Small epsilon to prevent log(0) (default: 1e-6).
        add_channel_dim: If True, returns shape (1, n_mels, time_frames).
        device: Computation device.

    Returns:
        PyTorch Tensor of shape (1, n_mels, time_frames) or (n_mels, time_frames).

    Raises:
        ValueError: If waveform is empty or has invalid dimensionality.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    waveform = waveform.to(dtype=torch.float32, device=device)

    # Flatten if mono with extra batch/channel dims
    while waveform.dim() > 1:
        if waveform.size(0) == 1:
            waveform = waveform.squeeze(0)
        else:
            # Average multi-channel to mono
            waveform = waveform.mean(dim=0)

    if waveform.numel() == 0:
        raise ValueError("Cannot extract spectrogram from empty waveform.")

    # Hann analysis window
    window = torch.hann_window(win_length, device=waveform.device)

    # Compute complex STFT: shape (n_fft // 2 + 1, time_frames)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Power spectrogram
    power_spec = torch.abs(stft) ** power

    # Mel filterbank: shape (n_mels, n_fft // 2 + 1)
    filterbank = create_mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        device=waveform.device,
    )

    # Apply filterbank: (n_mels, n_freqs) @ (n_freqs, time_frames) -> (n_mels, time_frames)
    mel_spec = torch.matmul(filterbank, power_spec)

    # Log compression in decibels
    log_mel = 10.0 * torch.log10(mel_spec + log_offset)

    # Strict finite verification
    if not torch.isfinite(log_mel).all():
        log_mel = torch.nan_to_num(log_mel, nan=np.log10(log_offset) * 10.0,
                                   posinf=80.0, neginf=-80.0)

    if add_channel_dim:
        log_mel = log_mel.unsqueeze(0)  # Shape: (1, n_mels, time_frames)

    return log_mel

