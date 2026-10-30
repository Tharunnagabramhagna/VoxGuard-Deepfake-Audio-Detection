"""Controlled acoustic noise injection for robustness stress-testing.

Implements Additive White Gaussian Noise (AWGN) at strictly calibrated
Signal-to-Noise Ratios (SNRs) in decibels (dB):
    SNR_dB = 10 * log10(P_signal / P_noise)

Documented benchmark SNR conditions:
    - Clean (No noise, original baseline)
    - 30 dB SNR: Very light room/environmental background noise
    - 20 dB SNR: Moderate office/indoor ambient noise
    - 10 dB SNR: Heavy noise / street ambiance
    - 0 dB SNR: Extreme adverse conditions (signal power = noise power)
"""

from typing import Optional, Tuple
import numpy as np


def apply_awgn(
    waveform: np.ndarray,
    snr_db: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Inject Additive White Gaussian Noise (AWGN) at a specified SNR.

    Args:
        waveform: 1D float32 audio waveform.
        snr_db: Target Signal-to-Noise Ratio in decibels.
        seed: Random seed for reproducible noise generation.

    Returns:
        Noisy waveform array with identical shape and dtype float32,
        clipped to [-1.0, 1.0].
    """
    if len(waveform) == 0:
        return waveform.copy()

    rng = np.random.RandomState(seed)

    signal_power = float(np.mean(waveform ** 2))

    # Guard against pure silence
    if signal_power <= 1e-12:
        return waveform.copy()

    # Calculate required noise power: P_n = P_s / 10^(SNR/10)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    noise = rng.normal(loc=0.0, scale=noise_std, size=waveform.shape).astype(np.float32)
    noisy_waveform = waveform.astype(np.float32) + noise

    # Clip to valid audio bounds [-1.0, 1.0]
    return np.clip(noisy_waveform, -1.0, 1.0).astype(np.float32)


def measure_snr_db(
    clean_waveform: np.ndarray,
    noisy_waveform: np.ndarray,
) -> float:
    """Compute empirical SNR in dB between clean reference and noisy audio.

    Args:
        clean_waveform: Original clean audio.
        noisy_waveform: Degraded/perturbed audio.

    Returns:
        Empirical SNR in decibels.
    """
    clean = clean_waveform.astype(np.float64)
    noise = (noisy_waveform - clean_waveform).astype(np.float64)

    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)

    if p_noise <= 1e-15:
        return float("inf")
    if p_clean <= 1e-15:
        return float("-inf")

    return float(10.0 * np.log10(p_clean / p_noise))

