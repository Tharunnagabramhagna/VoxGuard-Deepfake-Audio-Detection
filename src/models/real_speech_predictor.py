"""Predictor wrapper for fine-tuned real-speech deepfake detector.

Encapsulates the ResNet-18 + BiGRU + Multi-Head Attention deepfake detector
adapted on MLAAD-tiny. Incorporates:
1. Multi-window temporal aggregation across long utterances (avoids harsh crop artifacts).
2. Biological vocal tract periodicity verification (distinguishes human mimicry/exaggeration from neural vocoder artifacts).
3. Calibrated decision threshold (0.58) with an uncertainty band for vocal modulation.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from scripts.test_pretrained_detector import (
    PretrainedDeepfakeDetector,
    load_pretrained_detector,
)
from src.features.mel_spectrogram import extract_log_mel_spectrogram


def compute_vocal_harmonicity(wave: np.ndarray, sr: int = 16000) -> float:
    """Compute maximum normalized autocorrelation in biological pitch range (50-500 Hz).

    Human vocal cords naturally vibrate with periodic fundamental frequency (F0),
    yielding high harmonicity (R_max >= 0.35) even during vocal mimicry, funny voices,
    or comical impressions. Neural vocoders and phase-scrambled synthetic speech
    often exhibit disrupted glottal periodicity and phase anomalies.
    """
    if len(wave) < 640:
        return 0.5

    # Analyze central active speech portion
    n_samples = min(len(wave), int(2.0 * sr))
    center_start = (len(wave) - n_samples) // 2
    seg = wave[center_start : center_start + n_samples]

    corr = np.correlate(seg, seg, mode="full")
    corr = corr[len(corr) // 2 :]
    norm = corr[0]
    if norm <= 1e-8:
        return 0.0
    corr = corr / norm

    # Pitch search window: 50 Hz to 500 Hz (32 to 320 samples at 16 kHz)
    min_lag = int(sr / 500)
    max_lag = int(sr / 50)
    if max_lag < len(corr):
        return float(np.max(corr[min_lag:max_lag]))
    return 0.0


class RealSpeechPredictor:
    """Predictor for genuine audio deepfake detection using ResNet-BiGRU architecture."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        threshold: float = 0.58,
        model_name: str = "ResNet-BiGRU Detector (MLAAD Fine-Tuned)",
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.threshold = threshold
        self.model_name = model_name
        self.model = load_pretrained_detector(self.checkpoint_path, device=device)
        self.model.eval()

    def predict_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Run deepfake audio prediction on normalized 16 kHz waveform.

        Supports arbitrary duration audio via multi-window robust temporal pooling
        and biological vocal tract periodicity verification to safely handle
        human mimicry, character acting, and expressive voice modulations.

        Args:
            waveform: 1D normalized float32 waveform.
            sample_rate: Audio sampling rate (must be 16000).

        Returns:
            Dictionary with probability_spoof, raw_logit, is_spoof, and diagnostics.
        """
        t0 = time.perf_counter()

        # 1. Biological vocal tract harmonicity
        harmonicity = compute_vocal_harmonicity(waveform, sample_rate)

        # 2. Multi-window temporal evaluation
        window_size = int(4.0 * sample_rate)  # 64,000 samples (4.0s)
        step_size = int(2.0 * sample_rate)    # 32,000 samples (50% overlap)

        window_probs: List[float] = []

        if len(waveform) > window_size:
            # Multi-window sliding evaluation
            for start in range(0, len(waveform) - window_size + 1, step_size):
                chunk = waveform[start : start + window_size]
                mel_chunk = extract_log_mel_spectrogram(
                    chunk,
                    sample_rate=sample_rate,
                    n_mels=128,
                    n_fft=1024,
                    win_length=1024,
                    hop_length=512,
                    add_channel_dim=True,
                ).unsqueeze(0)
                with torch.no_grad():
                    l_chunk = self.model(mel_chunk.to(self.device)).item()
                    window_probs.append(float(torch.sigmoid(torch.tensor(l_chunk)).item()))

            # Evaluate full uncropped temporal representation
            mel_full = extract_log_mel_spectrogram(
                waveform,
                sample_rate=sample_rate,
                n_mels=128,
                n_fft=1024,
                win_length=1024,
                hop_length=512,
                add_channel_dim=True,
            ).unsqueeze(0)
            with torch.no_grad():
                l_full = self.model(mel_full.to(self.device)).item()
                window_probs.append(float(torch.sigmoid(torch.tensor(l_full)).item()))

            # Robust aggregation across temporal sequence
            raw_prob = float(np.median(window_probs))
            raw_logit = float(np.log(max(raw_prob, 1e-6) / max(1.0 - raw_prob, 1e-6)))
        else:
            # Single window (padded or exact)
            mel = extract_log_mel_spectrogram(
                waveform,
                sample_rate=sample_rate,
                n_mels=128,
                n_fft=1024,
                win_length=1024,
                hop_length=512,
                add_channel_dim=True,
            ).unsqueeze(0)
            with torch.no_grad():
                raw_logit = self.model(mel.to(self.device)).item()
                raw_prob = float(torch.sigmoid(torch.tensor(raw_logit)).item())
            window_probs = [raw_prob]

        # 3. Human Mimicry & Vocal Modulation Verification
        # If score is in the borderline region (0.45 <= p <= 0.62) AND strong human glottal
        # periodicity is verified (harmonicity >= 0.35), this is physical human vocal modulation.
        is_mimicry_detected = False
        final_prob = raw_prob

        if 0.45 <= raw_prob <= 0.62 and harmonicity >= 0.35:
            is_mimicry_detected = True
            # Calibrate score to reflect biological human evidence
            final_prob = min(raw_prob * 0.75, 0.42)

        is_spoof = final_prob >= self.threshold
        infer_ms = (time.perf_counter() - t0) * 1000

        return {
            "is_spoof": is_spoof,
            "probability_spoof": final_prob,
            "raw_probability": raw_prob,
            "raw_logit": raw_logit,
            "threshold": self.threshold,
            "harmonicity": round(harmonicity, 4),
            "is_mimicry_detected": is_mimicry_detected,
            "window_count": len(window_probs),
            "window_probabilities": [round(p, 4) for p in window_probs],
            "inference_ms": infer_ms,
            "model_name": self.model_name,
        }
