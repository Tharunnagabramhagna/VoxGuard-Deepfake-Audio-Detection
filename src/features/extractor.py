"""Unified feature extraction manager for VoxGuard.

Coordinates audio loading, preprocessing, Log-Mel spectrogram generation,
and MFCC summary extraction according to a project configuration dictionary.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

from src.data.preprocessor import load_audio_file, preprocess_audio
from src.features.mel_spectrogram import extract_log_mel_spectrogram
from src.features.mfcc import extract_mfcc_summary


class FeatureExtractor:
    """Config-driven feature extraction engine for VoxGuard."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature extractor with configuration parameters.

        Args:
            config: Full project config dictionary or 'features' sub-dictionary.
        """
        self.config = config or {}
        feat_cfg = self.config.get("features", self.config)
        audio_cfg = self.config.get("audio", {})

        # Audio parameters
        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.duration_sec = audio_cfg.get("duration_sec", 4.0)
        self.target_samples = audio_cfg.get("target_samples", 64000)

        # Log-Mel parameters
        mel_cfg = feat_cfg.get("mel", {})
        self.mel_n_mels = mel_cfg.get("n_mels", 64)
        self.mel_n_fft = mel_cfg.get("n_fft", 1024)
        self.mel_win_length = mel_cfg.get("win_length", 400)
        self.mel_hop_length = mel_cfg.get("hop_length", 160)
        self.mel_f_min = mel_cfg.get("f_min", 20.0)
        self.mel_f_max = mel_cfg.get("f_max", 8000.0)
        self.mel_power = mel_cfg.get("power", 2.0)
        self.mel_log_offset = mel_cfg.get("log_offset", 1e-6)

        # MFCC parameters
        mfcc_cfg = feat_cfg.get("mfcc", {})
        self.mfcc_n_mfcc = mfcc_cfg.get("n_mfcc", 40)
        self.mfcc_n_fft = mfcc_cfg.get("n_fft", 1024)
        self.mfcc_win_length = mfcc_cfg.get("win_length", 400)
        self.mfcc_hop_length = mfcc_cfg.get("hop_length", 160)
        self.mfcc_include_deltas = mfcc_cfg.get("include_deltas", False)

    def process_waveform(
        self, waveform: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, Dict]:
        """Preprocess raw audio to standardized 16 kHz mono normalized clip."""
        return preprocess_audio(waveform, sample_rate, self.config)

    def extract_mel(
        self, waveform: Union[np.ndarray, torch.Tensor], add_channel_dim: bool = True
    ) -> torch.Tensor:
        """Extract Log-Mel spectrogram tensor."""
        return extract_log_mel_spectrogram(
            waveform=waveform,
            sample_rate=self.sample_rate,
            n_mels=self.mel_n_mels,
            n_fft=self.mel_n_fft,
            win_length=self.mel_win_length,
            hop_length=self.mel_hop_length,
            f_min=self.mel_f_min,
            f_max=self.mel_f_max,
            power=self.mel_power,
            log_offset=self.mel_log_offset,
            add_channel_dim=add_channel_dim,
        )

    def extract_mfcc(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Extract 1D summary MFCC feature vector."""
        return extract_mfcc_summary(
            waveform=waveform,
            sample_rate=self.sample_rate,
            n_mfcc=self.mfcc_n_mfcc,
            n_mels=self.mel_n_mels,
            n_fft=self.mfcc_n_fft,
            win_length=self.mfcc_win_length,
            hop_length=self.mfcc_hop_length,
            f_min=self.mel_f_min,
            f_max=self.mel_f_max,
            include_deltas=self.mfcc_include_deltas,
        )

    def extract_from_file(
        self, file_path: str
    ) -> Dict[str, object]:
        """Load audio file, preprocess, and extract both representations.

        Args:
            file_path: Path to .wav or .flac audio file.

        Returns:
            Dictionary containing:
              - 'waveform': preprocessed audio waveform (64000,)
              - 'mel_spectrogram': PyTorch tensor (1, 64, 401)
              - 'mfcc_summary': NumPy vector (80,) or (160,)
              - 'meta': preprocessing and extraction metadata
        """
        raw_wave, sr, err = load_audio_file(file_path, target_sr=self.sample_rate)
        if raw_wave is None:
            raise RuntimeError(f"Failed to load audio {file_path}: {err}")

        clean_wave, prep_meta = self.process_waveform(raw_wave, sr)
        mel_spec = self.extract_mel(clean_wave, add_channel_dim=True)
        mfcc_feat, mfcc_meta = self.extract_mfcc(clean_wave)

        return {
            "waveform": clean_wave,
            "mel_spectrogram": mel_spec,
            "mfcc_summary": mfcc_feat,
            "preprocessing_meta": prep_meta,
            "mfcc_meta": mfcc_meta,
        }

