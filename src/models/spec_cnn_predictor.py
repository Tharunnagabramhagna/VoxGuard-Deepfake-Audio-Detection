"""Inference predictor for VoxGuard SpectrogramCNN.

Implements the single prediction interface for 2D CNN models:
  1. Loads model weights and embedded preprocessing config from .pt checkpoint.
  2. Applies standardized audio preprocessing (16 kHz mono, crop/pad, peak normalization).
  3. Extracts Log-Mel spectrogram.
  4. Computes calibrated probabilities via softmax.
  5. Returns standardized forensic prediction dictionary.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from src.data.preprocessor import load_audio_file, preprocess_audio
from src.features.mel_spectrogram import extract_log_mel_spectrogram
from src.models.spec_cnn import SpectrogramCNN


class SpecCNNPredictor:
    """Forensic prediction engine for trained SpectrogramCNN models."""

    LABEL_NAMES: Dict[int, str] = {0: "bonafide", 1: "spoof"}

    def __init__(
        self,
        model: SpectrogramCNN,
        model_name: str = "spec_cnn",
        preprocessing_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize predictor.

        Args:
            model: Trained SpectrogramCNN instance.
            model_name: Model identifier name.
            preprocessing_config: Preprocessing parameters dictionary.
            device: Computation device.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.model_name = model_name
        self.preprocessing_config = preprocessing_config or {}

        # Audio parameters
        audio_cfg = self.preprocessing_config.get("audio", {})
        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.target_samples = audio_cfg.get("target_samples", 64000)

        # Mel spectrogram parameters
        mel_cfg = self.preprocessing_config.get("features", {}).get("mel", {})
        self.n_mels = mel_cfg.get("n_mels", 64)
        self.n_fft = mel_cfg.get("n_fft", 1024)
        self.win_length = mel_cfg.get("win_length", 400)
        self.hop_length = mel_cfg.get("hop_length", 160)
        self.f_min = mel_cfg.get("f_min", 20.0)
        self.f_max = mel_cfg.get("f_max", 8000.0)
        self.log_offset = mel_cfg.get("log_offset", 1e-6)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str = "spec_cnn",
        device: Optional[torch.device] = None,
    ) -> "SpecCNNPredictor":
        """Load predictor from a saved .pt checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            model_name: Name for predictor.
            device: Computation device.

        Returns:
            Configured SpecCNNPredictor instance.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path.resolve()}")

        ckpt = torch.load(str(path), map_location="cpu")
        model_cfg = ckpt.get("model_config", {})
        prep_cfg = ckpt.get("preprocessing_config", {})

        # Reconstruct model architecture
        model = SpectrogramCNN(
            in_channels=model_cfg.get("in_channels", 1),
            conv_channels=model_cfg.get("conv_channels", [32, 64, 128, 256]),
            dense_dim=model_cfg.get("dense_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
            num_classes=model_cfg.get("num_classes", 2),
        )
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(
            model=model,
            model_name=model_name,
            preprocessing_config=prep_cfg,
            device=device,
        )

    @torch.no_grad()
    def predict_spectrogram(self, spec_tensor: torch.Tensor) -> Dict[str, Any]:
        """Predict on a pre-extracted Log-Mel spectrogram tensor.

        Args:
            spec_tensor: Tensor of shape (1, 1, n_mels, time_frames) or (1, n_mels, time_frames).

        Returns:
            Standardized forensic prediction dictionary.
        """
        if spec_tensor.dim() == 2:
            spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)
        elif spec_tensor.dim() == 3:
            spec_tensor = spec_tensor.unsqueeze(0)

        spec_tensor = spec_tensor.to(self.device)
        probs = self.model.predict_proba(spec_tensor)  # Shape: (1, 2)

        prob_bonafide = float(probs[0, 0].item())
        prob_spoof = float(probs[0, 1].item())
        pred_class = 1 if prob_spoof >= 0.5 else 0
        confidence = max(prob_bonafide, prob_spoof)

        return {
            "prediction": pred_class,
            "label": self.LABEL_NAMES.get(pred_class, "unknown"),
            "probability_spoof": prob_spoof,
            "probability_bonafide": prob_bonafide,
            "confidence": confidence,
            "model_name": self.model_name,
            "spectrogram_shape": tuple(spec_tensor.shape),
        }

    @torch.no_grad()
    def predict_spectrogram_proba(self, spec_tensor: torch.Tensor) -> np.ndarray:
        """Predict class probabilities on a batch of spectrograms.

        Args:
            spec_tensor: Tensor of shape (B, 1, F, T) or (B, F, T).

        Returns:
            Numpy array of shape (B, 2) with probabilities [P(bonafide), P(spoof)].
        """
        if spec_tensor.dim() == 3:
            spec_tensor = spec_tensor.unsqueeze(1)
        spec_tensor = spec_tensor.to(self.device)
        probs = self.model.predict_proba(spec_tensor)
        return probs.cpu().numpy()

    def predict_audio(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full end-to-end prediction from audio file or waveform.

        Steps:
          1. Load audio if string path is passed.
          2. Standardized preprocessing (16 kHz mono, crop/pad, normalize).
          3. Extract Log-Mel spectrogram.
          4. Predict probabilities.

        Args:
            audio_input: File path or 1D waveform array.
            sample_rate: Sampling rate if waveform array is passed.

        Returns:
            Standardized prediction dictionary.
        """
        if isinstance(audio_input, (str, Path)):
            raw_wave, sr, err = load_audio_file(str(audio_input), target_sr=self.sample_rate)
            if raw_wave is None:
                raise RuntimeError(f"Failed to load audio: {err}")
        else:
            raw_wave = np.asarray(audio_input, dtype=np.float32)
            sr = sample_rate or self.sample_rate

        clean_wave, _ = preprocess_audio(raw_wave, sr, self.preprocessing_config)

        mel_spec = extract_log_mel_spectrogram(
            waveform=clean_wave,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            log_offset=self.log_offset,
            add_channel_dim=True,
        )

        return self.predict_spectrogram(mel_spec)

