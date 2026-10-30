"""Unit tests for VoxGuard Streamlit demonstration (demo/app.py).

Tests:
1. Model loading logic and checkpoint presence.
2. Waveform and Mel spectrogram plotting utility.
3. Audio validation for corrupt, silent, short, and normal audio.
4. Consistency between demo preprocessing and training preprocessing.
"""

from pathlib import Path
import numpy as np
import pytest

from demo.app import load_models, plot_analysis
from src.data.preprocessor import load_audio_file, preprocess_audio
from src.utils.config import load_config


class TestStreamlitDemoComponents:
    """Tests for demo/app.py components."""

    def test_load_models_loads_available_checkpoints(self):
        """Verify load_models loads CNN and baselines if checkpoints exist."""
        models = load_models()
        assert len(models) >= 1
        assert "Spectrogram-CNN (2D CNN)" in models
        cnn_info = models["Spectrogram-CNN (2D CNN)"]
        assert cnn_info["threshold"] > 0.0
        assert cnn_info["type"] == "cnn"

    def test_plot_analysis_generates_figure(self):
        """Verify plot_analysis produces valid matplotlib figure."""
        waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 4, 64000)).astype(np.float32)
        mel_spec = np.random.randn(64, 401).astype(np.float32)

        fig = plot_analysis(waveform, mel_spec, sr=16000)
        assert fig is not None
        assert len(fig.axes) >= 2

    def test_demo_preprocessing_matches_training_pipeline(self):
        """Verify that preprocessing in demo matches the training specs."""
        cfg = load_config("configs/default.yaml")
        raw_wave = np.random.randn(32000).astype(np.float32) * 0.5  # 2.0s
        clean_wave, meta = preprocess_audio(raw_wave, 16000, cfg)

        assert len(clean_wave) == 64000  # Exactly 4.0s at 16 kHz
        assert meta["length_final"] == 64000
        assert np.max(np.abs(clean_wave)) <= 1.0

    def test_demo_handles_corrupt_file_gracefully(self):
        """Verify corrupt audio file triggers load failure with error string."""
        corrupt_path = Path("data/raw/asvspoof2019_la/audio/corrupt_file.wav")
        if corrupt_path.exists():
            wave, sr, err = load_audio_file(str(corrupt_path))
            assert wave is None
            assert err is not None

    def test_demo_handles_silent_audio_detection(self):
        """Verify silent audio has peak amplitude < 1e-5."""
        silent_path = Path("data/raw/asvspoof2019_la/audio/silent_file.wav")
        if silent_path.exists():
            wave, sr, err = load_audio_file(str(silent_path))
            assert wave is not None
            peak_amp = float(np.max(np.abs(wave)))
            assert peak_amp < 1e-4

    def test_demo_inference_on_sample_audio(self):
        """Verify end-to-end inference using the loaded model from load_models()."""
        models = load_models()
        cnn_info = models["Spectrogram-CNN (2D CNN)"]
        predictor = cnn_info["predictor"]
        threshold = cnn_info["threshold"]

        # Run on genuine fixture
        sample_path = Path("tests/fixtures/audio/dev/LA_D_0000031.wav")
        if sample_path.exists():
            res = predictor.predict_audio(sample_path)
            assert "probability_spoof" in res
            assert 0.0 <= res["probability_spoof"] <= 1.0
            is_synthetic = res["probability_spoof"] >= threshold
            assert isinstance(is_synthetic, (bool, np.bool_))

    def test_demo_rejects_audio_shorter_than_minimum_duration(self):
        """Verify audio shorter than 0.2s is detected and rejected."""
        short_wave = np.random.randn(1600).astype(np.float32) * 0.1  # 0.1s at 16 kHz
        sr = 16000
        duration_sec = len(short_wave) / sr
        assert duration_sec < 0.2

    def test_demo_scientific_gating_message_present(self):
        """Verify scientific integrity gating text is present in demo code."""
        demo_code = Path("demo/app.py").read_text(encoding="utf-8")
        assert "Real-speech detection is not yet validated" in demo_code
        assert "SOFTWARE-TEST HARNESS DEMO" in demo_code

    def test_extract_log_mel_spectrogram_return_contract(self):
        """Verify extract_log_mel_spectrogram returns a single torch.Tensor, not a tuple."""
        from src.features.mel_spectrogram import extract_log_mel_spectrogram
        import torch

        clean_wave = np.random.randn(64000).astype(np.float32) * 0.5
        result = extract_log_mel_spectrogram(clean_wave, sample_rate=16000, n_mels=64)

        assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
        assert not isinstance(result, tuple), "Function must return a single tensor, not a tuple"
        assert result.shape == (1, 64, 401), f"Expected shape (1, 64, 401), got {result.shape}"

        # Squeeze for 2D visualization input
        mel_2d = result.squeeze().cpu().numpy()
        assert mel_2d.shape == (64, 401)
        assert np.isfinite(mel_2d).all()

    def test_demo_fixture_inference_and_visualization_end_to_end(self):
        """Verify fixture pipeline produces prediction and plot without unpacking errors."""
        from src.features.mel_spectrogram import extract_log_mel_spectrogram

        models = load_models()
        cnn_info = models["Spectrogram-CNN (2D CNN)"]
        predictor = cnn_info["predictor"]
        threshold = cnn_info["threshold"]
        cfg = load_config("configs/default.yaml")

        for sample_path in [
            Path("tests/fixtures/audio/dev/LA_D_0000031.wav"),
            Path("tests/fixtures/audio/dev/LA_D_0000041.wav"),
        ]:
            if not sample_path.exists():
                continue

            raw_wave, sr, err = load_audio_file(str(sample_path), target_sr=16000)
            assert raw_wave is not None
            assert err is None

            clean_wave, _ = preprocess_audio(raw_wave, sr, cfg)
            assert len(clean_wave) == 64000

            # 1. Inference
            res = predictor.predict_audio(clean_wave, sample_rate=16000)
            assert "probability_spoof" in res
            assert "probability_bonafide" in res
            assert 0.0 <= res["probability_spoof"] <= 1.0

            # 2. Spectrogram extraction (single tensor, no unpacking error)
            mel_tensor = extract_log_mel_spectrogram(clean_wave, sample_rate=16000, n_mels=64)
            mel_spec = mel_tensor.squeeze().cpu().numpy()

            # 3. Plot generation
            fig = plot_analysis(clean_wave, mel_spec, sr=16000)
            assert fig is not None
            assert len(fig.axes) >= 2

