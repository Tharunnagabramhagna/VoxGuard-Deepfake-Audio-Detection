"""Unit tests for expanded audio format support, simplified UI logic, and scientific gating.

Tests cover:
1. WAV upload and decoding.
2. FLAC upload and decoding (with backend availability check).
3. MP3 upload and decoding (with backend availability check).
4. M4A upload and decoding (with backend availability check).
5. Unsupported format rejection (no silent interpretation).
6. Corrupt, empty, and silent audio handling.
7. Audio preprocessing consistency (16 kHz, mono, 4.0s normalization).
8. Scientific safety gating preventing fabricated real-speech predictions.
9. Confidence integrity (never labeled as accuracy; not fabricated for unvalidated models).
10. Evidence-based explanation generation based on measurable acoustic features.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from scipy.io import wavfile

from src.data.preprocessor import (
    SUPPORTED_AUDIO_EXTENSIONS,
    check_audio_decoders,
    load_audio_file,
    preprocess_audio,
)
from src.analysis.explanation import (
    compute_measurable_acoustic_features,
    generate_result_summary,
)
from src.utils.config import load_config
from demo.app import get_audio_mime_type


class TestAudioFormatSupport:
    """Tests for multi-format audio loading, decoding, and fallback handling."""

    def test_supported_extensions_set(self):
        """Verify the set of officially recognized formats."""
        assert ".wav" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".flac" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".mp3" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".m4a" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".ogg" not in SUPPORTED_AUDIO_EXTENSIONS

    def test_audio_mime_type_mapping(self):
        """Verify MIME type resolution for browser audio playback."""
        assert get_audio_mime_type("test.wav") == "audio/wav"
        assert get_audio_mime_type("test.flac") == "audio/flac"
        assert get_audio_mime_type("recording.mp3") == "audio/mpeg"
        assert get_audio_mime_type("speech.m4a") == "audio/mp4"
        assert get_audio_mime_type("UPPERCASE.WAV") == "audio/wav"

    def test_wav_loading_and_standardization(self, tmp_path):
        """WAV files must decode to 16 kHz mono float32 in [-1.0, 1.0]."""
        sample_rate = 44100
        duration = 1.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        stereo_audio = np.stack([
            np.sin(2 * np.pi * 440 * t) * 0.7,
            np.cos(2 * np.pi * 880 * t) * 0.7,
        ], axis=1)
        wav_file = tmp_path / "test_stereo_44k.wav"
        wavfile.write(str(wav_file), sample_rate, (stereo_audio * 32767).astype(np.int16))

        waveform, sr, error = load_audio_file(str(wav_file), target_sr=16000)
        assert error is None
        assert waveform is not None
        assert sr == 16000
        assert waveform.ndim == 1  # Standardized to mono
        assert np.max(np.abs(waveform)) <= 1.05
        assert len(waveform) == int(round(duration * 16000))

    def test_flac_decoding_or_graceful_missing_decoder_message(self, tmp_path):
        """FLAC loading succeeds if decoder present, or returns clear informative error."""
        flac_file = tmp_path / "sample.flac"
        flac_file.write_bytes(b"fLaC" + b"\x00" * 100)  # Minimal FLAC header
        decoders = check_audio_decoders()

        waveform, sr, error = load_audio_file(str(flac_file), target_sr=16000)
        if not decoders["flac"]:
            assert waveform is None
            assert error is not None
            assert "FLAC decoding requires" in error
        else:
            # If decoder is present, corrupted header is caught
            assert waveform is None or error is not None

    def test_mp3_decoding_or_graceful_missing_ffmpeg_message(self, tmp_path):
        """MP3 loading requires FFmpeg and returns clear diagnostic if missing."""
        mp3_file = tmp_path / "sample.mp3"
        mp3_file.write_bytes(b"\xff\xfb\x90\x64" + b"\x00" * 100)
        decoders = check_audio_decoders()

        waveform, sr, error = load_audio_file(str(mp3_file), target_sr=16000)
        if not decoders["mp3"]:
            assert waveform is None
            assert error is not None
            assert "MP3 decoding requires 'ffmpeg'" in error
        else:
            assert waveform is None or error is not None

    def test_m4a_decoding_or_graceful_missing_ffmpeg_message(self, tmp_path):
        """M4A loading requires FFmpeg and returns clear diagnostic if missing."""
        m4a_file = tmp_path / "sample.m4a"
        m4a_file.write_bytes(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 100)
        decoders = check_audio_decoders()

        waveform, sr, error = load_audio_file(str(m4a_file), target_sr=16000)
        if not decoders["m4a"]:
            assert waveform is None
            assert error is not None
            assert "M4A decoding requires 'ffmpeg'" in error
        else:
            assert waveform is None or error is not None

    def test_unsupported_format_rejected_without_silent_interpretation(self, tmp_path):
        """Unsupported extensions must be rejected with an explicit error."""
        unsupported = tmp_path / "unsupported_audio.ogg"
        unsupported.write_bytes(b"OggS" + b"\x00" * 50)

        waveform, sr, error = load_audio_file(str(unsupported), target_sr=16000)
        assert waveform is None
        assert error is not None
        assert "Unsupported audio format" in error
        assert ".ogg" in error

    def test_empty_audio_file_rejected(self, tmp_path):
        """0-byte files must be rejected immediately."""
        empty_wav = tmp_path / "empty.wav"
        empty_wav.write_bytes(b"")

        waveform, sr, error = load_audio_file(str(empty_wav), target_sr=16000)
        assert waveform is None
        assert error is not None
        assert "empty" in error.lower()

    def test_corrupt_audio_file_handled_gracefully(self, tmp_path):
        """Corrupt files with invalid bytes return error rather than unhandled crash."""
        corrupt_wav = tmp_path / "corrupt.wav"
        corrupt_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00random_garbage")

        waveform, sr, error = load_audio_file(str(corrupt_wav), target_sr=16000)
        assert waveform is None
        assert error is not None


class TestScientificSafetyAndResultInterface:
    """Tests for three-part simplified interface, gating, and evidence explanation."""

    @pytest.fixture
    def sample_audio(self):
        """Generate 4.0s test waveform at 16 kHz."""
        t = np.linspace(0, 4.0, 64000)
        wave = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.random.randn(64000)).astype(np.float32)
        return wave / np.max(np.abs(wave))

    def test_real_speech_upload_strictly_gated(self, sample_audio):
        """User audio uploads must NEVER fabricate AI-Generated or Human Voice labels."""
        summary = generate_result_summary(
            is_fixture=False,
            clean_wave=sample_audio,
            sample_rate=16000,
            model_name="Spectrogram-CNN (2D CNN)",
            raw_score=0.95,  # Even if an unvalidated model gives 0.95
            threshold=0.4604,
            is_real_speech_model=False,
        )

        assert summary["classification_result"] == "Unable to Determine – Real-Speech Classification Not Yet Available"
        assert summary["validation_status"] == "Real-Speech Classification Gated"
        assert summary["can_classify"] is False
        assert summary["confidence"] is None  # Confidence is NOT fabricated
        assert summary["confidence_label"] == "Not Available"
        assert "synthetic software-test fixtures" in summary["explanation"]

    def test_software_fixture_inference_classification_synthetic(self, sample_audio):
        """Software fixtures evaluated above threshold return Synthetic Test Fixture."""
        summary = generate_result_summary(
            is_fixture=True,
            clean_wave=sample_audio,
            sample_rate=16000,
            model_name="Spectrogram-CNN (2D CNN)",
            raw_score=0.88,
            threshold=0.4604,
            is_real_speech_model=False,
        )

        assert summary["classification_result"] == "Synthetic Test Fixture"
        assert summary["validation_status"] == "SOFTWARE-TEST HARNESS DEMO"
        assert summary["can_classify"] is True
        assert summary["confidence"] == pytest.approx(0.88, abs=1e-4)
        assert summary["confidence_label"] == "88.0% Confidence"
        assert "not individual accuracy" in summary["confidence_disclaimer"]
        assert len(summary["explanation"]) > 20

    def test_software_fixture_inference_classification_human(self, sample_audio):
        """Software fixtures evaluated below threshold return Bonafide Test Fixture."""
        summary = generate_result_summary(
            is_fixture=True,
            clean_wave=sample_audio,
            sample_rate=16000,
            model_name="Spectrogram-CNN (2D CNN)",
            raw_score=0.20,
            threshold=0.4604,
            is_real_speech_model=False,
        )

        assert summary["classification_result"] == "Bonafide Test Fixture"
        assert summary["validation_status"] == "SOFTWARE-TEST HARNESS DEMO"
        assert summary["confidence"] == pytest.approx(0.80, abs=1e-4)  # 1.0 - 0.20
        assert summary["confidence_label"] == "80.0% Confidence"

    def test_measurable_acoustic_features_computation(self, sample_audio):
        """Acoustic feature computation extracts valid finite measurements."""
        feats = compute_measurable_acoustic_features(sample_audio, sample_rate=16000)

        assert "rms_energy" in feats
        assert "peak_amplitude" in feats
        assert "zero_crossing_rate" in feats
        assert "spectral_centroid_hz" in feats
        assert "spectral_rolloff_hz" in feats
        assert "high_frequency_ratio" in feats

        assert feats["rms_energy"] > 0.0
        assert feats["peak_amplitude"] == pytest.approx(1.0, abs=1e-2)
        assert 0.0 <= feats["zero_crossing_rate"] <= 1.0
        assert 100.0 < feats["spectral_centroid_hz"] < 8000.0
        assert 0.0 <= feats["high_frequency_ratio"] <= 1.0

    def test_preprocessing_consistency(self, sample_audio):
        """Preprocessing pipeline standardizes length, sample rate, and peak amplitude."""
        cfg = load_config("configs/default.yaml")
        clean_wave, meta = preprocess_audio(sample_audio, 16000, cfg)

        assert len(clean_wave) == 64000  # Exactly 4.0s
        assert meta["length_final"] == 64000
        assert np.max(np.abs(clean_wave)) <= 1.0

