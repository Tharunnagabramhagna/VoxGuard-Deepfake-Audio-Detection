"""Comprehensive tests for VoxGuard Phase 3: audio features and baseline models.

Tests cover:
  1. MFCC feature extraction (40 coefficients, summary stats: mean & std).
  2. Log-Mel spectrogram generation (16 kHz mono, configurable Mel bins, initially 64).
  3. Output shapes and strict finite checks (no NaN, no Inf).
  4. Deterministic preprocessing and feature reproducibility.
  5. Scaler fitting on training data only (data leakage prevention).
  6. Logistic Regression and Random Forest pipeline training with balanced weights.
  7. Checkpoint persistence and reload via joblib.
  8. Single unified prediction interface (consistent inference and probability calibration).
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessor import preprocess_audio
from src.data.test_fixtures import create_test_fixtures, generate_sine_tone
from src.features.extractor import FeatureExtractor
from src.features.mel_spectrogram import (
    create_mel_filterbank,
    extract_log_mel_spectrogram,
)
from src.features.mfcc import (
    compute_deltas,
    extract_mfcc_frames,
    extract_mfcc_summary,
)
from src.models.baseline_models import (
    BaselinePredictor,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_baseline_pipeline,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


@pytest.fixture(scope="module")
def audio_fixture():
    """Generate a clean 4.0-second 16 kHz mono waveform."""
    sr = 16000
    duration = 4.0
    waveform = generate_sine_tone(duration_sec=duration, sample_rate=sr, frequency=440.0, seed=42)
    return waveform, sr


# ─────────────────────────────────────────────────────────────────────────────
# 1. MFCC Feature Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMFCCFeatures:
    """Tests for 40-coefficient MFCC extraction and summary statistics."""

    def test_mfcc_frames_shape_and_finite(self, audio_fixture):
        """Extracting 40 MFCCs should produce (40, 401) frame matrix with finite values."""
        waveform, sr = audio_fixture
        mfcc = extract_mfcc_frames(waveform, sample_rate=sr, n_mfcc=40, n_mels=64, hop_length=160)

        assert mfcc.shape == (40, 401), f"Expected shape (40, 401), got {mfcc.shape}"
        assert mfcc.dtype == np.float32
        assert np.isfinite(mfcc).all(), "MFCC frame matrix contains non-finite values (NaN/Inf)"

    def test_mfcc_summary_statistics_static(self, audio_fixture):
        """MFCC summary without deltas should return 80 features (40 mean + 40 std)."""
        waveform, sr = audio_fixture
        feats, meta = extract_mfcc_summary(waveform, sample_rate=sr, n_mfcc=40, include_deltas=False)

        assert feats.ndim == 1
        assert len(feats) == 80, f"Expected 80 features (40 mean + 40 std), got {len(feats)}"
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all(), "MFCC summary contains NaN or Inf"
        assert meta["summary_statistics"] == ["mean", "std"]
        assert len(meta["feature_names"]) == 80

    def test_mfcc_summary_statistics_with_deltas(self, audio_fixture):
        """MFCC summary with deltas should return 160 features (40 mean + 40 std + 40 d_mean + 40 d_std)."""
        waveform, sr = audio_fixture
        feats, meta = extract_mfcc_summary(waveform, sample_rate=sr, n_mfcc=40, include_deltas=True)

        assert len(feats) == 160, f"Expected 160 features with deltas, got {len(feats)}"
        assert np.isfinite(feats).all(), "MFCC delta summary contains NaN or Inf"
        assert "delta_mean" in meta["summary_statistics"]
        assert "delta_std" in meta["summary_statistics"]

    def test_compute_deltas_shape_and_finite(self):
        """compute_deltas should preserve (n_features, time_frames) shape."""
        frames = np.random.randn(40, 401).astype(np.float32)
        deltas = compute_deltas(frames, width=5)

        assert deltas.shape == frames.shape
        assert np.isfinite(deltas).all()

    def test_mfcc_determinism(self, audio_fixture):
        """Same input audio should yield bit-for-bit identical MFCC summary features."""
        waveform, sr = audio_fixture
        feats1, _ = extract_mfcc_summary(waveform, sample_rate=sr, n_mfcc=40)
        feats2, _ = extract_mfcc_summary(waveform, sample_rate=sr, n_mfcc=40)

        np.testing.assert_array_equal(feats1, feats2, err_msg="MFCC extraction is non-deterministic")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Log-Mel Spectrogram Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLogMelSpectrogram:
    """Tests for Log-Mel spectrogram generation (16 kHz mono, configurable Mel bins)."""

    def test_mel_spectrogram_shape_initially_64_bins(self, audio_fixture):
        """Initially configured 64 Mel bins should return tensor of shape (1, 64, 401)."""
        waveform, sr = audio_fixture
        mel = extract_log_mel_spectrogram(
            waveform, sample_rate=sr, n_mels=64, n_fft=1024, win_length=400, hop_length=160, add_channel_dim=True
        )

        assert isinstance(mel, torch.Tensor)
        assert mel.shape == (1, 64, 401), f"Expected shape (1, 64, 401), got {tuple(mel.shape)}"
        assert mel.dtype == torch.float32

    def test_mel_spectrogram_configurable_bins(self, audio_fixture):
        """Mel bin count should be fully configurable (e.g., 32, 80, 128)."""
        waveform, sr = audio_fixture

        mel_32 = extract_log_mel_spectrogram(waveform, sample_rate=sr, n_mels=32, hop_length=160)
        assert mel_32.shape == (1, 32, 401)

        mel_80 = extract_log_mel_spectrogram(waveform, sample_rate=sr, n_mels=80, hop_length=160)
        assert mel_80.shape == (1, 80, 401)

    def test_mel_spectrogram_finite_values(self, audio_fixture):
        """Log-Mel spectrogram tensor must contain zero NaN or Inf values."""
        waveform, sr = audio_fixture
        mel = extract_log_mel_spectrogram(waveform, sample_rate=sr, n_mels=64)

        assert torch.isfinite(mel).all().item(), "Log-Mel spectrogram contains NaN or Inf"

    def test_mel_spectrogram_silent_audio_finite(self):
        """Even completely silent audio must produce finite decibel values without log(0) crash."""
        silent = np.zeros(64000, dtype=np.float32)
        mel = extract_log_mel_spectrogram(silent, sample_rate=16000, n_mels=64)

        assert torch.isfinite(mel).all().item()

    def test_mel_filterbank_normalization(self):
        """Mel filterbank matrix must have non-negative weights and valid dimensions."""
        fb = create_mel_filterbank(sample_rate=16000, n_fft=1024, n_mels=64)
        assert fb.shape == (64, 513)
        assert (fb >= 0.0).all().item()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Traditional Baseline Classifiers (Logistic Regression & Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineModels:
    """Tests for baseline pipeline construction, training, and artifact persistence."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Generate balanced and imbalanced synthetic feature datasets."""
        set_seed(42)
        rng = np.random.RandomState(42)
        n_train = 60
        n_features = 80

        # Simulate 9:1 imbalanced training set (similar to ASVspoof)
        y_train = np.array([0] * 6 + [1] * 54)  # 6 bonafide, 54 spoof
        X_train = rng.randn(n_train, n_features).astype(np.float32)
        # Shift bonafide distribution slightly
        X_train[y_train == 0] += 0.8

        # Development set (unaltered official distribution)
        n_dev = 30
        y_dev = np.array([0] * 10 + [1] * 20)
        X_dev = rng.randn(n_dev, n_features).astype(np.float32)
        X_dev[y_dev == 0] += 0.8

        return X_train, y_train, X_dev, y_dev

    def test_logistic_regression_pipeline_training(self, synthetic_dataset):
        """Logistic Regression pipeline should fit on X_train and predict valid probabilities."""
        X_train, y_train, X_dev, y_dev = synthetic_dataset
        pipeline = train_baseline_pipeline("logistic_regression", X_train, y_train, random_state=42)

        # Preprocessing verification: scaler must have mean_ and var_ fitted on X_train
        scaler = pipeline.named_steps["scaler"]
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == 80

        # Class weight verification
        clf = pipeline.named_steps["classifier"]
        assert clf.class_weight == "balanced"

        # Predictions on dev set
        preds = pipeline.predict(X_dev)
        probs = pipeline.predict_proba(X_dev)

        assert preds.shape == (len(y_dev),)
        assert probs.shape == (len(y_dev), 2)
        assert np.all((preds == 0) | (preds == 1))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_random_forest_pipeline_training(self, synthetic_dataset):
        """Random Forest pipeline should fit on X_train and predict valid probabilities."""
        X_train, y_train, X_dev, y_dev = synthetic_dataset
        pipeline = train_baseline_pipeline(
            "random_forest", X_train, y_train, config={"n_estimators": 20, "max_depth": 5}, random_state=42
        )

        clf = pipeline.named_steps["classifier"]
        assert clf.class_weight == "balanced"

        preds = pipeline.predict(X_dev)
        probs = pipeline.predict_proba(X_dev)

        assert preds.shape == (len(y_dev),)
        assert probs.shape == (len(y_dev), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_scaler_fitted_on_training_data_only(self, synthetic_dataset):
        """Ensure scaler statistics strictly match X_train and do not include dev data."""
        X_train, y_train, X_dev, _ = synthetic_dataset
        pipeline = train_baseline_pipeline("logistic_regression", X_train, y_train, random_state=42)

        fitted_scaler = pipeline.named_steps["scaler"]
        expected_mean = np.mean(X_train, axis=0)
        np.testing.assert_allclose(fitted_scaler.mean_, expected_mean, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unified Prediction Interface & Artifact Reload
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionInterface:
    """Tests for single, tested prediction interface across both baseline classifiers."""

    @pytest.fixture
    def trained_artifacts(self, tmp_path):
        """Train and persist both pipelines to temporary directory."""
        set_seed(42)
        rng = np.random.RandomState(42)
        X_train = rng.randn(40, 80).astype(np.float32)
        y_train = np.array([0] * 10 + [1] * 30)

        lr_path = str(tmp_path / "lr_test.joblib")
        rf_path = str(tmp_path / "rf_test.joblib")

        train_baseline_pipeline("logistic_regression", X_train, y_train, save_path=lr_path, random_state=42)
        train_baseline_pipeline(
            "random_forest", X_train, y_train, config={"n_estimators": 10}, save_path=rf_path, random_state=42
        )

        return lr_path, rf_path

    def test_reload_from_checkpoint(self, trained_artifacts):
        """Predictor should reload seamlessly from saved joblib checkpoint."""
        lr_path, rf_path = trained_artifacts

        pred_lr = BaselinePredictor.from_checkpoint(lr_path, model_name="logistic_regression")
        pred_rf = BaselinePredictor.from_checkpoint(rf_path, model_name="random_forest")

        assert pred_lr.model_name == "logistic_regression"
        assert pred_rf.model_name == "random_forest"

    def test_predict_features_interface(self, trained_artifacts):
        """predict_features must return structured, calibrated prediction dictionary."""
        lr_path, _ = trained_artifacts
        predictor = BaselinePredictor.from_checkpoint(lr_path)

        x_test = np.random.randn(80).astype(np.float32)
        result = predictor.predict_features(x_test)

        assert "prediction" in result
        assert result["prediction"] in (0, 1)
        assert result["label"] in ("bonafide", "spoof")
        assert 0.0 <= result["probability_spoof"] <= 1.0
        assert 0.0 <= result["probability_bonafide"] <= 1.0
        assert abs(result["probability_spoof"] + result["probability_bonafide"] - 1.0) < 1e-5
        assert result["confidence"] == max(result["probability_spoof"], result["probability_bonafide"])
        assert result["n_features"] == 80

    def test_consistent_inference_on_same_audio(self, trained_artifacts, audio_fixture):
        """Evaluating the same audio clip twice must yield identical probabilities and labels."""
        lr_path, _ = trained_artifacts
        predictor = BaselinePredictor.from_checkpoint(lr_path)
        waveform, sr = audio_fixture

        res1 = predictor.predict_audio(waveform, sample_rate=sr)
        res2 = predictor.predict_audio(waveform, sample_rate=sr)

        assert res1["prediction"] == res2["prediction"]
        assert res1["label"] == res2["label"]
        assert abs(res1["probability_spoof"] - res2["probability_spoof"]) < 1e-6
        assert abs(res1["confidence"] - res2["confidence"]) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Extractor Unified API
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtractorAPI:
    """Tests for the high-level FeatureExtractor class."""

    def test_extractor_extract_from_waveform(self, audio_fixture):
        """FeatureExtractor should produce consistent Log-Mel tensor and MFCC vector."""
        waveform, sr = audio_fixture
        extractor = FeatureExtractor(config={"features": {"mel": {"n_mels": 64}, "mfcc": {"n_mfcc": 40}}})

        clean_wave, _ = extractor.process_waveform(waveform, sr)
        mel_spec = extractor.extract_mel(clean_wave)
        mfcc_vec, meta = extractor.extract_mfcc(clean_wave)

        assert mel_spec.shape == (1, 64, 401)
        assert mfcc_vec.shape == (80,)
        assert torch.isfinite(mel_spec).all().item()
        assert np.isfinite(mfcc_vec).all()
