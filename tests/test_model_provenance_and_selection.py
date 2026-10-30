"""Unit and integration tests for Model Provenance and Checkpoint Selection.

Verifies:
1. Checkpoint provenance validation and required fields.
2. Fixture/real-speech strict isolation (no silent fallback).
3. Missing real-speech checkpoint handling.
4. Rejection of invalid or cross-contaminated provenance.
5. Preserved execution of procedural software-test harness.
"""

from pathlib import Path
import pytest
import torch
import numpy as np

from src.models.provenance import (
    REQUIRED_PROVENANCE_FIELDS,
    build_checkpoint_provenance,
    extract_checkpoint_provenance,
    validate_checkpoint_provenance,
)
from src.models.model_loader import (
    get_checkpoint_path,
    load_calibrated_thresholds,
    load_fixture_models,
    load_real_speech_model,
)
from src.analysis.explanation import generate_result_summary


class TestModelProvenance:
    """Tests for checkpoint provenance construction and validation."""

    def test_build_checkpoint_provenance_contains_all_required_fields(self):
        """Constructed provenance must contain all 8 required fields."""
        prov = build_checkpoint_provenance(
            dataset_name="ASVspoof2019_LA",
            dataset_mode="real_speech",
            model_architecture="SpectrogramCNN",
            preprocessing_config={"sample_rate": 16000},
            training_partition="train",
            development_partition="dev",
            training_date="2026-10-31T00:00:00Z",
            model_version="1.0.0",
        )

        for field in REQUIRED_PROVENANCE_FIELDS:
            assert field in prov, f"Missing required provenance field: {field}"
            assert prov[field] is not None

        assert prov["dataset_name"] == "ASVspoof2019_LA"
        assert prov["dataset_mode"] == "real_speech"
        assert prov["model_architecture"] == "SpectrogramCNN"

    def test_fixture_checkpoint_rejects_real_speech_validation(self):
        """Fixture checkpoint must be strictly rejected when evaluated for real-speech mode."""
        fixture_path = Path("checkpoints/fixture/best_spec_cnn.pt")
        if not fixture_path.exists():
            fixture_path = Path("checkpoints/best_spec_cnn.pt")

        assert fixture_path.exists(), "Fixture checkpoint must exist."
        ckpt = torch.load(str(fixture_path), map_location="cpu")

        is_valid, msg, prov = validate_checkpoint_provenance(ckpt, expected_mode="real_speech")
        assert is_valid is False
        assert "Provenance mode mismatch" in msg or "synthetic" in msg.lower()

    def test_real_speech_mock_payload_passes_validation(self):
        """Mock payload with valid real-speech provenance passes validation."""
        valid_payload = {
            "epoch": 10,
            "val_loss": 0.25,
            "provenance": {
                "dataset_name": "ASVspoof2019_LA",
                "dataset_mode": "real_speech",
                "model_architecture": "SpectrogramCNN",
                "preprocessing_config": {"sample_rate": 16000},
                "training_partition": "train",
                "development_partition": "dev",
                "training_date": "2026-10-31T00:00:00Z",
                "model_version": "1.0.0",
            },
        }

        is_valid, msg, prov = validate_checkpoint_provenance(valid_payload, expected_mode="real_speech")
        assert is_valid is True
        assert "verified successfully" in msg.lower()
        assert prov["dataset_mode"] == "real_speech"

    def test_synthetic_name_in_real_speech_mode_rejected(self):
        """Checkpoints with synthetic dataset names in real_speech mode are rejected."""
        contaminated_payload = {
            "provenance": {
                "dataset_name": "SyntheticTestFixtures_V2",
                "dataset_mode": "real_speech",  # Contaminated spoof mode
                "model_architecture": "SpectrogramCNN",
                "preprocessing_config": {},
                "training_partition": "train",
                "development_partition": "dev",
                "training_date": "2026-10-31T00:00:00Z",
                "model_version": "1.0.0",
            }
        }

        is_valid, msg, _ = validate_checkpoint_provenance(contaminated_payload, expected_mode="real_speech")
        assert is_valid is False
        assert "synthetic dataset name" in msg.lower()

    def test_eval_partition_in_development_partition_rejected(self):
        """Tuning on held-out eval partition violates protocol and is rejected."""
        leaked_payload = {
            "provenance": {
                "dataset_name": "ASVspoof2019_LA",
                "dataset_mode": "real_speech",
                "model_architecture": "SpectrogramCNN",
                "preprocessing_config": {},
                "training_partition": "train",
                "development_partition": "eval",  # Protocol breach
                "training_date": "2026-10-31T00:00:00Z",
                "model_version": "1.0.0",
            }
        }

        is_valid, msg, _ = validate_checkpoint_provenance(leaked_payload, expected_mode="real_speech")
        assert is_valid is False
        assert "eval" in msg.lower()


class TestModelSelectionAndIsolation:
    """Tests for checkpoint paths, model selection, and isolation."""

    def test_checkpoint_paths_are_strictly_isolated(self):
        """Paths for fixture and real_speech modes must resolve to distinct subdirectories."""
        p_fixture = get_checkpoint_path(model_type="cnn", mode="fixture")
        p_real = get_checkpoint_path(model_type="cnn", mode="real_speech")

        assert "fixture" in str(p_fixture) or "best_spec_cnn.pt" in str(p_fixture)
        assert "real_speech" in str(p_real)
        assert p_fixture != p_real

    def test_missing_real_speech_checkpoint_returns_explicit_status(self):
        """Missing real-speech checkpoint must return available=False with clear status."""
        status = load_real_speech_model(checkpoint_path="checkpoints/real_speech/non_existent.pt")

        assert status["available"] is False
        assert status["status"] == "CHECKPOINT_NOT_FOUND"
        assert status["predictor"] is None
        assert status["is_real_speech_model"] is False
        assert "Real-speech checkpoint not found" in status["message"]
        assert "checkpoints" in status["checkpoint_path"]

    def test_no_silent_fallback_to_fixture(self):
        """Real-speech model loader must NEVER silently fall back to the fixture checkpoint."""
        status = load_real_speech_model(checkpoint_path="checkpoints/real_speech/non_existent.pt")

        # The fixture checkpoint best_spec_cnn.pt exists on disk, but load_real_speech_model must not load it
        assert status["available"] is False
        assert status["predictor"] is None

    def test_fixture_model_loader_preserves_test_harness(self):
        """Fixture model loader successfully loads test harness models."""
        models = load_fixture_models()

        assert "Spectrogram-CNN (2D CNN)" in models
        cnn_info = models["Spectrogram-CNN (2D CNN)"]
        assert cnn_info["is_real_speech_model"] is False
        assert cnn_info["dataset_mode"] == "synthetic_fixture"
        assert cnn_info["threshold"] > 0.0

        # Predictor runs on procedural test fixture
        sample_path = Path("tests/fixtures/audio/dev/LA_D_0000031.wav")
        if sample_path.exists():
            predictor = cnn_info["predictor"]
            res = predictor.predict_audio(str(sample_path))
            assert "probability_spoof" in res
            assert 0.0 <= res["probability_spoof"] <= 1.0

            # Explanatory summary works in fixture mode
            summary = generate_result_summary(
                is_fixture=True,
                clean_wave=np.zeros(64000, dtype=np.float32),
                raw_score=res["probability_spoof"],
                threshold=cnn_info["threshold"],
                is_real_speech_model=False,
            )
            assert summary["validation_status"] == "SOFTWARE-TEST HARNESS DEMO"
            assert "Test Fixture" in summary["classification_result"]

