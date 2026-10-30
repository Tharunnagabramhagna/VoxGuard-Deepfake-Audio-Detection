"""Unit tests for Phase 2 data integrity, label alignment, and manifest validation.

Verifies:
1. Manifest integrity: detection of missing columns, null values, duplicate IDs, and invalid labels.
2. Label alignment: explicit LabelAlignmentError on missing audio files (no silent fallback).
3. Class balance enforcement: rejection of single-class evaluation sets.
4. Paired integrity: ensuring clean and corrupted waveforms remain strictly paired to true labels.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    DatasetIntegrityError,
    LabelAlignmentError,
    load_aligned_evaluation_set,
    validate_manifest_integrity,
)
from src.robustness.noise_simulation import apply_awgn


class TestManifestIntegrityValidation:
    """Tests for validate_manifest_integrity."""

    def test_valid_manifest_passes(self):
        valid_df = pd.DataFrame([
            {"utterance_id": "u001", "label": 0, "partition": "eval"},
            {"utterance_id": "u002", "label": 1, "partition": "eval"},
        ])
        # Should not raise
        validate_manifest_integrity(valid_df)

    def test_missing_column_raises(self):
        missing_label_df = pd.DataFrame([
            {"utterance_id": "u001", "partition": "eval"},
        ])
        with pytest.raises(DatasetIntegrityError, match="missing required column: 'label'"):
            validate_manifest_integrity(missing_label_df)

        missing_id_df = pd.DataFrame([
            {"label": 0, "partition": "eval"},
        ])
        with pytest.raises(DatasetIntegrityError, match="missing required column: 'utterance_id'"):
            validate_manifest_integrity(missing_id_df)

    def test_empty_manifest_raises(self):
        empty_df = pd.DataFrame(columns=["utterance_id", "label"])
        with pytest.raises(DatasetIntegrityError, match="Manifest is empty"):
            validate_manifest_integrity(empty_df)

    def test_duplicate_utterance_ids_raises(self):
        dup_df = pd.DataFrame([
            {"utterance_id": "u001", "label": 0},
            {"utterance_id": "u001", "label": 1},  # Duplicate ID with conflicting label!
        ])
        with pytest.raises(DatasetIntegrityError, match="duplicate utterance IDs"):
            validate_manifest_integrity(dup_df)

    def test_null_labels_or_ids_raises(self):
        null_id_df = pd.DataFrame([
            {"utterance_id": None, "label": 0},
            {"utterance_id": "u002", "label": 1},
        ])
        with pytest.raises(DatasetIntegrityError, match="null or NaN"):
            validate_manifest_integrity(null_id_df)

        null_label_df = pd.DataFrame([
            {"utterance_id": "u001", "label": np.nan},
            {"utterance_id": "u002", "label": 1},
        ])
        with pytest.raises(DatasetIntegrityError, match="null or NaN"):
            validate_manifest_integrity(null_label_df)

    def test_non_binary_labels_raises(self):
        invalid_label_df = pd.DataFrame([
            {"utterance_id": "u001", "label": 0},
            {"utterance_id": "u002", "label": 99},  # Invalid non-binary label
        ])
        with pytest.raises(DatasetIntegrityError, match="invalid non-binary labels"):
            validate_manifest_integrity(invalid_label_df)

    def test_single_class_manifest_raises(self):
        single_class_df = pd.DataFrame([
            {"utterance_id": "u001", "label": 1},
            {"utterance_id": "u002", "label": 1},
        ])
        with pytest.raises(DatasetIntegrityError, match="must contain both bonafide"):
            validate_manifest_integrity(single_class_df, require_both_classes=True)


class TestAlignedEvaluationLoader:
    """Tests for load_aligned_evaluation_set and label pairing."""

    def test_missing_audio_file_raises_explicit_error_without_defaulting(self):
        """Verifies that missing audio raises LabelAlignmentError instead of defaulting to label 0."""
        # Create manifest with an utterance that has no audio file on disk
        ghost_manifest = pd.DataFrame([
            {"utterance_id": "NONEXISTENT_GHOST_AUDIO_ID", "label": 1, "partition": "eval"},
            {"utterance_id": "ANOTHER_NONEXISTENT_ID", "label": 0, "partition": "eval"},
        ])

        with pytest.raises(LabelAlignmentError, match="could not be found"):
            load_aligned_evaluation_set(
                manifest_df=ghost_manifest,
                raw_dir=Path("tests/fixtures/audio"),
            )

    def test_loaded_samples_have_strict_pairing_and_both_classes(self):
        """Verifies that waveforms and labels loaded from real manifest match 1-to-1."""
        manifest_path = Path("data/manifests/asvspoof_eval_mini.csv")
        if not manifest_path.exists():
            manifest_path = Path("tests/fixtures/manifests/asvspoof_eval.csv")

        df = pd.read_csv(manifest_path)
        waveforms, labels, meta = load_aligned_evaluation_set(
            manifest_df=df,
            raw_dir=Path("data/raw/asvspoof2019_la"),
            max_samples=10,
        )

        assert len(waveforms) == len(labels) == len(meta) == 10
        assert set(labels) == {0, 1}

        # Verify exact 1-to-1 metadata alignment
        for i in range(len(meta)):
            assert meta[i]["label"] == labels[i]
            assert Path(meta[i]["audio_path"]).exists()

    def test_noise_perturbation_preserves_paired_label_integrity(self):
        """Verifies that corrupted copies of waveforms remain paired with their exact label."""
        manifest_path = Path("data/manifests/asvspoof_eval_mini.csv")
        if not manifest_path.exists():
            manifest_path = Path("tests/fixtures/manifests/asvspoof_eval.csv")

        df = pd.read_csv(manifest_path)
        clean_waveforms, stress_y, _ = load_aligned_evaluation_set(
            manifest_df=df,
            raw_dir=Path("data/raw/asvspoof2019_la"),
            max_samples=10,
        )

        # Apply noise perturbation
        noisy_waveforms = [apply_awgn(wf, snr_db=20.0, seed=42) for wf in clean_waveforms]

        # Verify pairing invariants
        assert len(noisy_waveforms) == len(stress_y)
        assert (stress_y == 0).sum() > 0, "Must have genuine samples in stress test"
        assert (stress_y == 1).sum() > 0, "Must have spoof samples in stress test"

    def test_real_speech_mode_blocks_fixture_directory_fallback(self):
        """Verifies that in real_speech mode, fixture directories are never searched."""
        # Using a fixture ID that exists in tests/fixtures/audio/eval
        fixture_df = pd.DataFrame([
            {"utterance_id": "LA_E_0000001", "label": 0, "partition": "eval"},
            {"utterance_id": "LA_E_0000002", "label": 1, "partition": "eval"},
        ])

        # Pointing raw_dir to an empty non-existent real directory
        empty_real_dir = Path("data/raw/nonexistent_real_dataset")

        with pytest.raises(LabelAlignmentError, match="could not be found"):
            load_aligned_evaluation_set(
                manifest_df=fixture_df,
                raw_dir=empty_real_dir,
                dataset_mode="real_speech",
            )

    def test_accidental_fixture_use_in_real_speech_mode_raises(self):
        """Verifies that manifests marked as synthetic_fixture are rejected in real_speech mode."""
        fixture_manifest = pd.DataFrame([
            {"utterance_id": "u001", "label": 0, "dataset_mode": "synthetic_fixture"},
            {"utterance_id": "u002", "label": 1, "dataset_mode": "synthetic_fixture"},
        ])

        with pytest.raises(DatasetIntegrityError, match="Cross-mode contamination is strictly prohibited"):
            validate_manifest_integrity(fixture_manifest, expected_mode="real_speech")

    def test_mode_mismatch_raises_dataset_integrity_error(self):
        """Verifies that expected_mode check raises if dataset_mode column differs."""
        manifest = pd.DataFrame([
            {"utterance_id": "u001", "label": 0, "dataset_mode": "other_mode"},
            {"utterance_id": "u002", "label": 1, "dataset_mode": "other_mode"},
        ])

        with pytest.raises(DatasetIntegrityError, match="does not match expected mode"):
            validate_manifest_integrity(manifest, expected_mode="synthetic_fixture")

