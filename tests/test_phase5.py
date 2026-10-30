"""Unit tests for VoxGuard Phase 5: Comprehensive, Reproducible Model Evaluation.

Tests:
1. Equal Error Rate (EER) mathematical calculation and linear interpolation.
2. Validation-only threshold calibration.
3. Confusion matrix, precision, recall, F1 (positive class = spoof = 1).
4. Representative False Positive and False Negative mining.
5. AWGN noise simulation and empirical SNR calibration.
6. ITU-T G.711 mu-law compression and codec availability reporting (blocked when ffmpeg absent).
7. Speaker disjointness audit and leakage detection.
8. Group performance by system ID and acoustic shortcut correlation.
9. CD-ADD evaluation fixture generation and metadata integrity.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.metrics.evaluator import (
    calibrate_threshold,
    compute_eer,
    evaluate_predictions,
)
from src.metrics.error_analysis import (
    mine_errors,
    save_error_analysis,
)
from src.robustness.noise_simulation import (
    apply_awgn,
    measure_snr_db,
)
from src.robustness.compression import (
    CodecUnavailableError,
    apply_mp3_compression,
    apply_mulaw_compression,
    check_codec_availability,
)
from src.analysis.leakage_analysis import (
    analyze_acoustic_shortcuts,
    group_performance_by_speaker,
    group_performance_by_system,
    verify_speaker_disjointness,
)
from src.data.test_fixtures import create_cd_add_test_fixtures


class TestEERAndMetricEvaluator:
    """Tests for Equal Error Rate (EER) and evaluation metrics."""

    def test_compute_eer_perfect_separation(self):
        """EER should be 0.0 when bonafide and spoof distributions are cleanly separated."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.15, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95])
        eer_val, eer_th = compute_eer(y_true, y_scores)
        assert eer_val == 0.0
        assert 0.25 <= eer_th <= 0.8

    def test_compute_eer_random_guessing(self):
        """EER should be close to 0.5 for random guessing."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 500 + [1] * 500)
        y_scores = rng.uniform(0.0, 1.0, size=1000)
        eer_val, eer_th = compute_eer(y_true, y_scores)
        assert 0.40 <= eer_val <= 0.60
        assert 0.0 <= eer_th <= 1.0

    def test_compute_eer_linear_interpolation(self):
        """Verify linear interpolation crosses FAR and FRR accurately."""
        # Partially overlapping scores: bonafide=[0.1, 0.2, 0.3, 0.6], spoof=[0.4, 0.7, 0.8, 0.9]
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
        eer_val, eer_th = compute_eer(y_true, y_scores)
        assert 0.0 < eer_val < 0.5
        assert eer_val == 0.25
        assert 0.3 <= eer_th <= 0.7

    def test_evaluate_predictions_positive_class_is_spoof(self):
        """Verify that positive class is explicitly spoof (label 1)."""
        y_true = np.array([0, 0, 1, 1])
        # Perfect detection of spoof
        y_scores = np.array([0.1, 0.2, 0.9, 0.85])
        res = evaluate_predictions(y_true, y_scores, threshold=0.5)

        assert res["positive_class"] == "spoof (label 1)"
        assert res["accuracy"] == 1.0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["f1"] == 1.0
        assert res["roc_auc"] == 1.0
        assert res["confusion_matrix"]["tp"] == 2  # 2 spoofs detected
        assert res["confusion_matrix"]["tn"] == 2  # 2 bonafide detected
        assert res["confusion_matrix"]["fp"] == 0
        assert res["confusion_matrix"]["fn"] == 0

    def test_evaluate_predictions_schema(self):
        """Verify all required keys and types in evaluation dictionary."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_scores = np.array([0.2, 0.8, 0.4, 0.6, 0.7])
        res = evaluate_predictions(y_true, y_scores, threshold=0.5)

        required_keys = [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "eer", "eer_percent", "eer_threshold", "operating_threshold",
            "confusion_matrix", "far", "frr", "roc_curve",
            "n_samples", "n_bonafide", "n_spoof", "positive_class",
        ]
        for k in required_keys:
            assert k in res, f"Missing key {k} in evaluation results"


class TestThresholdCalibration:
    """Tests for validation-only threshold calibration."""

    def test_calibrate_threshold_eer_method(self):
        y_val = np.array([0, 0, 0, 1, 1, 1])
        scores_val = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        th = calibrate_threshold(y_val, scores_val, method="eer")
        assert 0.3 <= th <= 0.7

    def test_calibrate_threshold_youden_j(self):
        y_val = np.array([0, 0, 1, 1])
        scores_val = np.array([0.1, 0.4, 0.6, 0.9])
        th = calibrate_threshold(y_val, scores_val, method="youden_j")
        assert 0.0 <= th <= 1.0

    def test_calibrate_threshold_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown threshold calibration method"):
            calibrate_threshold([0, 1], [0.2, 0.8], method="unsupported_method")


class TestErrorAnalysis:
    """Tests for mining False Positives and False Negatives."""

    def test_mine_errors_identifies_and_ranks_errors(self):
        manifest_records = [
            {"utterance_id": "spk1_01", "speaker_id": "spk1", "system_id": "-", "label": 0, "label_str": "bonafide"},
            {"utterance_id": "spk1_02", "speaker_id": "spk1", "system_id": "-", "label": 0, "label_str": "bonafide"},
            {"utterance_id": "spk2_01", "speaker_id": "spk2", "system_id": "A01", "label": 1, "label_str": "spoof"},
            {"utterance_id": "spk2_02", "speaker_id": "spk2", "system_id": "A02", "label": 1, "label_str": "spoof"},
        ]
        # Make spk1_02 a False Positive (score 0.85 >= 0.5)
        # Make spk2_01 a False Negative (score 0.20 < 0.5)
        y_scores = np.array([0.10, 0.85, 0.20, 0.90])
        err_res = mine_errors(manifest_records, y_scores, threshold=0.5, top_k=5)

        assert err_res["total_false_positives"] == 1
        assert err_res["total_false_negatives"] == 1
        assert len(err_res["top_false_positives"]) == 1
        assert len(err_res["top_false_negatives"]) == 1

        fp = err_res["top_false_positives"][0]
        assert fp["utterance_id"] == "spk1_02"
        assert fp["prediction"] == 1
        assert fp["score_spoof"] == 0.85

        fn = err_res["top_false_negatives"][0]
        assert fn["utterance_id"] == "spk2_01"
        assert fn["prediction"] == 0
        assert fn["score_spoof"] == 0.20

    def test_save_error_analysis_writes_valid_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_records = [
                {"utterance_id": "u1", "speaker_id": "s1", "system_id": "-", "label": 0},
                {"utterance_id": "u2", "speaker_id": "s2", "system_id": "A01", "label": 1},
            ]
            y_scores = np.array([0.9, 0.1])  # Both are misclassified
            err_res = mine_errors(manifest_records, y_scores, threshold=0.5)
            paths = save_error_analysis(err_res, output_dir=tmpdir, prefix="test_err")

            assert Path(paths["json_path"]).exists()
            assert Path(paths["csv_path"]).exists()


class TestRobustnessSimulation:
    """Tests for AWGN noise injection and audio compression."""

    def test_apply_awgn_snr_level(self):
        """Verifies that injected noise power matches the target SNR within 2 dB."""
        rng = np.random.RandomState(42)
        n_samples = 48000
        clean_wf = (0.5 * np.sin(2 * np.pi * 440 * np.arange(n_samples) / 16000)).astype(np.float32)

        target_snr = 20.0
        noisy_wf = apply_awgn(clean_wf, snr_db=target_snr, seed=42)

        measured_snr = measure_snr_db(clean_wf, noisy_wf)
        assert abs(measured_snr - target_snr) < 1.5
        assert noisy_wf.shape == clean_wf.shape
        assert np.all(noisy_wf >= -1.0) and np.all(noisy_wf <= 1.0)

    def test_apply_awgn_silent_waveform(self):
        """Silent audio should not blow up or produce NaNs."""
        silent = np.zeros(16000, dtype=np.float32)
        noisy = apply_awgn(silent, snr_db=20.0, seed=42)
        assert not np.any(np.isnan(noisy))

    def test_apply_mulaw_compression_quantization(self):
        """ITU-T G.711 mu-law compression should produce bounded distortion."""
        clean_wf = np.linspace(-0.9, 0.9, 16000, dtype=np.float32)
        comp_wf = apply_mulaw_compression(clean_wf, mu=255)

        assert comp_wf.shape == clean_wf.shape
        assert np.all(comp_wf >= -1.0) and np.all(comp_wf <= 1.0)
        # 8-bit companding has finite reconstruction error (quantization noise)
        max_err = np.max(np.abs(comp_wf - clean_wf))
        assert 0.0 < max_err < 0.05

    def test_codec_availability_reporting(self):
        """Verify check_codec_availability returns structured status dict."""
        status = check_codec_availability()
        assert "mulaw_g711" in status
        assert "mp3" in status
        assert "ffmpeg_available" in status
        assert "mp3_aac_status" in status
        assert status["mulaw_g711"] is True  # Always supported natively

    def test_mp3_compression_raises_when_codec_unavailable(self):
        """Verify policy: when ffmpeg is missing, raises CodecUnavailableError rather than faking."""
        status = check_codec_availability()
        if not status["ffmpeg_available"]:
            dummy_wf = np.zeros(16000, dtype=np.float32)
            with pytest.raises(CodecUnavailableError, match="BLOCKED"):
                apply_mp3_compression(dummy_wf, bitrate_kbps=64)


class TestLeakageAndGroupAnalysis:
    """Tests for speaker disjointness, group breakdowns, and shortcut correlation."""

    def test_verify_speaker_disjointness_clean(self):
        tr = [{"speaker_id": "spk_1"}, {"speaker_id": "spk_2"}]
        dv = [{"speaker_id": "spk_3"}, {"speaker_id": "spk_4"}]
        ev = [{"speaker_id": "spk_5"}, {"speaker_id": "spk_6"}]
        report = verify_speaker_disjointness(tr, dv, ev)
        assert report["is_all_disjoint"] is True
        assert report["train_dev_leakage_count"] == 0

    def test_verify_speaker_disjointness_detects_leakage(self):
        tr = [{"speaker_id": "spk_1"}, {"speaker_id": "spk_2"}]
        dv = [{"speaker_id": "spk_2"}, {"speaker_id": "spk_3"}]  # spk_2 leaked!
        report = verify_speaker_disjointness(tr, dv)
        assert report["is_train_dev_disjoint"] is False
        assert report["train_dev_leakage_count"] == 1
        assert "spk_2" in report["train_dev_leaked_speakers"]

    def test_group_performance_by_system(self):
        manifest_df = pd.DataFrame([
            {"system_id": "-", "label": 0},
            {"system_id": "-", "label": 0},
            {"system_id": "A01", "label": 1},
            {"system_id": "A01", "label": 1},
            {"system_id": "A07", "label": 1},
        ])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3])  # A07 is misclassified
        res = group_performance_by_system(manifest_df, y_scores, threshold=0.5)

        assert "-" in res
        assert "A01" in res
        assert "A07" in res
        assert res["-"]["accuracy"] == 1.0
        assert res["A01"]["accuracy"] == 1.0
        assert res["A07"]["accuracy"] == 0.0

    def test_analyze_acoustic_shortcuts_correlation(self):
        waveforms = [
            np.ones(1000, dtype=np.float32) * 0.1,
            np.ones(1000, dtype=np.float32) * 0.5,
            np.ones(1000, dtype=np.float32) * 0.9,
        ]
        # Scores strongly correlated with energy
        y_scores = np.array([0.1, 0.5, 0.9])
        res = analyze_acoustic_shortcuts(waveforms, y_scores)
        assert abs(res["correlation_rms_energy"] - 1.0) < 1e-3
        assert res["potential_rms_shortcut"] is True


class TestCDADDFixtureGeneration:
    """Tests for external held-out CD-ADD benchmark fixtures."""

    def test_create_cd_add_test_fixtures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_manifest = Path(tmpdir) / "cd_add_test.csv"
            res = create_cd_add_test_fixtures(
                output_dir=tmpdir,
                manifest_path=str(out_manifest),
                n_bonafide=5,
                n_spoof=5,
                seed=42,
            )

            assert res["split"] == "test"
            assert res["total_samples"] == 10
            assert out_manifest.exists()

            df = pd.read_csv(out_manifest)
            assert len(df) == 10
            assert (df["label"] == 0).sum() == 5
            assert (df["label"] == 1).sum() == 5
            # Ensure strictly external speaker IDs
            assert all(s.startswith("CD_SPK_") for s in df["speaker_id"])

