"""Comprehensive tests for VoxGuard Phase 2: data ingestion, auditing, preprocessing.

Tests cover:
  - Protocol parsing and label mapping
  - Speaker disjointness across partitions
  - Duplicate utterance detection (cross-partition and within-partition)
  - Manifest creation and round-trip CSV persistence
  - Audio preprocessing (resample, mono, trim, crop/pad, normalize)
  - Corrupt and silent file handling
  - Quick-experiment subset creation and seed determinism
  - CD-ADD loader policy enforcement (eval-only, no remote code)
  - Metadata isolation (speaker/system IDs excluded from features)
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.data.protocol_parser import (
    LABEL_MAP,
    MANIFEST_COLUMNS,
    get_class_distribution,
    get_system_distribution,
    parse_protocol_file,
    parse_protocol_line,
    validate_no_duplicate_utterances,
    validate_speaker_disjointness,
)
from src.data.preprocessor import (
    normalize_duration,
    peak_normalize,
    to_mono,
    trim_silence,
    preprocess_audio,
)
from src.data.manifest import (
    create_quick_subset,
    load_manifest_csv,
    records_to_csv,
)
from src.data.cd_add_loader import CDADDEvalOnlyError, build_cd_add_manifest
from src.data.test_fixtures import (
    create_test_fixtures,
    generate_noise,
    generate_silent,
    generate_sine_tone,
    write_wav_file,
)
from src.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    """Create a temporary directory with complete synthetic test fixtures."""
    tmpdir = tmp_path_factory.mktemp("voxguard_fixtures")
    result = create_test_fixtures(
        output_dir=str(tmpdir),
        n_bonafide=10,
        n_spoof=20,
        n_speakers=4,
        seed=42,
    )
    result["base_dir"] = str(tmpdir)
    return result


@pytest.fixture(scope="module")
def protocol_dir(fixture_dir):
    """Return path to generated protocol files."""
    return str(Path(fixture_dir["base_dir"]) / "protocols")


@pytest.fixture(scope="module")
def audio_dir(fixture_dir):
    """Return path to generated audio files."""
    return str(Path(fixture_dir["base_dir"]) / "audio")


@pytest.fixture(scope="module")
def manifest_dir(fixture_dir):
    """Return path to generated manifest files."""
    return str(Path(fixture_dir["base_dir"]) / "manifests")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Protocol Parsing & Label Mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestProtocolParsing:
    """Tests for ASVspoof 2019 LA protocol file parsing."""

    def test_label_map_has_both_classes(self):
        """LABEL_MAP must contain exactly bonafide=0 and spoof=1."""
        assert LABEL_MAP["bonafide"] == 0
        assert LABEL_MAP["spoof"] == 1
        assert len(LABEL_MAP) == 2

    def test_parse_5_column_bonafide_line(self):
        """Parse a standard 5-column bonafide protocol line."""
        line = "LA_0079 LA_T_1138216 - - bonafide"
        result = parse_protocol_line(line)
        assert result is not None
        assert result["speaker_id"] == "LA_0079"
        assert result["utterance_id"] == "LA_T_1138216"
        assert result["system_id"] == "-"
        assert result["label_str"] == "bonafide"

    def test_parse_5_column_spoof_line(self):
        """Parse a standard 5-column spoof protocol line."""
        line = "LA_0079 LA_T_1138215 A04 - spoof"
        result = parse_protocol_line(line)
        assert result is not None
        assert result["speaker_id"] == "LA_0079"
        assert result["utterance_id"] == "LA_T_1138215"
        assert result["system_id"] == "A04"
        assert result["label_str"] == "spoof"

    def test_parse_4_column_line(self):
        """Parse a 4-column protocol line (alternative format)."""
        line = "LA_0079 LA_T_1138216 - bonafide"
        result = parse_protocol_line(line)
        assert result is not None
        assert result["label_str"] == "bonafide"

    def test_parse_blank_line_returns_none(self):
        """Blank lines should return None."""
        assert parse_protocol_line("") is None
        assert parse_protocol_line("   ") is None
        assert parse_protocol_line("\n") is None

    def test_parse_malformed_line_returns_none(self):
        """Lines with too few fields should return None."""
        assert parse_protocol_line("LA_0079 LA_T_1138216") is None

    def test_parse_unknown_label_returns_none(self):
        """Lines with invalid labels should return None."""
        assert parse_protocol_line("LA_0079 LA_T_1 A01 - unknown_label") is None

    def test_parse_protocol_file(self, protocol_dir):
        """Parse a complete protocol file and verify record structure."""
        train_proto = str(Path(protocol_dir) / "ASVspoof2019.LA.cm.train.trn.txt")
        records = parse_protocol_file(train_proto, "train")

        assert len(records) > 0
        for rec in records:
            assert "speaker_id" in rec
            assert "utterance_id" in rec
            assert "system_id" in rec
            assert "label_str" in rec
            assert "label" in rec
            assert rec["label"] in (0, 1)
            assert rec["partition"] == "train"
            assert rec["label_str"] in ("bonafide", "spoof")

    def test_missing_protocol_file_raises(self):
        """Attempting to parse a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_protocol_file("/nonexistent/protocol.txt", "train")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Partition Integrity & Speaker Disjointness
# ─────────────────────────────────────────────────────────────────────────────

class TestPartitionIntegrity:
    """Tests ensuring official partition boundaries are preserved."""

    def test_speaker_disjointness(self, fixture_dir):
        """Train, dev, and eval partitions must have completely disjoint speakers."""
        records = []
        for partition in ["train", "dev", "eval"]:
            manifest = Path(fixture_dir["base_dir"]) / "manifests" / f"asvspoof_{partition}.csv"
            rows = load_manifest_csv(str(manifest))
            for r in rows:
                r["partition"] = partition
            records.extend(rows)

        result = validate_speaker_disjointness(records)
        assert result["is_disjoint"], (
            f"Speaker overlap detected: {result['overlapping_speakers']}"
        )

    def test_fixture_speakers_are_known_disjoint(self, fixture_dir):
        """The fixture generator explicitly creates disjoint speaker pools across all 3 partitions."""
        train_spk = set(fixture_dir["train_speakers"])
        dev_spk = set(fixture_dir["dev_speakers"])
        eval_spk = set(fixture_dir.get("eval_speakers", []))
        assert train_spk & dev_spk == set(), f"Overlap train-dev: {train_spk & dev_spk}"
        if eval_spk:
            assert train_spk & eval_spk == set(), f"Overlap train-eval: {train_spk & eval_spk}"
            assert dev_spk & eval_spk == set(), f"Overlap dev-eval: {dev_spk & eval_spk}"

    def test_no_cross_partition_duplicate_utterances(self, fixture_dir):
        """No utterance ID should appear in multiple partitions."""
        records = []
        for partition in ["train", "dev", "eval"]:
            manifest = Path(fixture_dir["base_dir"]) / "manifests" / f"asvspoof_{partition}.csv"
            rows = load_manifest_csv(str(manifest))
            for r in rows:
                r["partition"] = partition
            records.extend(rows)

        result = validate_no_duplicate_utterances(records)
        assert not result["has_cross_partition_leaks"], (
            f"Cross-partition duplicates: {result['cross_partition_duplicates']}"
        )

    def test_injected_cross_partition_duplicate_detected(self):
        """Verify duplicate detection catches a planted cross-partition leak."""
        records = [
            {"utterance_id": "dup_001", "speaker_id": "S1", "partition": "train",
             "label_str": "bonafide", "label": 0, "system_id": "-"},
            {"utterance_id": "dup_001", "speaker_id": "S1", "partition": "dev",
             "label_str": "bonafide", "label": 0, "system_id": "-"},
            {"utterance_id": "unique_002", "speaker_id": "S2", "partition": "train",
             "label_str": "spoof", "label": 1, "system_id": "A01"},
        ]
        result = validate_no_duplicate_utterances(records)
        assert result["has_cross_partition_leaks"]
        assert "dup_001" in result["cross_partition_duplicates"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Audio Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioPreprocessing:
    """Tests for audio preprocessing pipeline operations."""

    def test_mono_conversion_from_stereo(self):
        """Stereo audio (2, N) should be averaged to mono (N,)."""
        stereo = np.random.randn(2, 16000).astype(np.float32)
        mono = to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == 16000
        # Verify it's the mean of channels
        expected = np.mean(stereo, axis=0)
        np.testing.assert_allclose(mono, expected, rtol=1e-5)

    def test_mono_passthrough(self):
        """Already-mono audio should pass through unchanged."""
        mono_in = np.random.randn(16000).astype(np.float32)
        mono_out = to_mono(mono_in)
        np.testing.assert_array_equal(mono_in, mono_out)

    def test_normalize_duration_pad_repeat(self):
        """Short waveform should be repeat-padded to target length."""
        short = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = normalize_duration(short, target_samples=10, pad_mode="repeat")
        assert len(result) == 10
        # First 4 samples should match the original
        np.testing.assert_array_equal(result[:4], short)
        # Repeat pattern
        np.testing.assert_array_equal(result[4:8], short)

    def test_normalize_duration_pad_zero(self):
        """Short waveform should be center zero-padded to target length."""
        short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = normalize_duration(short, target_samples=9, pad_mode="zero")
        assert len(result) == 9
        # Center position: (9-3)//2 = 3
        np.testing.assert_array_equal(result[3:6], short)
        # Padded regions should be zero
        np.testing.assert_array_equal(result[:3], np.zeros(3))
        np.testing.assert_array_equal(result[6:], np.zeros(3))

    def test_normalize_duration_center_crop(self):
        """Long waveform should be center-cropped to target length."""
        long = np.arange(100, dtype=np.float32)
        result = normalize_duration(long, target_samples=20)
        assert len(result) == 20
        # Center crop: start = (100-20)//2 = 40
        np.testing.assert_array_equal(result, long[40:60])

    def test_normalize_duration_exact_length_passthrough(self):
        """Waveform exactly at target length should pass through."""
        exact = np.ones(64000, dtype=np.float32)
        result = normalize_duration(exact, target_samples=64000)
        assert len(result) == 64000
        np.testing.assert_array_equal(result, exact)

    def test_normalize_duration_empty_waveform(self):
        """Empty waveform with repeat-pad should produce zeros."""
        empty = np.array([], dtype=np.float32)
        result = normalize_duration(empty, target_samples=100, pad_mode="repeat")
        assert len(result) == 100
        assert np.all(result == 0)

    def test_peak_normalize(self):
        """Peak normalization should scale max amplitude to 1.0."""
        signal = np.array([0.0, 0.5, -0.25, 0.3], dtype=np.float32)
        result = peak_normalize(signal)
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-6

    def test_peak_normalize_silent(self):
        """Silent audio should remain silent after peak normalization."""
        silent = np.zeros(100, dtype=np.float32)
        result = peak_normalize(silent)
        np.testing.assert_array_equal(result, silent)

    def test_trim_silence_removes_leading_trailing_zeros(self):
        """Silence trimming should remove leading/trailing silent segments."""
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)  # 1s silence
        tone = generate_sine_tone(1.0, sr, 440, 0.5, seed=0)
        # Create: 1s silence + 1s tone + 1s silence
        waveform = np.concatenate([silence, tone, silence])
        trimmed = trim_silence(waveform, top_db=30)
        # Trimmed should be significantly shorter than original
        assert len(trimmed) < len(waveform)
        # Trimmed should still contain most of the tone
        assert len(trimmed) >= sr * 0.5  # At least half the tone

    def test_full_preprocess_pipeline_output_shape(self):
        """Full pipeline should produce exactly target_samples output."""
        sr = 16000
        waveform = generate_sine_tone(2.5, sr, 300, 0.5, seed=123)
        config = {
            "sample_rate": 16000,
            "target_samples": 64000,
            "pad_mode": "repeat",
            "trim_silence": True,
            "silence_top_db": 30,
            "normalize_loudness": True,
        }
        result, meta = preprocess_audio(waveform, sr, config)
        assert len(result) == 64000
        assert result.dtype == np.float32
        assert np.max(np.abs(result)) <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 4. Corrupt & Silent File Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestCorruptFileHandling:
    """Tests for handling corrupt and problematic audio files."""

    def test_corrupt_file_detection(self, fixture_dir):
        """Loading a corrupt file should return None with an error message."""
        from src.data.preprocessor import load_audio_file
        corrupt_path = fixture_dir["corrupt_file"]
        waveform, sr, error = load_audio_file(corrupt_path)
        # Should fail gracefully
        assert waveform is None or error is not None

    def test_silent_file_detection(self, fixture_dir):
        """Silent files should be detected by validation."""
        from src.data.preprocessor import validate_audio_file
        silent_path = fixture_dir["silent_file"]
        report = validate_audio_file(silent_path)
        assert report["exists"]
        assert report["is_silent"]

    def test_nonexistent_file_detection(self):
        """Loading a nonexistent file should return an error."""
        from src.data.preprocessor import load_audio_file
        waveform, sr, error = load_audio_file("/nonexistent/fake_audio.wav")
        assert waveform is None
        assert error is not None
        assert "not found" in error.lower()

    def test_validate_nonexistent_file(self):
        """Validating a nonexistent file should report it missing."""
        from src.data.preprocessor import validate_audio_file
        report = validate_audio_file("/nonexistent/fake.wav")
        assert not report["exists"]
        assert report["is_corrupt"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Manifest Persistence & Round-Trip
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestPersistence:
    """Tests for CSV manifest creation and loading."""

    def test_manifest_round_trip(self, tmp_path):
        """Records written to CSV should load back identically."""
        records = [
            {"utterance_id": "U001", "speaker_id": "S1", "system_id": "-",
             "label_str": "bonafide", "label": 0, "partition": "train"},
            {"utterance_id": "U002", "speaker_id": "S2", "system_id": "A01",
             "label_str": "spoof", "label": 1, "partition": "train"},
        ]
        csv_path = str(tmp_path / "test_manifest.csv")
        n_written = records_to_csv(records, csv_path)
        assert n_written == 2

        loaded = load_manifest_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0]["utterance_id"] == "U001"
        assert loaded[0]["label_str"] == "bonafide"
        assert loaded[1]["system_id"] == "A01"

    def test_manifest_from_fixtures(self, manifest_dir):
        """Fixture-generated manifests should be loadable."""
        train_csv = str(Path(manifest_dir) / "asvspoof_train.csv")
        records = load_manifest_csv(train_csv)
        assert len(records) > 0
        # All records should have required columns
        for rec in records:
            assert "utterance_id" in rec
            assert "speaker_id" in rec
            assert "label_str" in rec

    def test_load_nonexistent_manifest_raises(self):
        """Loading a nonexistent manifest should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_manifest_csv("/nonexistent/manifest.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quick-Experiment Subset
# ─────────────────────────────────────────────────────────────────────────────

class TestQuickSubset:
    """Tests for seeded quick-experiment subset creation."""

    def test_subset_size_capped(self, manifest_dir, tmp_path):
        """Subset should not exceed samples_per_class per class."""
        train_csv = str(Path(manifest_dir) / "asvspoof_train.csv")
        subset_csv = str(tmp_path / "subset.csv")
        summary = create_quick_subset(train_csv, subset_csv,
                                      samples_per_class=5, seed=42)
        assert summary["total_subset_samples"] <= 5 * 2  # 2 classes

    def test_subset_is_strict_subset_of_original(self, manifest_dir, tmp_path):
        """Every utterance in the subset must exist in the original partition."""
        train_csv = str(Path(manifest_dir) / "asvspoof_train.csv")
        subset_csv = str(tmp_path / "subset.csv")
        create_quick_subset(train_csv, subset_csv, samples_per_class=5, seed=42)

        original = load_manifest_csv(train_csv)
        subset = load_manifest_csv(subset_csv)

        original_ids = {r["utterance_id"] for r in original}
        subset_ids = {r["utterance_id"] for r in subset}

        assert subset_ids.issubset(original_ids), (
            f"Subset contains utterances not in original: {subset_ids - original_ids}"
        )

    def test_subset_seed_determinism(self, manifest_dir, tmp_path):
        """Same seed should always produce the same subset."""
        train_csv = str(Path(manifest_dir) / "asvspoof_train.csv")

        subset1_csv = str(tmp_path / "subset1.csv")
        subset2_csv = str(tmp_path / "subset2.csv")

        create_quick_subset(train_csv, subset1_csv, samples_per_class=5, seed=42)
        create_quick_subset(train_csv, subset2_csv, samples_per_class=5, seed=42)

        subset1 = load_manifest_csv(subset1_csv)
        subset2 = load_manifest_csv(subset2_csv)

        ids1 = [r["utterance_id"] for r in subset1]
        ids2 = [r["utterance_id"] for r in subset2]
        assert ids1 == ids2, "Same seed should produce identical subsets."

    def test_different_seed_produces_different_subset(self, manifest_dir, tmp_path):
        """Different seeds should generally produce different subsets."""
        train_csv = str(Path(manifest_dir) / "asvspoof_train.csv")

        subset1_csv = str(tmp_path / "subset_a.csv")
        subset2_csv = str(tmp_path / "subset_b.csv")

        create_quick_subset(train_csv, subset1_csv, samples_per_class=5, seed=42)
        create_quick_subset(train_csv, subset2_csv, samples_per_class=5, seed=99)

        subset1 = load_manifest_csv(subset1_csv)
        subset2 = load_manifest_csv(subset2_csv)

        ids1 = {r["utterance_id"] for r in subset1}
        ids2 = {r["utterance_id"] for r in subset2}
        # With sufficient data and different seeds, subsets should differ
        # (not guaranteed for very small datasets, but highly likely)
        # We check they're at least not identical sets
        # For very small pools this might coincidentally match
        if len(ids1) > 3:
            assert ids1 != ids2 or len(ids1) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# 7. CD-ADD Eval-Only Policy Enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestCDADDPolicy:
    """Tests ensuring CD-ADD is strictly evaluation-only."""

    def test_cd_add_rejects_train_split(self):
        """CD-ADD loader must reject any split other than 'test'."""
        with pytest.raises(CDADDEvalOnlyError):
            build_cd_add_manifest(split="train")

    def test_cd_add_rejects_validation_split(self):
        """CD-ADD loader must reject 'validation' split."""
        with pytest.raises(CDADDEvalOnlyError):
            build_cd_add_manifest(split="validation")

    def test_cd_add_rejects_trust_remote_code(self):
        """CD-ADD loader must reject trust_remote_code=True."""
        with pytest.raises(RuntimeError, match="trust_remote_code"):
            build_cd_add_manifest(split="test", trust_remote_code=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Metadata Isolation (Prevent Leakage into Features)
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataIsolation:
    """Verify that metadata fields are not embedded in classifier inputs."""

    def test_manifest_columns_defined(self):
        """MANIFEST_COLUMNS should include metadata but NOT feature data."""
        # These columns exist for auditing/provenance only
        assert "speaker_id" in MANIFEST_COLUMNS
        assert "system_id" in MANIFEST_COLUMNS
        assert "partition" in MANIFEST_COLUMNS
        # The actual feature input to classifiers is the audio waveform/spectrogram,
        # which is extracted separately and never includes these text fields

    def test_preprocessed_audio_is_pure_numerical(self):
        """Preprocessed output should be a pure numerical array with no metadata."""
        waveform = generate_sine_tone(2.0, 16000, 440, 0.5, seed=0)
        config = {
            "sample_rate": 16000,
            "target_samples": 64000,
            "pad_mode": "repeat",
            "trim_silence": False,
            "normalize_loudness": True,
        }
        result, meta = preprocess_audio(waveform, 16000, config)

        # Result should be a pure float array
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        # Metadata is returned separately, never embedded in the waveform
        assert isinstance(meta, dict)
        assert "speaker_id" not in meta  # No speaker info in feature output
        assert "system_id" not in meta  # No system info in feature output


# ─────────────────────────────────────────────────────────────────────────────
# 9. Synthetic Test Fixture Generator
# ─────────────────────────────────────────────────────────────────────────────

class TestFixtureGenerator:
    """Tests for the synthetic test fixture generation utility."""

    def test_sine_tone_shape_and_dtype(self):
        """Sine tone should have correct length and float32 dtype."""
        tone = generate_sine_tone(1.0, 16000, 440, 0.5, seed=0)
        assert len(tone) == 16000
        assert tone.dtype == np.float32

    def test_noise_shape(self):
        """Noise should have correct length."""
        noise = generate_noise(0.5, 16000, seed=0)
        assert len(noise) == 8000

    def test_silent_is_all_zeros(self):
        """Silent audio should be exactly zero."""
        silent = generate_silent(1.0, 16000)
        assert len(silent) == 16000
        assert np.all(silent == 0)

    def test_fixture_creates_audio_files(self, fixture_dir):
        """Fixture generator should create actual audio files on disk."""
        audio_dir = Path(fixture_dir["base_dir"]) / "audio"
        wav_files = list(audio_dir.rglob("*.wav"))
        # Should have files for both partitions + corrupt + silent
        assert len(wav_files) >= fixture_dir["total_audio_files"]

    def test_fixture_creates_protocol_files(self, fixture_dir):
        """Fixture generator should create protocol text files."""
        proto_dir = Path(fixture_dir["base_dir"]) / "protocols"
        proto_files = list(proto_dir.glob("*.txt"))
        assert len(proto_files) >= 2  # train and dev

    def test_fixture_wav_files_are_readable(self, fixture_dir):
        """Generated WAV files should be loadable."""
        from src.data.preprocessor import load_audio_file
        audio_dir = Path(fixture_dir["base_dir"]) / "audio" / "train"
        wav_files = list(audio_dir.glob("*.wav"))
        assert len(wav_files) > 0

        waveform, sr, error = load_audio_file(str(wav_files[0]))
        assert waveform is not None, f"Failed to load {wav_files[0]}: {error}"
        assert sr == 16000
        assert len(waveform) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 10. Class Distribution & Balance Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestClassDistribution:
    """Tests for class distribution computation."""

    def test_class_distribution_counts(self, fixture_dir):
        """Class distribution should reflect fixture generation parameters."""
        proto_dir = Path(fixture_dir["base_dir"]) / "protocols"
        records = parse_protocol_file(
            str(proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"), "train"
        )
        dist = get_class_distribution(records)

        assert "train" in dist
        assert dist["train"]["bonafide"] == 10  # n_bonafide
        assert dist["train"]["spoof"] == 20     # n_spoof

    def test_system_distribution(self, fixture_dir):
        """System distribution should reflect generated spoofing algorithms."""
        proto_dir = Path(fixture_dir["base_dir"]) / "protocols"
        records = parse_protocol_file(
            str(proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"), "train"
        )
        sys_dist = get_system_distribution(records)

        assert "train" in sys_dist
        # Bonafide uses "-", spoof uses A01-A04
        assert "-" in sys_dist["train"]
        assert sys_dist["train"]["-"] == 10  # All bonafide samples


# ─────────────────────────────────────────────────────────────────────────────
# 11. Global Seed Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedReproducibility:
    """Tests for deterministic seed behavior."""

    def test_seed_produces_identical_random_sequences(self):
        """Setting the same seed must produce identical random outputs."""
        import random

        set_seed(42)
        seq1 = [random.random() for _ in range(10)]
        np_seq1 = np.random.rand(10).tolist()

        set_seed(42)
        seq2 = [random.random() for _ in range(10)]
        np_seq2 = np.random.rand(10).tolist()

        assert seq1 == seq2
        assert np_seq1 == np_seq2

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds should produce different random outputs."""
        set_seed(42)
        seq1 = [np.random.random() for _ in range(10)]

        set_seed(99)
        seq2 = [np.random.random() for _ in range(10)]

        assert seq1 != seq2

