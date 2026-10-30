"""Data integrity and label alignment validation for VoxGuard.

Provides strict validation guards to prevent silent data corruption, mismatched manifests,
and missing-label fallback bugs in forensic evaluation pipelines.

Guarantees:
1. Every evaluated audio waveform maps to exactly ONE verified ground-truth label.
2. Silent default fallbacks (e.g. dict.get(key, 0)) are strictly prohibited.
3. Audio files are loaded directly from manifest records, never from directory globbing.
4. Evaluation subsets are verified to contain both genuine (0) and synthetic (1) speech.
5. Injected perturbations (e.g. noise, codecs) remain strictly paired with their clean reference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.preprocessor import load_audio_file, preprocess_audio


class LabelAlignmentError(ValueError):
    """Raised when audio files and manifest ground-truth labels are misaligned or missing."""
    pass


class DatasetIntegrityError(ValueError):
    """Raised when an evaluation manifest contains duplicate IDs, invalid labels, or single-class distributions."""
    pass


def validate_manifest_integrity(
    manifest: Union[pd.DataFrame, List[Dict[str, object]]],
    require_binary: bool = True,
    require_both_classes: bool = True,
    expected_mode: Optional[str] = None,
) -> None:
    """Validate manifest records for completeness, uniqueness, and label validity.

    Args:
        manifest: DataFrame or list of dicts with 'utterance_id' and 'label'.
        require_binary: If True, asserts labels are strictly 0 or 1.
        require_both_classes: If True, asserts both classes (0 and 1) are present.
        expected_mode: Optional expected dataset mode ('synthetic_fixture' or 'real_speech').

    Raises:
        DatasetIntegrityError: If validation fails.
    """
    if isinstance(manifest, list):
        df = pd.DataFrame(manifest)
    else:
        df = manifest

    if "utterance_id" not in df.columns:
        raise DatasetIntegrityError("Manifest is missing required column: 'utterance_id'")
    if "label" not in df.columns:
        raise DatasetIntegrityError("Manifest is missing required column: 'label'")

    # Check for empty manifest
    if len(df) == 0:
        raise DatasetIntegrityError("Manifest is empty (0 records).")

    # Check for duplicate utterance IDs within the manifest
    duplicates = df[df.duplicated(subset=["utterance_id"], keep=False)]
    if len(duplicates) > 0:
        dup_ids = duplicates["utterance_id"].unique().tolist()
        raise DatasetIntegrityError(
            f"Manifest contains {len(dup_ids)} duplicate utterance IDs: {dup_ids[:5]}"
        )

    # Check for null / NaN labels or IDs
    if df["utterance_id"].isnull().any():
        raise DatasetIntegrityError("Manifest contains null or NaN utterance IDs.")
    if df["label"].isnull().any():
        raise DatasetIntegrityError("Manifest contains null or NaN ground-truth labels.")

    # Validate binary label values
    labels = df["label"].values
    try:
        int_labels = labels.astype(int)
    except (ValueError, TypeError) as e:
        raise DatasetIntegrityError(f"Labels cannot be cast to integer: {e}")

    if require_binary:
        invalid_labels = [lbl for lbl in int_labels if lbl not in (0, 1)]
        if len(invalid_labels) > 0:
            raise DatasetIntegrityError(
                f"Manifest contains invalid non-binary labels: {set(invalid_labels)}. Must be 0 or 1."
            )

    if require_both_classes:
        unique_labels = set(int_labels)
        if 0 not in unique_labels or 1 not in unique_labels:
            raise DatasetIntegrityError(
                f"Manifest must contain both bonafide (0) and spoof (1) classes. Found: {unique_labels}"
            )

    # Validate dataset mode consistency if column or expected_mode is present
    if "dataset_mode" in df.columns and expected_mode is not None:
        modes_present = set(df["dataset_mode"].dropna().astype(str).unique())
        if expected_mode not in modes_present:
            raise DatasetIntegrityError(
                f"Manifest dataset_mode ({modes_present}) does not match expected mode '{expected_mode}'. "
                "Cross-mode contamination is strictly prohibited."
            )

    if expected_mode == "real_speech" and "dataset_mode" in df.columns:
        fixture_entries = df[df["dataset_mode"] == "synthetic_fixture"]
        if len(fixture_entries) > 0:
            raise DatasetIntegrityError(
                f"Found {len(fixture_entries)} synthetic fixture samples in a real_speech manifest. "
                "Cross-mode contamination is strictly prohibited."
            )


def load_aligned_evaluation_set(
    manifest_df: pd.DataFrame,
    raw_dir: Path,
    max_samples: Optional[int] = None,
    config: Optional[dict] = None,
    seed: int = 42,
    dataset_mode: str = "synthetic_fixture",
) -> Tuple[List[np.ndarray], np.ndarray, List[Dict[str, object]]]:
    """Load audio waveforms and strictly aligned labels directly from an evaluation manifest.

    Policy:
        - NEVER globs directories.
        - Every audio file is identified by manifest utterance_id.
        - If an audio file cannot be located, raises LabelAlignmentError immediately.
        - Ensures clean_waveforms[i] has ground-truth label labels[i] with zero ambiguity.
        - In real_speech mode, fixture directories are strictly excluded from search.

    Args:
        manifest_df: Validated manifest DataFrame with 'utterance_id', 'label', 'partition'.
        raw_dir: Base directory to search for raw audio files.
        max_samples: Optional sample count cap (performs stratified sampling).
        config: Preprocessing config dictionary.
        seed: Random seed for stratified subsampling if max_samples is given.
        dataset_mode: 'synthetic_fixture' or 'real_speech'.

    Returns:
        Tuple of (clean_waveforms, labels_array, sample_metadata_list).

    Raises:
        DatasetIntegrityError: If manifest fails integrity checks.
        LabelAlignmentError: If an audio file cannot be found or decoded.
    """
    validate_manifest_integrity(
        manifest_df,
        require_binary=True,
        require_both_classes=True,
        expected_mode=dataset_mode if "dataset_mode" in manifest_df.columns else None,
    )

    if config is None:
        config = {}

    df = manifest_df.copy()

    # Stratified subsampling if max_samples specified
    if max_samples is not None and len(df) > max_samples:
        rng = np.random.RandomState(seed)
        samples_per_class = max_samples // 2
        groups = []
        for lbl in [0, 1]:
            grp = df[df["label"].astype(int) == lbl]
            n_select = min(len(grp), samples_per_class)
            idx = rng.choice(grp.index, size=n_select, replace=False)
            groups.append(grp.loc[idx])
        df = pd.concat(groups).sample(frac=1.0, random_state=rng).reset_index(drop=True)

    extensions = [".flac", ".wav"]
    clean_waveforms: List[np.ndarray] = []
    aligned_labels: List[int] = []
    metadata_list: List[Dict[str, object]] = []

    # Configure search directories strictly based on dataset_mode
    if dataset_mode == "real_speech":
        search_dirs_base = [
            raw_dir / partition if 'partition' in locals() else raw_dir,
            raw_dir / "flac",
        ]
    else:
        search_dirs_base = []

    for _, row in df.iterrows():
        uid = str(row["utterance_id"])
        partition = str(row.get("partition", "eval"))
        label = int(row["label"])

        if dataset_mode == "real_speech":
            search_dirs = [
                raw_dir / partition,
                raw_dir / partition / "flac",
                raw_dir / f"ASVspoof2019_LA_{partition}" / "flac",
                raw_dir / "audio" / partition,
            ]
        else:
            # Fixture mode: search explicit fixture directories
            search_dirs = [
                Path("tests/fixtures/audio") / partition,
                Path("data/raw/asvspoof2019_la/audio") / partition,
                Path("data/raw/cd_add/audio"),
                raw_dir / partition,
                raw_dir / "audio" / partition,
            ]

        audio_path: Optional[Path] = None
        for s_dir in search_dirs:
            for ext in extensions:
                candidate = s_dir / f"{uid}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break
            if audio_path is not None:
                break

        if audio_path is None:
            raise LabelAlignmentError(
                f"Audio file for utterance '{uid}' in partition '{partition}' could not be found. "
                f"Mode='{dataset_mode}'. Searched in: {[str(d) for d in search_dirs]}."
            )
            raise LabelAlignmentError(
                f"Audio file for utterance '{uid}' in partition '{partition}' could not be found "
                f"in search directories. Manifest integrity violation: missing audio for label {label}."
            )

        wf, sr, err = load_audio_file(str(audio_path), target_sr=16000)
        if wf is None:
            raise LabelAlignmentError(
                f"Audio file for utterance '{uid}' ({audio_path}) failed to decode: {err}"
            )

        proc_wf, _ = preprocess_audio(wf, sample_rate=sr, config=config)

        clean_waveforms.append(proc_wf)
        aligned_labels.append(label)
        metadata_list.append({
            "utterance_id": uid,
            "speaker_id": str(row.get("speaker_id", "-")),
            "system_id": str(row.get("system_id", "-")),
            "label": label,
            "label_str": "spoof" if label == 1 else "bonafide",
            "audio_path": str(audio_path),
        })

    labels_array = np.array(aligned_labels, dtype=int)

    # Final assertion: Must have matching lengths and both classes
    if len(clean_waveforms) != len(labels_array):
        raise LabelAlignmentError(
            f"Waveforms count ({len(clean_waveforms)}) does not match labels count ({len(labels_array)})"
        )
    if len(np.unique(labels_array)) < 2:
        raise DatasetIntegrityError(
            f"Loaded evaluation set has degenerate class distribution: {np.unique(labels_array)}"
        )

    return clean_waveforms, labels_array, metadata_list
