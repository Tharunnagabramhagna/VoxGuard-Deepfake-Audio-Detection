"""Feature extraction script for VoxGuard.

Extracts Log-Mel spectrograms and MFCC summary feature vectors from audio
files referenced in partition manifests, saving processed arrays to data/processed/.

Usage:
    python scripts/extract_features.py --config configs/default.yaml --subset mini
    python scripts/extract_features.py --config configs/default.yaml --subset full
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import load_manifest_csv
from src.data.preprocessor import load_audio_file, preprocess_audio
from src.features.extractor import FeatureExtractor
from src.utils.config import load_config
from src.utils.seed import set_seed


def find_audio_file(
    utterance_id: str, partition: str, raw_dir: Path
) -> Optional[Path]:
    """Locate an audio file by utterance ID in expected directories."""
    extensions = [".flac", ".wav"]
    # Check partition root, flac subfolder, audio subfolder
    subdirs = [
        raw_dir / partition,
        raw_dir / partition / "flac",
        raw_dir / "audio" / partition,
        Path("tests/fixtures/audio") / partition,
    ]
    for subdir in subdirs:
        for ext in extensions:
            candidate = subdir / f"{utterance_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def extract_features_for_partition(
    manifest_path: str,
    raw_dir: Path,
    extractor: FeatureExtractor,
    desc: str = "Extracting",
) -> Tuple[np.ndarray, torch.Tensor, np.ndarray, List[str]]:
    """Extract MFCC summary and Log-Mel spectrograms for all utterances in a manifest.

    Args:
        manifest_path: Path to manifest CSV.
        raw_dir: Base directory containing raw audio.
        extractor: FeatureExtractor instance.
        desc: Progress bar description.

    Returns:
        Tuple of (mfcc_features, mel_tensors, labels, utterance_ids).
    """
    records = load_manifest_csv(manifest_path)
    mfcc_list: List[np.ndarray] = []
    mel_list: List[torch.Tensor] = []
    labels: List[int] = []
    valid_uids: List[str] = []

    missing = 0
    corrupt = 0

    for rec in tqdm(records, desc=desc):
        uid = rec["utterance_id"]
        partition = rec.get("partition", "train")
        label = int(rec["label"])

        audio_path = find_audio_file(uid, partition, raw_dir)
        if audio_path is None:
            missing += 1
            continue

        try:
            feats = extractor.extract_from_file(str(audio_path))
            mfcc_list.append(feats["mfcc_summary"])
            mel_list.append(feats["mel_spectrogram"])
            labels.append(label)
            valid_uids.append(uid)
        except Exception:
            corrupt += 1
            continue

    if not mfcc_list:
        raise RuntimeError(
            f"Zero valid audio files found for {manifest_path} "
            f"(missing={missing}, corrupt={corrupt})."
        )

    X_mfcc = np.stack(mfcc_list, axis=0).astype(np.float32)
    X_mel = torch.stack(mel_list, dim=0)  # Shape: (N, 1, n_mels, time_frames)
    y = np.array(labels, dtype=np.int64)

    return X_mfcc, X_mel, y, valid_uids


def run_extraction(config_path: str = "configs/default.yaml", subset: str = "mini"):
    """Execute full feature extraction pipeline."""
    print("=" * 70)
    print(" VoxGuard Acoustic Feature Extraction")
    print("=" * 70)

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    manifest_dir = Path(cfg["paths"]["manifest_dir"])
    raw_dir = Path(cfg["paths"]["raw_asvspoof_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor(cfg)
    print(f"  Configuration:")
    print(f"    - Sample Rate: {extractor.sample_rate} Hz")
    print(f"    - Target Duration: {extractor.duration_sec}s ({extractor.target_samples} samples)")
    print(f"    - Log-Mel Bins: {extractor.mel_n_mels} (FFT={extractor.mel_n_fft}, Hop={extractor.mel_hop_length})")
    print(f"    - MFCC Coefficients: {extractor.mfcc_n_mfcc} (Deltas={extractor.mfcc_include_deltas})")

    suffix = "_mini.csv" if subset == "mini" else ".csv"
    partitions = ["train", "dev", "eval"]

    for partition in partitions:
        manifest_file = manifest_dir / f"asvspoof_{partition}{suffix}"
        if not manifest_file.exists():
            print(f"  [WARN] Manifest {manifest_file} not found, skipping partition.")
            continue

        print(f"\nProcessing {partition.upper()} ({manifest_file.name})...")
        try:
            X_mfcc, X_mel, y, uids = extract_features_for_partition(
                str(manifest_file), raw_dir, extractor, desc=f"{partition.capitalize()}"
            )
            # Save artifacts
            prefix = f"{partition}_{subset}" if subset != "full" else partition
            mfcc_file = processed_dir / f"{prefix}_mfcc.npy"
            labels_file = processed_dir / f"{prefix}_labels.npy"
            mel_file = processed_dir / f"{prefix}_mel.pt"

            np.save(str(mfcc_file), X_mfcc)
            np.save(str(labels_file), y)
            torch.save(X_mel, str(mel_file))

            print(f"  [OK] Saved {partition} features:")
            print(f"    - MFCC: {X_mfcc.shape} -> {mfcc_file}")
            print(f"    - Log-Mel: {list(X_mel.shape)} -> {mel_file}")
            print(f"    - Labels: {y.shape} (Bonafide={int((y==0).sum())}, Spoof={int((y==1).sum())})")
        except Exception as e:
            print(f"  [FAIL] Error processing {partition}: {e}")

    print("\n" + "=" * 70)
    print(" Feature extraction completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Feature Extraction")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--subset", default="mini", choices=["mini", "full"], help="Subset type")
    args = parser.parse_args()
    run_extraction(args.config, args.subset)
