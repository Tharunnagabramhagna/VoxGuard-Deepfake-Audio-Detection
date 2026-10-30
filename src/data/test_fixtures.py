"""Synthetic audio test fixture generator for VoxGuard.

Creates deterministic, reproducible synthetic audio files and protocol files
for offline testing without requiring real dataset downloads.  Uses only
NumPy and scipy (both available in the base environment).

Generated audio is NOT realistic speech — it consists of sine tones and noise
at controlled frequencies and amplitudes, sufficient to test pipeline logic
(loading, resampling, trimming, crop/pad, feature extraction shapes) but
NOT suitable for acoustic model evaluation.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def generate_sine_tone(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sine tone with optional additive noise.

    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        frequency: Sine frequency in Hz.
        amplitude: Peak amplitude [0.0, 1.0].
        seed: Random seed for noise component.

    Returns:
        Float32 waveform array.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    noise = rng.randn(n_samples).astype(np.float32) * 0.01
    return (tone + noise).astype(np.float32)


def generate_noise(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate white noise.

    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: RMS amplitude.
        seed: Random seed.

    Returns:
        Float32 noise array.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * sample_rate)
    return (rng.randn(n_samples) * amplitude).astype(np.float32)


def generate_silent(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate silent audio (all zeros).

    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Zero-valued float32 array.
    """
    n_samples = int(duration_sec * sample_rate)
    return np.zeros(n_samples, dtype=np.float32)


def write_wav_file(
    file_path: str,
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> None:
    """Write a waveform to a WAV file using scipy.

    Args:
        file_path: Output file path.
        waveform: Float32 waveform array.
        sample_rate: Sample rate in Hz.
    """
    from scipy.io import wavfile

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to int16 for WAV
    waveform_int16 = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform_int16 * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, waveform_int16)


def create_test_fixtures(
    output_dir: str = "tests/fixtures",
    n_bonafide: int = 10,
    n_spoof: int = 20,
    n_speakers: int = 6,
    sample_rate: int = 16000,
    seed: int = 42,
) -> Dict[str, object]:
    """Create a complete set of synthetic test fixtures.

    Generates:
        - WAV audio files in output_dir/audio/
        - Protocol files matching ASVspoof 2019 format in output_dir/protocols/
        - Manifest CSV files in output_dir/manifests/

    Creates THREE partitions (train, dev, eval) with strictly disjoint speakers.
    Train & dev use known spoof algorithms (A01-A04); eval uses unseen (A07-A10).

    Args:
        output_dir: Root directory for test fixtures.
        n_bonafide: Number of bonafide samples per partition.
        n_spoof: Number of spoof samples per partition.
        n_speakers: Total number of speakers (divided among partitions, min 6).
        sample_rate: Audio sample rate.
        seed: Global random seed.

    Returns:
        Summary of generated fixtures.
    """
    rng = np.random.RandomState(seed)
    out = Path(output_dir)
    audio_dir = out / "audio"
    proto_dir = out / "protocols"
    manifest_dir = out / "manifests"

    for d in [audio_dir, proto_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create speaker pools — ensure strictly disjoint partitions
    n_spk = max(n_speakers, 6)
    speakers = [f"LA_{i:04d}" for i in range(n_spk)]
    spk_per_part = len(speakers) // 3
    train_speakers = speakers[:spk_per_part]
    dev_speakers = speakers[spk_per_part : 2 * spk_per_part]
    eval_speakers = speakers[2 * spk_per_part :]

    train_dev_systems = ["-", "A01", "A02", "A03", "A04"]
    eval_systems = ["-", "A07", "A08", "A09", "A10"]  # Unseen algorithms

    partition_configs = [
        ("train", "T", train_speakers, train_dev_systems),
        ("dev", "D", dev_speakers, train_dev_systems),
        ("eval", "E", eval_speakers, eval_systems),
    ]

    all_records: Dict[str, List[Dict]] = {"train": [], "dev": [], "eval": []}
    file_counter = 0

    for partition, partition_prefix, partition_speakers, systems in partition_configs:
        audio_subdir = audio_dir / partition
        audio_subdir.mkdir(parents=True, exist_ok=True)

        for label_str, n_samples in [("bonafide", n_bonafide), ("spoof", n_spoof)]:
            for i in range(n_samples):
                file_counter += 1
                utterance_id = f"LA_{partition_prefix}_{file_counter:07d}"
                speaker_id = partition_speakers[i % len(partition_speakers)]

                if label_str == "bonafide":
                    system_id = "-"
                    freq = 200 + rng.randint(0, 300)
                    duration = 1.0 + rng.random() * 5.0
                    waveform = generate_sine_tone(
                        duration, sample_rate, freq, 0.5, seed=seed + file_counter
                    )
                else:
                    system_id = systems[1 + (i % (len(systems) - 1))]
                    freq = 300 + rng.randint(0, 500)
                    duration = 1.0 + rng.random() * 5.0
                    waveform = generate_sine_tone(
                        duration, sample_rate, freq, 0.4, seed=seed + file_counter
                    )

                wav_path = audio_subdir / f"{utterance_id}.wav"
                write_wav_file(str(wav_path), waveform, sample_rate)

                all_records[partition].append({
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "system_id": system_id,
                    "label_str": label_str,
                    "label": 0 if label_str == "bonafide" else 1,
                    "partition": partition,
                })

    # Write protocol files (ASVspoof format)
    for partition in ["train", "dev", "eval"]:
        suffix = "trn" if partition == "train" else "trl"
        proto_filename = f"ASVspoof2019.LA.cm.{partition}.{suffix}.txt"
        proto_path = proto_dir / proto_filename

        with open(proto_path, "w", encoding="utf-8") as f:
            for rec in all_records[partition]:
                # ASVspoof format: speaker_id utterance_id system_id - label_str
                f.write(
                    f"{rec['speaker_id']} {rec['utterance_id']} "
                    f"{rec['system_id']} - {rec['label_str']}\n"
                )

    # Write manifest CSVs
    columns = ["utterance_id", "speaker_id", "system_id", "label_str", "label", "partition"]
    for partition in ["train", "dev", "eval"]:
        csv_path = manifest_dir / f"asvspoof_{partition}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records[partition])

    # Create one corrupt file for testing
    corrupt_path = audio_dir / "corrupt_file.wav"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_WAV_FILE_HEADER_GARBAGE_DATA")

    # Create one silent file for testing
    silent_waveform = generate_silent(2.0, sample_rate)
    silent_path = audio_dir / "silent_file.wav"
    write_wav_file(str(silent_path), silent_waveform, sample_rate)

    return {
        "output_dir": str(out),
        "partitions": {
            p: len(recs) for p, recs in all_records.items()
        },
        "train_speakers": train_speakers,
        "dev_speakers": dev_speakers,
        "eval_speakers": eval_speakers,
        "total_audio_files": file_counter,
        "corrupt_file": str(corrupt_path),
        "silent_file": str(silent_path),
    }


def create_cd_add_test_fixtures(
    output_dir: str = "data/raw/cd_add",
    manifest_path: str = "data/manifests/cd_add_test.csv",
    n_bonafide: int = 20,
    n_spoof: int = 20,
    sample_rate: int = 16000,
    seed: int = 42,
) -> Dict[str, object]:
    """Generate reproducible CD-ADD held-out test fixtures for offline evaluation.

    Strict Evaluation-Only Guarantee:
        - Designated exclusively for post-training generalization evaluation.
        - NEVER used for model training, validation early stopping, or threshold tuning.
        - Uses distinct unseen external speaker IDs and external generator IDs.

    Args:
        output_dir: Audio output folder.
        manifest_path: Output CSV manifest file path.
        n_bonafide: Count of genuine speech samples (default 20).
        n_spoof: Count of synthetic speech samples (default 20).
        sample_rate: Standard 16 kHz sampling rate.
        seed: Random seed for deterministic generation.

    Returns:
        Summary dict of generated CD-ADD test partition.
    """
    rng = np.random.RandomState(seed)
    out_audio = Path(output_dir) / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)

    records = []
    file_id = 0

    external_speakers = [f"CD_SPK_{i:02d}" for i in range(1, 9)]
    external_systems = ["CD_TTS_FastPitch", "CD_TTS_VITS", "CD_VC_FreeVC", "CD_Diffusion_DiffWave"]

    for label_str, count in [("bonafide", n_bonafide), ("spoof", n_spoof)]:
        for i in range(count):
            file_id += 1
            utterance_id = f"CD_TEST_{file_id:05d}"
            speaker_id = external_speakers[i % len(external_speakers)]

            if label_str == "bonafide":
                system_id = "-"
                freq = 180 + rng.randint(0, 220)
                duration = 2.0 + rng.random() * 2.5
                waveform = generate_sine_tone(duration, sample_rate, freq, 0.5, seed=seed + file_id)
            else:
                system_id = external_systems[i % len(external_systems)]
                freq = 280 + rng.randint(0, 350)
                duration = 2.0 + rng.random() * 2.5
                waveform = generate_sine_tone(duration, sample_rate, freq, 0.45, seed=seed + file_id)

            wav_path = out_audio / f"{utterance_id}.wav"
            write_wav_file(str(wav_path), waveform, sample_rate)

            records.append({
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "system_id": system_id,
                "label_str": label_str,
                "label": 0 if label_str == "bonafide" else 1,
                "partition": "cd_add_test",
                "audio_path": str(wav_path),
            })

    # Save manifest CSV
    m_path = Path(manifest_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["utterance_id", "speaker_id", "system_id", "label_str", "label", "partition", "audio_path"]
    with open(m_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    return {
        "dataset": "CD-ADD",
        "split": "test",
        "total_samples": len(records),
        "class_distribution": {
            "bonafide": n_bonafide,
            "spoof": n_spoof,
        },
        "seed": seed,
        "selection_method": "deterministic_held_out_fixture",
        "manifest_path": str(m_path),
        "audio_dir": str(out_audio),
    }

