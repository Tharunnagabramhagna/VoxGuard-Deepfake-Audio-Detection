"""CD-ADD external evaluation dataset loader.

This module provides EVALUATION-ONLY access to the Cross-Domain Audio
Deepfake Detection (CD-ADD) benchmark hosted on Hugging Face.

╔══════════════════════════════════════════════════════════════════════╗
║  STRICT POLICY: EVALUATION ONLY — NEVER USE FOR TRAINING           ║
║                                                                    ║
║  1. CD-ADD data must NEVER be used for training or fine-tuning.    ║
║  2. CD-ADD data must NEVER be used to fit preprocessing params     ║
║     (e.g., normalizer means, scaler statistics).                   ║
║  3. CD-ADD data must NEVER be used for model selection or          ║
║     decision threshold tuning.                                     ║
║  4. EER operating thresholds must be derived ONLY from the         ║
║     ASVspoof development partition.                                ║
║  5. CD-ADD results report empirical out-of-domain performance.     ║
╚══════════════════════════════════════════════════════════════════════╝

Security Note:
    The CD-ADD dataset uses standard Parquet format on Hugging Face and
    does NOT require trust_remote_code=True. We explicitly pass
    trust_remote_code=False to prevent execution of any bundled scripts.
    If a future version of the dataset adds a custom loading script that
    requires trust_remote_code=True, loading will deliberately FAIL
    rather than silently executing unreviewed code.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class CDADDEvalOnlyError(Exception):
    """Raised when CD-ADD data is used outside of evaluation context."""

    pass


def load_cd_add_manifest(
    manifest_path: str,
) -> List[Dict[str, object]]:
    """Load a pre-built CD-ADD test manifest CSV.

    Args:
        manifest_path: Path to the CD-ADD manifest CSV.

    Returns:
        List of record dictionaries.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"CD-ADD manifest not found: {path.resolve()}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))

    return records


def build_cd_add_manifest(
    dataset_name: str = "SpeechAntiSpoofingBenchmarks/CD-ADD",
    split: str = "test",
    output_path: str = "data/manifests/cd_add_test.csv",
    max_samples: Optional[int] = None,
    trust_remote_code: bool = False,
    seed: int = 42,
) -> Dict[str, object]:
    """Download and build a manifest from the CD-ADD Hugging Face dataset.

    IMPORTANT: This function requires network access and the `datasets`
    library. It is designed to be run ONCE to create a manifest, not
    during training loops.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to load (must be 'test').
        output_path: Path to write the manifest CSV.
        max_samples: Maximum number of samples to include (None = all).
        trust_remote_code: Must remain False for security.
        seed: Random seed for sampling if max_samples is set.

    Returns:
        Summary dictionary with record counts and metadata.

    Raises:
        CDADDEvalOnlyError: If split is not 'test'.
        RuntimeError: If trust_remote_code is True (blocked by policy).
    """
    # Enforce evaluation-only policy
    if split not in ("test",):
        raise CDADDEvalOnlyError(
            f"CD-ADD may only be loaded with split='test', got split='{split}'. "
            "CD-ADD is strictly held-out evaluation data."
        )

    # Enforce no remote code execution
    if trust_remote_code:
        raise RuntimeError(
            "trust_remote_code=True is blocked by VoxGuard security policy. "
            "The CD-ADD dataset uses standard Parquet format and does not "
            "require remote code execution. If this has changed, review the "
            "loading script manually before enabling."
        )

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to download CD-ADD. "
            "Install with: pip install datasets huggingface_hub"
        )

    # Load the dataset (Parquet format, no remote code needed)
    ds = load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=False,
    )

    records = []
    for i, example in enumerate(ds):
        record = {
            "index": i,
            "partition": "cd_add_test",
        }

        # Extract label
        if "label" in example:
            label_val = example["label"]
            if isinstance(label_val, int):
                record["label"] = label_val
                record["label_str"] = "bonafide" if label_val == 0 else "spoof"
            else:
                record["label_str"] = str(label_val).lower()
                record["label"] = 0 if "bonafide" in record["label_str"] else 1

        # Extract path
        if "path" in example:
            record["utterance_id"] = Path(example["path"]).stem
        elif "audio" in example and isinstance(example["audio"], dict):
            if "path" in example["audio"]:
                record["utterance_id"] = Path(example["audio"]["path"]).stem
            else:
                record["utterance_id"] = f"cd_add_{i:06d}"
        else:
            record["utterance_id"] = f"cd_add_{i:06d}"

        # Extract metadata from notes field if present
        if "notes" in example:
            try:
                notes = json.loads(example["notes"])
                record["speaker_id"] = notes.get("speaker_id", "unknown")
                record["system_id"] = notes.get("system_id", "unknown")
            except (json.JSONDecodeError, TypeError):
                record["speaker_id"] = "unknown"
                record["system_id"] = "unknown"
        else:
            record["speaker_id"] = "unknown"
            record["system_id"] = "unknown"

        records.append(record)

    # Subsample if requested
    if max_samples is not None and len(records) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(records), size=max_samples, replace=False)
        records = [records[i] for i in sorted(indices)]

    # Write manifest
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if records:
        columns = list(records[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

    # Class distribution
    class_dist = {}
    for rec in records:
        label = rec.get("label_str", "unknown")
        class_dist[label] = class_dist.get(label, 0) + 1

    return {
        "total_records": len(records),
        "class_distribution": class_dist,
        "output_path": str(out),
        "dataset_name": dataset_name,
        "split": split,
        "trust_remote_code": False,
    }

