"""Dataset manifest creation and management.

Generates, validates, and persists CSV manifests for the ASVspoof 2019 LA
dataset from parsed protocol records.  Also provides seeded subset creation
that preserves official partition boundaries.
"""

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data.protocol_parser import (
    MANIFEST_COLUMNS,
    get_class_distribution,
    load_all_partitions,
    parse_protocol_file,
    validate_no_duplicate_utterances,
    validate_speaker_disjointness,
)


def records_to_csv(
    records: List[Dict[str, object]],
    output_path: str,
    columns: Optional[List[str]] = None,
) -> int:
    """Write parsed protocol records to a CSV manifest file.

    Args:
        records: List of record dictionaries.
        output_path: Destination CSV file path.
        columns: Column ordering. Defaults to MANIFEST_COLUMNS.

    Returns:
        Number of records written.
    """
    if columns is None:
        columns = MANIFEST_COLUMNS

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    return len(records)


def load_manifest_csv(manifest_path: str) -> List[Dict[str, str]]:
    """Load a CSV manifest file into a list of dictionaries.

    Args:
        manifest_path: Path to the CSV file.

    Returns:
        List of record dictionaries with string values.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_manifests(
    protocol_dir: str,
    manifest_dir: str,
) -> Dict[str, object]:
    """Parse all ASVspoof 2019 LA protocols and write partition manifests.

    Creates three CSV files:
        - asvspoof_train.csv
        - asvspoof_dev.csv
        - asvspoof_eval.csv

    Args:
        protocol_dir: Directory containing the three protocol .txt files.
        manifest_dir: Output directory for CSV manifests.

    Returns:
        Audit summary with record counts, class distributions, and validations.
    """
    all_records = load_all_partitions(protocol_dir)

    # Partition records
    partitioned: Dict[str, list] = {"train": [], "dev": [], "eval": []}
    for rec in all_records:
        partitioned[rec["partition"]].append(rec)

    # Write partition CSVs
    counts = {}
    for partition, records in partitioned.items():
        out_path = str(Path(manifest_dir) / f"asvspoof_{partition}.csv")
        counts[partition] = records_to_csv(records, out_path)

    # Validations
    speaker_check = validate_speaker_disjointness(all_records)
    dup_check = validate_no_duplicate_utterances(all_records)
    class_dist = get_class_distribution(all_records)

    return {
        "partition_counts": counts,
        "class_distribution": class_dist,
        "speaker_disjointness": speaker_check,
        "duplicate_check": dup_check,
        "total_records": len(all_records),
    }


def create_quick_subset(
    manifest_path: str,
    output_path: str,
    samples_per_class: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    """Create a stratified, seeded mini-subset from a partition manifest.

    Preserves class proportions and speaker composition. Selects a fixed,
    reproducible subset for quick CPU experiments without modifying official
    partition boundaries (the subset is always a strict subset of the
    original partition).

    Args:
        manifest_path: Path to the source partition CSV manifest.
        output_path: Path for the output subset CSV.
        samples_per_class: Maximum samples per class (bonafide/spoof).
        seed: Random seed for reproducibility.

    Returns:
        Summary dictionary with subset statistics.
    """
    records = load_manifest_csv(manifest_path)
    rng = np.random.RandomState(seed)

    # Split by label
    class_groups: Dict[str, list] = {}
    for rec in records:
        label = rec.get("label_str", "unknown")
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(rec)

    subset: List[Dict[str, str]] = []
    class_counts: Dict[str, int] = {}

    for label, group in class_groups.items():
        n = min(len(group), samples_per_class)
        indices = rng.choice(len(group), size=n, replace=False)
        selected = [group[i] for i in sorted(indices)]
        subset.extend(selected)
        class_counts[label] = n

    # Write subset
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if subset:
        columns = list(subset[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(subset)

    # Gather speakers in subset
    subset_speakers = set(r.get("speaker_id", "") for r in subset)

    return {
        "total_subset_samples": len(subset),
        "class_counts": class_counts,
        "speakers_in_subset": len(subset_speakers),
        "seed": seed,
        "source_manifest": manifest_path,
    }

