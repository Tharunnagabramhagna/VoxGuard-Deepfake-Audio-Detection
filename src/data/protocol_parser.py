"""ASVspoof 2019 LA protocol file parser.

Parses the official space-delimited protocol/trial text files into
structured manifest DataFrames suitable for downstream data loaders.

ASVspoof 2019 LA protocol line format (5 columns, space-separated):
    SPEAKER_ID  UTTERANCE_ID  SYSTEM_ID  UNUSED_KEY  LABEL

Example:
    LA_0079  LA_T_1138215  A04  -  spoof
    LA_0079  LA_T_1138216  -    -  bonafide
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Canonical label mapping ──────────────────────────────────────────────────
LABEL_MAP: Dict[str, int] = {"bonafide": 0, "spoof": 1}

# ── Official ASVspoof 2019 LA partition protocol filenames ────────────────────
PROTOCOL_FILES: Dict[str, str] = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

# ── Column names for parsed manifests ─────────────────────────────────────────
MANIFEST_COLUMNS: List[str] = [
    "utterance_id",
    "speaker_id",
    "system_id",
    "label_str",
    "label",
    "partition",
]


def parse_protocol_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a single ASVspoof 2019 LA protocol line.

    Args:
        line: A whitespace-separated line from a protocol file.

    Returns:
        Dictionary with speaker_id, utterance_id, system_id, label_str,
        or None if the line is blank / malformed.
    """
    parts = line.strip().split()
    if len(parts) < 4:
        return None

    # Handle both 4-column (older) and 5-column (standard) formats
    if len(parts) == 4:
        speaker_id, utterance_id, system_id, label_str = parts
    else:
        # Standard 5-column: speaker_id utterance_id system_id _key label_str
        speaker_id = parts[0]
        utterance_id = parts[1]
        system_id = parts[2]
        # parts[3] is the unused key field (usually "-")
        label_str = parts[4]

    label_str = label_str.lower().strip()
    if label_str not in LABEL_MAP:
        return None

    return {
        "speaker_id": speaker_id,
        "utterance_id": utterance_id,
        "system_id": system_id,
        "label_str": label_str,
    }


def parse_protocol_file(
    protocol_path: str,
    partition: str,
) -> List[Dict[str, object]]:
    """Parse an entire ASVspoof 2019 LA protocol file into a list of records.

    Args:
        protocol_path: Path to the protocol text file.
        partition: Partition name to tag each record with ('train', 'dev', 'eval').

    Returns:
        List of dictionaries, each containing utterance metadata plus integer label.

    Raises:
        FileNotFoundError: If the protocol file does not exist.
        ValueError: If the protocol file has no valid records.
    """
    path = Path(protocol_path)
    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path.resolve()}")

    records: List[Dict[str, object]] = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parsed = parse_protocol_line(line)
            if parsed is None:
                skipped += 1
                continue
            parsed["label"] = LABEL_MAP[parsed["label_str"]]
            parsed["partition"] = partition
            records.append(parsed)

    if not records:
        raise ValueError(
            f"Protocol file {path.name} yielded zero valid records "
            f"({skipped} lines skipped)."
        )

    return records


def load_all_partitions(
    protocol_dir: str,
) -> List[Dict[str, object]]:
    """Load train, dev, and eval protocol files and merge into a single list.

    Args:
        protocol_dir: Directory containing the three ASVspoof protocol text files.

    Returns:
        Combined list of all records across all three partitions.

    Raises:
        FileNotFoundError: If any of the three protocol files is missing.
    """
    all_records: List[Dict[str, object]] = []
    proto_dir = Path(protocol_dir)

    for partition, filename in PROTOCOL_FILES.items():
        proto_path = proto_dir / filename
        records = parse_protocol_file(str(proto_path), partition)
        all_records.extend(records)

    return all_records


def validate_speaker_disjointness(
    records: List[Dict[str, object]],
) -> Dict[str, object]:
    """Check that speakers do not appear in multiple partitions.

    Args:
        records: List of parsed protocol records.

    Returns:
        Dictionary with validation results including any overlapping speakers.
    """
    partition_speakers: Dict[str, set] = {}
    for rec in records:
        part = rec["partition"]
        if part not in partition_speakers:
            partition_speakers[part] = set()
        partition_speakers[part].add(rec["speaker_id"])

    overlaps: Dict[str, set] = {}
    partitions = list(partition_speakers.keys())
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            p1, p2 = partitions[i], partitions[j]
            common = partition_speakers[p1] & partition_speakers[p2]
            if common:
                key = f"{p1}_vs_{p2}"
                overlaps[key] = common

    return {
        "is_disjoint": len(overlaps) == 0,
        "partition_speaker_counts": {
            p: len(s) for p, s in partition_speakers.items()
        },
        "overlapping_speakers": {
            k: list(v) for k, v in overlaps.items()
        },
    }


def validate_no_duplicate_utterances(
    records: List[Dict[str, object]],
) -> Dict[str, object]:
    """Check that no utterance ID appears in multiple partitions.

    Args:
        records: List of parsed protocol records.

    Returns:
        Dictionary with validation results including any cross-partition duplicates.
    """
    utterance_partitions: Dict[str, set] = {}
    for rec in records:
        uid = rec["utterance_id"]
        if uid not in utterance_partitions:
            utterance_partitions[uid] = set()
        utterance_partitions[uid].add(rec["partition"])

    cross_partition = {
        uid: list(parts)
        for uid, parts in utterance_partitions.items()
        if len(parts) > 1
    }

    # Also check for duplicate utterance_id within the same partition
    from collections import Counter

    within_partition_dups: Dict[str, list] = {}
    for partition in ["train", "dev", "eval"]:
        partition_uids = [
            r["utterance_id"] for r in records if r["partition"] == partition
        ]
        counts = Counter(partition_uids)
        dups = {uid: cnt for uid, cnt in counts.items() if cnt > 1}
        if dups:
            within_partition_dups[partition] = list(dups.keys())

    return {
        "cross_partition_duplicates": cross_partition,
        "within_partition_duplicates": within_partition_dups,
        "total_utterances": len(records),
        "unique_utterances": len(utterance_partitions),
        "has_cross_partition_leaks": len(cross_partition) > 0,
        "has_within_partition_duplicates": len(within_partition_dups) > 0,
    }


def get_class_distribution(
    records: List[Dict[str, object]],
) -> Dict[str, Dict[str, int]]:
    """Compute class distribution per partition.

    Args:
        records: List of parsed protocol records.

    Returns:
        Nested dictionary: partition -> label_str -> count.
    """
    dist: Dict[str, Dict[str, int]] = {}
    for rec in records:
        part = rec["partition"]
        label = rec["label_str"]
        if part not in dist:
            dist[part] = {}
        dist[part][label] = dist[part].get(label, 0) + 1
    return dist


def get_system_distribution(
    records: List[Dict[str, object]],
) -> Dict[str, Dict[str, int]]:
    """Compute spoofing system/algorithm distribution per partition.

    Args:
        records: List of parsed protocol records.

    Returns:
        Nested dictionary: partition -> system_id -> count.
    """
    dist: Dict[str, Dict[str, int]] = {}
    for rec in records:
        part = rec["partition"]
        sys_id = rec["system_id"]
        if part not in dist:
            dist[part] = {}
        dist[part][sys_id] = dist[part].get(sys_id, 0) + 1
    return dist

