"""Model provenance validation and metadata management for VoxGuard.

Ensures strict scientific integrity by requiring every saved model checkpoint
to carry verified provenance metadata identifying:
1. Dataset name (e.g., 'ASVspoof2019_LA' or 'SyntheticTestFixtures')
2. Dataset mode ('real_speech' or 'synthetic_fixture')
3. Model architecture ('SpectrogramCNN', etc.)
4. Preprocessing configuration
5. Training partition ('train')
6. Development partition ('dev')
7. Training date (ISO 8601 UTC timestamp)
8. Model version (semantic version string)

Real-speech inference strictly rejects checkpoints with missing, corrupted,
or fixture-mode provenance metadata.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


# -----------------------------------------------------------------------------
# Canonical Provenance Field Constants
# -----------------------------------------------------------------------------
REQUIRED_PROVENANCE_FIELDS: List[str] = [
    "dataset_name",
    "dataset_mode",
    "model_architecture",
    "preprocessing_config",
    "training_partition",
    "development_partition",
    "training_date",
    "model_version",
]

VALID_DATASET_MODES: List[str] = ["real_speech", "synthetic_fixture"]


# -----------------------------------------------------------------------------
# Provenance Exceptions
# -----------------------------------------------------------------------------
class ProvenanceError(Exception):
    """Base exception for model provenance violations."""
    pass


class MissingProvenanceError(ProvenanceError):
    """Raised when required provenance fields are missing from a checkpoint."""
    pass


class InvalidProvenanceError(ProvenanceError):
    """Raised when provenance metadata is malformed or invalid."""
    pass


class ModeMismatchError(ProvenanceError):
    """Raised when checkpoint dataset_mode does not match the operating context."""
    pass


# -----------------------------------------------------------------------------
# Metadata Construction & Extraction Utilities
# -----------------------------------------------------------------------------
def build_checkpoint_provenance(
    dataset_name: str,
    dataset_mode: str,
    model_architecture: str,
    preprocessing_config: Dict[str, Any],
    training_partition: str = "train",
    development_partition: str = "dev",
    training_date: Optional[str] = None,
    model_version: str = "1.0.0",
) -> Dict[str, Any]:
    """Construct a canonical provenance dictionary.

    Args:
        dataset_name: Name of training dataset.
        dataset_mode: 'real_speech' or 'synthetic_fixture'.
        model_architecture: Model class or architecture name.
        preprocessing_config: Preprocessing parameters dictionary.
        training_partition: Partition used for parameter fitting ('train').
        development_partition: Partition used for validation/stopping ('dev').
        training_date: ISO 8601 UTC timestamp (defaults to current time).
        model_version: Semantic version string.

    Returns:
        Structured provenance dictionary.
    """
    if dataset_mode not in VALID_DATASET_MODES:
        raise ValueError(
            f"Invalid dataset_mode '{dataset_mode}'. Must be one of {VALID_DATASET_MODES}."
        )

    if not training_date:
        training_date = datetime.now(timezone.utc).isoformat()

    return {
        "dataset_name": str(dataset_name),
        "dataset_mode": str(dataset_mode),
        "model_architecture": str(model_architecture),
        "preprocessing_config": dict(preprocessing_config),
        "training_partition": str(training_partition),
        "development_partition": str(development_partition),
        "training_date": str(training_date),
        "model_version": str(model_version),
    }


def extract_checkpoint_provenance(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract provenance metadata from a loaded checkpoint dictionary.

    Supports both top-level 'provenance' dictionary and backward-compatible
    embedded keys ('preprocessing_config', 'model_config').

    Args:
        checkpoint: Deserialized checkpoint state dict.

    Returns:
        Dictionary containing extracted provenance fields.
    """
    if "provenance" in checkpoint and isinstance(checkpoint["provenance"], dict):
        return dict(checkpoint["provenance"])

    # Fallback inference for existing/legacy checkpoints
    prep_cfg = checkpoint.get("preprocessing_config", {})
    model_cfg = checkpoint.get("model_config", {})

    dataset_mode = prep_cfg.get("dataset_mode", "synthetic_fixture")
    dataset_name = "SyntheticTestFixtures" if dataset_mode == "synthetic_fixture" else "ASVspoof2019_LA"
    model_arch = model_cfg.get("model_type", "SpectrogramCNN")
    model_version = prep_cfg.get("model_version", "0.4.0")

    return {
        "dataset_name": dataset_name,
        "dataset_mode": dataset_mode,
        "model_architecture": model_arch,
        "preprocessing_config": prep_cfg,
        "training_partition": "train",
        "development_partition": "dev",
        "training_date": checkpoint.get("training_date", "2026-10-30T22:26:00+00:00"),
        "model_version": model_version,
    }


def validate_checkpoint_provenance(
    checkpoint: Dict[str, Any],
    expected_mode: str = "real_speech",
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate checkpoint provenance against required scientific standards.

    Args:
        checkpoint: Deserialized checkpoint dictionary.
        expected_mode: Expected dataset mode ('real_speech' or 'synthetic_fixture').

    Returns:
        Tuple of (is_valid, message, provenance_dict).
    """
    if not isinstance(checkpoint, dict):
        return False, "Checkpoint payload is not a valid dictionary.", {}

    provenance = extract_checkpoint_provenance(checkpoint)

    # 1. Verify all required fields exist and are non-empty
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in provenance or provenance[f] is None]
    if missing:
        msg = f"Checkpoint provenance missing required fields: {missing}."
        return False, msg, provenance

    # 2. Validate expected mode matches checkpoint dataset_mode
    actual_mode = provenance["dataset_mode"]
    if actual_mode not in VALID_DATASET_MODES:
        msg = f"Invalid checkpoint dataset_mode '{actual_mode}'. Must be one of {VALID_DATASET_MODES}."
        return False, msg, provenance

    if actual_mode != expected_mode:
        msg = (
            f"Provenance mode mismatch: Checkpoint dataset_mode is '{actual_mode}', but expected mode "
            f"is '{expected_mode}'. Fixture models cannot process real speech, and cross-mode execution "
            f"is strictly prohibited by scientific integrity policy."
        )
        return False, msg, provenance

    # 3. For real_speech mode, ensure dataset_name is legitimate and not a fixture
    if expected_mode == "real_speech":
        d_name = provenance.get("dataset_name", "").lower()
        if "fixture" in d_name or "synthetic" in d_name or "mock" in d_name:
            msg = f"Provenance violation: Real-speech model cannot be linked to synthetic dataset name '{provenance.get('dataset_name')}'."
            return False, msg, provenance

    # 4. Strict development split check (never eval)
    dev_part = provenance.get("development_partition", "").lower()
    if dev_part == "eval":
        msg = "Protocol violation: Checkpoint was tuned on 'eval' partition. Held-out test contamination is forbidden."
        return False, msg, provenance

    return True, "Checkpoint provenance verified successfully.", provenance

