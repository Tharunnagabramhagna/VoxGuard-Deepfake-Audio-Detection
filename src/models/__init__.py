"""Model architectures and baseline classifiers for VoxGuard."""

from src.models.baseline_models import (
    BaselinePredictor,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_baseline_pipeline,
)
from src.models.spec_cnn import SpectrogramCNN
from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.models.trainer import train_model
from src.models.provenance import (
    REQUIRED_PROVENANCE_FIELDS,
    ProvenanceError,
    build_checkpoint_provenance,
    extract_checkpoint_provenance,
    validate_checkpoint_provenance,
)
from src.models.model_loader import (
    get_checkpoint_path,
    load_calibrated_thresholds,
    load_fixture_models,
    load_real_speech_model,
)

__all__ = [
    "BaselinePredictor",
    "build_logistic_regression_pipeline",
    "build_random_forest_pipeline",
    "train_baseline_pipeline",
    "SpectrogramCNN",
    "SpecCNNPredictor",
    "train_model",
    "REQUIRED_PROVENANCE_FIELDS",
    "ProvenanceError",
    "build_checkpoint_provenance",
    "extract_checkpoint_provenance",
    "validate_checkpoint_provenance",
    "get_checkpoint_path",
    "load_calibrated_thresholds",
    "load_fixture_models",
    "load_real_speech_model",
]
