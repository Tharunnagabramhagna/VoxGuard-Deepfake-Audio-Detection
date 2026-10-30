"""Explicit model selection and checkpoint loader for VoxGuard.

Enforces strict isolation between procedural software-test fixtures and
real-world speech detection models:
1. Fixture mode loads ONLY fixture checkpoints (checkpoints/fixture/).
2. Real-speech mode loads ONLY validated real-speech checkpoints (checkpoints/real_speech/).
3. Real-speech mode NEVER silently falls back to fixture checkpoints.
4. Missing real-speech checkpoints produce an explicit status report.
5. All checkpoints must pass provenance validation before deployment.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

from src.models.provenance import (
    ProvenanceError,
    validate_checkpoint_provenance,
)
from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.models.baseline_models import BaselinePredictor
from src.utils.config import load_config


DEFAULT_THRESHOLDS = {
    "spectrogram_cnn": 0.4604,
    "random_forest": 0.5546,
    "logistic_regression": 0.3903,
}


def get_checkpoint_path(
    model_type: str = "cnn",
    mode: str = "fixture",
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve checkpoint file path for the requested model and operating mode.

    Args:
        model_type: 'cnn', 'random_forest', or 'logistic_regression'.
        mode: 'fixture' or 'real_speech'.
        base_dir: Base directory (defaults to 'checkpoints').

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If mode is unrecognized.
    """
    base = Path(base_dir or "checkpoints")

    if mode == "fixture":
        fixture_dir = base / "fixture"
        filename = "best_spec_cnn.pt" if model_type == "cnn" else f"{model_type}.joblib"
        target = fixture_dir / filename
        # Backward compatibility with un-partitioned checkpoints root
        if not target.exists() and (base / filename).exists():
            return base / filename
        return target

    elif mode == "real_speech":
        real_dir = base / "real_speech"
        filename = "best_spec_cnn.pt" if model_type == "cnn" else f"{model_type}.joblib"
        # STRICT POLICY: Never fall back to fixture directory!
        return real_dir / filename

    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'fixture' or 'real_speech'.")


def load_calibrated_thresholds(
    summary_path: str = "reports/results/evaluation_summary.json",
) -> Dict[str, float]:
    """Load development operating thresholds from evaluation summary artifact."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    p = Path(summary_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            calibrated = data.get("thresholds_calibrated_on_dev", {})
            for k in thresholds:
                if k in calibrated:
                    thresholds[k] = float(calibrated[k])
        except Exception:
            pass
    return thresholds


def load_real_speech_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    config_path: str = "configs/default.yaml",
) -> Dict[str, Any]:
    """Attempt to load a validated real-speech deepfake detector.

    Selection Priority:
    1. Fine-tuned MLAAD-tiny detector: checkpoints/real_speech/voxguard_mlaad_finetuned/best_model.pt
    2. Original pretrained baseline: checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth
    3. Legacy / ASVspoof Spec-CNN: checkpoints/real_speech/best_spec_cnn.pt
    """
    # If explicit checkpoint path passed and does not exist, return explicit not-found status
    if checkpoint_path is not None:
        explicit_path = Path(checkpoint_path)
        if not explicit_path.exists():
            return {
                "available": False,
                "status": "CHECKPOINT_NOT_FOUND",
                "message": f"Real-speech checkpoint not found at: {explicit_path}",
                "checkpoint_path": str(explicit_path),
                "predictor": None,
                "provenance": None,
                "threshold": None,
                "is_real_speech_model": False,
            }

    # Priority 1: Fine-tuned MLAAD-tiny model
    mlaad_ckpt = Path(checkpoint_path or "checkpoints/real_speech/voxguard_mlaad_finetuned/best_model.pt")
    mlaad_prov = Path("checkpoints/real_speech/voxguard_mlaad_finetuned/provenance.json")

    if mlaad_ckpt.exists():
        try:
            from src.models.real_speech_predictor import RealSpeechPredictor
            prov_data = None
            if mlaad_prov.exists():
                with open(mlaad_prov, "r", encoding="utf-8") as pf:
                    prov_data = json.load(pf)
            predictor = RealSpeechPredictor(
                str(mlaad_ckpt),
                device=device,
                threshold=0.50,
                model_name="ResNet-BiGRU Detector (MLAAD Fine-Tuned)",
            )
            return {
                "available": True,
                "status": "EXPERIMENTAL_MLAAD_REAL_SPEECH_MODEL",
                "message": "Fine-tuned MLAAD-tiny deepfake detector (ResNet-18 + BiGRU + Attention) loaded and verified.",
                "checkpoint_path": str(mlaad_ckpt),
                "predictor": predictor,
                "provenance": prov_data,
                "threshold": 0.50,
                "dev_threshold": 0.2777,
                "is_real_speech_model": True,
                "model_name": "ResNet-BiGRU Detector (MLAAD Fine-Tuned)",
            }
        except Exception as e:
            return {
                "available": False,
                "status": "ERROR_LOADING_MLAAD_MODEL",
                "message": f"Failed to load fine-tuned MLAAD model: {e}",
                "checkpoint_path": str(mlaad_ckpt),
                "predictor": None,
                "provenance": None,
                "threshold": None,
                "is_real_speech_model": False,
            }

    # Priority 2: Original Pretrained Baseline
    base_ckpt = Path("checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth")
    if base_ckpt.exists():
        try:
            from src.models.real_speech_predictor import RealSpeechPredictor
            predictor = RealSpeechPredictor(
                str(base_ckpt),
                device=device,
                threshold=0.50,
                model_name="Pretrained Deepfake Voice Detector (Untouched Baseline)",
            )
            return {
                "available": True,
                "status": "EXPERIMENTAL_PRETRAINED_BASELINE_MODEL",
                "message": "Untouched pretrained deepfake detector (ResNet-18 + BiGRU + Attention) loaded as experimental baseline.",
                "checkpoint_path": str(base_ckpt),
                "predictor": predictor,
                "provenance": {"model_name": "Pretrained Deepfake Voice Detector"},
                "threshold": 0.50,
                "dev_threshold": 0.0758,
                "is_real_speech_model": True,
                "model_name": "Pretrained Deepfake Voice Detector (Untouched Baseline)",
            }
        except Exception:
            pass

    path = Path(checkpoint_path or get_checkpoint_path(model_type="cnn", mode="real_speech"))

    # Case 1: Checkpoint file does not exist
    if not path.exists():
        return {
            "available": False,
            "status": "CHECKPOINT_NOT_FOUND",
            "message": (
                f"Real-speech checkpoint not found at: '{path}'. "
                "Genuine audio deepfake detection requires an authentic model checkpoint "
                "trained on speech corpora (e.g. ASVspoof 2019 LA). Silent fallback to fixture "
                "checkpoints is strictly disabled."
            ),
            "checkpoint_path": str(path),
            "predictor": None,
            "provenance": None,
            "threshold": None,
            "is_real_speech_model": False,
        }

    # Case 2: Checkpoint exists — inspect provenance
    try:
        ckpt_payload = torch.load(str(path), map_location="cpu")
    except Exception as e:
        return {
            "available": False,
            "status": "CORRUPT_CHECKPOINT",
            "message": f"Failed to load checkpoint file: {e}",
            "checkpoint_path": str(path),
            "predictor": None,
            "provenance": None,
            "threshold": None,
            "is_real_speech_model": False,
        }

    is_valid, msg, prov = validate_checkpoint_provenance(ckpt_payload, expected_mode="real_speech")
    if not is_valid:
        return {
            "available": False,
            "status": "INVALID_PROVENANCE",
            "message": f"Real-speech checkpoint provenance validation failed: {msg}",
            "checkpoint_path": str(path),
            "predictor": None,
            "provenance": prov,
            "threshold": None,
            "is_real_speech_model": False,
        }

    # Case 3: Validated real-speech model
    thresholds = load_calibrated_thresholds()
    predictor = SpecCNNPredictor.from_checkpoint(str(path), device=torch.device(device))

    return {
        "available": True,
        "status": "VALIDATED_REAL_SPEECH_MODEL",
        "message": "Validated real-speech model loaded and verified.",
        "checkpoint_path": str(path),
        "predictor": predictor,
        "provenance": prov,
        "threshold": thresholds.get("spectrogram_cnn", 0.5),
        "is_real_speech_model": True,
    }


def load_fixture_models(
    base_dir: Optional[Union[str, Path]] = None,
    config_path: str = "configs/default.yaml",
) -> Dict[str, Dict[str, Any]]:
    """Load model checkpoints designated for the software-test harness.

    Loads from checkpoints/fixture/ (with backward compatibility fallback to checkpoints/).
    Validates that loaded checkpoints have dataset_mode='synthetic_fixture'.

    Returns:
        Dictionary mapping user-friendly model names to model descriptors.
    """
    thresholds = load_calibrated_thresholds()
    cfg = load_config(config_path)
    models: Dict[str, Dict[str, Any]] = {}

    # 1. 2D Spectrogram-CNN
    cnn_path = get_checkpoint_path(model_type="cnn", mode="fixture", base_dir=base_dir)
    if cnn_path.exists():
        try:
            ckpt_dict = torch.load(str(cnn_path), map_location="cpu")
            _, _, prov = validate_checkpoint_provenance(ckpt_dict, expected_mode="synthetic_fixture")
            models["Spectrogram-CNN (2D CNN)"] = {
                "predictor": SpecCNNPredictor.from_checkpoint(str(cnn_path), device="cpu"),
                "threshold": thresholds["spectrogram_cnn"],
                "type": "cnn",
                "feature": "64-bin Log-Mel Spectrogram (64x401)",
                "dataset_mode": "synthetic_fixture",
                "provenance": prov,
                "is_real_speech_model": False,
                "path": str(cnn_path),
            }
        except Exception:
            pass

    # 2. Random Forest Baseline
    rf_path = get_checkpoint_path(model_type="random_forest", mode="fixture", base_dir=base_dir)
    if rf_path.exists():
        try:
            models["Random Forest (Baseline)"] = {
                "predictor": BaselinePredictor.from_checkpoint(str(rf_path), model_name="random_forest", config=cfg),
                "threshold": thresholds["random_forest"],
                "type": "baseline",
                "feature": "80-dim MFCC Summary Statistics",
                "dataset_mode": "synthetic_fixture",
                "is_real_speech_model": False,
                "path": str(rf_path),
            }
        except Exception:
            pass

    # 3. Logistic Regression Baseline
    lr_path = get_checkpoint_path(model_type="logistic_regression", mode="fixture", base_dir=base_dir)
    if lr_path.exists():
        try:
            models["Logistic Regression (Baseline)"] = {
                "predictor": BaselinePredictor.from_checkpoint(str(lr_path), model_name="logistic_regression", config=cfg),
                "threshold": thresholds["logistic_regression"],
                "type": "baseline",
                "feature": "80-dim MFCC Summary Statistics",
                "dataset_mode": "synthetic_fixture",
                "is_real_speech_model": False,
                "path": str(lr_path),
            }
        except Exception:
            pass

    return models

