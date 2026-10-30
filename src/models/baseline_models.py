"""Traditional machine learning baseline models for VoxGuard.

Implements:
1. Logistic Regression with MFCC summary features.
2. Balanced Random Forest with the same MFCC summary representation.

Uses scikit-learn Pipelines combining feature standardization (fitted strictly
on training data) with class-weighted classifiers to address the ~9:1 class
imbalance without altering validation or evaluation set distributions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import load_audio_file, preprocess_audio
from src.features.mfcc import extract_mfcc_summary


def build_logistic_regression_pipeline(
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: str = "balanced",
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn Pipeline for Logistic Regression.

    Pipeline:
      Step 1: StandardScaler() - fits mean/std strictly on X_train.
      Step 2: LogisticRegression(class_weight='balanced') - penalizes minority class errors.

    Args:
        C: Inverse regularization strength.
        solver: Optimization algorithm ('lbfgs').
        max_iter: Maximum solver iterations.
        class_weight: 'balanced' adjusts weights inversely to class frequencies.
        random_state: Seed for reproducibility.
        **kwargs: Additional kwargs passed to LogisticRegression.

    Returns:
        Configured scikit-learn Pipeline.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    classifier = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )
    return Pipeline([("scaler", scaler), ("classifier", classifier)])


def build_random_forest_pipeline(
    n_estimators: int = 150,
    max_depth: Optional[int] = 15,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build a scikit-learn Pipeline for Random Forest.

    Pipeline:
      Step 1: StandardScaler() - standardized inputs.
      Step 2: RandomForestClassifier(class_weight='balanced') - bootstrap tree ensemble.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        class_weight: 'balanced' adjusts weights inversely to class frequencies.
        random_state: Seed for reproducibility.
        n_jobs: Parallel worker threads (-1 for all cores).

    Returns:
        Configured scikit-learn Pipeline.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline([("scaler", scaler), ("classifier", classifier)])


def train_baseline_pipeline(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> Pipeline:
    """Train a baseline classifier pipeline on training data.

    Enforces that preprocessing (StandardScaler) is fitted strictly on X_train.
    The validation/eval data is never used during fitting.

    Args:
        model_type: 'logistic_regression' or 'random_forest'.
        X_train: 2D feature matrix of shape (n_samples, n_features).
        y_train: 1D array of binary labels {0: bonafide, 1: spoof}.
        config: Hyperparameter dictionary.
        save_path: Path to save the serialized pipeline via joblib.
        random_state: Random seed.

    Returns:
        Fitted Pipeline.
    """
    model_type = model_type.lower().strip()
    cfg = config or {}

    if model_type == "logistic_regression":
        lr_cfg = cfg.get("logistic_regression", cfg)
        pipeline = build_logistic_regression_pipeline(
            C=lr_cfg.get("C", 1.0),
            solver=lr_cfg.get("solver", "lbfgs"),
            max_iter=lr_cfg.get("max_iter", 1000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            random_state=random_state,
        )
    elif model_type in ("random_forest", "rf"):
        rf_cfg = cfg.get("random_forest", cfg)
        pipeline = build_random_forest_pipeline(
            n_estimators=rf_cfg.get("n_estimators", 150),
            max_depth=rf_cfg.get("max_depth", 15),
            min_samples_split=rf_cfg.get("min_samples_split", 4),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            class_weight=rf_cfg.get("class_weight", "balanced"),
            random_state=random_state,
            n_jobs=rf_cfg.get("n_jobs", -1),
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'logistic_regression' or 'random_forest'.")

    # Fit pipeline on training data ONLY
    pipeline.fit(X_train, y_train)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, str(out))

    return pipeline


class BaselinePredictor:
    """Unified, single prediction interface for all VoxGuard baseline classifiers.

    Handles raw audio input, preprocessing, MFCC summary feature extraction,
    feature scaling, and model inference with calibrated probability output.
    """

    LABEL_NAMES: Dict[int, str] = {0: "bonafide", 1: "spoof"}

    def __init__(
        self,
        pipeline: Pipeline,
        model_name: str = "baseline",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize predictor with a fitted scikit-learn Pipeline.

        Args:
            pipeline: A fitted scikit-learn Pipeline with a classifier step.
            model_name: Name of the model for logging and reports.
            config: Optional project configuration dictionary.
        """
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config or {}

        # Audio and feature config
        audio_cfg = self.config.get("audio", {})
        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.target_samples = audio_cfg.get("target_samples", 64000)

        feat_cfg = self.config.get("features", {}).get("mfcc", {})
        self.n_mfcc = feat_cfg.get("n_mfcc", 40)
        self.include_deltas = feat_cfg.get("include_deltas", False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "BaselinePredictor":
        """Load a fitted pipeline from a joblib checkpoint file.

        Args:
            checkpoint_path: Path to .joblib file.
            model_name: Model name (inferred from filename if None).
            config: Project configuration.

        Returns:
            Configured BaselinePredictor instance.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

        pipeline = joblib.load(str(path))
        if model_name is None:
            model_name = path.stem.replace("_pipeline", "")

        return cls(pipeline=pipeline, model_name=model_name, config=config)

    def extract_features(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Extract MFCC summary vector from an audio waveform."""
        feat, _ = extract_mfcc_summary(
            waveform=waveform,
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            include_deltas=self.include_deltas,
        )
        return feat

    def predict_features(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict on a pre-extracted 1D or 2D feature array.

        Args:
            X: Feature vector of shape (n_features,) or matrix (n_samples, n_features).

        Returns:
            Dictionary with prediction class, label, probabilities, and confidence.
        """
        X_in = np.atleast_2d(X)

        preds = self.pipeline.predict(X_in)
        # Check if classifier supports predict_proba
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X_in)
            prob_bonafide = float(probs[0, 0])
            prob_spoof = float(probs[0, 1])
        else:
            pred_val = int(preds[0])
            prob_spoof = float(pred_val)
            prob_bonafide = 1.0 - prob_spoof

        pred_class = int(preds[0])
        confidence = max(prob_bonafide, prob_spoof)

        return {
            "prediction": pred_class,
            "label": self.LABEL_NAMES.get(pred_class, "unknown"),
            "probability_spoof": prob_spoof,
            "probability_bonafide": prob_bonafide,
            "confidence": confidence,
            "model_name": self.model_name,
            "n_features": X_in.shape[1],
        }

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities on a 2D feature matrix.

        Args:
            X: 2D array of shape (n_samples, n_features).

        Returns:
            2D array of probabilities of shape (n_samples, 2).
        """
        X_in = np.atleast_2d(X)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X_in)
        else:
            preds = self.pipeline.predict(X_in)
            p1 = preds.astype(float)
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

    def predict_audio(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict whether an audio recording is genuine or deepfake.

        Full end-to-end inference:
          1. Load audio if string path is provided.
          2. Apply standardized preprocessing (16 kHz mono, crop/pad, peak normalize).
          3. Extract 40 MFCC temporal summary statistics.
          4. Predict using the fitted pipeline.

        Args:
            audio_input: Path to audio file (.wav/.flac) or 1D waveform array.
            sample_rate: Sampling rate if waveform array is passed.

        Returns:
            Inference result dictionary with prediction, label, probabilities,
            and confidence score.
        """
        if isinstance(audio_input, (str, Path)):
            raw_wave, sr, err = load_audio_file(str(audio_input), target_sr=self.sample_rate)
            if raw_wave is None:
                raise RuntimeError(f"Failed to load audio: {err}")
        else:
            raw_wave = np.asarray(audio_input, dtype=np.float32)
            sr = sample_rate or self.sample_rate

        # Standardized preprocessing
        clean_wave, _ = preprocess_audio(raw_wave, sr, self.config)

        # Feature extraction
        features = self.extract_features(clean_wave)

        # Predict
        result = self.predict_features(features)
        return result
