"""Model training and evaluation script for VoxGuard.

Trains:
  1. Logistic Regression on MFCC summary features.
  2. Balanced Random Forest on MFCC summary features.
  3. 2D Spectrogram-CNN on Log-Mel spectrogram tensors.

Supports:
  --mode quick: 5 epochs, short run ideal for initial CPU or Colab GPU tests.
  --mode full:  25 epochs with early stopping on validation loss.

Usage:
    python scripts/train_models.py --model baselines --mode quick
    python scripts/train_models.py --model cnn --mode quick
    python scripts/train_models.py --model all --mode quick
    python scripts/train_models.py --model all --mode full
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import SpectrogramDataset
from src.models.baseline_models import (
    BaselinePredictor,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_baseline_pipeline,
)
from src.models.spec_cnn import SpectrogramCNN
from src.models.trainer import train_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def load_processed_mfcc(
    processed_dir: Path, subset: str = "mini"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted training and development MFCC features."""
    prefix = f"_{subset}" if subset != "full" else ""
    train_mfcc = processed_dir / f"train{prefix}_mfcc.npy"
    train_labels = processed_dir / f"train{prefix}_labels.npy"
    dev_mfcc = processed_dir / f"dev{prefix}_mfcc.npy"
    dev_labels = processed_dir / f"dev{prefix}_labels.npy"

    for f in [train_mfcc, train_labels, dev_mfcc, dev_labels]:
        if not f.exists():
            raise FileNotFoundError(
                f"Feature cache missing: {f}. Run python scripts/extract_features.py first."
            )

    return (
        np.load(str(train_mfcc)),
        np.load(str(train_labels)),
        np.load(str(dev_mfcc)),
        np.load(str(dev_labels)),
    )


def load_processed_spectrograms(
    processed_dir: Path, subset: str = "mini"
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Load pre-extracted training and development Log-Mel spectrogram tensors."""
    prefix = f"_{subset}" if subset != "full" else ""
    train_mel = processed_dir / f"train{prefix}_mel.pt"
    train_labels = processed_dir / f"train{prefix}_labels.npy"
    dev_mel = processed_dir / f"dev{prefix}_mel.pt"
    dev_labels = processed_dir / f"dev{prefix}_labels.npy"

    for f in [train_mel, train_labels, dev_mel, dev_labels]:
        if not f.exists():
            raise FileNotFoundError(
                f"Spectrogram cache missing: {f}. Run python scripts/extract_features.py first."
            )

    return (
        torch.load(str(train_mel), map_location="cpu"),
        np.load(str(train_labels)),
        torch.load(str(dev_mel), map_location="cpu"),
        np.load(str(dev_labels)),
    )


def evaluate_sklearn_pipeline(
    pipeline, X_eval: np.ndarray, y_eval: np.ndarray, model_name: str
) -> Dict[str, Any]:
    """Evaluate a trained scikit-learn pipeline on evaluation data."""
    y_pred = pipeline.predict(X_eval)
    y_proba = (
        pipeline.predict_proba(X_eval)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else y_pred.astype(float)
    )

    acc = float(accuracy_score(y_eval, y_pred))
    prec = float(precision_score(y_eval, y_pred, zero_division=0))
    rec = float(recall_score(y_eval, y_pred, zero_division=0))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_eval, y_proba))
    except Exception:
        auc = float("nan")

    return {
        "model": model_name,
        "representation": "MFCC Summary (80-dim)",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def train_all(
    config_path: str = "configs/default.yaml",
    model_target: str = "all",
    subset: str = "mini",
    mode: str = "quick",
    checkpoint_dir: Optional[str] = None,
):
    """Run model training for traditional baselines and/or 2D SpectrogramCNN."""
    cfg = load_config(config_path)
    d_mode = cfg.get("dataset_mode", "synthetic_fixture")
    print("=" * 70)
    print(" VoxGuard Deepfake Audio Classifier Training Suite")
    print(f" Target: {model_target.upper()} | Mode: {mode.upper()} | Subset: {subset.upper()} | Dataset Mode: {d_mode.upper()}")
    print("=" * 70)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
    elif d_mode == "real_speech":
        ckpt_dir = Path(cfg["paths"].get("checkpoint_real_speech_dir", "checkpoints/real_speech"))
    else:
        ckpt_dir = Path(cfg["paths"].get("checkpoint_fixture_dir", "checkpoints/fixture"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    comparison_results: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Traditional Baseline Classifiers (Logistic Regression & Random Forest)
    # ─────────────────────────────────────────────────────────────────────────
    if model_target in ("baselines", "logistic_regression", "random_forest", "all"):
        print("\n" + "-" * 70)
        print(" [SECTION 1] Training Traditional Baseline Classifiers")
        print("-" * 70)
        X_train_mfcc, y_train_mfcc, X_dev_mfcc, y_dev_mfcc = load_processed_mfcc(
            processed_dir, subset=subset
        )
        print(f"  Loaded MFCC Data:")
        print(f"    - Train: {X_train_mfcc.shape} (Bonafide={(y_train_mfcc==0).sum()}, Spoof={(y_train_mfcc==1).sum()})")
        print(f"    - Dev:   {X_dev_mfcc.shape} (Bonafide={(y_dev_mfcc==0).sum()}, Spoof={(y_dev_mfcc==1).sum()})")

        sub_models = []
        if model_target in ("baselines", "all"):
            sub_models = ["logistic_regression", "random_forest"]
        elif model_target in ("logistic_regression", "lr"):
            sub_models = ["logistic_regression"]
        elif model_target in ("random_forest", "rf"):
            sub_models = ["random_forest"]

        for m_name in sub_models:
            save_path = str(ckpt_dir / f"{m_name}.joblib")
            pipeline = train_baseline_pipeline(
                model_type=m_name,
                X_train=X_train_mfcc,
                y_train=y_train_mfcc,
                config=cfg.get("models", {}),
                save_path=save_path,
                random_state=seed,
            )
            # Also copy to root checkpoints/ for backward compatibility if fixture mode
            if d_mode == "synthetic_fixture":
                root_path = Path("checkpoints") / f"{m_name}.joblib"
                import shutil
                shutil.copyfile(save_path, str(root_path))

            metrics = evaluate_sklearn_pipeline(pipeline, X_dev_mfcc, y_dev_mfcc, model_name=m_name)
            comparison_results.append(metrics)
            print(f"  [OK] {m_name.upper()} Dev Acc: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 2D Spectrogram-CNN Model
    # ─────────────────────────────────────────────────────────────────────────
    if model_target in ("cnn", "all"):
        print("\n" + "-" * 70)
        print(" [SECTION 2] Training 2D Spectrogram-CNN")
        print("-" * 70)
        X_train_mel, y_train_mel, X_dev_mel, y_dev_mel = load_processed_spectrograms(
            processed_dir, subset=subset
        )
        print(f"  Loaded Spectrogram Data:")
        print(f"    - Train: {list(X_train_mel.shape)} (Bonafide={(y_train_mel==0).sum()}, Spoof={(y_train_mel==1).sum()})")
        print(f"    - Dev:   {list(X_dev_mel.shape)} (Bonafide={(y_dev_mel==0).sum()}, Spoof={(y_dev_mel==1).sum()})")

        train_ds = SpectrogramDataset(X_train_mel, y_train_mel)
        dev_ds = SpectrogramDataset(X_dev_mel, y_dev_mel)

        cnn_cfg = cfg.get("models", {}).get("cnn", {})
        epochs = 5 if mode == "quick" else cnn_cfg.get("epochs", 25)
        batch_size = 16 if mode == "quick" else cnn_cfg.get("batch_size", 32)
        lr = cnn_cfg.get("learning_rate", 0.0005)
        weight_decay = cnn_cfg.get("weight_decay", 0.0001)
        patience = 3 if mode == "quick" else cnn_cfg.get("early_stopping_patience", 5)

        cnn_model = SpectrogramCNN(
            in_channels=cnn_cfg.get("in_channels", 1),
            conv_channels=cnn_cfg.get("conv_channels", [32, 64, 128, 256]),
            dense_dim=cnn_cfg.get("dense_dim", 128),
            dropout=cnn_cfg.get("dropout", 0.3),
            num_classes=2,
        )

        cnn_save_path = str(ckpt_dir / "best_spec_cnn.pt")

        train_res = train_model(
            model=cnn_model,
            train_dataset=train_ds,
            val_dataset=dev_ds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            early_stopping_patience=patience,
            save_path=cnn_save_path,
            preprocessing_config=cfg,
            seed=seed,
            val_partition_name="dev",  # Security check: eval is rejected
        )

        # Also copy to root checkpoints/ for backward compatibility if fixture mode
        if d_mode == "synthetic_fixture":
            root_cnn_path = Path("checkpoints") / "best_spec_cnn.pt"
            import shutil
            shutil.copyfile(cnn_save_path, str(root_cnn_path))

        best_m = train_res["best_metrics"]
        comparison_results.append({
            "model": "spectrogram_cnn",
            "representation": f"Log-Mel ({X_train_mel.shape[2]}x{X_train_mel.shape[3]})",
            "accuracy": best_m["accuracy"],
            "precision": best_m["precision"],
            "recall": best_m["recall"],
            "f1": best_m["f1"],
            "roc_auc": best_m["roc_auc"],
        })

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Comprehensive Performance Comparison Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" EMPIRICAL PERFORMANCE COMPARISON (Held-Out Development Partition)")
    print("=" * 80)
    print(f"{'Model Architecture':<20} | {'Representation':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'ROC-AUC':<9}")
    print("-" * 96)
    for r in comparison_results:
        print(
            f"{r['model']:<20} | "
            f"{r['representation']:<22} | "
            f"{r['accuracy']*100:>7.2f}% | "
            f"{r['precision']:>9.4f} | "
            f"{r['recall']:>9.4f} | "
            f"{r['f1']:>9.4f} | "
            f"{r['roc_auc']:>9.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Model Training Suite")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration YAML path")
    parser.add_argument("--model", default="all", choices=["baselines", "logistic_regression", "random_forest", "cnn", "all"], help="Model target")
    parser.add_argument("--subset", default="mini", choices=["mini", "full"], help="Feature cache subset")
    parser.add_argument("--mode", default="quick", choices=["quick", "full"], help="Training mode (quick=5 epochs, full=25 epochs)")
    parser.add_argument("--checkpoint-dir", default=None, help="Custom checkpoint destination directory")
    args = parser.parse_args()
    train_all(args.config, args.model, args.subset, args.mode, checkpoint_dir=args.checkpoint_dir)
