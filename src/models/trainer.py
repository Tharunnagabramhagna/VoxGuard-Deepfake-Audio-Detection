"""PyTorch training and validation engine for VoxGuard deep learning models.

Implements:
  - Deterministic training with global seed configuration.
  - Class-weighted Cross-Entropy loss to counteract ~9:1 training imbalance without
    altering validation or evaluation set distributions.
  - Epoch-by-epoch tracking of training and validation loss, accuracy, precision, recall, F1, ROC-AUC.
  - Model selection based strictly on validation partition performance (eval partition is never used).
  - Early stopping with configurable patience.
  - Complete checkpoint serialization including model architecture and preprocessing parameters.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SpectrogramDataset
from src.models.spec_cnn import SpectrogramCNN
from src.utils.seed import set_seed


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation split.

    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * len(y_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(spoof)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_targets.extend(y_batch.cpu().numpy().tolist())
        total += len(y_batch)

    val_loss = running_loss / max(total, 1)
    metrics = compute_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
    )
    return val_loss, metrics


def train_model(
    model: nn.Module,
    train_dataset: SpectrogramDataset,
    val_dataset: SpectrogramDataset,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.0001,
    early_stopping_patience: int = 4,
    save_path: Optional[str] = "checkpoints/best_spec_cnn.pt",
    preprocessing_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    val_partition_name: str = "dev",
) -> Dict[str, Any]:
    """Train SpectrogramCNN with validation model selection and class imbalance weighting.

    Args:
        model: SpectrogramCNN instance.
        train_dataset: SpectrogramDataset for training.
        val_dataset: SpectrogramDataset for validation.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization penalty.
        early_stopping_patience: Epochs to wait without val_loss improvement before stopping.
        save_path: Destination path for the best model checkpoint.
        preprocessing_config: Acoustic preprocessing parameters to embed in checkpoint.
        device: PyTorch device (CPU or CUDA).
        seed: Random seed for deterministic reproducibility.
        val_partition_name: Name of validation split (must NOT be 'eval').

    Returns:
        Dictionary containing training history and best validation metrics.
    """
    # Strict validation partition security guardrail
    if val_partition_name.lower().strip() == "eval":
        raise ValueError(
            "CRITICAL PROTOCOL VIOLATION: The evaluation partition ('eval') must NEVER "
            "be used for validation, early stopping, or checkpoint selection."
        )

    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Class imbalance handling: compute inverse-frequency weights from training set ONLY
    class_weights = train_dataset.compute_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    best_val_loss = float("inf")
    best_metrics = {}
    best_epoch = -1
    patience_counter = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_roc_auc": [],
    }

    print(f"Starting training on {device} (Epochs={epochs}, BatchSize={batch_size}, LR={learning_rate})")
    print(f"Class weights (bonafide, spoof): {class_weights.cpu().numpy().tolist()}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']*100:.1f}% "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['roc_auc']:.4f}"
        )

        # Model selection: save checkpoint based STRICTLY on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_epoch = epoch
            patience_counter = 0

            if save_path:
                out_path = Path(save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                from src.models.provenance import build_checkpoint_provenance

                prep_cfg = preprocessing_config or {}
                d_mode = prep_cfg.get("dataset_mode", "synthetic_fixture")
                d_name = prep_cfg.get("dataset_name", "ASVspoof2019_LA" if d_mode == "real_speech" else "SyntheticTestFixtures")
                model_cfg = model.get_config() if hasattr(model, "get_config") and callable(model.get_config) else {}
                model_arch = model_cfg.get("model_type", model.__class__.__name__)

                provenance = build_checkpoint_provenance(
                    dataset_name=d_name,
                    dataset_mode=d_mode,
                    model_architecture=model_arch,
                    preprocessing_config=prep_cfg,
                    training_partition="train",
                    development_partition=val_partition_name,
                    model_version=prep_cfg.get("model_version", "1.0.0"),
                )

                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "class_weights": class_weights.cpu().numpy().tolist(),
                    "model_config": model_cfg,
                    "preprocessing_config": prep_cfg,
                    "provenance": provenance,
                    "seed": seed,
                }
                torch.save(checkpoint_payload, str(out_path))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}).")
                break

    print(f"Training completed. Best checkpoint at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "history": history,
        "save_path": save_path,
    }

