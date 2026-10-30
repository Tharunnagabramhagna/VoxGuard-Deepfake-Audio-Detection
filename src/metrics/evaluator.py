"""Comprehensive evaluation metrics for VoxGuard deepfake audio detection.

Defines standard forensic and biometric evaluation metrics:
- Precision, Recall, F1-Score (Positive class = Synthetic/Spoof, label 1).
- ROC-AUC (Area Under Receiver Operating Characteristic Curve).
- Equal Error Rate (EER) with linear interpolation between operating points.
- Confusion Matrix (TN, FP, FN, TP).
- Validation-only threshold calibration.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
) -> Tuple[float, float]:
    """Calculate Equal Error Rate (EER) and optimal operating threshold.

    The positive class is explicitly defined as Synthetic/Spoof (label 1).
    EER is the operating point where False Acceptance Rate (FAR / FPR) equals
    False Rejection Rate (FRR / FNR).

    Numerical Method:
        1. Computes ROC curve operating points (FPR, TPR, thresholds).
        2. Derives FNR = 1.0 - TPR.
        3. Identifies the crossover interval where FPR and FNR intersect.
        4. Applies linear interpolation across the interval to find the exact EER
           and corresponding decision threshold.

    Args:
        y_true: Ground truth binary labels (0 = bonafide, 1 = spoof).
        y_scores: Continuous prediction scores or probabilities for spoof (class 1).

    Returns:
        Tuple of (eer_value, eer_threshold), where eer_value is in [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    # Edge case: empty or single-class array
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return 0.0, 0.5

    # Edge case: perfect separation
    scores_bonafide = y_scores[y_true == 0]
    scores_spoof = y_scores[y_true == 1]
    if np.max(scores_bonafide) < np.min(scores_spoof):
        eer_threshold = float((np.max(scores_bonafide) + np.min(scores_spoof)) / 2.0)
        return 0.0, eer_threshold

    # Calculate ROC operating points (thresholds sorted descending)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find where FPR and FNR cross: fpr increases with index, fnr decreases
    diff = fpr - fnr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Linear interpolation between idx and idx+1
        x1, y1_fpr, y1_fnr = thresholds[idx], fpr[idx], fnr[idx]
        x2, y2_fpr, y2_fnr = thresholds[idx + 1], fpr[idx + 1], fnr[idx + 1]

        denom = (y2_fpr - y1_fpr) - (y2_fnr - y1_fnr)
        if abs(denom) > 1e-12:
            alpha = (y1_fnr - y1_fpr) / denom
            eer_threshold = float(x1 + alpha * (x2 - x1))
            eer_val = float(y1_fpr + alpha * (y2_fpr - y1_fpr))
        else:
            eer_threshold = float((x1 + x2) / 2.0)
            eer_val = float((y1_fpr + y2_fpr) / 2.0)
    else:
        # Fallback to closest point
        min_idx = np.argmin(np.abs(diff))
        eer_val = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
        eer_threshold = float(thresholds[min_idx])

    eer_val = float(np.clip(eer_val, 0.0, 1.0))
    return eer_val, eer_threshold


def calibrate_threshold(
    y_val_true: Union[np.ndarray, list],
    y_val_scores: Union[np.ndarray, list],
    method: str = "eer",
) -> float:
    """Derive classification decision threshold using VALIDATION data only.

    CRITICAL POLICY:
        Threshold selection must be performed strictly on validation data (e.g. dev).
        The returned threshold must be frozen and evaluated as-is on test / eval data.
        Never tune thresholds on test partitions.

    Args:
        y_val_true: Validation ground truth binary labels (0 = bonafide, 1 = spoof).
        y_val_scores: Validation prediction scores/probabilities for spoof.
        method: Calibration criterion ('eer', 'youden_j', or 'f1').

    Returns:
        Frozen operating threshold (float).
    """
    y_val_true = np.asarray(y_val_true, dtype=int)
    y_val_scores = np.asarray(y_val_scores, dtype=float)

    if method == "eer":
        _, threshold = compute_eer(y_val_true, y_val_scores)
        return float(threshold)

    elif method == "youden_j":
        fpr, tpr, thresholds = roc_curve(y_val_true, y_val_scores, pos_label=1)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    elif method == "f1":
        # Search thresholds to maximize F1 on validation
        candidate_thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = -1.0
        best_th = 0.5
        for th in candidate_thresholds:
            preds = (y_val_scores >= th).astype(int)
            score = f1_score(y_val_true, preds, pos_label=1, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        return float(best_th)

    else:
        raise ValueError(f"Unknown threshold calibration method: {method}. Choose 'eer', 'youden_j', or 'f1'.")


def evaluate_predictions(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """Calculate comprehensive evaluation metrics on held-out predictions.

    Positive class is explicitly defined as synthetic/spoof (label 1).

    Args:
        y_true: Ground truth binary labels (0 = bonafide, 1 = spoof).
        y_scores: Continuous prediction scores or probabilities for spoof.
        threshold: Operating decision threshold. If None, computes EER threshold.
                   Note: For test/eval splits, this must be a frozen threshold
                   calibrated on the validation set.

    Returns:
        Structured metric dictionary.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    # 1. EER and EER threshold
    eer_val, eer_th = compute_eer(y_true, y_scores)

    # 2. Operating threshold
    operating_threshold = float(threshold if threshold is not None else eer_th)

    # 3. Discrete predictions at operating threshold
    y_pred = (y_scores >= operating_threshold).astype(int)

    # 4. Standard classification metrics (positive class = spoof = 1)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # 5. ROC-AUC
    unique_labels = np.unique(y_true)
    if len(unique_labels) >= 2:
        auc = float(roc_auc_score(y_true, y_scores))
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fpr_list = [float(x) for x in fpr]
        tpr_list = [float(x) for x in tpr]
    else:
        auc = float("nan")
        fpr_list, tpr_list = [], []

    # 6. Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # 7. Error rates
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False Acceptance Rate of spoof
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0  # False Rejection Rate of spoof

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "eer": eer_val,
        "eer_percent": round(eer_val * 100.0, 3),
        "eer_threshold": float(eer_th),
        "operating_threshold": operating_threshold,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "far": far,
        "frr": frr,
        "roc_curve": {
            "fpr": fpr_list,
            "tpr": tpr_list,
        },
        "n_samples": len(y_true),
        "n_bonafide": int(np.sum(y_true == 0)),
        "n_spoof": int(np.sum(y_true == 1)),
        "positive_class": "spoof (label 1)",
    }

