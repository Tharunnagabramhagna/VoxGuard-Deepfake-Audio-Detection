"""Evaluation script for MLAAD-tiny deepfake detection models.

Evaluates an untouched pretrained or fine-tuned model against:
1. Development Set (300 samples) -> Computes ROC, EER, and optimal decision threshold.
2. Held-Out Test Set (300 samples) -> Computes:
   - Accuracy, Precision, Recall, F1-score, ROC-AUC, EER
   - Confusion matrix (TN, FP, FN, TP)
   - Breakdown by Seen vs Unseen generators
   - Breakdown per individual synthetic generator
   - Detailed list of False Positives and False Negatives
   - Inference latency measurements
Saves results to JSON (e.g. reports/results/pretrained_baseline_mlaad.json).
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.test_pretrained_detector import PretrainedDeepfakeDetector, load_pretrained_detector, predict_audio_deepfake


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the corresponding threshold.

    EER occurs where False Positive Rate (FPR) == False Negative Rate (FNR = 1 - TPR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where |fpr - fnr| is minimized
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_thresh = thresholds[idx]
    return float(eer), float(eer_thresh)


def evaluate_dataset(
    model: PretrainedDeepfakeDetector,
    manifest_path: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, object]:
    """Run full evaluation on a manifest of audio files."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    y_true = []
    y_scores = []
    y_preds = []
    latencies = []

    detailed_results = []
    per_generator = {}

    for item in records:
        fpath = item["local_path"]
        label = item["label"]  # 0 = bonafide, 1 = spoof
        system_id = item.get("system_id", "unknown")
        gen_status = item.get("generator_status", "unknown")

        res = predict_audio_deepfake(model, fpath, device=device)

        prob_fake = res["prob_fake"]
        pred_label = 1 if prob_fake >= threshold else 0
        infer_ms = res["timing_ms"]["model_inference"]

        y_true.append(label)
        y_scores.append(prob_fake)
        y_preds.append(pred_label)
        latencies.append(infer_ms)

        is_correct = (pred_label == label)
        err_type = "correct"
        if not is_correct:
            err_type = "false_positive" if pred_label == 1 else "false_negative"

        record_res = {
            "utterance_id": item["utterance_id"],
            "system_id": system_id,
            "generator_status": gen_status,
            "ground_truth": label,
            "predicted": pred_label,
            "prob_fake": prob_fake,
            "raw_logit": res["raw_logit"],
            "inference_ms": infer_ms,
            "error_type": err_type,
            "local_path": fpath,
        }
        detailed_results.append(record_res)

        # Aggregate per generator
        if system_id not in per_generator:
            per_generator[system_id] = {
                "system_id": system_id,
                "generator_status": gen_status,
                "ground_truth": label,
                "total": 0,
                "correct": 0,
                "probs": [],
            }
        per_generator[system_id]["total"] += 1
        if is_correct:
            per_generator[system_id]["correct"] += 1
        per_generator[system_id]["probs"].append(prob_fake)

    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)
    y_preds_np = np.array(y_preds)

    acc = float(accuracy_score(y_true_np, y_preds_np))
    prec = float(precision_score(y_true_np, y_preds_np, zero_division=0))
    rec = float(recall_score(y_true_np, y_preds_np, zero_division=0))
    f1 = float(f1_score(y_true_np, y_preds_np, zero_division=0))
    auc = float(roc_auc_score(y_true_np, y_scores_np)) if len(np.unique(y_true_np)) > 1 else 0.5
    eer, eer_thresh = compute_eer(y_true_np, y_scores_np)

    cm = confusion_matrix(y_true_np, y_preds_np, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0

    # Per generator metrics
    gen_metrics = {}
    for gid, gdata in per_generator.items():
        g_acc = gdata["correct"] / gdata["total"] if gdata["total"] > 0 else 0.0
        g_mean_prob = float(np.mean(gdata["probs"]))
        gen_metrics[gid] = {
            "system_id": gid,
            "generator_status": gdata["generator_status"],
            "ground_truth": gdata["ground_truth"],
            "total_samples": gdata["total"],
            "correct_predictions": gdata["correct"],
            "accuracy": round(g_acc, 4),
            "mean_prob_fake": round(g_mean_prob, 4),
        }

    # Breakdown by generator_status
    status_metrics = {}
    for st in ["human", "seen_generator", "unseen_generator", "seen_in_train", "unseen_in_train"]:
        st_items = [d for d in detailed_results if d["generator_status"] == st]
        if st_items:
            st_correct = sum(1 for d in st_items if d["error_type"] == "correct")
            status_metrics[st] = {
                "total": len(st_items),
                "correct": st_correct,
                "accuracy": round(st_correct / len(st_items), 4),
                "mean_prob_fake": round(float(np.mean([d["prob_fake"] for d in st_items])), 4),
            }

    # Error instances
    false_positives = [d for d in detailed_results if d["error_type"] == "false_positive"]
    false_negatives = [d for d in detailed_results if d["error_type"] == "false_negative"]

    return {
        "threshold_used": threshold,
        "sample_count": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_thresh, 4),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "labels": ["0: Bonafide", "1: Spoof"],
        },
        "status_breakdown": status_metrics,
        "per_generator": gen_metrics,
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
        },
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positive_samples": false_positives[:10],
        "false_negative_samples": false_negatives[:10],
        "y_true": [int(x) for x in y_true],
        "y_scores": [round(float(x), 5) for x in y_scores],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector on MLAAD-tiny")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth")
    parser.add_argument("--dev-manifest", type=str, default="data/manifests/mlaad_dev.json")
    parser.add_argument("--test-manifest", type=str, default="data/manifests/mlaad_test.json")
    parser.add_argument("--output", type=str, default="reports/results/pretrained_baseline_mlaad.json")
    args = parser.parse_args()

    print("=" * 80)
    print("VOXGUARD MODEL EVALUATION ON MLAAD-TINY")
    print("=" * 80)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_pretrained_detector(args.checkpoint, device="cpu")
    print("Model loaded successfully.")

    # 1. Dev Evaluation -> Determine optimal threshold
    print(f"\n[Step 1] Evaluating Development Set ({args.dev_manifest})...")
    dev_results_def = evaluate_dataset(model, args.dev_manifest, threshold=0.5)
    eer_thresh = dev_results_def["eer_threshold"]
    print(f"  Dev Accuracy (default 0.50): {dev_results_def['accuracy']*100:.2f}% | EER: {dev_results_def['eer']*100:.2f}%")
    print(f"  Dev EER Operating Threshold: {eer_thresh:.4f}")

    # Also evaluate dev with eer_threshold
    dev_results_opt = evaluate_dataset(model, args.dev_manifest, threshold=eer_thresh)
    print(f"  Dev Accuracy (tuned {eer_thresh:.4f}): {dev_results_opt['accuracy']*100:.2f}% | F1: {dev_results_opt['f1_score']:.4f}")

    # 2. Test Evaluation -> Evaluate on unseen test set using both thresholds
    print(f"\n[Step 2] Evaluating Held-Out Test Set ({args.test_manifest})...")
    print("  Evaluating with default threshold 0.50:")
    test_results_def = evaluate_dataset(model, args.test_manifest, threshold=0.5)
    print(f"    - Accuracy:  {test_results_def['accuracy']*100:.2f}%")
    print(f"    - Precision: {test_results_def['precision']*100:.2f}%")
    print(f"    - Recall:    {test_results_def['recall']*100:.2f}%")
    print(f"    - F1-Score:  {test_results_def['f1_score']:.4f}")
    print(f"    - ROC-AUC:   {test_results_def['roc_auc']:.4f}")
    print(f"    - EER:       {test_results_def['eer']*100:.2f}%")
    print(f"    - Confusion: TN={test_results_def['confusion_matrix']['tn']}, FP={test_results_def['confusion_matrix']['fp']}, FN={test_results_def['confusion_matrix']['fn']}, TP={test_results_def['confusion_matrix']['tp']}")

    print(f"\n  Evaluating with Dev-calibrated threshold {eer_thresh:.4f}:")
    test_results_calib = evaluate_dataset(model, args.test_manifest, threshold=eer_thresh)
    print(f"    - Accuracy:  {test_results_calib['accuracy']*100:.2f}%")
    print(f"    - Precision: {test_results_calib['precision']*100:.2f}%")
    print(f"    - Recall:    {test_results_calib['recall']*100:.2f}%")
    print(f"    - F1-Score:  {test_results_calib['f1_score']:.4f}")
    print(f"    - Confusion: TN={test_results_calib['confusion_matrix']['tn']}, FP={test_results_calib['confusion_matrix']['fp']}, FN={test_results_calib['confusion_matrix']['fn']}, TP={test_results_calib['confusion_matrix']['tp']}")

    # Save unified evaluation report
    report = {
        "checkpoint": args.checkpoint,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dev_tuning": {
            "dev_manifest": args.dev_manifest,
            "default_threshold": 0.50,
            "dev_eer": dev_results_def["eer"],
            "optimal_threshold_from_dev": eer_thresh,
            "dev_accuracy_default": dev_results_def["accuracy"],
            "dev_accuracy_tuned": dev_results_opt["accuracy"],
        },
        "test_evaluation_default_threshold": test_results_def,
        "test_evaluation_dev_calibrated_threshold": test_results_calib,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved evaluation report: {args.output}")


if __name__ == "__main__":
    main()

