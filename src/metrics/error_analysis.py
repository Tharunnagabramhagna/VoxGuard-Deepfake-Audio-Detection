"""Forensic error analysis for VoxGuard deepfake audio detection.

Inspects and categorizes representative False Positives and False Negatives:
- False Positives (FP): Genuine human audio falsely flagged as deepfake.
  High-risk failure mode in forensic/verification contexts (false accusation).
- False Negatives (FN): Synthetic speech misclassified as genuine human speech.
  High-risk failure mode in authentication/spoof prevention (security breach).

Records sample identifiers, speaker IDs, generator systems, and prediction confidence.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def mine_errors(
    manifest_records: Union[pd.DataFrame, List[Dict[str, object]]],
    y_scores: Union[np.ndarray, list],
    threshold: float,
    top_k: int = 10,
) -> Dict[str, object]:
    """Mine and rank representative False Positives and False Negatives.

    Args:
        manifest_records: Manifest DataFrame or list of dicts with keys:
                          'utterance_id', 'speaker_id', 'system_id', 'label'.
        y_scores: Predicted probabilities of spoof (class 1).
        threshold: Decision threshold for classification.
        top_k: Maximum number of top error samples to extract per category.

    Returns:
        Structured dictionary containing ranked false positives, false negatives,
        and forensic error summaries.
    """
    if isinstance(manifest_records, list):
        df = pd.DataFrame(manifest_records)
    else:
        df = manifest_records.copy()

    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= threshold).astype(int)

    df["score_spoof"] = y_scores
    df["score_bonafide"] = 1.0 - y_scores
    df["prediction"] = y_pred
    df["pred_label"] = np.where(y_pred == 1, "spoof", "bonafide")

    # Invariants check
    if "label" not in df.columns:
        raise KeyError("Manifest missing required 'label' column.")

    y_true = df["label"].astype(int).values

    # Mine False Positives: true = 0 (bonafide), pred = 1 (spoof)
    fp_mask = (y_true == 0) & (y_pred == 1)
    df_fp = df[fp_mask].sort_values(by="score_spoof", ascending=False)

    # Mine False Negatives: true = 1 (spoof), pred = 0 (bonafide)
    fn_mask = (y_true == 1) & (y_pred == 0)
    df_fn = df[fn_mask].sort_values(by="score_spoof", ascending=True)

    cols_to_export = [
        "utterance_id",
        "speaker_id",
        "system_id",
        "label",
        "label_str",
        "prediction",
        "pred_label",
        "score_spoof",
        "score_bonafide",
    ]
    cols_present = [c for c in cols_to_export if c in df.columns]

    fp_list = df_fp[cols_present].head(top_k).to_dict(orient="records")
    fn_list = df_fn[cols_present].head(top_k).to_dict(orient="records")

    # Group statistics on errors
    fp_by_speaker = df_fp["speaker_id"].value_counts().to_dict() if "speaker_id" in df_fp.columns else {}
    fn_by_system = df_fn["system_id"].value_counts().to_dict() if "system_id" in df_fn.columns else {}

    return {
        "threshold": float(threshold),
        "total_samples": len(df),
        "total_false_positives": int(fp_mask.sum()),
        "total_false_negatives": int(fn_mask.sum()),
        "false_positive_rate": float(fp_mask.sum() / (y_true == 0).sum()) if (y_true == 0).sum() > 0 else 0.0,
        "false_negative_rate": float(fn_mask.sum() / (y_true == 1).sum()) if (y_true == 1).sum() > 0 else 0.0,
        "top_false_positives": fp_list,
        "top_false_negatives": fn_list,
        "fp_speaker_breakdown": fp_by_speaker,
        "fn_system_breakdown": fn_by_system,
    }


def save_error_analysis(
    error_summary: Dict[str, object],
    output_dir: str = "reports/results",
    prefix: str = "error_analysis",
) -> Dict[str, str]:
    """Save error analysis results to machine-readable JSON and CSV.

    Args:
        error_summary: Output dictionary from mine_errors.
        output_dir: Target directory.
        prefix: Filename prefix.

    Returns:
        Dictionary of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)

    # Save CSV of all mined errors
    all_errors = []
    for fp in error_summary.get("top_false_positives", []):
        row = dict(fp)
        row["error_type"] = "FALSE_POSITIVE"
        all_errors.append(row)

    for fn in error_summary.get("top_false_negatives", []):
        row = dict(fn)
        row["error_type"] = "FALSE_NEGATIVE"
        all_errors.append(row)

    csv_path = out_dir / f"{prefix}_samples.csv"
    if all_errors:
        fieldnames = ["error_type"] + [k for k in all_errors[0].keys() if k != "error_type"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_errors)
    else:
        # Create empty placeholder file if zero errors
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("error_type,utterance_id,label,prediction,score_spoof\n")

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }

