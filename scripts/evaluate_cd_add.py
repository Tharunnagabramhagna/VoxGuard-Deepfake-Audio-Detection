"""Standalone CD-ADD benchmark evaluation wrapper for VoxGuard.

Evaluates trained VoxGuard classifiers on the external CD-ADD held-out test split.
Enforces zero training, zero normalizer tuning, and frozen development thresholds.

Usage:
    python scripts/evaluate_cd_add.py --config configs/default.yaml
    python scripts/evaluate_cd_add.py --manifest data/manifests/cd_add_test.csv
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline_models import BaselinePredictor
from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.metrics.evaluator import evaluate_predictions
from src.data.test_fixtures import create_cd_add_test_fixtures
from src.data.validation import validate_manifest_integrity
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate VoxGuard on External CD-ADD Benchmark")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--manifest", default="data/manifests/cd_add_test.csv", help="CD-ADD manifest path")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--mode", default="fixture", choices=["fixture", "real"],
                        help="Mode: 'fixture' for synthetic tests, 'real' for real speech")
    parser.add_argument("--threshold-source", default="dev", choices=["dev", "default"],
                        help="Threshold calibration source: 'dev' (uses frozen development operating threshold from evaluation_summary.json) or 'default' (uncalibrated 0.5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        if args.mode == "real":
            raise FileNotFoundError(
                f"Real CD-ADD test manifest missing: {manifest_path}. "
                "Acquire real CD-ADD via 'python scripts/download_data.py --subset cd_add'."
            )
        else:
            print("[INFO] Generating CD-ADD synthetic test fixtures...")
            create_cd_add_test_fixtures(
                output_dir="data/raw/cd_add",
                manifest_path=str(manifest_path),
                n_bonafide=25,
                n_spoof=25,
                seed=42,
            )

    df = pd.read_csv(manifest_path)
    validate_manifest_integrity(df)

    # Resolve operating thresholds
    thresholds = {}
    if args.threshold_source == "dev":
        summary_path = Path("reports/results/evaluation_summary.json")
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                thresholds = summary.get("thresholds_calibrated_on_dev", {})
                print(f"  [INFO] Loaded frozen development operating thresholds from {summary_path}:")
                for m, th in thresholds.items():
                    print(f"    - {m}: {th:.4f}")
            except Exception as e:
                print(f"  [WARN] Failed to load dev thresholds from {summary_path}: {e}")
        if not thresholds:
            print("  [WARN] Dev thresholds not found; falling back to default threshold 0.5.")
            thresholds = {"logistic_regression": 0.5, "random_forest": 0.5, "spectrogram_cnn": 0.5}
    else:
        print("  [INFO] Using uncalibrated default decision threshold 0.5 for all models.")
        thresholds = {"logistic_regression": 0.5, "random_forest": 0.5, "spectrogram_cnn": 0.5}

    ckpt_dir = Path(args.checkpoints_dir)
    lr_pred = BaselinePredictor.from_checkpoint(str(ckpt_dir / "logistic_regression.joblib"), model_name="logistic_regression", config=cfg)
    rf_pred = BaselinePredictor.from_checkpoint(str(ckpt_dir / "random_forest.joblib"), model_name="random_forest", config=cfg)
    cnn_pred = SpecCNNPredictor.from_checkpoint(str(ckpt_dir / "best_spec_cnn.pt"), device="cpu")

    print("=" * 75)
    print(f" External Benchmark Evaluation: CD-ADD Test Split (Mode: {args.mode.upper()})")
    print(f" Total Samples: {len(df)} | Bonafide: {(df['label'] == 0).sum()} | Spoof: {(df['label'] == 1).sum()}")
    print(f" Threshold Source: {args.threshold_source.upper()}")
    print("=" * 75)

    scores = {"logistic_regression": [], "random_forest": [], "spectrogram_cnn": []}
    y_true = df["label"].values

    for _, row in df.iterrows():
        p = Path(row["audio_path"])
        scores["logistic_regression"].append(lr_pred.predict_audio(p)["probability_spoof"])
        scores["random_forest"].append(rf_pred.predict_audio(p)["probability_spoof"])
        scores["spectrogram_cnn"].append(cnn_pred.predict_audio(p)["probability_spoof"])

    for k in scores:
        scores[k] = np.array(scores[k])

    print(f"\n  {'Model':20s} | {'Threshold':9s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'F1':9s} | {'ROC-AUC':9s} | {'EER (%)':8s}")
    print("  " + "-" * 88)
    for m in ["logistic_regression", "random_forest", "spectrogram_cnn"]:
        th = thresholds.get(m, 0.5)
        met = evaluate_predictions(y_true, scores[m], threshold=th)
        print(f"  {m:20s} | {th:9.4f} | {met['accuracy']*100:8.2f}% | {met['precision']:9.4f} | {met['recall']:9.4f} | {met['f1']:9.4f} | {met['roc_auc']:9.4f} | {met['eer_percent']:7.2f}%")
    print("=" * 88)


if __name__ == "__main__":
    main()

