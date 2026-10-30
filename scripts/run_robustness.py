"""Robustness and acoustic perturbation stress-testing script for VoxGuard.

Evaluates trained classifiers under Additive White Gaussian Noise (AWGN at 30, 20, 10, 0 dB SNR)
and ITU-T G.711 mu-law telephony companding compression.

Guarantees:
- Strict 1-to-1 pairing of clean reference audio and perturbed signals.
- Zero silent fallbacks to arbitrary labels.

Usage:
    python scripts/run_robustness.py --config configs/default.yaml
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
from src.data.validation import load_aligned_evaluation_set, validate_manifest_integrity
from src.robustness.noise_simulation import apply_awgn
from src.robustness.compression import apply_mulaw_compression, check_codec_availability
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Robustness & Acoustic Stress Testing")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--manifest", default="data/manifests/asvspoof_eval_mini.csv", help="Evaluation manifest path")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--max-samples", type=int, default=30, help="Number of aligned samples to stress test")
    parser.add_argument("--mode", default="fixture", choices=["fixture", "real"],
                        help="Mode: 'fixture' for synthetic tests, 'real' for real speech")
    parser.add_argument("--threshold-source", default="dev", choices=["dev", "default"],
                        help="Threshold calibration source: 'dev' (uses frozen development operating threshold from evaluation_summary.json) or 'default' (uncalibrated 0.5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest missing: {manifest_path}")

    eval_df = pd.read_csv(manifest_path)
    validate_manifest_integrity(eval_df)

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

    m_thresh = {
        "lr": thresholds.get("logistic_regression", 0.5),
        "rf": thresholds.get("random_forest", 0.5),
        "cnn": thresholds.get("spectrogram_cnn", 0.5),
    }

    ckpt_dir = Path(args.checkpoints_dir)
    lr_pred = BaselinePredictor.from_checkpoint(str(ckpt_dir / "logistic_regression.joblib"), model_name="logistic_regression", config=cfg)
    rf_pred = BaselinePredictor.from_checkpoint(str(ckpt_dir / "random_forest.joblib"), model_name="random_forest", config=cfg)
    cnn_pred = SpecCNNPredictor.from_checkpoint(str(ckpt_dir / "best_spec_cnn.pt"), device="cpu")

    raw_dir = Path(cfg["paths"]["raw_asvspoof_dir"])
    clean_waveforms, stress_y, _ = load_aligned_evaluation_set(
        manifest_df=eval_df,
        raw_dir=raw_dir,
        max_samples=args.max_samples,
        config=cfg,
        seed=42,
        dataset_mode="synthetic_fixture" if args.mode == "fixture" else "real_speech",
    )

    print("=" * 75)
    print(f" VoxGuard Robustness Stress Testing (Mode: {args.mode.upper()})")
    print(f" Aligned Samples: N={len(clean_waveforms)} | Bonafide: {(stress_y == 0).sum()} | Spoof: {(stress_y == 1).sum()}")
    print(f" Threshold Source: {args.threshold_source.upper()} (LR={m_thresh['lr']:.4f}, RF={m_thresh['rf']:.4f}, Spec-CNN={m_thresh['cnn']:.4f})")
    print("=" * 75)

    snr_levels = [30.0, 20.0, 10.0, 0.0]
    conditions = [("Clean Reference", lambda wf: wf)]
    for snr in snr_levels:
        conditions.append((f"AWGN {int(snr)}dB SNR", lambda wf, s=snr: apply_awgn(wf, snr_db=s, seed=42)))
    conditions.append(("ITU-T G.711 mu-law", lambda wf: apply_mulaw_compression(wf)))

    print(f"\n  {'Condition':25s} | {'Logistic Reg Acc':17s} | {'Random Forest Acc':18s} | {'Spec-CNN Acc':13s}")
    print("  " + "-" * 75)

    for cond_name, perturb_fn in conditions:
        accuracies = {}
        for m_name, predictor in [("lr", lr_pred), ("rf", rf_pred), ("cnn", cnn_pred)]:
            scores = []
            for wf in clean_waveforms:
                mod_wf = perturb_fn(wf)
                res = predictor.predict_audio(mod_wf, sample_rate=16000)
                scores.append(res["probability_spoof"])
            preds = (np.array(scores) >= m_thresh[m_name]).astype(int)
            acc = float(np.mean(preds == stress_y)) * 100.0
            accuracies[m_name] = acc
        print(f"  {cond_name:25s} | {accuracies['lr']:15.2f}% | {accuracies['rf']:16.2f}% | {accuracies['cnn']:12.2f}%")

    codec_status = check_codec_availability()
    print("  " + "-" * 75)
    print(f"  Lossy Codecs (MP3/AAC): {codec_status['mp3_aac_status']} ({codec_status['blocked_reason']})")
    print("=" * 75)


if __name__ == "__main__":
    main()

