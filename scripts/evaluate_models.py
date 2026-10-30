"""Comprehensive, reproducible model evaluation script for VoxGuard.

Evaluates trained models:
1. Logistic Regression (MFCC Summary)
2. Random Forest (MFCC Summary)
3. 2D Spectrogram-CNN (Log-Mel Spectrogram)

Evaluation Protocols:
- Internal Development Partition (Validation): Calibration of operating threshold via EER.
- Internal Evaluation Partition (ASVspoof Eval): Unseen spoofing attacks evaluated with frozen threshold.
- External Benchmark (CD-ADD Test): Out-of-domain evaluation strictly held-out (zero training/tuning).
- Robustness Stress Tests: Clean vs AWGN noise (30, 20, 10, 0 dB SNR) and ITU-T G.711 mu-law compression.
- Forensic Error Analysis: Mining top False Positives and False Negatives with sample IDs.
- Shortcut & Leakage Analysis: Speaker disjointness audit and acoustic shortcut correlations.

Usage:
    python scripts/evaluate_models.py --subset mini
    python scripts/evaluate_models.py --subset full
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for experiment provenance tracking."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="VoxGuard Comprehensive Model Evaluation Suite")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--subset", type=str, default="mini", choices=["mini", "full"], help="Data subset")
    parser.add_argument("--dataset-mode", type=str, default="fixture", choices=["fixture", "real"],
                        help="Operating mode: 'fixture' for synthetic tests, 'real' for real speech research")
    parser.add_argument("--output-dir", type=str, default="reports/results", help="Metrics output directory")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Figures output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Torch execution device")
    return parser.parse_args()

from src.models.baseline_models import BaselinePredictor
from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.metrics.evaluator import calibrate_threshold, evaluate_predictions
from src.metrics.error_analysis import mine_errors, save_error_analysis
from src.metrics.plots import (
    plot_confusion_matrices,
    plot_robustness_curves,
    plot_roc_curves,
    plot_score_distributions,
)
from src.analysis.leakage_analysis import (
    analyze_acoustic_shortcuts,
    group_performance_by_speaker,
    group_performance_by_system,
    verify_speaker_disjointness,
)
from src.robustness.noise_simulation import apply_awgn
from src.robustness.compression import (
    apply_mulaw_compression,
    check_codec_availability,
)
from src.data.test_fixtures import create_cd_add_test_fixtures
from src.data.preprocessor import load_audio_file, preprocess_audio
from src.data.validation import (
    LabelAlignmentError,
    load_aligned_evaluation_set,
    validate_manifest_integrity,
)
from src.utils.config import load_config
from src.utils.seed import set_seed





def load_partition_features(
    partition: str,
    subset: str = "mini",
    processed_dir: Path = Path("data/processed"),
) -> Tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Load cached pre-extracted features and labels."""
    mfcc_path = processed_dir / f"{partition}_{subset}_mfcc.npy"
    mel_path = processed_dir / f"{partition}_{subset}_mel.pt"
    labels_path = processed_dir / f"{partition}_{subset}_labels.npy"

    if not mfcc_path.exists() or not mel_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Pre-extracted features missing for '{partition}_{subset}'. "
            f"Run scripts/extract_features.py first."
        )

    X_mfcc = np.load(str(mfcc_path))
    X_mel = torch.load(str(mel_path), map_location="cpu", weights_only=True)
    y = np.load(str(labels_path))
    return X_mfcc, X_mel, y


def load_manifest(partition: str, subset: str = "mini") -> pd.DataFrame:
    """Load manifest dataframe for a partition."""
    suffix = f"_{subset}" if subset == "mini" else ""
    manifest_path = Path(f"data/manifests/asvspoof_{partition}{suffix}.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def run_evaluation():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    out_dir = Path(args.output_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" VoxGuard Comprehensive Forensic & Robustness Evaluation Suite")
    print(f" Subset: {args.subset.upper()} | Output: {out_dir} | Figures: {fig_dir}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load Trained Model Checkpoints
    # -------------------------------------------------------------------------
    ckpt_dir = Path(args.checkpoints_dir)
    lr_path = ckpt_dir / "logistic_regression.joblib"
    rf_path = ckpt_dir / "random_forest.joblib"
    cnn_path = ckpt_dir / "best_spec_cnn.pt"

    for p in [lr_path, rf_path, cnn_path]:
        if not p.exists():
            print(f"[FAIL] Checkpoint not found: {p}. Run scripts/train_models.py first.")
            sys.exit(1)

    print("\n[1/7] Loading trained model predictors...")
    lr_pred = BaselinePredictor.from_checkpoint(str(lr_path), model_name="logistic_regression", config=cfg)
    rf_pred = BaselinePredictor.from_checkpoint(str(rf_path), model_name="random_forest", config=cfg)
    cnn_pred = SpecCNNPredictor.from_checkpoint(str(cnn_path), device=args.device)
    print("  [OK] Loaded: Logistic Regression, Random Forest, Spectrogram-CNN.")

    # -------------------------------------------------------------------------
    # 2. Validation (Dev) Partition: Threshold Calibration
    # -------------------------------------------------------------------------
    print("\n[2/7] Evaluating Development Partition (Validation) & Calibrating Thresholds...")
    dev_mfcc, dev_mel, dev_y = load_partition_features("dev", args.subset)
    dev_df = load_manifest("dev", args.subset)
    validate_manifest_integrity(dev_df)

    # Predictions on Dev
    lr_dev_scores = lr_pred.predict_proba_features(dev_mfcc)[:, 1]
    rf_dev_scores = rf_pred.predict_proba_features(dev_mfcc)[:, 1]
    cnn_dev_scores = cnn_pred.predict_spectrogram_proba(dev_mel)[:, 1]

    # Threshold calibration strictly on validation data
    thresholds = {
        "logistic_regression": calibrate_threshold(dev_y, lr_dev_scores, method="eer"),
        "random_forest": calibrate_threshold(dev_y, rf_dev_scores, method="eer"),
        "spectrogram_cnn": calibrate_threshold(dev_y, cnn_dev_scores, method="eer"),
    }

    dev_metrics = {
        "logistic_regression": evaluate_predictions(dev_y, lr_dev_scores, threshold=thresholds["logistic_regression"]),
        "random_forest": evaluate_predictions(dev_y, rf_dev_scores, threshold=thresholds["random_forest"]),
        "spectrogram_cnn": evaluate_predictions(dev_y, cnn_dev_scores, threshold=thresholds["spectrogram_cnn"]),
    }

    print("  Calibrated Operating Thresholds (from Dev EER):")
    for m, th in thresholds.items():
        print(f"    - {m:20s}: Threshold = {th:.4f} | Dev EER = {dev_metrics[m]['eer_percent']:.2f}%")

    # -------------------------------------------------------------------------
    # 3. Held-Out Evaluation Partition (Internal Unseen Benchmark)
    # -------------------------------------------------------------------------
    print("\n[3/7] Evaluating Internal Held-Out Evaluation Partition (ASVspoof Eval)...")
    eval_mfcc, eval_mel, eval_y = load_partition_features("eval", args.subset)
    eval_df = load_manifest("eval", args.subset)
    validate_manifest_integrity(eval_df)

    lr_eval_scores = lr_pred.predict_proba_features(eval_mfcc)[:, 1]
    rf_eval_scores = rf_pred.predict_proba_features(eval_mfcc)[:, 1]
    cnn_eval_scores = cnn_pred.predict_spectrogram_proba(eval_mel)[:, 1]

    eval_scores_dict = {
        "logistic_regression": lr_eval_scores,
        "random_forest": rf_eval_scores,
        "spectrogram_cnn": cnn_eval_scores,
    }

    # Evaluate with FROZEN validation threshold
    eval_metrics = {}
    for m in ["logistic_regression", "random_forest", "spectrogram_cnn"]:
        eval_metrics[m] = evaluate_predictions(
            eval_y,
            eval_scores_dict[m],
            threshold=thresholds[m],  # Frozen validation threshold
        )

    print("\n  Internal Evaluation Benchmark Results (Held-Out Eval Split):")
    print("  " + "-" * 75)
    print(f"  {'Model':20s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'F1':9s} | {'ROC-AUC':9s} | {'EER (%)':8s}")
    print("  " + "-" * 75)
    for m, met in eval_metrics.items():
        print(f"  {m:20s} | {met['accuracy']*100:8.2f}% | {met['precision']:9.4f} | {met['recall']:9.4f} | {met['f1']:9.4f} | {met['roc_auc']:9.4f} | {met['eer_percent']:7.2f}%")
    print("  " + "-" * 75)

    # -------------------------------------------------------------------------
    # 4. External Held-Out Benchmark: CD-ADD (Zero Training/Tuning)
    # -------------------------------------------------------------------------
    print("\n[4/7] Evaluating External Out-of-Domain Benchmark: CD-ADD Test...")
    cd_add_manifest_path = Path("data/manifests/cd_add_test.csv")
    if not cd_add_manifest_path.exists():
        if args.dataset_mode == "real":
            raise FileNotFoundError(
                f"Real CD-ADD test manifest missing: {cd_add_manifest_path}. "
                "Silent fallback to fixtures is blocked by scientific integrity policy. "
                "Acquire real CD-ADD via 'python scripts/download_data.py --subset cd_add'."
            )
        else:
            print("  Generating CD-ADD reproducible held-out test fixtures (software verification mode)...")
            create_cd_add_test_fixtures(
                output_dir="data/raw/cd_add",
                manifest_path="data/manifests/cd_add_test.csv",
                n_bonafide=25,
                n_spoof=25,
                seed=42,
            )

    cd_add_df = pd.read_csv(cd_add_manifest_path)
    validate_manifest_integrity(
        cd_add_df,
        expected_mode="synthetic_fixture" if args.dataset_mode == "fixture" else "real_speech" if "dataset_mode" in cd_add_df.columns else None,
    )
    print(f"  CD-ADD Test Set: N={len(cd_add_df)} (Bonafide={(cd_add_df['label']==0).sum()}, Spoof={(cd_add_df['label']==1).sum()})")

    # Run inference across CD-ADD audio files
    cd_add_scores = {"logistic_regression": [], "random_forest": [], "spectrogram_cnn": []}
    cd_add_y = cd_add_df["label"].values

    for idx, row in cd_add_df.iterrows():
        audio_path = Path(row["audio_path"])
        # Baselines
        res_lr = lr_pred.predict_audio(audio_path)
        res_rf = rf_pred.predict_audio(audio_path)
        res_cnn = cnn_pred.predict_audio(audio_path)

        cd_add_scores["logistic_regression"].append(res_lr["probability_spoof"])
        cd_add_scores["random_forest"].append(res_rf["probability_spoof"])
        cd_add_scores["spectrogram_cnn"].append(res_cnn["probability_spoof"])

    for k in cd_add_scores:
        cd_add_scores[k] = np.array(cd_add_scores[k])

    cd_add_metrics = {}
    for m in ["logistic_regression", "random_forest", "spectrogram_cnn"]:
        cd_add_metrics[m] = evaluate_predictions(
            cd_add_y,
            cd_add_scores[m],
            threshold=thresholds[m],  # Frozen validation threshold
        )

    benchmark_title = "Synthetic Fixture Verification" if args.dataset_mode == "fixture" else "Real Speech Evaluation"
    print(f"\n  External Benchmark Results (CD-ADD Split — {benchmark_title}):")
    print("  " + "-" * 75)
    print(f"  {'Model':20s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'F1':9s} | {'ROC-AUC':9s} | {'EER (%)':8s}")
    print("  " + "-" * 75)
    for m, met in cd_add_metrics.items():
        print(f"  {m:20s} | {met['accuracy']*100:8.2f}% | {met['precision']:9.4f} | {met['recall']:9.4f} | {met['f1']:9.4f} | {met['roc_auc']:9.4f} | {met['eer_percent']:7.2f}%")
    print("  " + "-" * 75)

    # -------------------------------------------------------------------------
    # 5. Robustness & Stress Experiments: Noise (AWGN) & Codec Compression
    # -------------------------------------------------------------------------
    print("\n[5/7] Executing Robustness Stress Tests (Additive Noise & Compression)...")
    snr_levels = [30.0, 20.0, 10.0, 0.0]
    robustness_results = {"Clean": {}}
    for m in ["logistic_regression", "random_forest", "spectrogram_cnn"]:
        robustness_results["Clean"][m] = {"accuracy": eval_metrics[m]["accuracy"], "eer": eval_metrics[m]["eer"]}

    # Load strictly aligned evaluation subset directly from eval_df manifest (NEVER glob or default)
    eval_raw_dir = Path("data/raw/asvspoof2019_la")
    clean_waveforms, stress_y, stress_samples_meta = load_aligned_evaluation_set(
        manifest_df=eval_df,
        raw_dir=eval_raw_dir,
        max_samples=30,
        config=cfg,
        seed=42,
        dataset_mode="synthetic_fixture" if args.dataset_mode == "fixture" else "real_speech",
    )
    print(f"  Robustness evaluation subset loaded: N={len(clean_waveforms)} (Bonafide={(stress_y == 0).sum()}, Spoof={(stress_y == 1).sum()})")

    # AWGN noise test across SNRs
    for snr in snr_levels:
        cond_name = f"{int(snr)}dB SNR"
        robustness_results[cond_name] = {}
        for m_name, predictor in [("logistic_regression", lr_pred), ("random_forest", rf_pred), ("spectrogram_cnn", cnn_pred)]:
            scores = []
            for wf in clean_waveforms:
                noisy_wf = apply_awgn(wf, snr_db=snr, seed=42)
                res = predictor.predict_audio(noisy_wf, sample_rate=16000)
                scores.append(res["probability_spoof"])
            scores = np.array(scores)
            preds = (scores >= thresholds[m_name]).astype(int)
            acc = float(np.mean(preds == stress_y))
            robustness_results[cond_name][m_name] = {"accuracy": acc}

    # Compression test: ITU-T G.711 mu-law & MP3/AAC check
    codec_status = check_codec_availability()
    print(f"  Codec Availability: mu-law=OK | MP3={codec_status['mp3_aac_status']} ({codec_status['blocked_reason']})")

    # mu-law compression test
    robustness_results["G.711 mu-law (64kbps)"] = {}
    for m_name, predictor in [("logistic_regression", lr_pred), ("random_forest", rf_pred), ("spectrogram_cnn", cnn_pred)]:
        scores = []
        for wf in clean_waveforms:
            compressed_wf = apply_mulaw_compression(wf)
            res = predictor.predict_audio(compressed_wf, sample_rate=16000)
            scores.append(res["probability_spoof"])
        scores = np.array(scores)
        preds = (scores >= thresholds[m_name]).astype(int)
        acc = float(np.mean(preds == stress_y))
        robustness_results["G.711 mu-law (64kbps)"][m_name] = {"accuracy": acc}

    # Print Robustness Summary
    print("\n  Noise & Compression Robustness (Accuracy across Conditions):")
    print("  " + "-" * 75)
    print(f"  {'Condition':25s} | {'Logistic Reg':14s} | {'Random Forest':14s} | {'Spectrogram-CNN':14s}")
    print("  " + "-" * 75)
    for cond, res in robustness_results.items():
        lr_acc = res.get("logistic_regression", {}).get("accuracy", 0.0) * 100.0
        rf_acc = res.get("random_forest", {}).get("accuracy", 0.0) * 100.0
        cnn_acc = res.get("spectrogram_cnn", {}).get("accuracy", 0.0) * 100.0
        print(f"  {cond:25s} | {lr_acc:13.2f}% | {rf_acc:13.2f}% | {cnn_acc:13.2f}%")
    print("  " + "-" * 75)

    # -------------------------------------------------------------------------
    # 6. Forensic Error Analysis & Leakage Investigation
    # -------------------------------------------------------------------------
    print("\n[6/7] Mining Forensic Errors and Analyzing Shortcut / Leakage Biases...")

    # A. Error Mining for Spectrogram-CNN and Baselines
    error_summaries = {}
    for m in ["logistic_regression", "random_forest", "spectrogram_cnn"]:
        err_res = mine_errors(eval_df, eval_scores_dict[m], threshold=thresholds[m], top_k=5)
        error_summaries[m] = err_res
        save_error_analysis(err_res, output_dir=str(out_dir), prefix=f"error_analysis_{m}")

    print(f"  Mined Error Samples saved to {out_dir}/error_analysis_*.json and .csv")

    # B. Speaker Disjointness Audit
    train_df = load_manifest("train", args.subset)
    spk_audit = verify_speaker_disjointness(train_df, dev_df, eval_df)
    print(f"  Speaker Disjointness Audit: Zero Train/Dev/Eval Speaker Leakage? -> {spk_audit['is_all_disjoint']}")

    # C. Group Breakdown by System ID (Seen vs Unseen Attacks)
    system_breakdown = group_performance_by_system(eval_df, eval_scores_dict["spectrogram_cnn"], threshold=thresholds["spectrogram_cnn"])

    # D. Acoustic Shortcut Correlations
    shortcut_analysis = analyze_acoustic_shortcuts(clean_waveforms, eval_scores_dict["spectrogram_cnn"][:len(clean_waveforms)])
    print(f"  Acoustic Shortcut Correlations (Spectrogram-CNN):")
    print(f"    - RMS Energy Correlation:         r = {shortcut_analysis['correlation_rms_energy']} (Shortcut: {shortcut_analysis['potential_rms_shortcut']})")
    print(f"    - Zero-Crossing Rate Correlation: r = {shortcut_analysis['correlation_zero_crossing_rate']} (Shortcut: {shortcut_analysis['potential_zcr_shortcut']})")

    # -------------------------------------------------------------------------
    # 7. Generate Figures and Save Machine-Readable Summaries
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating Artifacts, Visualizations, and Machine-Readable Metrics...")

    # Plot ROC curves
    plot_roc_curves(eval_metrics, output_path=str(fig_dir / "roc_curves.png"),
                    title=f"Internal Held-Out Evaluation ROC Curves ({args.subset.upper()})")

    # Plot Confusion Matrices
    plot_confusion_matrices(eval_metrics, output_path=str(fig_dir / "confusion_matrices.png"))

    # Plot Score Distributions
    score_plot_data = {
        m: {"y_true": eval_y, "y_scores": eval_scores_dict[m], "threshold": thresholds[m]}
        for m in ["logistic_regression", "random_forest", "spectrogram_cnn"]
    }
    plot_score_distributions(score_plot_data, output_path=str(fig_dir / "score_distributions.png"))

    # Plot Robustness Degradation
    plot_robustness_curves(robustness_results, output_path=str(fig_dir / "robustness_snr_curve.png"))

    # Save comprehensive evaluation summary JSON with complete provenance
    mode_tag = "synthetic_fixture" if args.dataset_mode == "fixture" else "real_speech"
    audio_type = "procedurally_generated_sine_noise_fixture" if args.dataset_mode == "fixture" else "human_speech_recording"
    disclaimer = (
        "SOFTWARE TEST HARNESS ONLY: These metrics were computed on procedurally generated synthetic audio fixtures "
        "(sine waves and white noise) designed to verify mathematical pipeline correctness, EER calculation, and software execution. "
        "They do NOT establish or measure real-world human speech deepfake detection capability."
        if args.dataset_mode == "fixture" else
        "REAL SPEECH RESEARCH: Empirical metrics computed on genuine human speech and spoofing attack benchmarks."
    )

    summary_report = {
        "metadata": {
            "experiment_id": f"eval_{mode_tag}_{args.subset}_{int(time.time())}",
            "dataset_mode": mode_tag,
            "audio_reality": audio_type,
            "is_synthetic_fixture": (args.dataset_mode == "fixture"),
            "status": "COMPLETED_FIXTURE_VERIFICATION_ONLY" if args.dataset_mode == "fixture" else "COMPLETED_REAL_SPEECH_RESEARCH",
            "scientific_disclaimer": disclaimer,
            "subset": args.subset,
            "seed": cfg.get("seed", 42),
            "device": args.device,
            "manifest_paths": {
                "dev": str(Path(f"data/manifests/asvspoof_dev_{args.subset}.csv")),
                "eval": str(Path(f"data/manifests/asvspoof_eval_{args.subset}.csv")),
                "cd_add": str(cd_add_manifest_path),
            },
            "sample_counts": {
                "dev": len(dev_df),
                "eval": len(eval_df),
                "cd_add": len(cd_add_df),
                "robustness_evaluated_samples": len(clean_waveforms),
            },
            "class_distributions": {
                "dev": {"bonafide": int((dev_df["label"] == 0).sum()), "spoof": int((dev_df["label"] == 1).sum())},
                "eval": {"bonafide": int((eval_df["label"] == 0).sum()), "spoof": int((eval_df["label"] == 1).sum())},
                "cd_add": {"bonafide": int((cd_add_df["label"] == 0).sum()), "spoof": int((cd_add_df["label"] == 1).sum())},
                "robustness_clean_vs_perturbed": {"bonafide": int((stress_y == 0).sum()), "spoof": int((stress_y == 1).sum())},
            },
            "checkpoint_identities_sha256": {
                "logistic_regression": compute_file_sha256(lr_path),
                "random_forest": compute_file_sha256(rf_path),
                "spectrogram_cnn": compute_file_sha256(cnn_path),
            },
            "software_environment": {
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "platform": sys.platform,
            },
            "clean_vs_perturbed_alignment": "strictly_paired_1_to_1",
            "codec_status": codec_status,
        },
        "thresholds_calibrated_on_dev": thresholds,
        "development_metrics": {m: {k: v for k, v in met.items() if k != "roc_curve"} for m, met in dev_metrics.items()},
        "internal_eval_metrics": {m: {k: v for k, v in met.items() if k != "roc_curve"} for m, met in eval_metrics.items()},
        "external_cd_add_metrics": {m: {k: v for k, v in met.items() if k != "roc_curve"} for m, met in cd_add_metrics.items()},
        "robustness_sweep": robustness_results,
        "speaker_disjointness_audit": spk_audit,
        "system_id_breakdown_cnn": system_breakdown,
        "shortcut_correlation_cnn": shortcut_analysis,
    }

    summary_json_path = out_dir / "evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Save comparison CSV with explicit dataset_mode and audio_reality columns
    summary_csv_path = out_dir / "evaluation_metrics.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_mode",
            "audio_reality",
            "benchmark",
            "model",
            "sample_count",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "eer_percent",
            "operating_threshold",
        ])
        internal_bench = f"{mode_tag}_internal_eval"
        for m, met in eval_metrics.items():
            writer.writerow([
                mode_tag,
                audio_type,
                internal_bench,
                m,
                len(eval_df),
                met["accuracy"],
                met["precision"],
                met["recall"],
                met["f1"],
                met["roc_auc"],
                met["eer_percent"],
                met["operating_threshold"],
            ])
        external_bench = f"{mode_tag}_cd_add"
        for m, met in cd_add_metrics.items():
            writer.writerow([
                mode_tag,
                audio_type,
                external_bench,
                m,
                len(cd_add_df),
                met["accuracy"],
                met["precision"],
                met["recall"],
                met["f1"],
                met["roc_auc"],
                met["eer_percent"],
                met["operating_threshold"],
            ])

    print(f"\n  [SUCCESS] Evaluation Complete!")
    print(f"  - Metrics JSON:    {summary_json_path}")
    print(f"  - Metrics CSV:     {summary_csv_path}")
    print(f"  - ROC Curves:      {fig_dir / 'roc_curves.png'}")
    print(f"  - Confusion Mat:   {fig_dir / 'confusion_matrices.png'}")
    print(f"  - Score Histograms:{fig_dir / 'score_distributions.png'}")
    print(f"  - Robustness Plot: {fig_dir / 'robustness_snr_curve.png'}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()

