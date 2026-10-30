"""Acoustic shortcut and dataset bias analysis script for VoxGuard.

Evaluates potential non-causal acoustic shortcuts (RMS energy, zero-crossing rate)
and verifies strict speaker disjointness across partitions.

Usage:
    python scripts/run_shortcut_analysis.py --config configs/default.yaml
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

from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.data.validation import load_aligned_evaluation_set, validate_manifest_integrity
from src.analysis.leakage_analysis import (
    analyze_acoustic_shortcuts,
    group_performance_by_system,
    verify_speaker_disjointness,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Acoustic Shortcut & Confounder Analysis")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--subset", default="mini", choices=["mini", "full"], help="Partition subset")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Checkpoints directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    suffix = f"_{args.subset}" if args.subset == "mini" else ""
    manifest_dir = Path("data/manifests")

    train_df = pd.read_csv(manifest_dir / f"asvspoof_train{suffix}.csv")
    dev_df = pd.read_csv(manifest_dir / f"asvspoof_dev{suffix}.csv")
    eval_df = pd.read_csv(manifest_dir / f"asvspoof_eval{suffix}.csv")

    for d in [train_df, dev_df, eval_df]:
        validate_manifest_integrity(d)

    print("=" * 75)
    print(" VoxGuard Forensic Shortcut & Dataset Leakage Analysis")
    print("=" * 75)

    # 1. Speaker Disjointness
    print("\n[1] Partition Speaker Disjointness Audit:")
    spk_audit = verify_speaker_disjointness(train_df, dev_df, eval_df)
    print(f"  Train Speakers: {spk_audit['n_train_speakers']}")
    print(f"  Dev Speakers:   {spk_audit['n_dev_speakers']}")
    print(f"  Eval Speakers:  {spk_audit['n_eval_speakers']}")
    print(f"  Zero Contamination Verified? -> {spk_audit['is_all_disjoint']}")

    # 2. CNN Checkpoint & Predictions for Correlation Analysis
    cnn_path = Path(args.checkpoints_dir) / "best_spec_cnn.pt"
    if not cnn_path.exists():
        print(f"[WARN] CNN checkpoint missing at {cnn_path}. Skipping acoustic shortcut correlation.")
        return

    cnn_pred = SpecCNNPredictor.from_checkpoint(str(cnn_path), device="cpu")
    raw_dir = Path(cfg["paths"]["raw_asvspoof_dir"])
    clean_wfs, eval_y, _ = load_aligned_evaluation_set(
        manifest_df=eval_df,
        raw_dir=raw_dir,
        max_samples=30,
        config=cfg,
        seed=42,
        dataset_mode="synthetic_fixture",
    )

    scores = []
    for wf in clean_wfs:
        res = cnn_pred.predict_audio(wf, sample_rate=16000)
        scores.append(res["probability_spoof"])
    scores = np.array(scores)

    # 3. Acoustic Shortcut Correlations
    print("\n[2] Acoustic Shortcut Correlation Analysis (Spectrogram-CNN):")
    shortcut_res = analyze_acoustic_shortcuts(clean_wfs, scores)
    print(f"  RMS Energy vs P(Spoof):         r = {shortcut_res['correlation_rms_energy']:+.4f} (Shortcut: {shortcut_res['potential_rms_shortcut']})")
    print(f"  Zero-Crossing Rate vs P(Spoof): r = {shortcut_res['correlation_zero_crossing_rate']:+.4f} (Shortcut: {shortcut_res['potential_zcr_shortcut']})")

    # 4. Group Performance by System
    summary_path = Path("reports/results/evaluation_summary.json")
    th_cnn = 0.5
    th_source = "DEFAULT"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            th_cnn = summary.get("thresholds_calibrated_on_dev", {}).get("spectrogram_cnn", 0.5)
            th_source = "DEV"
        except Exception:
            pass

    print(f"\n[3] Performance Grouped by Spoofing System (Threshold={th_cnn:.4f} from {th_source}):")
    system_res = group_performance_by_system(eval_df.iloc[:len(clean_wfs)], scores, threshold=th_cnn)
    for sys_id, metrics in system_res.items():
        print(f"  System {sys_id:>4s}: N={metrics['n_samples']:<3d} | Acc={metrics['accuracy']*100:6.2f}% | Mean P(Spoof)={metrics['mean_score']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

