"""End-to-end inference smoke test for VoxGuard baseline models.

Tests that both trained checkpoints (Logistic Regression and Random Forest)
can be reloaded and perform deterministic inference on sample audio files
through the single, unified prediction interface.

Usage:
    python scripts/smoke_test_baseline.py
"""

from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline_models import BaselinePredictor
from src.utils.config import load_config


def run_smoke_test():
    print("=" * 70)
    print(" VoxGuard Baseline Single Prediction Interface Smoke Test")
    print("=" * 70)

    cfg = load_config("configs/default.yaml")
    ckpt_dir = Path("checkpoints")
    lr_ckpt = ckpt_dir / "logistic_regression.joblib"
    rf_ckpt = ckpt_dir / "random_forest.joblib"

    if not lr_ckpt.exists() or not rf_ckpt.exists():
        print("  [FAIL] Missing model checkpoints. Run scripts/train_models.py first.")
        sys.exit(1)

    print("\n[1] Reloading checkpoints via BaselinePredictor...")
    pred_lr = BaselinePredictor.from_checkpoint(str(lr_ckpt), model_name="logistic_regression", config=cfg)
    pred_rf = BaselinePredictor.from_checkpoint(str(rf_ckpt), model_name="random_forest", config=cfg)
    print("  [OK] Successfully loaded Logistic Regression and Random Forest pipelines.")

    # Locate test audio samples
    audio_dir = Path("data/raw/asvspoof2019_la/audio/dev")
    if not audio_dir.exists():
        audio_dir = Path("tests/fixtures/audio/dev")

    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac"))
    if not audio_files:
        print(f"  [FAIL] No audio files found in {audio_dir}")
        sys.exit(1)

    sample_audio = audio_files[0]
    print(f"\n[2] Running inference on sample audio clip: {sample_audio.name}")

    for predictor in [pred_lr, pred_rf]:
        res = predictor.predict_audio(sample_audio)
        print(f"\n  Model: {predictor.model_name}")
        print(f"    - Prediction:           {res['prediction']} ({res['label'].upper()})")
        print(f"    - Probability (Spoof):  {res['probability_spoof']:.4f}")
        print(f"    - Probability (Human):  {res['probability_bonafide']:.4f}")
        print(f"    - Confidence Score:     {res['confidence'] * 100:.2f}%")
        print(f"    - Input Features:       {res['n_features']}-dim MFCC summary")

        # Invariant assertions
        assert 0.0 <= res["probability_spoof"] <= 1.0, "Probability out of [0, 1] range"
        assert 0.0 <= res["probability_bonafide"] <= 1.0, "Probability out of [0, 1] range"
        assert abs(res["probability_spoof"] + res["probability_bonafide"] - 1.0) < 1e-4, "Probabilities do not sum to 1"
        assert res["confidence"] >= 0.5, "Confidence should be >= 0.5 for binary classification"

    # Determinism test on same audio
    res_a = pred_lr.predict_audio(sample_audio)
    res_b = pred_lr.predict_audio(sample_audio)
    assert res_a["prediction"] == res_b["prediction"], "Inference non-deterministic"
    assert abs(res_a["probability_spoof"] - res_b["probability_spoof"]) < 1e-6, "Probabilities differ across runs"
    print("\n  [PASS] Deterministic inference verified across consecutive calls.")

    print("\n" + "=" * 70)
    print(" SMOKE TEST PASSED: Single prediction interface verified.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()

