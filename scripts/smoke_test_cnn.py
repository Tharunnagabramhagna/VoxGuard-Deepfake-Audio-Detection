"""End-to-end inference smoke test for VoxGuard 2D Spectrogram-CNN.

Tests that the trained PyTorch checkpoint (best_spec_cnn.pt)
can be reloaded and perform deterministic inference on sample audio files
through the SpecCNNPredictor interface.

Usage:
    python scripts/smoke_test_cnn.py
"""

from pathlib import Path
import sys
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.utils.config import load_config


def run_cnn_smoke_test():
    print("=" * 70)
    print(" VoxGuard 2D Spectrogram-CNN Prediction Interface Smoke Test")
    print("=" * 70)

    cfg = load_config("configs/default.yaml")
    ckpt_dir = Path("checkpoints")
    cnn_ckpt = ckpt_dir / "best_spec_cnn.pt"

    if not cnn_ckpt.exists():
        print(f"  [FAIL] Missing model checkpoint at {cnn_ckpt}. Run scripts/train_models.py --model cnn first.")
        sys.exit(1)

    print(f"\n[1] Reloading CNN checkpoint via SpecCNNPredictor from {cnn_ckpt}...")
    predictor = SpecCNNPredictor.from_checkpoint(str(cnn_ckpt), device="cpu")
    print("  [OK] Successfully loaded Spectrogram-CNN checkpoint and preprocessing metadata.")

    # Locate sample audio files
    audio_dir = Path("data/raw/asvspoof2019_la/audio/dev")
    if not audio_dir.exists():
        audio_dir = Path("tests/fixtures/audio/dev")

    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac"))
    if not audio_files:
        print(f"  [FAIL] No audio files found in {audio_dir}")
        sys.exit(1)

    sample_audio = audio_files[0]
    print(f"\n[2] Running end-to-end inference on sample audio: {sample_audio.name}")

    res = predictor.predict_audio(sample_audio)
    print(f"\n  Model: {predictor.model_name}")
    print(f"    - Prediction:           {res['prediction']} ({res['label'].upper()})")
    print(f"    - Probability (Spoof):  {res['probability_spoof']:.4f}")
    print(f"    - Probability (Human):  {res['probability_bonafide']:.4f}")
    print(f"    - Confidence Score:     {res['confidence'] * 100:.2f}%")
    print(f"    - Spectrogram Shape:    {res['spectrogram_shape']}")

    # Assertions
    assert 0.0 <= res["probability_spoof"] <= 1.0, "Probability out of [0, 1] range"
    assert 0.0 <= res["probability_bonafide"] <= 1.0, "Probability out of [0, 1] range"
    assert abs(res["probability_spoof"] + res["probability_bonafide"] - 1.0) < 1e-4, "Probabilities do not sum to 1"
    assert res["confidence"] >= 0.5, "Confidence score must be >= 0.5"

    # Determinism test
    print("\n[3] Testing inference determinism across repeat calls...")
    res_a = predictor.predict_audio(sample_audio)
    res_b = predictor.predict_audio(sample_audio)
    assert res_a["prediction"] == res_b["prediction"], "Inference non-deterministic"
    assert abs(res_a["probability_spoof"] - res_b["probability_spoof"]) < 1e-6, "Probabilities differ across runs"
    print("  [PASS] Deterministic inference verified.")

    print("\n" + "=" * 70)
    print(" SMOKE TEST PASSED: Spectrogram-CNN prediction interface verified.")
    print("=" * 70)


if __name__ == "__main__":
    run_cnn_smoke_test()

