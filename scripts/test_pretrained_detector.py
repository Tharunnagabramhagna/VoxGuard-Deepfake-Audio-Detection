"""Standalone verification and inference test for the pretrained real-speech deepfake detector.

Model Provenance:
- Source: Hugging Face koyelog/deepfake-voice-detector-sota
- Architecture: ResNet-18 Front-End (3 residual stages) + BiGRU (2-layer, 512 hidden)
                + Multi-Head Self-Attention (8 heads) + MLP Classification Head
- Acoustic Input: 128-bin Log-Mel Spectrogram (16 kHz, n_fft=1024, hop=512, duration=4.0s)
- Output: Single scalar logit -> Sigmoid probability P(Fake/Spoof) in [0.0, 1.0]
- Decision Threshold: 0.50 (Default: < 0.5 = Bonafide / Real, >= 0.5 = Spoof / Fake)
- Safe Storage Location: checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth
"""

import argparse
import ctypes
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import load_audio_file, preprocess_audio
from src.features.mel_spectrogram import extract_log_mel_spectrogram


# -----------------------------------------------------------------------------
# Hardware and RAM Measurement Utilities
# -----------------------------------------------------------------------------

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_ram_info() -> Tuple[float, float, float]:
    """Return (total_gb, used_gb, free_gb) using Windows API."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    total_gb = stat.ullTotalPhys / (1024**3)
    avail_gb = stat.ullAvailPhys / (1024**3)
    used_gb = total_gb - avail_gb
    return total_gb, used_gb, avail_gb


# -----------------------------------------------------------------------------
# Pretrained Architecture Definition
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual building block matching the pretrained checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = downsample if downsample is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class PretrainedDeepfakeDetector(nn.Module):
    """Exact PyTorch model reconstruction matching koyelog/deepfake-voice-detector-sota.

    Consists of:
      1. Initial 7x7 Conv2d (stride 2) + BatchNorm2d
      2. Layer 1: 2 ResidualBlocks (64 channels)
      3. Layer 2: 2 ResidualBlocks (128 channels, 1x1 downsampling conv)
      4. Layer 3: 2 ResidualBlocks (256 channels, 1x1 downsampling conv)
      5. Frequency pooling + 2-layer Bidirectional GRU (hidden_size 256 -> 512)
      6. Multi-Head Self-Attention (8 heads, embed_dim 512)
      7. Dense classifier MLP with BatchNorm and Dropout
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 1: 64 -> 64
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        # Layer 2: 64 -> 128
        downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, downsample=downsample2),
            ResidualBlock(128, 128),
        )

        # Layer 3: 128 -> 256
        downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, downsample=downsample3),
            ResidualBlock(256, 256),
        )

        # BiGRU
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # Classifier MLP Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-Mel spectrogram tensor of shape (batch, 1, 128, time_frames).

        Returns:
            Scalar logit tensor of shape (batch, 1).
        """
        # Feature extraction front-end
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # shape: (B, 256, H', T')

        # Pool along frequency axis (dim=2) to get temporal representations
        x = x.mean(dim=2)  # shape: (B, 256, T')
        x = x.permute(0, 2, 1)  # shape: (B, T', 256)

        # Temporal sequence modeling
        gru_out, _ = self.gru(x)  # shape: (B, T', 512)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)  # shape: (B, T', 512)
        pooled = attn_out.mean(dim=1)  # shape: (B, 512)

        # Classifier
        logits = self.classifier(pooled)  # shape: (B, 1)
        return logits


def load_pretrained_detector(checkpoint_path: str, device: str = "cpu") -> PretrainedDeepfakeDetector:
    """Load model architecture and weights from disk."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = PretrainedDeepfakeDetector()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state

    msg = model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def predict_audio_deepfake(
    model: PretrainedDeepfakeDetector,
    audio_path: str,
    device: str = "cpu",
    target_sr: int = 16000,
    target_duration: float = 4.0,
) -> Dict[str, object]:
    """Run full forensic deepfake audio analysis on a single audio file."""
    t_start = time.perf_counter()

    # 1. Load audio
    t0 = time.perf_counter()
    raw_wave, orig_sr, err = load_audio_file(audio_path, target_sr=target_sr)
    if err or raw_wave is None:
        raise ValueError(f"Failed to load audio: {err}")
    t_load = (time.perf_counter() - t0) * 1000

    # 2. Preprocess to 16 kHz mono normalized
    t0 = time.perf_counter()
    cfg = {
        "sample_rate": target_sr,
        "duration_seconds": target_duration,
        "target_samples": int(target_sr * target_duration),
        "trim_silence": False,
        "peak_norm": True,
    }
    prep_wave, meta = preprocess_audio(raw_wave, orig_sr, config=cfg)
    t_prep = (time.perf_counter() - t0) * 1000

    # 3. Extract 128-bin Log-Mel Spectrogram (n_fft=1024, hop=512)
    t0 = time.perf_counter()
    mel = extract_log_mel_spectrogram(
        prep_wave,
        sample_rate=target_sr,
        n_mels=128,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        add_channel_dim=True,
    )  # shape: (1, 128, 126)
    t_feat = (time.perf_counter() - t0) * 1000

    # 4. Model inference
    t0 = time.perf_counter()
    inp = mel.unsqueeze(0).to(device)  # shape: (1, 1, 128, 126)
    with torch.no_grad():
        logit = model(inp).item()
        prob_fake = torch.sigmoid(torch.tensor(logit)).item()
    t_infer = (time.perf_counter() - t0) * 1000

    total_time = (time.perf_counter() - t_start) * 1000

    is_spoof = prob_fake >= 0.5
    verdict = "AI-Generated Voice" if is_spoof else "Human Voice"
    confidence = prob_fake if is_spoof else (1.0 - prob_fake)

    return {
        "audio_path": audio_path,
        "classification": verdict,
        "confidence": confidence,
        "prob_fake": prob_fake,
        "prob_bonafide": 1.0 - prob_fake,
        "raw_logit": logit,
        "audio_duration_sec": len(prep_wave) / target_sr,
        "spectrogram_shape": list(inp.shape),
        "timing_ms": {
            "load": t_load,
            "preprocess": t_prep,
            "feature_extraction": t_feat,
            "model_inference": t_infer,
            "total": total_time,
        },
    }


def run_verification_suite():
    """Run full verification suite answering all Phase 1 research requirements."""
    ckpt_path = "checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth"

    print("=" * 80)
    print("VOXGUARD PRETRAINED REAL-SPEECH DETECTOR VERIFICATION")
    print("=" * 80)

    # 1. Hardware & Environment Check
    total_ram, used_ram, free_ram = get_ram_info()
    print(f"\n[1] Environment & Hardware:")
    print(f"  - Python Version: {sys.version.split()[0]}")
    print(f"  - PyTorch Version: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    print(f"  - CPU Logical Cores: {os.cpu_count()}")
    print(f"  - System RAM: {total_ram:.2f} GB total, {used_ram:.2f} GB used, {free_ram:.2f} GB available")

    # 2. Model Checkpoint Loading
    print(f"\n[2] Checkpoint Inspection:")
    print(f"  - Path: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        print(f"  [ERROR] Checkpoint not found at {ckpt_path}")
        return

    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  - File Size: {ckpt_size_mb:.2f} MB")

    t0 = time.perf_counter()
    model = load_pretrained_detector(ckpt_path, device="cpu")
    t_load = (time.perf_counter() - t0) * 1000

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    print(f"  - Model Load Time: {t_load:.2f} ms")

    total_ram, used_ram_post, free_ram_post = get_ram_info()
    ram_delta = (used_ram_post - used_ram) * 1024
    print(f"  - RAM Consumption of Model: {ram_delta:.2f} MB")

    # 3. Latency Benchmarking (Cold vs Warm)
    print(f"\n[3] Inference Latency Benchmarking (CPU):")
    dummy_input = torch.randn(1, 1, 128, 126)

    # Cold run
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    cold_latency = (time.perf_counter() - t0) * 1000
    print(f"  - Cold Forward Pass: {cold_latency:.2f} ms")

    # Warm runs (average over 5 runs)
    warm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        warm_times.append((time.perf_counter() - t0) * 1000)
    avg_warm = sum(warm_times) / len(warm_times)
    print(f"  - Warm Forward Pass (avg of 5): {avg_warm:.2f} ms (min: {min(warm_times):.2f} ms, max: {max(warm_times):.2f} ms)")

    # 4. Audio Inference Test
    print(f"\n[4] Audio Sample Inference Results:")
    sample_files = [
        ("data/raw/audio_fixtures/sample_01.wav", "Sine Wave Software Test Fixture", "N/A"),
    ]

    # Check cached Hugging Face samples if downloaded
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--mueller91--MLAAD-tiny")
    if os.path.isdir(hf_cache_dir):
        for root, _, files in os.walk(hf_cache_dir):
            if ".no_exist" in root:
                continue
            for f in files:
                if f.endswith(".wav"):
                    fpath = os.path.join(root, f)
                    if os.path.isfile(fpath) and os.path.getsize(fpath) > 100:
                        if "original" in fpath and len([s for s in sample_files if "Genuine" in s[1]]) == 0:
                            sample_files.append((fpath, f"Genuine Human Speech ({f})", "Bonafide (Real)"))
                        elif "fake" in fpath and len([s for s in sample_files if "Synthetic" in s[1]]) == 0:
                            sample_files.append((fpath, f"AI Synthetic Voice ({f})", "Spoof (Fake)"))

    for path, desc, ground_truth in sample_files:
        if not os.path.isfile(path):
            continue
        try:
            res = predict_audio_deepfake(model, path)
            print(f"\n  * File: {os.path.basename(path)}")
            print(f"    Description: {desc}")
            print(f"    Ground Truth: {ground_truth}")
            print(f"    Classification: {res['classification']} (Confidence: {res['confidence']*100:.2f}%)")
            print(f"    P(Fake): {res['prob_fake']*100:.2f}% | P(Bonafide): {res['prob_bonafide']*100:.2f}%")
            print(f"    Raw Logit: {res['raw_logit']:+.4f}")
            print(f"    Latency: Inference={res['timing_ms']['model_inference']:.2f} ms, Total={res['timing_ms']['total']:.2f} ms")
        except Exception as e:
            print(f"  [ERROR] Failed to analyze {path}: {e}")

    print("\n" + "=" * 80)
    print("VERIFICATION SUITE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Pretrained Deepfake Detector")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file to test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth")
    args = parser.parse_args()

    if args.audio:
        model = load_pretrained_detector(args.checkpoint)
        res = predict_audio_deepfake(model, args.audio)
        print(f"File: {args.audio}")
        print(f"Classification: {res['classification']} (Confidence: {res['confidence']*100:.2f}%)")
        print(f"P(Fake): {res['prob_fake']*100:.2f}% | P(Real): {res['prob_bonafide']*100:.2f}%")
        print(f"Inference Latency: {res['timing_ms']['model_inference']:.2f} ms")
    else:
        run_verification_suite()
