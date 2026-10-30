"""Download script for external pretrained baseline checkpoint.

Source: https://huggingface.co/koyelog/deepfake-voice-detector-sota
File: pytorch_model.pth (70.27 MB)
Destination: checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth
License: Apache-2.0
"""

import os
from pathlib import Path
import sys
import urllib.request

HF_URL = "https://huggingface.co/koyelog/deepfake-voice-detector-sota/resolve/main/pytorch_model.pth"
TARGET_DIR = Path("checkpoints/real_speech/wav2vec2_deepfake_detector")
TARGET_FILE = TARGET_DIR / "pytorch_model.pth"
EXPECTED_SIZE_BYTES = 73685957  # ~70.27 MB


def download_pretrained_model(force: bool = False):
    """Download pretrained checkpoint from Hugging Face if not already present."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not force:
        size = TARGET_FILE.stat().st_size
        print(f"Pretrained checkpoint already exists at {TARGET_FILE} ({size / (1024*1024):.2f} MB).")
        return

    print(f"Downloading pretrained model from {HF_URL}...")
    print(f"Destination: {TARGET_FILE}")

    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(HF_URL, str(TARGET_FILE), reporthook=report_hook)
        print("\nDownload completed successfully.")
        print(f"Verified size: {TARGET_FILE.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"\nError downloading checkpoint: {e}", file=sys.stderr)
        print("Manual download instructions:", file=sys.stderr)
        print(f"1. Download {HF_URL}", file=sys.stderr)
        print(f"2. Place at {TARGET_FILE}", file=sys.stderr)
        raise


if __name__ == "__main__":
    download_pretrained_model()

