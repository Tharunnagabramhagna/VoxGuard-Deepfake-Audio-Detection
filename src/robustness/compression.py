"""Audio compression robustness evaluation for VoxGuard.

Implements real standard speech compression codecs and rigorous codec availability checks:
1. ITU-T G.711 mu-law Telephony Codec:
   - Compresses 16-bit linear PCM down to 8-bit log-quantized PCM (64 kbps).
   - Standard international telecommunications codec (North America, Japan, VoIP).
   - Operates natively without external executable binaries.

2. MP3 / AAC Lossy Codecs:
   - Checks for external encoders (ffmpeg / lame).
   - If ffmpeg is absent, strictly reports the MP3/AAC evaluation as BLOCKED
     per policy ("If the required codec is unavailable, report the test as
     blocked instead of pretending it ran").
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class CodecUnavailableError(RuntimeError):
    """Raised when an audio compression codec is requested but unavailable on the host system."""
    pass


def check_codec_availability() -> Dict[str, object]:
    """Inspect the host environment for available real audio compression codecs.

    Returns:
        Dictionary of codec availability and diagnostic status.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    has_ffmpeg = ffmpeg_path is not None

    status = {
        "mulaw_g711": True,  # ITU-T G.711 mu-law is always supported natively
        "mp3": has_ffmpeg,
        "aac": has_ffmpeg,
        "ffmpeg_available": has_ffmpeg,
        "ffmpeg_path": ffmpeg_path,
        "mp3_aac_status": "AVAILABLE" if has_ffmpeg else "BLOCKED",
        "blocked_reason": None if has_ffmpeg else "ffmpeg not found on system PATH; MP3/AAC encoding is blocked.",
    }
    return status


def apply_mulaw_compression(
    waveform: np.ndarray,
    mu: int = 255,
) -> np.ndarray:
    """Apply real ITU-T G.711 mu-law speech companding and 8-bit quantization.

    Standard telecommunication compression:
        Encodes 16-bit linear PCM down to 8-bit non-linear logarithmic PCM (64 kbps),
        then decodes back to linear representation, introducing authentic
        quantization distortion and dynamic range companding.

    Args:
        waveform: Float32 waveform array in [-1.0, 1.0].
        mu: Companding parameter (standard ITU-T value is 255).

    Returns:
        Decompressed float32 waveform reflecting 8-bit mu-law quantization distortion.
    """
    if len(waveform) == 0:
        return waveform.copy()

    # Clip input to standard range [-1.0, 1.0]
    x = np.clip(waveform.astype(np.float64), -1.0, 1.0)

    # 1. Non-linear mu-law compression equation:
    #    F(x) = sgn(x) * ln(1 + mu * |x|) / ln(1 + mu)
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    # 2. Quantize into 8-bit uniform integer levels (0 to 255)
    quantized_8bit = np.round(((fx + 1.0) / 2.0) * mu).astype(np.uint8)

    # 3. De-quantize back to normalized [-1.0, 1.0] domain
    fx_rec = (quantized_8bit.astype(np.float64) / mu) * 2.0 - 1.0

    # 4. Inverse mu-law expansion equation:
    #    x_rec = sgn(fx) * (1 / mu) * ((1 + mu)^|fx| - 1)
    x_rec = np.sign(fx_rec) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(fx_rec)) - 1.0)

    return x_rec.astype(np.float32)


def apply_mp3_compression(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    bitrate_kbps: int = 64,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Encode audio to MP3 at specified bitrate using ffmpeg and decode back to PCM.

    Strict Policy:
        If ffmpeg is missing, raises CodecUnavailableError rather than faking compression.

    Args:
        waveform: Float32 waveform array.
        sample_rate: Audio sampling rate (default 16000 Hz).
        bitrate_kbps: Target MP3 bitrate (e.g. 128, 64, 32 kbps).

    Returns:
        Tuple of (compressed_and_decompressed_waveform, metadata_dict).

    Raises:
        CodecUnavailableError: If ffmpeg is not available on PATH.
    """
    from scipy.io import wavfile

    codec_info = check_codec_availability()
    if not codec_info["ffmpeg_available"]:
        raise CodecUnavailableError(
            f"MP3 compression at {bitrate_kbps} kbps BLOCKED: {codec_info['blocked_reason']}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        in_wav = tmp_dir / "input.wav"
        out_mp3 = tmp_dir / f"compressed_{bitrate_kbps}k.mp3"
        out_wav = tmp_dir / "decompressed.wav"

        # Save input WAV
        wavfile.write(str(in_wav), sample_rate, (waveform * 32767).astype(np.int16))

        # Encode to MP3 via ffmpeg
        cmd_enc = [
            "ffmpeg", "-y", "-i", str(in_wav),
            "-codec:a", "libmp3lame",
            "-b:a", f"{bitrate_kbps}k",
            str(out_mp3),
        ]
        res_enc = subprocess.run(cmd_enc, capture_output=True, text=True)
        if res_enc.returncode != 0:
            raise RuntimeError(f"ffmpeg MP3 encoding failed: {res_enc.stderr}")

        # Decode back to WAV
        cmd_dec = [
            "ffmpeg", "-y", "-i", str(out_mp3),
            "-ar", str(sample_rate), "-ac", "1",
            str(out_wav),
        ]
        res_dec = subprocess.run(cmd_dec, capture_output=True, text=True)
        if res_dec.returncode != 0:
            raise RuntimeError(f"ffmpeg MP3 decoding failed: {res_dec.stderr}")

        sr_dec, data_dec = wavfile.read(str(out_wav))
        dec_waveform = (data_dec.astype(np.float32) / 32767.0).copy()

        return dec_waveform, {
            "codec": "mp3",
            "bitrate_kbps": bitrate_kbps,
            "status": "SUCCESS",
        }

