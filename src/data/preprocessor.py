"""Audio preprocessing pipeline for VoxGuard.

Handles loading, resampling, mono conversion, duration normalization (crop/pad),
silence trimming, and loudness normalization for raw audio waveforms.

Preprocessing Policy (documented decisions):
─────────────────────────────────────────────
1. RESAMPLING: All audio is resampled to 16,000 Hz using scipy.signal.resample
   (or librosa if available). This matches ASVspoof 2019 native sampling rate.

2. CHANNEL CONVERSION: Stereo or multi-channel audio is averaged to mono.

3. SILENCE TRIMMING: Leading and trailing silence below the configured dB
   threshold (default 30 dB) is removed BEFORE duration normalization.
   Rationale: Prevents models from learning silence-duration shortcuts.

4. DURATION NORMALIZATION (deterministic crop/pad):
   - Target duration: 4.0 seconds = 64,000 samples at 16 kHz.
   - If waveform < target: REPEAT-PAD by tiling the waveform cyclically,
     then truncating to exact target length. This preserves spectral character
     better than zero-padding, which introduces artificial silence.
   - If waveform > target: CENTER-CROP by taking the middle segment.
     Rationale: Energy and articulation tend to concentrate in the center
     of recorded utterances. Deterministic center selection ensures
     reproducibility (no random offsets).
   - If waveform is exactly target length: pass through unchanged.

5. AMPLITUDE NORMALIZATION: Peak-normalize waveform to [-1.0, 1.0] range.
   This prevents loudness variation from becoming a shortcut feature.

6. CORRUPT FILE HANDLING: Files that fail to decode or produce all-zero
   waveforms after trimming are flagged and excluded from manifests.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Set, Tuple

import numpy as np

SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".flac", ".mp3", ".m4a"}

__all__ = [
    "SUPPORTED_AUDIO_EXTENSIONS",
    "check_audio_decoders",
    "load_audio_file",
    "preprocess_audio",
    "normalize_duration",
    "peak_normalize",
    "trim_silence",
    "validate_audio_file",
]


def check_audio_decoders() -> Dict[str, bool]:
    """Check availability of audio decoders for supported formats.

    Returns:
        Dictionary mapping codec formats to boolean availability flags.
    """
    import importlib.util

    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_soundfile = importlib.util.find_spec("soundfile") is not None
    has_soundfile_mp3 = False
    if has_soundfile:
        try:
            import soundfile as sf
            has_soundfile_mp3 = "MP3" in sf.available_formats()
        except Exception:
            pass

    return {
        "wav": True,  # scipy.io.wavfile is built into environment
        "flac": has_soundfile or has_ffmpeg,
        "mp3": has_soundfile_mp3 or has_ffmpeg,
        "m4a": has_ffmpeg,
        "ffmpeg": has_ffmpeg,
        "soundfile": has_soundfile,
    }


def _decode_with_ffmpeg(path: Path, target_sr: int) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    """Decode audio using FFmpeg subprocess to 16 kHz mono WAV."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        return None, 0, "FFmpeg executable not found on system PATH."

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_wav = Path(tmp_dir) / "decoded.wav"
        cmd = [
            ffmpeg_exe, "-y", "-i", str(path),
            "-ar", str(target_sr), "-ac", "1",
            "-f", "wav", str(out_wav)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode != 0:
                return None, 0, f"FFmpeg decoding failed: {res.stderr.strip()[:200]}"
            if not out_wav.exists() or out_wav.stat().st_size == 0:
                return None, 0, "FFmpeg produced empty audio output."

            from scipy.io import wavfile
            orig_sr, data = wavfile.read(str(out_wav))
            if data.dtype == np.int16:
                waveform = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                waveform = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.float32:
                waveform = data
            elif data.dtype == np.uint8:
                waveform = (data.astype(np.float32) - 128.0) / 128.0
            else:
                waveform = data.astype(np.float32)

            return waveform, orig_sr, None
        except subprocess.TimeoutExpired:
            return None, 0, "FFmpeg decoding timed out after 15 seconds."
        except Exception as e:
            return None, 0, f"FFmpeg execution error: {e}"


def load_audio_file(
    file_path: str,
    target_sr: int = 16000,
) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    """Load an audio file and resample to target sample rate.

    Supports: WAV, FLAC, MP3, M4A.
    Standardizes output to 16 kHz mono float32 in [-1.0, 1.0].

    Args:
        file_path: Path to the audio file.
        target_sr: Desired output sample rate (default: 16000 Hz).

    Returns:
        Tuple of (waveform, sample_rate, error_message).
        If loading fails, waveform is None and error_message describes the issue.
    """
    path = Path(file_path)
    if not path.exists():
        return None, 0, f"File not found: {file_path}"

    ext = path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        return None, 0, f"Unsupported audio format: '{ext}'. Supported formats: WAV, FLAC, MP3, M4A."

    if path.stat().st_size == 0:
        return None, 0, f"Audio file is empty (0 bytes): {file_path}"

    waveform = None
    original_sr = 0
    error = None

    # Strategy 1: scipy.io.wavfile (native for .wav)
    if ext == ".wav":
        try:
            from scipy.io import wavfile

            original_sr, data = wavfile.read(str(path))
            if data.dtype == np.int16:
                waveform = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                waveform = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.float32:
                waveform = data
            elif data.dtype == np.uint8:
                waveform = (data.astype(np.float32) - 128.0) / 128.0
            else:
                waveform = data.astype(np.float32)
        except Exception as e:
            error = f"scipy.io.wavfile error: {e}"

    # Strategy 2: soundfile (handles .wav, .flac, .mp3 natively without FFmpeg)
    if waveform is None and ext in {".wav", ".flac", ".mp3", ".ogg"}:
        try:
            import soundfile as sf
            data, original_sr = sf.read(str(path), dtype="float32")
            waveform = data
            error = None
        except Exception as e:
            error = f"soundfile error: {e}"

    # Strategy 3: FFmpeg decoder (universal fallback for .mp3, .m4a, .flac, and fallback for .wav)
    if waveform is None:
        dec_wave, dec_sr, dec_err = _decode_with_ffmpeg(path, target_sr)
        if dec_wave is not None:
            return dec_wave, target_sr, None
        elif ext in {".mp3", ".m4a"}:
            decoders = check_audio_decoders()
            if ext == ".mp3" and not decoders["mp3"]:
                return None, 0, "MP3 decoding requires 'soundfile' with MP3 support or 'ffmpeg' on system PATH. Please provide audio in WAV format or install FFmpeg."
            elif ext == ".m4a" and not decoders["m4a"]:
                return None, 0, "M4A decoding requires 'ffmpeg' with AAC support, which was not found on system PATH. Please provide audio in WAV, FLAC, or MP3 format."
            return None, 0, f"{ext.upper().replace('.', '')} decoding failed: {dec_err or error}"
        elif ext == ".flac":
            decoders = check_audio_decoders()
            if not decoders["flac"]:
                return None, 0, "FLAC decoding requires 'soundfile' or 'ffmpeg', neither of which is available in the current environment. Please install FFmpeg or soundfile, or upload a WAV file."

    # Strategy 4: librosa fallback (if available)
    if waveform is None:
        try:
            import importlib
            import importlib.util

            if importlib.util.find_spec("librosa") is not None:
                librosa = importlib.import_module("librosa")
                waveform, original_sr = librosa.load(str(path), sr=target_sr, mono=False)
                if waveform.ndim == 1:
                    return waveform, target_sr, None
                waveform = np.mean(waveform, axis=0)
                return waveform, target_sr, None
        except Exception as e:
            error = f"librosa error: {e}"

    if waveform is None:
        return None, 0, error or f"Audio decoding failed for {ext.upper()} file: unsupported format or corrupt data."

    # Convert to mono if multi-channel
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1) if waveform.shape[-1] <= waveform.shape[0] else np.mean(waveform, axis=0)

    # Resample if needed
    if original_sr != target_sr and original_sr > 0:
        waveform = _resample(waveform, original_sr, target_sr)

    return waveform, target_sr, None


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return waveform

    # Compute the number of target samples
    duration = len(waveform) / orig_sr
    target_length = int(round(duration * target_sr))

    try:
        from scipy.signal import resample as scipy_resample
        return scipy_resample(waveform, target_length).astype(np.float32)
    except ImportError:
        pass

    # Linear interpolation fallback
    indices = np.linspace(0, len(waveform) - 1, target_length)
    return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels.

    Args:
        waveform: Audio array, shape (samples,) or (channels, samples)
                  or (samples, channels).

    Returns:
        Mono waveform of shape (samples,).
    """
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        # Heuristic: the longer axis is samples
        if waveform.shape[0] > waveform.shape[1]:
            return np.mean(waveform, axis=1).astype(np.float32)
        else:
            return np.mean(waveform, axis=0).astype(np.float32)
    raise ValueError(f"Unexpected waveform shape: {waveform.shape}")


def trim_silence(
    waveform: np.ndarray,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim leading and trailing silence from a waveform.

    Uses energy-based detection: frames with RMS energy below
    `top_db` dB relative to peak are considered silent.

    Args:
        waveform: 1D audio array.
        top_db: Threshold in dB below peak to consider as silence.
        frame_length: Analysis frame length in samples.
        hop_length: Hop between frames in samples.

    Returns:
        Trimmed waveform. Returns original if all frames are below threshold
        (prevents returning empty arrays).
    """
    if len(waveform) == 0:
        return waveform

    # Try librosa first if installed
    try:
        import importlib
        import importlib.util

        if importlib.util.find_spec("librosa") is not None:
            librosa = importlib.import_module("librosa")
            trimmed, _ = librosa.effects.trim(
                waveform, top_db=top_db, frame_length=frame_length, hop_length=hop_length
            )
            if len(trimmed) > 0:
                return trimmed
            return waveform
    except Exception:
        pass

    # Manual energy-based trimming fallback
    if np.max(np.abs(waveform)) < 1e-10:
        return waveform

    # Compute frame energies
    n_frames = 1 + (len(waveform) - frame_length) // hop_length
    if n_frames <= 0:
        return waveform

    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start : start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        energies[i] = rms

    # Convert to dB relative to peak
    peak_rms = np.max(energies)
    if peak_rms < 1e-10:
        return waveform

    energies_db = 20 * np.log10(energies / peak_rms + 1e-10)

    # Find first and last frames above threshold
    above = np.where(energies_db > -top_db)[0]
    if len(above) == 0:
        return waveform

    start_frame = above[0]
    end_frame = above[-1]

    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length + frame_length, len(waveform))

    return waveform[start_sample:end_sample]


def normalize_duration(
    waveform: np.ndarray,
    target_samples: int = 64000,
    pad_mode: str = "repeat",
) -> np.ndarray:
    """Normalize waveform to exactly target_samples length.

    Policy (deterministic, no randomness):
        - Short waveforms: repeat-pad (tile cyclically then truncate)
          or zero-pad depending on pad_mode.
        - Long waveforms: center-crop (take middle segment).
        - Exact length: pass through unchanged.

    Args:
        waveform: 1D audio array.
        target_samples: Desired number of samples (default: 64000 = 4.0s at 16kHz).
        pad_mode: 'repeat' for cyclic tiling, 'zero' for zero-padding.

    Returns:
        Waveform of exactly target_samples length.
    """
    current_length = len(waveform)

    if current_length == target_samples:
        return waveform

    if current_length > target_samples:
        # Center-crop: take the middle segment
        start = (current_length - target_samples) // 2
        return waveform[start : start + target_samples]

    # Pad: waveform is shorter than target
    if pad_mode == "repeat":
        # Tile cyclically
        if current_length == 0:
            return np.zeros(target_samples, dtype=np.float32)
        repeats = (target_samples // current_length) + 1
        tiled = np.tile(waveform, repeats)
        return tiled[:target_samples]
    elif pad_mode == "zero":
        # Center the waveform, zero-pad both sides
        padded = np.zeros(target_samples, dtype=np.float32)
        start = (target_samples - current_length) // 2
        padded[start : start + current_length] = waveform
        return padded
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}. Use 'repeat' or 'zero'.")


def peak_normalize(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize waveform to [-1.0, 1.0] range.

    Args:
        waveform: 1D audio array.

    Returns:
        Peak-normalized waveform. Returns zeros if input is all-zero.
    """
    peak = np.max(np.abs(waveform))
    if peak < 1e-10:
        return waveform
    return (waveform / peak).astype(np.float32)


def preprocess_audio(
    waveform: np.ndarray,
    sample_rate: int,
    config: Optional[dict] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the full VoxGuard preprocessing pipeline to a waveform.

    Pipeline steps (in order):
        1. Mono conversion
        2. Resample to target sample rate
        3. Silence trimming (if enabled)
        4. Duration normalization (crop/pad)
        5. Peak amplitude normalization (if enabled)

    Args:
        waveform: Raw audio array.
        sample_rate: Original sample rate of the input.
        config: Audio configuration dictionary from default.yaml.

    Returns:
        Tuple of (preprocessed_waveform, metadata_dict).
    """
    if config is None:
        config = {}
    audio_cfg = config if "sample_rate" in config else config.get("audio", config)
    target_sr = audio_cfg.get("sample_rate", 16000)
    target_samples = audio_cfg.get("target_samples", 64000)
    pad_mode = audio_cfg.get("pad_mode", "repeat")
    do_trim = audio_cfg.get("trim_silence", True)
    silence_db = audio_cfg.get("silence_top_db", 30)
    do_normalize = audio_cfg.get("normalize_loudness", True)
    do_norm_duration = audio_cfg.get("normalize_duration", True)

    meta = {
        "original_sr": sample_rate,
        "original_length": len(waveform),
    }

    # 1. Mono
    waveform = to_mono(waveform)

    # 2. Resample
    if sample_rate != target_sr:
        waveform = _resample(waveform, sample_rate, target_sr)

    meta["length_after_resample"] = len(waveform)

    # 3. Silence trimming
    if do_trim:
        waveform = trim_silence(waveform, top_db=silence_db)
        meta["length_after_trim"] = len(waveform)

    # 4. Duration normalization (crop/pad)
    if do_norm_duration and target_samples is not None:
        waveform = normalize_duration(waveform, target_samples, pad_mode)
        meta["length_final"] = len(waveform)
        meta["was_padded"] = meta.get("length_after_trim", meta["length_after_resample"]) < target_samples
        meta["was_cropped"] = meta.get("length_after_trim", meta["length_after_resample"]) > target_samples
    else:
        meta["length_final"] = len(waveform)
        meta["was_padded"] = False
        meta["was_cropped"] = False

    # 5. Peak normalization
    if do_normalize:
        waveform = peak_normalize(waveform)

    return waveform, meta


def validate_audio_file(
    file_path: str,
    target_sr: int = 16000,
) -> Dict[str, object]:
    """Validate an audio file for integrity, duration, and format.

    Args:
        file_path: Path to the audio file.
        target_sr: Expected sample rate.

    Returns:
        Validation report dictionary.
    """
    report: Dict[str, object] = {
        "file_path": file_path,
        "exists": False,
        "readable": False,
        "sample_rate": None,
        "channels": None,
        "duration_sec": None,
        "samples": None,
        "is_corrupt": True,
        "is_silent": False,
        "error": None,
    }

    path = Path(file_path)
    if not path.exists():
        report["error"] = "File not found"
        return report

    report["exists"] = True

    waveform, sr, error = load_audio_file(str(path), target_sr)
    if waveform is None:
        report["error"] = error
        return report

    report["readable"] = True
    report["sample_rate"] = sr
    report["samples"] = len(waveform)
    report["duration_sec"] = len(waveform) / sr if sr > 0 else 0
    report["channels"] = 1  # Always converted to mono
    report["is_corrupt"] = False

    # Check for all-silent audio
    if np.max(np.abs(waveform)) < 1e-8:
        report["is_silent"] = True

    return report

