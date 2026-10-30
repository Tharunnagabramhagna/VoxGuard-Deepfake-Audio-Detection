"""VoxGuard Interactive Deepfake Audio Detection Demonstration.

Streamlit application providing a clean, modern, minimal dark-themed interface
for deepfake audio detection verification and software harness demonstration.

Key Features:
1. Compact Navbar with logo, version, and status badge.
2. Drag-and-drop audio uploader supporting WAV, FLAC, MP3, and M4A with 15 MB limit.
3. Audio Preview Player preserving original file playback.
4. Simplified Three-Part Primary Result Area:
   - A. Classification Result: "AI-Generated Voice", "Human Voice", or "Unable to Determine / Model Not Validated"
   - B. Confidence: Genuine model confidence when available (never labeled as accuracy; not fabricated)
   - C. Simple Explanation: Evidence-based summary grounded in measurable acoustic properties
5. Scientific Safety Gate: Real-world speech detection remains strictly gated
   until models trained and benchmarked on genuine human speech corpora (e.g. ASVspoof 2019 LA) exist.
6. Expandable Technical Section: Waveform, Log-Mel Spectrogram, and signal preprocessing metrics.

Usage:
    streamlit run demo/app.py
"""

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline_models import BaselinePredictor
from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.models.model_loader import load_fixture_models, load_real_speech_model
from src.data.preprocessor import (
    SUPPORTED_AUDIO_EXTENSIONS,
    check_audio_decoders,
    load_audio_file,
    preprocess_audio,
)
from src.features.mel_spectrogram import extract_log_mel_spectrogram
from src.analysis.explanation import (
    compute_measurable_acoustic_features,
    generate_result_summary,
)
from src.utils.config import load_config


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Dark Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VoxGuard — Audio Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
/* Modern Dark Theme Palette */
:root {
    --bg-dark: #0d1117;
    --card-bg: #161b22;
    --border-color: #30363d;
    --accent-blue: #58a6ff;
    --accent-green: #238636;
    --accent-red: #da3633;
    --accent-amber: #d29922;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
}

/* Base resets & typography */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text-primary);
}

/* Navbar Header */
.navbar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.25rem;
    background-color: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    margin-bottom: 1.5rem;
}

.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}

.badge-tag {
    font-size: 0.72rem;
    padding: 0.2rem 0.55rem;
    background-color: rgba(88, 166, 255, 0.15);
    color: var(--accent-blue);
    border: 1px solid rgba(88, 166, 255, 0.4);
    border-radius: 12px;
    font-weight: 600;
}

/* Card Containers */
.vg-card {
    background-color: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.15rem;
    margin-bottom: 1rem;
}

.vg-card-title {
    font-size: 0.92rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.65rem;
}

/* Metric Pill Badges */
.status-pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}
.pill-active { background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636; }
.pill-warning { background: rgba(210, 153, 34, 0.2); color: #e3b341; border: 1px solid #9e6a03; }
.pill-danger { background: rgba(218, 54, 51, 0.2); color: #f85149; border: 1px solid #da3633; }
.pill-info { background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid #1f6feb; }

/* File Preview Meta Table */
.preview-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem 1rem;
    font-size: 0.85rem;
    margin: 0.75rem 0;
}
.preview-label { color: var(--text-secondary); font-weight: 500; }
.preview-value { color: var(--text-primary); font-family: monospace; font-weight: 600; }

/* Footer */
.vg-footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-secondary);
    border-top: 1px solid var(--border-color);
    padding-top: 1rem;
    margin-top: 2rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_audio_mime_type(filename: str) -> str:
    """Return the MIME type for browser audio playback."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
    }
    return mime_map.get(ext, "audio/wav")


# -----------------------------------------------------------------------------
# Model Checkpoint Management & Architecture Isolation
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models() -> Dict[str, Dict]:
    """Load model checkpoints designated for the software-test harness.

    Enforces that loaded models originate from checkpoints/fixture/ (or legacy checkpoints/)
    and carry verified synthetic_fixture provenance metadata.
    """
    return load_fixture_models()


@st.cache_resource
def get_real_speech_detector() -> Dict[str, Any]:
    """Inspect and load validated real-speech detector from checkpoints/real_speech/.

    Returns status dictionary. Never silently falls back to fixture checkpoints.
    """
    return load_real_speech_model()


# -----------------------------------------------------------------------------
# Visualization Utilities
# -----------------------------------------------------------------------------
def plot_analysis(waveform: np.ndarray, mel_spec: Optional[np.ndarray], sr: int = 16000) -> plt.Figure:
    """Generate high-contrast waveform and Log-Mel spectrogram analysis plot."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), dpi=140)
    fig.patch.set_facecolor("#161b22")

    # 1. Normalized Waveform
    axes[0].set_facecolor("#0d1117")
    time_axis = np.linspace(0, len(waveform) / sr, len(waveform))
    axes[0].plot(time_axis, waveform, color="#58a6ff", lw=0.75)
    axes[0].set_title("Normalized Audio Waveform (4.0s)", fontsize=9, fontweight="bold", color="#e6edf3")
    axes[0].set_xlabel("Time (s)", fontsize=8, color="#8b949e")
    axes[0].set_ylabel("Amplitude", fontsize=8, color="#8b949e")
    axes[0].set_ylim([-1.05, 1.05])
    axes[0].tick_params(colors="#8b949e", labelsize=7)
    axes[0].grid(True, linestyle=":", alpha=0.3, color="#30363d")

    # 2. Log-Mel Spectrogram
    axes[1].set_facecolor("#0d1117")
    if mel_spec is not None:
        im = axes[1].imshow(
            mel_spec,
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[1].set_title(f"{mel_spec.shape[0]}-bin Log-Mel Spectrogram", fontsize=9, fontweight="bold", color="#e6edf3")
        axes[1].set_xlabel("Time Frames", fontsize=8, color="#8b949e")
        axes[1].set_ylabel("Mel Bins (0–8 kHz)", fontsize=8, color="#8b949e")
        axes[1].tick_params(colors="#8b949e", labelsize=7)
        cbar = fig.colorbar(im, ax=axes[1], format="%+2.0f dB", pad=0.03)
        cbar.ax.tick_params(colors="#8b949e", labelsize=6)
    else:
        axes[1].text(0.5, 0.5, "MFCC Summary Features Active", ha="center", va="center", color="#8b949e", fontsize=8)
        axes[1].axis("off")

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Application Interface
# -----------------------------------------------------------------------------
def main():
    # 1. Compact Navbar
    st.markdown(
        """
        <div class="navbar-header">
            <div class="navbar-brand">
                <span>🛡️ VoxGuard</span>
                <span class="badge-tag">v0.4</span>
                <span class="badge-tag">Forensic Verification</span>
            </div>
            <div style="font-size: 0.8rem; color: #8b949e;">
                Deepfake Audio Detection & Forensic Verification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = load_models()
    if not models:
        st.error(
            "⚠️ **Model Checkpoints Not Found!**\n\n"
            "Trained model artifacts were not found in `checkpoints/`. "
            "Please run: `python scripts/train_models.py --model all --mode quick --subset mini`"
        )
        st.stop()

    # Layout: Balanced 2 Columns
    col_upload, col_output = st.columns([1, 1], gap="medium")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Audio Input & Preview
    # -------------------------------------------------------------------------
    with col_upload:
        st.markdown('<div class="vg-card-title">1. Audio Input Selection</div>', unsafe_allow_html=True)

        # Check query params for deep-linking
        default_mode_idx = 0
        try:
            if st.query_params.get("mode") == "fixture":
                default_mode_idx = 1
        except Exception:
            pass

        mode_choice = st.radio(
            "Select Audio Source:",
            ["Upload Audio File (WAV, FLAC, MP3, M4A)", "Load Software-Test Fixture"],
            index=default_mode_idx,
            horizontal=True,
            label_visibility="collapsed",
        )

        audio_bytes = None
        audio_name = None
        is_fixture = False
        fixture_label = None

        if mode_choice == "Upload Audio File (WAV, FLAC, MP3, M4A)":
            uploaded_file = st.file_uploader(
                "Upload Audio Recording",
                type=["wav", "flac", "mp3", "m4a"],
                help="Supported formats: WAV, FLAC, MP3, M4A. Maximum file size: 15 MB.",
            )
            st.caption("Accepted: **WAV, FLAC, MP3, M4A** | Max size: **15 MB** | Standardized to: **16 kHz Mono**")

            if uploaded_file is not None:
                audio_name = uploaded_file.name
                ext = Path(audio_name).suffix.lower()

                # Guard against unsupported file extensions
                if ext not in SUPPORTED_AUDIO_EXTENSIONS:
                    st.error(
                        f"❌ **Unsupported File Format:** `{ext}`\n\n"
                        "VoxGuard only supports: WAV, FLAC, MP3, and M4A."
                    )
                    st.stop()

                if uploaded_file.size > 15 * 1024 * 1024:
                    st.error("❌ File size exceeds 15 MB limit. Please provide a shorter audio clip.")
                    st.stop()
                if uploaded_file.size == 0:
                    st.error("❌ Uploaded audio file is empty (0 bytes).")
                    st.stop()

                audio_bytes = uploaded_file.read()

        else:
            is_fixture = True
            fixture_options = {
                "Bonafide Fixture (Dev Partition)": ("tests/fixtures/audio/dev/LA_D_0000031.wav", 0),
                "Spoof Fixture A01 (Dev Partition)": ("tests/fixtures/audio/dev/LA_D_0000041.wav", 1),
                "Spoof Fixture A02 (Dev Partition)": ("tests/fixtures/audio/dev/LA_D_0000042.wav", 1),
                "Edge Case: Silent Audio": ("tests/fixtures/audio/silent_file.wav", None),
                "Edge Case: Corrupt Header": ("tests/fixtures/audio/corrupt_file.wav", None),
            }
            fixture_keys = list(fixture_options.keys())
            default_fixture_idx = 0
            try:
                if st.query_params.get("fixture") == "spoof":
                    default_fixture_idx = 1
            except Exception:
                pass

            selected_fixture = st.selectbox(
                "Choose procedural test fixture:",
                fixture_keys,
                index=default_fixture_idx,
            )
            rel_path, fixture_label = fixture_options[selected_fixture]
            fixture_path = Path(rel_path)

            if fixture_path.exists():
                with open(fixture_path, "rb") as f:
                    audio_bytes = f.read()
                audio_name = fixture_path.name
                st.caption(f"📁 Loaded: `{fixture_path}`")
            else:
                st.warning(f"⚠️ Test fixture not found at `{fixture_path}`. Generate fixtures via `python scripts/download_data.py --subset fixtures`.")

        # Audio Preview Card
        if audio_bytes is not None:
            st.markdown('<div class="vg-card-title">Audio Preview</div>', unsafe_allow_html=True)
            
            # Preserve actual file extension for accurate decoding
            file_ext = Path(audio_name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = Path(tmp_file.name)

            raw_wave, sr, decode_err = load_audio_file(str(tmp_path), target_sr=16000)

            # File metadata
            size_kb = len(audio_bytes) / 1024
            dur_str = f"{len(raw_wave)/sr:.2f}s" if raw_wave is not None else "N/A"
            ext_str = file_ext.upper().replace(".", "")

            st.markdown(
                f"""
                <div class="vg-card" style="padding: 0.85rem;">
                    <div class="preview-grid">
                        <div><span class="preview-label">File:</span> <span class="preview-value">{audio_name}</span></div>
                        <div><span class="preview-label">Size:</span> <span class="preview-value">{size_kb:.1f} KB</span></div>
                        <div><span class="preview-label">Format:</span> <span class="preview-value">{ext_str}</span></div>
                        <div><span class="preview-label">Duration:</span> <span class="preview-value">{dur_str}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Audio Player preserving original format
            playback_mime = get_audio_mime_type(audio_name)
            st.audio(audio_bytes, format=playback_mime)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Simplified Primary Result & Forensic Verification
    # -------------------------------------------------------------------------
    with col_output:
        st.markdown('<div class="vg-card-title">2. Analysis & Verdict</div>', unsafe_allow_html=True)

        if audio_bytes is None:
            st.info("👈 Select or upload an audio recording on the left to analyze.")
            st.stop()

        # Check decoding
        if raw_wave is None:
            st.error(
                f"❌ **Audio Decoding Failed**\n\n"
                f"{decode_err}\n\n"
                "Please verify that the audio file is not corrupt and is encoded in a supported format."
            )
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            st.stop()

        # Silence validation
        peak_amp = float(np.max(np.abs(raw_wave)))
        if peak_amp < 1e-5:
            st.warning(
                "⚠️ **Silent Audio Signal Detected**\n\n"
                "The audio file contains near-zero energy (amplitude < 1e-5). "
                "Forensic classification cannot run on pure silence."
            )
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            st.stop()

        # Duration validation
        duration_sec = len(raw_wave) / sr
        if duration_sec < 0.2:
            st.warning(
                "⚠️ **Audio Duration Too Short**\n\n"
                f"Recorded duration is {duration_sec:.2f}s (< 0.2s minimum required). "
                "Feature extraction requires sufficient acoustic speech frames."
            )
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            st.stop()

        # Standardized Preprocessing
        cfg = load_config("configs/default.yaml")

        if is_fixture:
            clean_wave, prep_meta = preprocess_audio(raw_wave, sr, cfg)
        else:
            # For user audio upload: preserve full waveform duration for multi-window deep neural analysis
            user_cfg = dict(cfg)
            user_cfg["normalize_duration"] = False
            user_cfg["target_samples"] = None
            clean_wave, prep_meta = preprocess_audio(raw_wave, sr, user_cfg)

        # Model Selection & Inference Execution
        p_spoof = None
        is_real_speech_model = False
        classifier_name = "Spectrogram-CNN (2D CNN)"

        if is_fixture:
            # Mode A: Procedural Software-Test Fixture Harness
            if classifier_name not in models:
                classifier_name = list(models.keys())[0]
            selected_info = models[classifier_name]
            predictor = selected_info["predictor"]
            calibrated_threshold = selected_info["threshold"]

            res = predictor.predict_audio(clean_wave, sample_rate=16000)
            p_spoof = res["probability_spoof"]

            summary = generate_result_summary(
                is_fixture=True,
                clean_wave=clean_wave,
                sample_rate=16000,
                model_name=classifier_name,
                raw_score=p_spoof,
                threshold=calibrated_threshold,
                is_real_speech_model=False,
            )
        else:
            # Mode B: User Audio Upload (Strict Isolation from Fixtures)
            real_speech_info = get_real_speech_detector()
            if real_speech_info["available"]:
                # Genuine real-speech model available and validated
                real_predictor = real_speech_info["predictor"]
                real_threshold = real_speech_info["threshold"]
                res = real_predictor.predict_audio(clean_wave, sample_rate=16000)
                p_spoof = res["probability_spoof"]
                is_real_speech_model = True

                summary = generate_result_summary(
                    is_fixture=False,
                    clean_wave=clean_wave,
                    sample_rate=16000,
                    model_name=real_speech_info.get("model_name", "ResNet-BiGRU Detector (MLAAD Fine-Tuned)"),
                    raw_score=p_spoof,
                    threshold=real_threshold,
                    is_real_speech_model=True,
                    is_mimicry_detected=res.get("is_mimicry_detected", False),
                    harmonicity=res.get("harmonicity", None),
                )
            else:
                # Real-speech model not deployed or invalid provenance -> strictly gated (no fallback!)
                summary = generate_result_summary(
                    is_fixture=False,
                    clean_wave=clean_wave,
                    sample_rate=16000,
                    model_name=None,
                    raw_score=None,
                    threshold=None,
                    is_real_speech_model=False,
                )
                summary["validation_status"] = "Real-Speech Model Not Deployed"
                summary["explanation"] = (
                    "Real-speech detection is not yet validated. "
                    + real_speech_info["message"]
                    + f" Diagnostic observations: Preprocessed audio has an RMS energy of {summary['measurable_features']['rms_energy']:.3f} "
                    f"and a spectral centroid of {summary['measurable_features']['spectral_centroid_hz']:.0f} Hz (observations are for signal inspection only)."
                )

        # ---------------------------------------------------------------------
        # PRIMARY RESULT SECTION (Classification, Confidence, Explanation)
        # ---------------------------------------------------------------------
        res_class = summary["classification_result"]
        status_badge = summary["validation_status"]

        if "Unable to Determine" in res_class:
            border_color = "var(--accent-amber)"
            text_color = "var(--accent-amber)"
            pill_class = "pill-warning"
            icon = "⚠️"
        elif res_class == "Synthetic Test Fixture" or "AI-Generated" in res_class or "Spoof" in res_class:
            border_color = "var(--accent-red)"
            text_color = "var(--accent-red)"
            pill_class = "pill-danger"
            icon = "🚨"
        elif res_class == "Bonafide Test Fixture" or "Human Speech" in res_class or "Human Voice" in res_class or "Mimicry" in res_class:
            border_color = "var(--accent-green)"
            text_color = "var(--accent-green)"
            pill_class = "pill-active"
            icon = "✅"
        else:
            border_color = "var(--accent-blue)"
            text_color = "var(--accent-blue)"
            pill_class = "pill-info"
            icon = "ℹ️"

        # A. Classification Result Card
        st.markdown(
            f"""
            <div class="vg-card" style="border-left: 4px solid {border_color}; margin-bottom: 0.85rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); font-weight: 600;">
                        Classification Result
                    </span>
                    <span class="status-pill {pill_class}">{status_badge}</span>
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; color: {text_color};">
                    {icon} {res_class}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # B. Confidence Card
        conf_lbl = summary["confidence_label"]
        conf_disclaimer = summary["confidence_disclaimer"]
        st.markdown(
            f"""
            <div class="vg-card" style="margin-bottom: 0.85rem;">
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.35rem;">
                    Confidence
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary); font-family: monospace;">
                    {conf_lbl}
                </div>
                <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem; line-height: 1.4;">
                    {conf_disclaimer}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # C. Simple Evidence-Based Explanation Card
        expl_text = summary["explanation"]
        st.markdown(
            f"""
            <div class="vg-card" style="margin-bottom: 0.85rem;">
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.35rem;">
                    Forensic Explanation
                </div>
                <div style="font-size: 0.88rem; line-height: 1.55; color: var(--text-primary);">
                    {expl_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------------------
        # EXPANDABLE TECHNICAL SECTION (Waveform, Mel-Spec, Diagnostics)
        # ---------------------------------------------------------------------
        with st.expander("🔬 Technical Signal Diagnostics & Forensic Visualizations", expanded=False):
            # Model selection inside expander for advanced technical inspection
            adv_classifier = st.selectbox(
                "Classifier Architecture for Diagnostics:",
                list(models.keys()),
                index=0,
                help="Inspect acoustic representation and threshold calibration for selected model.",
            )
            adv_info = models[adv_classifier]

            # Acoustic Visualizations
            mel_spec = None
            try:
                n_mels_to_extract = 128 if is_real_speech_model else 64
                mel_tensor = extract_log_mel_spectrogram(clean_wave, sample_rate=16000, n_mels=n_mels_to_extract)
                mel_spec = mel_tensor.squeeze().cpu().numpy()
                fig = plot_analysis(clean_wave, mel_spec, sr=16000)
                st.pyplot(fig)
            except Exception as viz_err:
                st.warning(f"⚠️ Acoustic visualization unavailable: {str(viz_err)}")

            # Measurable signal diagnostics table
            feats = summary["measurable_features"]
            disp_threshold = adv_info.get("threshold", 0.5) if is_fixture else (res.get("threshold", 0.58) if is_real_speech_model else 0.5)
            harmonicity_val = res.get("harmonicity") if is_real_speech_model else None
            harmonicity_display = f"{harmonicity_val:.3f}" if isinstance(harmonicity_val, (int, float)) else "N/A"
            window_cnt = res.get("window_count", 1) if is_real_speech_model else 1
            duration_s = len(clean_wave) / 16000
            window_display = f"{duration_s:.2f}s ({len(clean_wave):,} samples) | {window_cnt} analysis window{'s' if window_cnt > 1 else ''}" if not is_fixture else "Normalized to 4.0s (64,000 samples)"
            
            if is_fixture:
                pill_type = "pill-info"
                status_txt = "SOFTWARE-TEST HARNESS DEMO"
            elif is_real_speech_model:
                if res.get("is_mimicry_detected"):
                    pill_type = "pill-active"
                    status_txt = "HUMAN VOCAL MIMICRY VERIFIED"
                else:
                    pill_type = "pill-active"
                    status_txt = "EXPERIMENTAL MLAAD REAL-SPEECH MODEL"
            else:
                pill_type = "pill-warning"
                status_txt = "GATED / UNVALIDATED"

            st.markdown(
                f"""
                <div class="vg-card" style="margin-top: 0.5rem; padding: 0.85rem;">
                    <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 0.4rem; color: var(--accent-blue);">
                        Signal Preprocessing Diagnostics (Verified Pipeline)
                    </div>
                    <div style="font-size: 0.82rem; line-height: 1.6; color: var(--text-primary);">
                        • <b>Decoding:</b> Decoded PCM 16 kHz Mono ({ext_str})<br>
                        • <b>Windowing:</b> {window_display} | FFT 1024, Hop 160<br>
                        • <b>Signal Energy (RMS):</b> {feats['rms_energy']:.4f} (Peak Amplitude: {feats['peak_amplitude']:.4f})<br>
                        • <b>Spectral Centroid:</b> {feats['spectral_centroid_hz']:.1f} Hz (Rolloff: {feats['spectral_rolloff_hz']:.1f} Hz)<br>
                        • <b>Biological Vocal Harmonicity:</b> {harmonicity_display} (Pitch lag correlation R<sub>max</sub>)<br>
                        • <b>Model Operating Threshold:</b> &theta; = {disp_threshold:.4f}<br>
                        • <b>Inference Status:</b> <span class="status-pill {pill_type}">{status_txt}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Cleanup temp file
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Footer & Limitations
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="vg-footer">
            VoxGuard Research Prototype &bull; Scientific Test Harness Mode &bull; 
            Deterministic 16 kHz Preprocessing &bull; Real-Speech Models Gated Until Validated
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
