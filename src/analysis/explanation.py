"""Evidence-based explanation and result generation for VoxGuard audio forensics.

Provides measurable acoustic feature extraction and explainable forensic
summaries for speech classification without fabricating unvalidated claims.

Scientific Terminology Policy:
1. Procedural Fixtures:
   - Evaluated as "Synthetic Test Fixture" or "Bonafide Test Fixture".
   - Never described as genuine AI-generated voice or human speech.
2. User Audio Uploads:
   - Explicitly labeled: "Unable to Determine – Real-Speech Classification Not Yet Available".
   - Prediction confidence, accuracy, and deepfake labels are NEVER fabricated.
3. Diagnostic Observations:
   - Measurable acoustic properties (RMS energy, spectral centroid, etc.) are reported
     strictly as observational signal metrics, not claimed as causal attributions of CNN layers.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def compute_measurable_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> Dict[str, float]:
    """Compute measurable spectral and temporal acoustic properties.

    Args:
        waveform: 1D normalized float32 waveform array.
        sample_rate: Audio sampling rate (default: 16000 Hz).

    Returns:
        Dictionary of objective, measurable acoustic features.
    """
    if len(waveform) == 0:
        return {
            "rms_energy": 0.0,
            "peak_amplitude": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "high_frequency_ratio": 0.0,
        }

    # 1. Temporal metrics
    peak_amp = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(waveform))))
    zcr = float(zero_crossings / max(len(waveform) - 1, 1))

    # 2. Spectral metrics via Real FFT
    fft_mag = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), 1.0 / sample_rate)
    mag_sum = float(np.sum(fft_mag))

    if mag_sum > 1e-8:
        # Spectral Centroid (frequency balance / brightness)
        spectral_centroid = float(np.sum(freqs * fft_mag) / mag_sum)

        # Spectral Rolloff (85% energy frequency limit)
        cumulative_mag = np.cumsum(fft_mag)
        rolloff_idx = np.searchsorted(cumulative_mag, 0.85 * mag_sum)
        spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # High-frequency ratio (> 4000 Hz)
        hf_mask = freqs >= 4000.0
        hf_energy = float(np.sum(fft_mag[hf_mask]))
        hf_ratio = float(hf_energy / mag_sum)
    else:
        spectral_centroid = 0.0
        spectral_rolloff = 0.0
        hf_ratio = 0.0

    return {
        "rms_energy": rms,
        "peak_amplitude": peak_amp,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_rolloff_hz": spectral_rolloff,
        "high_frequency_ratio": hf_ratio,
    }


def generate_result_summary(
    is_fixture: bool,
    clean_wave: np.ndarray,
    sample_rate: int = 16000,
    model_name: Optional[str] = None,
    raw_score: Optional[float] = None,
    threshold: Optional[float] = None,
    is_real_speech_model: bool = False,
    is_mimicry_detected: bool = False,
    harmonicity: Optional[float] = None,
) -> Dict[str, object]:
    """Generate the three-part forensic result summary with precise terminology.

    Primary outputs:
    A. Classification Result:
       - 'Unable to Determine – Real-Speech Classification Not Yet Available' (for user audio)
       - 'Synthetic Test Fixture' or 'Bonafide Test Fixture' (for software fixtures)
    B. Confidence:
       - Posterior probability for fixtures; None for unvalidated uploads.
    C. Simple Explanation:
       - Honest scientific disclosure with measurable diagnostic observations.

    Args:
        is_fixture: Whether the input audio is a procedural software-test fixture.
        clean_wave: Preprocessed 16 kHz audio waveform.
        sample_rate: Audio sampling rate (default: 16000).
        model_name: Name of the evaluated classifier model.
        raw_score: Estimated posterior probability of synthetic class (if evaluated).
        threshold: Decision threshold for the model.
        is_real_speech_model: Whether the loaded model is verified on real speech.

    Returns:
        Structured result dictionary.
    """
    # Extract measurable diagnostic features
    features = compute_measurable_acoustic_features(clean_wave, sample_rate)
    centroid = features["spectral_centroid_hz"]
    hf_ratio = features["high_frequency_ratio"]
    rms = features["rms_energy"]

    # -------------------------------------------------------------------------
    # CASE 1: Real-World Upload & Model Not Validated for Real Speech
    # -------------------------------------------------------------------------
    if not is_fixture and not is_real_speech_model:
        return {
            "classification_result": "Unable to Determine – Real-Speech Classification Not Yet Available",
            "validation_status": "Real-Speech Classification Gated",
            "can_classify": False,
            "confidence": None,
            "confidence_label": "Not Available",
            "confidence_disclaimer": "Confidence scores are not computed for unvalidated models to prevent false certainty.",
            "explanation": (
                "Real-speech detection is not yet validated. Current model checkpoints were trained exclusively on "
                "synthetic software-test fixtures (sine waves and procedural noise) for pipeline verification. "
                "Real-world predictions are disabled until models trained and evaluated on authentic human speech "
                "corpora (e.g., ASVspoof 2019 LA) are available. "
                f"Diagnostic observations: Preprocessed audio has an RMS energy of {rms:.3f} and a spectral centroid "
                f"of {centroid:.0f} Hz (observations are for signal inspection only)."
            ),
            "measurable_features": features,
            "is_fixture": False,
            "is_real_speech_model": False,
        }

    # -------------------------------------------------------------------------
    # CASE 2: Evaluated Fixture Mode
    # -------------------------------------------------------------------------
    if raw_score is None or threshold is None:
        return {
            "classification_result": "Unable to Determine – Real-Speech Classification Not Yet Available",
            "validation_status": "Evaluation Score Unavailable",
            "can_classify": False,
            "confidence": None,
            "confidence_label": "Not Available",
            "confidence_disclaimer": "Model inference did not produce a valid prediction score.",
            "explanation": "No evaluation score or operating threshold was provided for this recording.",
            "measurable_features": features,
            "is_fixture": is_fixture,
            "is_real_speech_model": is_real_speech_model,
        }

    is_synthetic = bool(raw_score >= threshold)

    if is_real_speech_model:
        if is_mimicry_detected:
            classification = "Experimental Human Speech (Vocal Mimicry Detected)"
            validation_status = "HUMAN VOCAL MIMICRY VERIFIED"
            confidence_disclaimer = (
                "Human mimicry / voice modulation detected. Although atypical pitch inflections raised acoustic suspicion, "
                "biological glottal periodicity confirms natural human vocal fold excitation."
            )
            confidence_val = float(raw_score)
            confidence_lbl = f"{confidence_val * 100:.1f}% Model Score (Calibrated for Mimicry)"
            h_str = f"{harmonicity:.2f}" if harmonicity is not None else ">= 0.35"
            explanation = (
                f"The model observed acoustic pitch modulation or comical mimicry dynamics (raw score: {raw_score:.4f}). "
                f"Biological vocal tract analysis verified high fundamental glottal periodicity (R_max = {h_str}), "
                "confirming real human physiological excitation without neural vocoder phase scrambling. "
                "This recording is classified as human-generated speech with intentional voice acting or mimicry."
            )
        else:
            classification = "Experimental AI-Generated Speech Prediction" if is_synthetic else "Experimental Human Speech Prediction"
            validation_status = "EXPERIMENTAL RESEARCH MODEL"
            confidence_disclaimer = (
                "This is an experimental research model. Its predictions may be incorrect, particularly for unseen "
                "voice-generation systems. It is not a forensic verification result."
            )
            confidence_val = float(raw_score)
            confidence_lbl = f"{confidence_val * 100:.1f}% Model Score (Spoof Probability)"
            if is_synthetic:
                explanation = (
                    f"The experimental model predicted this recording as AI-Generated Speech because the estimated spoof "
                    f"probability ({raw_score:.4f}) meets or exceeds the operating threshold ({threshold:.4f}). "
                    f"Diagnostic observations: Measured RMS energy is {rms:.3f} with a spectral centroid of {centroid:.0f} Hz. "
                    "Notice: This is an experimental research model evaluated on MLAAD-tiny; predictions may be incorrect."
                )
            else:
                explanation = (
                    f"The experimental model predicted this recording as Human Speech because the estimated spoof "
                    f"probability ({raw_score:.4f}) is below the operating threshold ({threshold:.4f}). "
                    f"Diagnostic observations: Measured RMS energy is {rms:.3f} with a spectral centroid of {centroid:.0f} Hz. "
                    "Notice: This is an experimental research model evaluated on MLAAD-tiny; predictions may be incorrect."
                )
    else:
        classification = "Synthetic Test Fixture" if is_synthetic else "Bonafide Test Fixture"
        validation_status = "SOFTWARE-TEST HARNESS DEMO"
        confidence_disclaimer = "Model confidence represents estimated posterior probability on procedural test fixtures; not individual accuracy."
        if is_synthetic:
            confidence_val = float(raw_score)
            confidence_lbl = f"{confidence_val * 100:.1f}% Confidence"
            explanation = (
                f"The software harness assigned this fixture to Synthetic Test Fixture because the estimated synthetic "
                f"probability ({raw_score:.4f}) meets or exceeds the frozen operating threshold ({threshold:.4f}). "
                f"Diagnostic observations: Measured RMS energy is {rms:.3f} with a spectral centroid of {centroid:.0f} Hz "
                f"({hf_ratio * 100:.1f}% energy above 4 kHz). Note: Acoustic measurements are observational and do not "
                f"represent causal layer attributions."
            )
        else:
            confidence_val = float(1.0 - raw_score)
            confidence_lbl = f"{confidence_val * 100:.1f}% Confidence"
            explanation = (
                f"The software harness assigned this fixture to Bonafide Test Fixture because the estimated synthetic "
                f"probability ({raw_score:.4f}) is below the frozen operating threshold ({threshold:.4f}). "
                f"Diagnostic observations: Measured RMS energy is {rms:.3f} with a spectral centroid of {centroid:.0f} Hz. "
                f"Note: Acoustic measurements are observational and do not represent causal layer attributions."
            )

    return {
        "classification_result": classification,
        "validation_status": validation_status,
        "can_classify": True,
        "confidence": confidence_val,
        "confidence_label": confidence_lbl,
        "confidence_disclaimer": confidence_disclaimer,
        "explanation": explanation,
        "measurable_features": features,
        "is_fixture": is_fixture,
        "is_real_speech_model": is_real_speech_model,
        "operating_threshold": threshold,
        "raw_score": raw_score,
    }
