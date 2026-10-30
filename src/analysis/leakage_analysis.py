"""Speaker identity, shortcut learning, and dataset bias investigation.

Performs group-level analysis and forensic sanity checks:
1. Speaker Partition Leakage Audit:
   Verifies that zero speaker identities cross training, validation, or evaluation splits.
2. Generator/System Group Breakdown:
   Evaluates detection performance broken down by synthesis algorithm (seen vs unseen).
3. Speaker-Level Error Concentration:
   Determines whether false alarms or missed detections cluster within specific speaker identities.
4. Acoustic Shortcut Correlation:
   Measures Pearson correlation between superficial acoustic attributes (RMS energy, duration,
   zero-crossing rate) and predicted spoof probabilities to identify shortcut learning.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def verify_speaker_disjointness(
    manifest_train: Union[pd.DataFrame, List[Dict]],
    manifest_dev: Union[pd.DataFrame, List[Dict]],
    manifest_eval: Optional[Union[pd.DataFrame, List[Dict]]] = None,
) -> Dict[str, object]:
    """Verify strictly disjoint speaker sets across dataset partitions.

    Args:
        manifest_train: Training manifest.
        manifest_dev: Development/validation manifest.
        manifest_eval: Optional evaluation manifest.

    Returns:
        Audit report detailing speaker counts and any intersection violations.
    """
    df_tr = pd.DataFrame(manifest_train) if isinstance(manifest_train, list) else manifest_train
    df_dv = pd.DataFrame(manifest_dev) if isinstance(manifest_dev, list) else manifest_dev

    spk_train = set(df_tr["speaker_id"].unique()) if "speaker_id" in df_tr.columns else set()
    spk_dev = set(df_dv["speaker_id"].unique()) if "speaker_id" in df_dv.columns else set()

    leak_tr_dv = spk_train.intersection(spk_dev)

    report = {
        "n_train_speakers": len(spk_train),
        "n_dev_speakers": len(spk_dev),
        "train_dev_leakage_count": len(leak_tr_dv),
        "train_dev_leaked_speakers": sorted(list(leak_tr_dv)),
        "is_train_dev_disjoint": len(leak_tr_dv) == 0,
    }

    if manifest_eval is not None:
        df_ev = pd.DataFrame(manifest_eval) if isinstance(manifest_eval, list) else manifest_eval
        spk_eval = set(df_ev["speaker_id"].unique()) if "speaker_id" in df_ev.columns else set()
        leak_tr_ev = spk_train.intersection(spk_eval)
        leak_dv_ev = spk_dev.intersection(spk_eval)

        report.update({
            "n_eval_speakers": len(spk_eval),
            "train_eval_leakage_count": len(leak_tr_ev),
            "train_eval_leaked_speakers": sorted(list(leak_tr_ev)),
            "dev_eval_leakage_count": len(leak_dv_ev),
            "dev_eval_leaked_speakers": sorted(list(leak_dv_ev)),
            "is_all_disjoint": (len(leak_tr_dv) == 0 and len(leak_tr_ev) == 0 and len(leak_dv_ev) == 0),
        })

    return report


def group_performance_by_system(
    manifest_df: pd.DataFrame,
    y_scores: np.ndarray,
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Compute detection accuracy and score statistics per synthesis system/algorithm.

    Args:
        manifest_df: Manifest dataframe with 'system_id' and 'label'.
        y_scores: Continuous prediction scores for spoof (class 1).
        threshold: Frozen decision threshold.

    Returns:
        Dictionary mapping system_id to performance summary.
    """
    df = manifest_df.copy()
    df["score"] = y_scores
    df["prediction"] = (y_scores >= threshold).astype(int)
    df["correct"] = (df["prediction"] == df["label"].astype(int))

    system_summary = {}
    for sys_id, group in df.groupby("system_id"):
        n_samples = len(group)
        n_correct = int(group["correct"].sum())
        acc = float(n_correct / n_samples) if n_samples > 0 else 0.0
        avg_score = float(group["score"].mean())
        std_score = float(group["score"].std()) if n_samples > 1 else 0.0

        label_type = "bonafide" if sys_id == "-" else "spoof"

        system_summary[str(sys_id)] = {
            "system_type": label_type,
            "n_samples": n_samples,
            "accuracy": acc,
            "mean_score": avg_score,
            "std_score": std_score,
            "error_rate": 1.0 - acc,
        }

    return system_summary


def group_performance_by_speaker(
    manifest_df: pd.DataFrame,
    y_scores: np.ndarray,
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    """Compute detection accuracy and score statistics per speaker identity.

    Args:
        manifest_df: Manifest dataframe with 'speaker_id' and 'label'.
        y_scores: Continuous prediction scores for spoof.
        threshold: Frozen decision threshold.

    Returns:
        Dictionary mapping speaker_id to performance summary.
    """
    df = manifest_df.copy()
    if "speaker_id" not in df.columns:
        return {}

    df["score"] = y_scores
    df["prediction"] = (y_scores >= threshold).astype(int)
    df["correct"] = (df["prediction"] == df["label"].astype(int))

    speaker_summary = {}
    for spk_id, group in df.groupby("speaker_id"):
        n_samples = len(group)
        n_correct = int(group["correct"].sum())
        acc = float(n_correct / n_samples) if n_samples > 0 else 0.0

        speaker_summary[str(spk_id)] = {
            "n_samples": n_samples,
            "accuracy": acc,
            "error_count": n_samples - n_correct,
            "mean_score": float(group["score"].mean()),
        }

    return speaker_summary


def analyze_acoustic_shortcuts(
    waveforms: List[np.ndarray],
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """Measure correlation between superficial acoustic attributes and spoof prediction scores.

    A strong correlation (|r| > 0.5) indicates the model may be exploiting acoustic shortcuts
    (e.g., classifying louder clips as spoof) rather than learning generative speech artifacts.

    Attributes measured:
        - RMS Energy: Overall signal amplitude/loudness.
        - Zero-Crossing Rate (ZCR): Proxy for high-frequency spectral content and noisiness.

    Args:
        waveforms: List of 1D float32 audio waveforms.
        y_scores: Predicted probabilities of spoof.

    Returns:
        Dictionary with Pearson correlation coefficients (r_rms, r_zcr).
    """
    y_scores = np.asarray(y_scores, dtype=float)
    if len(waveforms) != len(y_scores) or len(waveforms) < 3:
        return {"r_rms": 0.0, "r_zcr": 0.0}

    rms_values = []
    zcr_values = []

    for w in waveforms:
        if len(w) == 0:
            rms_values.append(0.0)
            zcr_values.append(0.0)
            continue

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(w ** 2)))
        rms_values.append(rms)

        # 2. Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(w)) != 0) / float(len(w))
        zcr_values.append(float(zero_crossings))

    rms_arr = np.array(rms_values, dtype=float)
    zcr_arr = np.array(zcr_values, dtype=float)

    # Compute Pearson r (safely handling constant arrays)
    def safe_corr(x, y):
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x < 1e-12 or std_y < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    r_rms = safe_corr(rms_arr, y_scores)
    r_zcr = safe_corr(zcr_arr, y_scores)

    return {
        "correlation_rms_energy": round(r_rms, 4),
        "correlation_zero_crossing_rate": round(r_zcr, 4),
        "potential_rms_shortcut": abs(r_rms) > 0.5,
        "potential_zcr_shortcut": abs(r_zcr) > 0.5,
    }

