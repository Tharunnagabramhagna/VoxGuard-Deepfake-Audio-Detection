"""Publication-quality visualization suite for deepfake audio evaluation.

Generates reproducible figures for:
- Overlaid Receiver Operating Characteristic (ROC) curves with EER points.
- Confusion matrices evaluated at the calibrated operating threshold.
- Actual-versus-predicted probability score distributions (Separation Histograms).
- Robustness degradation curves under additive noise (SNR sweeps).
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curves(
    model_eval_results: Dict[str, Dict[str, object]],
    output_path: str = "reports/figures/roc_curves.png",
    title: str = "Receiver Operating Characteristic (ROC) Curves",
) -> str:
    """Plot overlaid ROC curves for multiple models with EER annotations.

    Args:
        model_eval_results: Dictionary mapping model_name -> evaluation result dict
                            containing 'roc_curve' (with 'fpr' and 'tpr'), 'roc_auc', and 'eer'.
        output_path: Destination image path.
        title: Plot title.

    Returns:
        Path to saved PNG figure.
    """
    plt.figure(figsize=(8, 7), dpi=300)

    colors = ["#2b5c8f", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (model_name, res) in enumerate(model_eval_results.items()):
        roc = res.get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc = res.get("roc_auc", 0.0)
        eer = res.get("eer", 0.0)

        color = colors[i % len(colors)]
        label = f"{model_name.replace('_', ' ').title()} (AUC = {auc:.4f}, EER = {eer * 100:.2f}%)"

        if len(fpr) > 1 and len(tpr) > 1:
            plt.plot(fpr, tpr, color=color, lw=2, label=label)

            # Mark EER operating point on the curve
            # At EER, FPR ~= EER and TPR ~= 1 - EER
            plt.scatter([eer], [1.0 - eer], color=color, s=50, zorder=5)

    # Plot diagonal reference line (random guess)
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (False Acceptance of Spoof)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Correct Detection of Spoof)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)


def plot_confusion_matrices(
    model_eval_results: Dict[str, Dict[str, object]],
    output_path: str = "reports/figures/confusion_matrices.png",
) -> str:
    """Plot side-by-side confusion matrices at calibrated operating thresholds.

    Args:
        model_eval_results: Dict mapping model_name -> eval result with 'confusion_matrix'.
        output_path: Destination image path.

    Returns:
        Path to saved PNG figure.
    """
    n_models = len(model_eval_results)
    if n_models == 0:
        return ""

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=300, squeeze=False)
    axes = axes[0]

    for idx, (model_name, res) in enumerate(model_eval_results.items()):
        ax = axes[idx]
        cm_dict = res.get("confusion_matrix", {})
        matrix = cm_dict.get("matrix", [[0, 0], [0, 0]])
        cm_arr = np.array(matrix)

        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{model_name.replace('_', ' ').title()}\n(Th = {res.get('operating_threshold', 0.5):.3f})",
                     fontsize=11, fontweight="bold")

        tick_marks = np.arange(2)
        classes = ["Bonafide (0)", "Spoof (1)"]
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9)

        # Annotate text counts
        thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
        labels_sub = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = cm_arr[i, j]
                text_color = "white" if val > thresh else "black"
                ax.text(j, i, f"{labels_sub[i][j]}\n{val}",
                        ha="center", va="center", color=text_color,
                        fontsize=11, fontweight="bold")

        ax.set_ylabel("True Category", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Category", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)


def plot_score_distributions(
    model_score_data: Dict[str, Dict[str, object]],
    output_path: str = "reports/figures/score_distributions.png",
) -> str:
    """Plot separation histograms of predicted spoof probabilities.

    Args:
        model_score_data: Dict mapping model_name -> dict with 'y_true', 'y_scores',
                          and 'threshold'.
        output_path: Destination image path.

    Returns:
        Path to saved PNG figure.
    """
    n_models = len(model_score_data)
    if n_models == 0:
        return ""

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), dpi=300, squeeze=False)
    axes = axes[0]

    for idx, (model_name, data) in enumerate(model_score_data.items()):
        ax = axes[idx]
        y_true = np.asarray(data["y_true"], dtype=int)
        y_scores = np.asarray(data["y_scores"], dtype=float)
        th = float(data.get("threshold", 0.5))

        scores_bona = y_scores[y_true == 0]
        scores_spoof = y_scores[y_true == 1]

        bins = np.linspace(0.0, 1.0, 25)
        ax.hist(scores_bona, bins=bins, alpha=0.65, color="#2ca02c", label=f"Bonafide (N={len(scores_bona)})", density=False)
        ax.hist(scores_spoof, bins=bins, alpha=0.65, color="#d62728", label=f"Spoof (N={len(scores_spoof)})", density=False)

        ax.axvline(th, color="black", linestyle="--", lw=2, label=f"Threshold = {th:.3f}")

        ax.set_title(f"{model_name.replace('_', ' ').title()}\nScore Separation", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Spoof Probability P(Spoof)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Utterance Count", fontsize=10, fontweight="bold")
        ax.legend(loc="upper center", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)


def plot_robustness_curves(
    robustness_results: Dict[str, Dict[str, float]],
    output_path: str = "reports/figures/robustness_snr_curve.png",
) -> str:
    """Plot model accuracy and EER degradation across varying SNR levels.

    Args:
        robustness_results: Dict mapping condition_str (e.g. 'Clean', '30dB', ...)
                            to dict containing metrics for models.
        output_path: Destination path.

    Returns:
        Path to saved PNG figure.
    """
    plt.figure(figsize=(8, 5.5), dpi=300)

    conditions = list(robustness_results.keys())

    # Collect models present
    models = set()
    for cond in conditions:
        models.update(robustness_results[cond].keys())

    colors = ["#2b5c8f", "#2ca02c", "#d62728"]

    for idx, model in enumerate(sorted(list(models))):
        accuracies = []
        for cond in conditions:
            acc = robustness_results[cond].get(model, {}).get("accuracy", 0.0)
            accuracies.append(acc * 100.0)

        plt.plot(conditions, accuracies, marker="o", lw=2.2, color=colors[idx % len(colors)],
                 label=f"{model.replace('_', ' ').title()}")

    plt.xlabel("Acoustic Condition (Additive Noise Level)", fontsize=11, fontweight="bold")
    plt.ylabel("Classification Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("VoxGuard Noise Robustness Stress Curve", fontsize=13, fontweight="bold", pad=12)
    plt.ylim([0, 105])
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)

