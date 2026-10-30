"""Generate publication-quality evaluation figures for VoxGuard MLAAD-tiny experiment.

Produces:
1. reports/figures/mlaad_confusion_matrices.png (Side-by-side Baseline vs Fine-Tuned)
2. reports/figures/mlaad_roc_curves.png (Baseline vs Fine-Tuned ROC Curves)
3. reports/figures/mlaad_generator_comparison.png (Per-generator breakdown)
4. reports/figures/mlaad_training_curves.png (Loss and EER curves across epochs)
"""

import json
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

FIG_DIR = os.path.abspath("reports/figures")


def plot_confusion_matrices(baseline_data: dict, finetuned_data: dict, output_path: str):
    """Plot side-by-side confusion matrices for baseline and fine-tuned models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = [
        ("Untouched Pretrained Baseline", baseline_data["test_evaluation_default_threshold"], axes[0]),
        ("VoxGuard Fine-Tuned (MLAAD)", finetuned_data["test_evaluation_default_threshold"], axes[1]),
    ]

    for title, eval_dict, ax in models:
        cm_dict = eval_dict["confusion_matrix"]
        cm = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]]
        ])
        total = cm.sum()
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color="black", fontsize=12, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Bonafide (0)", "Spoof (1)"], fontsize=10)
        ax.set_yticklabels(["Bonafide (0)", "Spoof (1)"], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"{title}\nAcc: {eval_dict['accuracy']*100:.1f}% | EER: {eval_dict['eer']*100:.1f}%", fontsize=12, pad=12)

    plt.suptitle("VoxGuard Held-Out Test Set Performance (MLAAD-tiny, N=300)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices figure: {output_path}")


def plot_generator_breakdown(baseline_data: dict, finetuned_data: dict, output_path: str):
    """Plot bar chart comparing accuracy across generators for baseline and fine-tuned models."""
    base_gens = baseline_data["test_evaluation_default_threshold"]["per_generator"]
    fine_gens = finetuned_data["test_evaluation_default_threshold"]["per_generator"]

    all_gens = sorted(list(base_gens.keys()))
    # Separate human from spoof generators
    all_gens = [g for g in all_gens if g != "human_bonafide"] + ["human_bonafide"]

    labels = []
    base_accs = []
    fine_accs = []
    colors = []

    for g in all_gens:
        labels.append(g)
        base_accs.append(base_gens[g]["accuracy"] * 100)
        fine_accs.append(fine_gens[g]["accuracy"] * 100)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, base_accs, width, label="Pretrained Baseline", color="#90caf9", edgecolor="#1565c0")
    rects2 = ax.bar(x + width/2, fine_accs, width, label="VoxGuard Fine-Tuned", color="#81c784", edgecolor="#2e7d32")

    ax.set_ylabel("Detection Accuracy (%)", fontsize=11)
    ax.set_title("Detection Accuracy by Synthetic Generator (Held-Out Test Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved generator comparison figure: {output_path}")


def plot_training_curves(history_data: dict, output_path: str):
    """Plot training and development loss, accuracy, and EER over epochs."""
    epochs = history_data["history"]["epoch"]
    train_loss = history_data["history"]["train_loss"]
    dev_loss = history_data["history"]["dev_loss"]
    dev_acc = [x * 100 for x in history_data["history"]["dev_acc"]]
    dev_eer = [x * 100 for x in history_data["history"]["dev_eer"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Curves
    ax1.plot(epochs, train_loss, "o-", label="Train BCE Loss", color="#1976d2", linewidth=2)
    ax1.plot(epochs, dev_loss, "s--", label="Dev BCE Loss", color="#d32f2f", linewidth=2)
    best_ep = history_data["best_epoch"]
    ax1.axvline(best_ep, color="green", linestyle=":", label=f"Best Model (Epoch {best_ep})")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Loss Curves (ResNet-18 Frozen)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy & EER Curves
    ax2.plot(epochs, dev_acc, "o-", label="Dev Accuracy (%)", color="#388e3c", linewidth=2)
    ax2.plot(epochs, dev_eer, "^--", label="Dev EER (%)", color="#f57c00", linewidth=2)
    ax2.axvline(best_ep, color="green", linestyle=":", label=f"Best Model (Epoch {best_ep})")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_title("Development Metrics Across Epochs", fontsize=12, fontweight="bold")
    ax2.legend(loc="center right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("VoxGuard Fine-Tuning Convergence History", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves figure: {output_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    baseline_path = "reports/results/pretrained_baseline_mlaad.json"
    finetuned_path = "reports/results/finetuned_mlaad_eval.json"
    history_path = "checkpoints/real_speech/voxguard_mlaad_finetuned/training_history.json"

    if not os.path.isfile(baseline_path) or not os.path.isfile(finetuned_path):
        print(f"Waiting for results files ({baseline_path}, {finetuned_path})...")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(finetuned_path, "r", encoding="utf-8") as f:
        finetuned_data = json.load(f)

    plot_confusion_matrices(baseline_data, finetuned_data, os.path.join(FIG_DIR, "mlaad_confusion_matrices.png"))
    plot_generator_breakdown(baseline_data, finetuned_data, os.path.join(FIG_DIR, "mlaad_generator_comparison.png"))

    if os.path.isfile(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        plot_training_curves(history_data, os.path.join(FIG_DIR, "mlaad_training_curves.png"))


if __name__ == "__main__":
    main()

