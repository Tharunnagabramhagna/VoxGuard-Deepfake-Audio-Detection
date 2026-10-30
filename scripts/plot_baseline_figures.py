"""Generate baseline evaluation figures for untouched pretrained model on MLAAD-tiny."""

import json
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

FIG_DIR = os.path.abspath("reports/figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    baseline_path = "reports/results/pretrained_baseline_mlaad.json"

    if not os.path.isfile(baseline_path):
        print(f"Results file not found: {baseline_path}")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_res_def = data["test_evaluation_default_threshold"]
    test_res_calib = data["test_evaluation_dev_calibrated_threshold"]
    eer_thresh = data["dev_tuning"]["optimal_threshold_from_dev"]

    # 1. Side-by-side Confusion Matrices (Default 0.50 vs Calibrated)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for title, res, ax in [
        ("Default Threshold (θ = 0.50)", test_res_def, ax1),
        (f"Dev-Calibrated Threshold (θ = {eer_thresh:.4f})", test_res_calib, ax2),
    ]:
        cm_dict = res["confusion_matrix"]
        cm = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]]
        ])
        total = cm.sum()
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

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
        ax.set_ylabel("True Ground Truth Label", fontsize=11)
        ax.set_title(f"{title}\nAcc: {res['accuracy']*100:.1f}% | Prec: {res['precision']*100:.1f}% | Rec: {res['recall']*100:.1f}%", fontsize=11, pad=12)

    plt.suptitle("Untouched Pretrained Baseline – Confusion Matrices on Held-Out Test Set (N=300)", fontsize=13, fontweight="bold", y=1.03)
    cm_path = os.path.join(FIG_DIR, "baseline_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline confusion matrix figure: {cm_path}")

    # 2. ROC Curve
    # To plot exact ROC curve, let's load test manifest and get y_true, y_scores
    test_manifest = "data/manifests/mlaad_test.json"
    with open(test_manifest, "r", encoding="utf-8") as f:
        test_items = json.load(f)

    # We need y_scores for test set
    # Load from detailed results if saved or re-extract from report
    y_true = [x["label"] for x in test_items]

    # In predict_audio_deepfake, let's inspect if detailed results are in json
    # Let's extract y_scores by running a quick inference collection if needed, or plotting from false positive/negatives
    # Actually, evaluate_dataset computed roc_auc = 0.6958 and eer = 35.33%
    # Let's compute exact roc curve from test set
    y_scores = []
    from scripts.test_pretrained_detector import load_pretrained_detector, predict_audio_deepfake
    model = load_pretrained_detector("checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth")
    for item in test_items:
        res = predict_audio_deepfake(model, item["local_path"])
        y_scores.append(res["prob_fake"])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer = test_res_def["eer"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565c0", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    ax.plot([0, 1], [1, 0], color="#e65100", lw=1.2, linestyle=":", label=f"EER Diagonal (EER = {eer*100:.1f}%)")
    ax.scatter([eer], [1 - eer], color="#d32f2f", s=80, zorder=5, label=f"EER Operating Point ({eer*100:.1f}%)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("Untouched Pretrained Baseline ROC Curve\nMLAAD-tiny Held-Out Test Set (N=300)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    roc_path = os.path.join(FIG_DIR, "baseline_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline ROC curve figure: {roc_path}")

    # 3. Per-generator breakdown
    gens = test_res_def["per_generator"]
    all_gens = sorted(list(gens.keys()))
    all_gens = [g for g in all_gens if g != "human_bonafide"] + ["human_bonafide"]

    labels = []
    accs_def = []
    accs_calib = []

    for g in all_gens:
        labels.append(g)
        accs_def.append(gens[g]["accuracy"] * 100)
        accs_calib.append(test_res_calib["per_generator"][g]["accuracy"] * 100)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width/2, accs_def, width, label="Default Threshold (θ=0.50)", color="#90caf9", edgecolor="#1565c0")
    rects2 = ax.bar(x + width/2, accs_calib, width, label=f"Calibrated Threshold (θ={eer_thresh:.4f})", color="#ffb74d", edgecolor="#e65100")

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Baseline Detection Accuracy by Generator (Held-Out Test Set)\nDefault Threshold vs. Dev-Calibrated Threshold", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    gen_path = os.path.join(FIG_DIR, "baseline_generator_accuracy.png")
    plt.tight_layout()
    plt.savefig(gen_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline generator accuracy figure: {gen_path}")


if __name__ == "__main__":
    main()
