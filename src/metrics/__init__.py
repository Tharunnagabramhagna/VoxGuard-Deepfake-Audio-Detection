"""VoxGuard metrics package."""

from src.metrics.evaluator import calibrate_threshold, compute_eer, evaluate_predictions
from src.metrics.error_analysis import mine_errors, save_error_analysis
from src.metrics.plots import (
    plot_confusion_matrices,
    plot_robustness_curves,
    plot_roc_curves,
    plot_score_distributions,
)

__all__ = [
    "compute_eer",
    "calibrate_threshold",
    "evaluate_predictions",
    "mine_errors",
    "save_error_analysis",
    "plot_roc_curves",
    "plot_confusion_matrices",
    "plot_score_distributions",
    "plot_robustness_curves",
]
