"""VoxGuard analysis package."""

from src.analysis.leakage_analysis import (
    analyze_acoustic_shortcuts,
    group_performance_by_speaker,
    group_performance_by_system,
    verify_speaker_disjointness,
)
from src.analysis.explanation import (
    compute_measurable_acoustic_features,
    generate_result_summary,
)

__all__ = [
    "verify_speaker_disjointness",
    "group_performance_by_system",
    "group_performance_by_speaker",
    "analyze_acoustic_shortcuts",
    "compute_measurable_acoustic_features",
    "generate_result_summary",
]
