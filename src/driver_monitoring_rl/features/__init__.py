"""Feature extraction and feature analysis modules.

Includes:
- HRV/capsule feature extraction for drowsiness modeling.
- Cross-validated feature importance benchmarking (RFE, MI, RF, PI).
"""

from driver_monitoring_rl.features.feature_importance import (
    FeatureImportanceBenchmarkResult,
    FeatureImportanceMethodResult,
    build_raw_hrv_feature_dataset,
    run_feature_importance_benchmark,
)

__all__ = [
    "FeatureImportanceBenchmarkResult",
    "FeatureImportanceMethodResult",
    "build_raw_hrv_feature_dataset",
    "run_feature_importance_benchmark",
]
