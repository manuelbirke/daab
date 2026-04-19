from __future__ import annotations

"""Feature importance benchmarking for ECG-HRV drowsiness modeling.

This module provides a complete, reproducible benchmark of common feature
importance estimators for binary drowsiness classification using *raw* HRV
features extracted by :func:`driver_monitoring_rl.features.hrv_features.extract_hrv_features`.

Implemented methods
-------------------
1. **RFE (Recursive Feature Elimination)**
   - Base estimator: L2-regularized Logistic Regression.
   - RFE removes less useful features iteratively and returns a final ranking.
   - Interpretation: lower rank is better; this module maps rank to a normalized
     importance score in ``[0, 1]`` where larger means more important.

2. **MI (Mutual Information)**
   - Measures nonlinear dependence between each feature and the target label.
   - Interpretation: larger MI indicates stronger information shared with class.

3. **RF (Random Forest impurity importance)**
   - Uses mean decrease in impurity from a RandomForestClassifier.
   - Interpretation: larger value means larger contribution to tree split quality.

4. **PI (Permutation Importance)**
   - Measures validation performance drop when one feature is shuffled.
   - Interpretation: larger drop means the model relied more on that feature.

Cross-validation robustness
---------------------------
All methods are estimated fold-by-fold using StratifiedKFold and then aggregated
(mean and standard deviation). This reduces sensitivity to a single split and
supports more robust comparative analysis.

Important caveats
-----------------
- Different importance methods answer different questions; disagreements are
  expected and often informative.
- RF impurity importance can favor high-cardinality/noisy features.
- MI estimates can be unstable for tiny datasets.
- PI depends on the chosen predictive model and metric.

Use :func:`run_feature_importance_benchmark` for the unified workflow, and
:func:`build_raw_hrv_feature_dataset` to create a tabular HRV dataset from
recordings.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from driver_monitoring_rl.data.dddb_loader import Recording
from driver_monitoring_rl.data.preprocessing import preprocess_to_rr
from driver_monitoring_rl.features.hrv_features import FEATURE_NAMES, extract_hrv_features


@dataclass
class FeatureImportanceMethodResult:
    """Fold-aggregated output for one importance method."""

    method: str
    feature_names: List[str]
    scores_mean: np.ndarray
    scores_std: np.ndarray
    fold_scores: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return long-form summary sorted by descending mean importance."""
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_mean": self.scores_mean,
                "importance_std": self.scores_std,
            }
        )
        return df.sort_values("importance_mean", ascending=False, ignore_index=True)


@dataclass
class FeatureImportanceBenchmarkResult:
    """Unified output of all feature importance methods."""

    feature_names: List[str]
    methods: Dict[str, FeatureImportanceMethodResult]

    def comparison_frame(self) -> pd.DataFrame:
        """Wide comparison matrix with one column per method."""
        frame = pd.DataFrame({"feature": self.feature_names})
        for method_name, result in self.methods.items():
            frame[method_name] = result.scores_mean
        return frame

    def rank_frame(self) -> pd.DataFrame:
        """Return feature ranks per method (1 = most important)."""
        comp = self.comparison_frame()
        out = pd.DataFrame({"feature": comp["feature"]})
        for method_name in self.methods:
            out[f"{method_name}_rank"] = comp[method_name].rank(ascending=False, method="min").astype(int)
        return out


def _extract_windows(
    recording: Recording,
    window_seconds: int = 120,
    min_gap_seconds: int = 120,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract drowsy and non-drowsy ECG windows from one recording."""
    fs = recording.fs
    w = window_seconds * fs

    event_samples = np.array(recording.drowsy_event_seconds * fs, dtype=int)
    if len(event_samples) == 0:
        return [], []

    sparse_events = [event_samples[0]]
    for sample in event_samples[1:]:
        if (sample - sparse_events[-1]) >= min_gap_seconds * fs:
            sparse_events.append(sample)

    drowsy_windows: List[np.ndarray] = []
    normal_windows: List[np.ndarray] = []
    n = len(recording.ecg)

    for event_sample in sparse_events:
        start = max(0, event_sample - w // 2)
        end = min(n, start + w)
        start = max(0, end - w)
        if end - start == w:
            drowsy_windows.append(recording.ecg[start:end])

    candidate_starts = np.arange(0, max(1, n - w), w)
    for start in candidate_starts:
        end = start + w
        if end > n:
            continue
        center = (start + end) // 2
        if np.all(np.abs(np.array(sparse_events) - center) >= min_gap_seconds * fs):
            normal_windows.append(recording.ecg[start:end])
        if len(normal_windows) >= len(drowsy_windows):
            break

    return drowsy_windows, normal_windows


def build_raw_hrv_feature_dataset(
    recordings: Sequence[Recording],
    window_seconds: int = 120,
    min_gap_seconds: int = 120,
    balance_classes: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Build a tabular HRV dataset using raw features from each ECG window.

    Returns
    -------
    X, y, feature_names
        ``X`` shape is ``[num_samples, num_features]`` and uses the HRV features
        from :func:`extract_hrv_features` directly (no capsule/RNN sequencing).
    """
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for rec in recordings:
        drowsy_windows, normal_windows = _extract_windows(
            rec,
            window_seconds=window_seconds,
            min_gap_seconds=min_gap_seconds,
        )

        for label, windows in ((1, drowsy_windows), (0, normal_windows)):
            for ecg_window in windows:
                _, rr = preprocess_to_rr(ecg_window, fs=rec.fs)
                feature_vec = extract_hrv_features(rr).values
                if not np.all(np.isfinite(feature_vec)):
                    continue
                x_rows.append(feature_vec)
                y_rows.append(label)

    if not x_rows:
        raise RuntimeError("No HRV samples generated. Check recording quality/events/config.")

    X = np.stack(x_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int64)

    if balance_classes:
        rng = np.random.default_rng(random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise RuntimeError("Balanced sampling requires both positive and negative samples.")
        k = min(len(pos_idx), len(neg_idx))
        keep = np.concatenate(
            [
                rng.choice(pos_idx, size=k, replace=False),
                rng.choice(neg_idx, size=k, replace=False),
            ]
        )
        rng.shuffle(keep)
        X = X[keep]
        y = y[keep]

    return X, y, list(FEATURE_NAMES)


def _safe_cv_splits(y: np.ndarray, requested_splits: int) -> int:
    class_counts = np.bincount(y)
    non_zero = class_counts[class_counts > 0]
    if len(non_zero) < 2:
        raise ValueError("At least two classes are required for feature importance benchmarking.")
    max_valid = int(np.min(non_zero))
    splits = min(requested_splits, max_valid)
    if splits < 2:
        raise ValueError("Not enough samples per class for cross-validation (need at least 2).")
    return splits


def run_feature_importance_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    cv_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 400,
    permutation_repeats: int = 20,
    scoring: str = "f1",
) -> FeatureImportanceBenchmarkResult:
    """Run RFE, MI, RF, and PI with cross-validation and aggregate results."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [n_samples, n_features], got shape={X.shape}")
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match X.shape[1]")

    n_splits = _safe_cv_splits(y, cv_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_features = X.shape[1]
    fold_rfe: List[np.ndarray] = []
    fold_mi: List[np.ndarray] = []
    fold_rf: List[np.ndarray] = []
    fold_pi: List[np.ndarray] = []

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        x_train, y_train = X[tr_idx], y[tr_idx]
        x_valid, y_valid = X[va_idx], y[va_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)

        rfe_estimator = LogisticRegression(max_iter=2500, solver="liblinear", random_state=random_state + fold_id)
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=1, step=1)
        rfe.fit(x_train_scaled, y_train)
        ranks = rfe.ranking_.astype(np.float32)
        rfe_scores = (n_features - ranks + 1.0) / n_features
        fold_rfe.append(rfe_scores)

        mi_scores = mutual_info_classif(x_train, y_train, random_state=random_state + fold_id)
        fold_mi.append(mi_scores.astype(np.float32))

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state + fold_id,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(x_train, y_train)
        fold_rf.append(rf.feature_importances_.astype(np.float32))

        pi = permutation_importance(
            rf,
            x_valid,
            y_valid,
            n_repeats=permutation_repeats,
            random_state=random_state + fold_id,
            n_jobs=-1,
            scoring=scoring,
        )
        fold_pi.append(pi.importances_mean.astype(np.float32))

    def _pack(method_name: str, fold_values: List[np.ndarray]) -> FeatureImportanceMethodResult:
        arr = np.stack(fold_values, axis=0)
        return FeatureImportanceMethodResult(
            method=method_name,
            feature_names=list(feature_names),
            scores_mean=np.mean(arr, axis=0),
            scores_std=np.std(arr, axis=0),
            fold_scores=arr,
        )

    methods = {
        "RFE": _pack("RFE", fold_rfe),
        "MI": _pack("MI", fold_mi),
        "RF": _pack("RF", fold_rf),
        "PI": _pack("PI", fold_pi),
    }

    return FeatureImportanceBenchmarkResult(feature_names=list(feature_names), methods=methods)
