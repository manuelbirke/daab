from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from driver_monitoring_rl.config import CapsuleConfig
from driver_monitoring_rl.features.hrv_features import extract_hrv_features


@dataclass
class SequenceExample:
    sequence: np.ndarray  # shape [N, feature_dim]
    label: int


def enumerate_valid_cnm_configs(
    window_samples: int,
    c_range_seconds: Tuple[int, int],
    n_range: Tuple[int, int],
    fs: int,
    decimal_places: int = 2,
) -> List[CapsuleConfig]:
    configs: List[CapsuleConfig] = []
    c_min_s, c_max_s = c_range_seconds
    for c_s in range(c_min_s, c_max_s + 1):
        C = c_s * fs
        for N in range(n_range[0], n_range[1] + 1):
            denom = (N - 1) * C
            if denom <= 0:
                continue
            M = (C * N - window_samples) / denom
            if 0 < M < 1:
                M_rounded = round(float(M), decimal_places)
                if np.isclose(M, M_rounded, atol=10 ** (-decimal_places)):
                    configs.append(CapsuleConfig(C=C, N=N, M=M_rounded))
    return configs


def split_window_into_capsules(ecg_window: np.ndarray, config: CapsuleConfig) -> List[np.ndarray]:
    C, N, M = config.C, config.N, config.M
    if len(ecg_window) < C:
        return []
    stride = int(round(C * (1 - M)))
    if stride <= 0:
        return []

    capsules = []
    for i in range(N):
        start = i * stride
        end = start + C
        if end > len(ecg_window):
            return []
        capsules.append(ecg_window[start:end])
    return capsules


def build_sequence_from_capsules(
    capsule_rr_intervals: Iterable[np.ndarray],
) -> np.ndarray:
    feature_seq = []
    for rr in capsule_rr_intervals:
        feature_seq.append(extract_hrv_features(rr).values)
    if not feature_seq:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(feature_seq, axis=0).astype(np.float32)
