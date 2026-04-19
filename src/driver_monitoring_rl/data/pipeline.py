from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from driver_monitoring_rl.config import CapsuleConfig
from driver_monitoring_rl.data.dddb_loader import Recording
from driver_monitoring_rl.data.preprocessing import preprocess_to_rr
from driver_monitoring_rl.features.capsule_shift import (
    build_sequence_from_capsules,
    split_window_into_capsules,
)


@dataclass
class DatasetBundle:
    x: np.ndarray  # [samples, sequence_len, feature_dim]
    y: np.ndarray  # [samples]


def _extract_windows(
    recording: Recording,
    window_seconds: int = 120,
    min_gap_seconds: int = 120,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    fs = recording.fs
    w = window_seconds * fs

    event_samples = np.array(recording.drowsy_event_seconds * fs, dtype=int)
    if len(event_samples) == 0:
        return [], []

    # Keep sparse events, at least min_gap apart.
    sparse_events = [event_samples[0]]
    for e in event_samples[1:]:
        if (e - sparse_events[-1]) >= min_gap_seconds * fs:
            sparse_events.append(e)

    dews: List[np.ndarray] = []
    nsrws: List[np.ndarray] = []
    n = len(recording.ecg)

    for e in sparse_events:
        start = max(0, e - w // 2)
        end = min(n, start + w)
        start = max(0, end - w)
        if end - start == w:
            dews.append(recording.ecg[start:end])

    # Match non-drowsy windows count 1:1 from non-event zones.
    candidate_starts = np.arange(0, max(1, n - w), w)
    for s in candidate_starts:
        e = s + w
        if e > n:
            continue
        # reject if close to any drowsy event
        center = (s + e) // 2
        if np.all(np.abs(np.array(sparse_events) - center) >= min_gap_seconds * fs):
            nsrws.append(recording.ecg[s:e])
        if len(nsrws) >= len(dews):
            break

    return dews, nsrws


def build_drowsiness_dataset(
    recordings: List[Recording],
    capsule_config: CapsuleConfig,
    window_seconds: int = 120,
) -> DatasetBundle:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []

    for rec in recordings:
        dews, nsrws = _extract_windows(rec, window_seconds=window_seconds)

        for label, windows in [(1, dews), (0, nsrws)]:
            for ecg_window in windows:
                capsules = split_window_into_capsules(ecg_window, capsule_config)
                if len(capsules) != capsule_config.N:
                    continue

                rr_capsules = []
                for cap in capsules:
                    _, rr = preprocess_to_rr(cap, fs=rec.fs)
                    rr_capsules.append(rr)

                seq = build_sequence_from_capsules(rr_capsules)
                if seq.shape[0] != capsule_config.N or seq.shape[1] == 0:
                    continue
                x_list.append(seq)
                y_list.append(label)

    if not x_list:
        raise RuntimeError("No sequence samples generated. Check events/configuration/data.")

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Balance classes 1:1 by downsampling majority.
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    k = min(len(pos_idx), len(neg_idx))
    rng = np.random.default_rng(42)
    pos_keep = rng.choice(pos_idx, size=k, replace=False)
    neg_keep = rng.choice(neg_idx, size=k, replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(keep)

    return DatasetBundle(x=x[keep], y=y[keep])
