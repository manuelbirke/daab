from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_ecg(ecg: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, ecg)


def detect_r_peaks(filtered_ecg: np.ndarray, fs: int) -> np.ndarray:
    """Simple robust R-peak detector for baseline experiments."""
    distance = int(0.35 * fs)
    height = np.percentile(filtered_ecg, 85)
    peaks, _ = find_peaks(filtered_ecg, distance=distance, height=height)
    return peaks


def compute_rr_intervals_seconds(r_peaks: np.ndarray, fs: int) -> np.ndarray:
    if len(r_peaks) < 3:
        return np.array([], dtype=float)
    rr = np.diff(r_peaks) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    return rr


def preprocess_to_rr(ecg: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    filtered = bandpass_ecg(ecg, fs=fs)
    peaks = detect_r_peaks(filtered, fs=fs)
    rr = compute_rr_intervals_seconds(peaks, fs=fs)
    return peaks, rr
