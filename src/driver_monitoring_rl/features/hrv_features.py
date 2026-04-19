from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.signal import welch


@dataclass
class HRVFeatureVector:
    values: np.ndarray
    names: List[str]


FEATURE_NAMES = [
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "hr_mean",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
]


def _time_domain(rr: np.ndarray) -> Dict[str, float]:
    if len(rr) < 3:
        return {k: 0.0 for k in ["mean_rr", "sdnn", "rmssd", "pnn50", "hr_mean"]}
    diff_rr = np.diff(rr)
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.05))
    return {
        "mean_rr": float(np.mean(rr)),
        "sdnn": float(np.std(rr, ddof=1)),
        "rmssd": float(np.sqrt(np.mean(diff_rr**2))),
        "pnn50": pnn50,
        "hr_mean": float(60.0 / np.mean(rr)),
    }


def _freq_domain(rr: np.ndarray) -> Dict[str, float]:
    if len(rr) < 8:
        return {"lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0}

    # RR is unevenly sampled; for a practical baseline we treat it as sequence.
    freqs, psd = welch(rr - np.mean(rr), fs=4.0, nperseg=min(256, len(rr)))
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
    ratio = float(lf_power / (hf_power + 1e-8))
    return {"lf_power": lf_power, "hf_power": hf_power, "lf_hf_ratio": ratio}


def extract_hrv_features(rr: np.ndarray) -> HRVFeatureVector:
    td = _time_domain(rr)
    fd = _freq_domain(rr)
    merged = {**td, **fd}
    values = np.array([merged[n] for n in FEATURE_NAMES], dtype=np.float32)
    return HRVFeatureVector(values=values, names=FEATURE_NAMES)
