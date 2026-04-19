from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Recording:
    subject_id: str
    ecg: np.ndarray
    fs: int
    drowsy_event_seconds: np.ndarray


class DDDBLoader:
    """Loader for Drivers Drowsiness DataBase-like CSV format.

    Expected minimal format per subject file:
      - ecg column (float)
      - optional drowsy_event column (0/1 marker per sample)

    For public placeholder experiments without dataset access, use synthetic
    generation via :meth:`generate_synthetic_recordings`.
    """

    def __init__(self, data_dir: str | Path, fs: int = 128) -> None:
        self.data_dir = Path(data_dir)
        self.fs = fs

    def load(self) -> List[Recording]:
        files = sorted(self.data_dir.glob("*.csv"))
        recordings: List[Recording] = []

        for f in files:
            df = pd.read_csv(f)
            if "ecg" not in df.columns:
                raise ValueError(f"Missing 'ecg' column in {f}")
            ecg = df["ecg"].to_numpy(dtype=float)

            if "drowsy_event" in df.columns:
                event_samples = np.flatnonzero(df["drowsy_event"].to_numpy() > 0)
                event_seconds = np.unique((event_samples / self.fs).astype(int))
            else:
                event_seconds = np.array([], dtype=int)

            recordings.append(
                Recording(
                    subject_id=f.stem,
                    ecg=ecg,
                    fs=self.fs,
                    drowsy_event_seconds=event_seconds,
                )
            )
        return recordings

    @staticmethod
    def generate_synthetic_recordings(
        n_subjects: int = 10,
        duration_seconds: int = 2 * 60 * 60,
        fs: int = 128,
        seed: int = 42,
    ) -> List[Recording]:
        """Generate synthetic ECG-like waveforms with sparse drowsy events.

        This is a placeholder pipeline for reproducibility when DD-DB files are
        unavailable.
        """
        rng = np.random.default_rng(seed)
        recordings: List[Recording] = []
        n_samples = duration_seconds * fs
        t = np.arange(n_samples) / fs

        for i in range(n_subjects):
            baseline_hr_hz = rng.uniform(0.95, 1.25)
            ecg = 0.8 * np.sin(2 * np.pi * baseline_hr_hz * t)
            ecg += 0.15 * np.sin(2 * np.pi * 2 * baseline_hr_hz * t)
            ecg += rng.normal(scale=0.05, size=n_samples)

            # Add drowsy phases with lower-frequency variability.
            n_events = rng.integers(8, 16)
            event_seconds = np.sort(rng.integers(180, duration_seconds - 180, size=n_events))
            for es in event_seconds:
                start = max(0, (es - 60) * fs)
                end = min(n_samples, (es + 60) * fs)
                ecg[start:end] += 0.08 * np.sin(2 * np.pi * 0.15 * t[start:end])

            recordings.append(
                Recording(
                    subject_id=f"subject_{i+1:02d}",
                    ecg=ecg.astype(np.float32),
                    fs=fs,
                    drowsy_event_seconds=event_seconds.astype(int),
                )
            )
        return recordings
