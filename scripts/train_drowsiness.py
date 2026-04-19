#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from driver_monitoring_rl.config import DrowsinessTrainConfig
from driver_monitoring_rl.data.dddb_loader import DDDBLoader
from driver_monitoring_rl.data.pipeline import build_drowsiness_dataset
from driver_monitoring_rl.models.drowsiness_rnn import RNNTrainer
from driver_monitoring_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ECG-HRV drowsiness RNN model")
    p.add_argument("--data-dir", type=str, default="", help="Optional path to DD-DB CSV files")
    p.add_argument("--out-dir", type=str, default="artifacts/drowsiness")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DrowsinessTrainConfig()
    set_global_seed(cfg.random_state)

    if args.data_dir:
        recordings = DDDBLoader(args.data_dir, fs=cfg.sampling_rate_hz).load()
    else:
        recordings = DDDBLoader.generate_synthetic_recordings(fs=cfg.sampling_rate_hz, seed=cfg.random_state)

    data = build_drowsiness_dataset(
        recordings,
        capsule_config=cfg.selected_capsule,
        window_seconds=cfg.window_seconds,
    )

    trainer = RNNTrainer(cfg=cfg, device=args.device)
    result = trainer.cross_validate(data.x, data.y)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scaler_mean.npy", result.scaler_mean)
    np.save(out / "scaler_scale.npy", result.scaler_scale)
    (out / "metrics.txt").write_text("\n".join(f"{k}: {v:.4f}" for k, v in result.metrics.items()))

    print("Drowsiness model CV metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Saved artifacts to {out}")


if __name__ == "__main__":
    main()
