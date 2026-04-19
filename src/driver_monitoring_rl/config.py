from dataclasses import dataclass, field
from typing import List


@dataclass
class CapsuleConfig:
    """Capsule-shifting configuration C, N, M.

    C is in samples; N is capsule count; M is overlap fraction in [0, 1).
    """

    C: int
    N: int
    M: float


@dataclass
class DrowsinessTrainConfig:
    sampling_rate_hz: int = 128
    window_seconds: int = 120
    window_overlap: float = 0.5
    random_state: int = 42
    batch_size: int = 48
    epochs: int = 100
    learning_rate: float = 2e-3
    hidden_size: int = 40
    num_rnn_layers: int = 3
    dropout: float = 0.25
    weight_decay: float = 0.0
    l2_dense: float = 0.28
    early_stopping_patience: int = 12
    cv_folds: int = 5
    selected_capsule: CapsuleConfig = field(
        default_factory=lambda: CapsuleConfig(C=6400, N=6, M=0.72)
    )


@dataclass
class RLConfig:
    seed: int = 42
    episodes: int = 500
    max_steps: int = 300
    dt: float = 0.1
    gamma: float = 0.90
    lr: float = 1e-3
    batch_size: int = 128
    replay_size: int = 1_000_000
    min_replay_size: int = 5_000
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    guided_exploration_episodes: int = 50
    guided_accel_prob: float = 0.8
    drowsy_delay_seconds: float = 0.5
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 256, 128])
    value_stream_sizes: List[int] = field(default_factory=lambda: [128, 64])
    advantage_stream_sizes: List[int] = field(default_factory=lambda: [128, 64])
