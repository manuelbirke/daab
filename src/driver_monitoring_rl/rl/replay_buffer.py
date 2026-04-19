from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int) -> TransitionBatch:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return TransitionBatch(
            states=np.stack(s).astype(np.float32),
            actions=np.array(a, dtype=np.int64),
            rewards=np.array(r, dtype=np.float32),
            next_states=np.stack(s2).astype(np.float32),
            dones=np.array(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
