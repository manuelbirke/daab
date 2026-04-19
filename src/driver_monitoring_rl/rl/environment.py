from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EnvState:
    v_ego: float
    action_idx: int
    d_rel: float
    v_rel: float
    drowsy: int


class DrowsyBrakingEnv:
    """Simplified longitudinal car-following environment.

    State: [v_t, a_t, d_rel_t, v_rel_t, theta_t]
    Actions:
      0: brake 100%
      1: brake 70%
      2: brake 40%
      3: brake 20%
      4: neutral
      5: acceleration 100%
    """

    ACTION_TO_CONTROL = {
        0: (0.0, 1.0),
        1: (0.0, 0.7),
        2: (0.0, 0.4),
        3: (0.0, 0.2),
        4: (0.0, 0.0),
        5: (1.0, 0.0),
    }

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 300,
        drowsy_delay_s: float = 0.5,
        seed: int = 42,
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.delay_steps = max(1, int(round(drowsy_delay_s / dt)))
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.prev_action = 4
        self.v_front = 12.0
        self.state = EnvState(0.0, 4, 30.0, 0.0, 0)

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def action_dim(self) -> int:
        return 6

    def reset(self, drowsy: int | None = None) -> np.ndarray:
        self.step_count = 0
        self.prev_action = 4
        self.v_front = float(self.rng.uniform(8.0, 16.0))
        d = float(self.rng.uniform(20.0, 70.0))
        v = float(self.rng.uniform(0.0, 4.0))
        theta = int(self.rng.integers(0, 2) if drowsy is None else drowsy)
        self.state = EnvState(v_ego=v, action_idx=4, d_rel=d, v_rel=v - self.v_front, drowsy=theta)
        return self._to_obs(self.state)

    def _safe_range(self, v_ego: float) -> Tuple[float, float]:
        dmin = max(5.0, 2.0 * v_ego)
        return dmin, dmin + 10.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1

        # Drowsiness-induced execution delay uses previously executed action.
        exec_action = self.prev_action if self.state.drowsy == 1 else action
        self.prev_action = action

        throttle, brake = self.ACTION_TO_CONTROL[exec_action]
        accel = 3.0 * throttle - 6.0 * brake

        # stochastic front-vehicle behavior
        front_accel = float(self.rng.normal(loc=0.0, scale=0.35))
        self.v_front = float(np.clip(self.v_front + front_accel * self.dt, 2.0, 20.0))

        v_ego = float(np.clip(self.state.v_ego + accel * self.dt, 0.0, 30.0))
        d_rel = float(self.state.d_rel + (self.v_front - v_ego) * self.dt)
        v_rel = v_ego - self.v_front

        # Reward terms inspired by paper's finite-state formulation.
        dmin, dmax = self._safe_range(v_ego)
        collision = d_rel <= 0.0
        delta_pressure = abs(self.ACTION_TO_CONTROL[exec_action][1] - self.ACTION_TO_CONTROL[self.state.action_idx][1])

        reward = 0.0
        reward += -200.0 if collision else 0.0  # Pf
        reward += -2.0 * delta_pressure          # P1 smoothness
        if d_rel < dmin:
            reward += -10.0                      # P2 too close
        elif d_rel > dmax:
            reward += -2.0                       # P1 too far
        else:
            reward += +10.0                      # R2 safe zone

        if exec_action == 5 and self.state.v_ego < 0.5:
            reward += +1.0                       # R1 standstill -> acceleration
        if dmin <= d_rel <= dmax and exec_action in (1, 2, 3):
            reward += +2.0                       # R3 braking in safety
        if dmin <= d_rel <= dmax and abs(v_rel) < 1.0:
            reward += +5.0                       # R4 stable following

        done = collision or self.step_count >= self.max_steps
        if done and not collision:
            reward += +1.0                       # Rf

        # Optional random drowsiness transitions for realism.
        theta = self.state.drowsy
        if self.rng.random() < 0.015:
            theta = 1 - theta

        self.state = EnvState(v_ego=v_ego, action_idx=exec_action, d_rel=d_rel, v_rel=v_rel, drowsy=theta)
        return self._to_obs(self.state), float(reward), bool(done), {
            "collision": collision,
            "safe_min": dmin,
            "safe_max": dmax,
            "throttle": throttle,
            "brake": brake,
            "v_front": self.v_front,
            "delay_active": self.state.drowsy == 1,
        }

    def _to_obs(self, s: EnvState) -> np.ndarray:
        return np.array([s.v_ego, s.action_idx, s.d_rel, s.v_rel, s.drowsy], dtype=np.float32)
