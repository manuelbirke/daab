from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from driver_monitoring_rl.config import RLConfig
from driver_monitoring_rl.rl.networks import DuelingQNetwork
from driver_monitoring_rl.rl.replay_buffer import ReplayBuffer


@dataclass
class TrainingStats:
    rewards: List[float]
    collisions: List[int]


class DDDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: RLConfig, device: str = "cpu"):
        self.cfg = cfg
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.policy_net = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            shared_sizes=cfg.hidden_sizes,
            value_sizes=cfg.value_stream_sizes,
            adv_sizes=cfg.advantage_stream_sizes,
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            shared_sizes=cfg.hidden_sizes,
            value_sizes=cfg.value_stream_sizes,
            adv_sizes=cfg.advantage_stream_sizes,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(cfg.replay_size)
        self.epsilon = cfg.epsilon_start
        self.update_counter = 0

    def select_action(self, obs: np.ndarray, episode: int) -> int:
        # Guided exploration for first episodes: acceleration preferred with 80%.
        if episode < self.cfg.guided_exploration_episodes and np.random.rand() < self.cfg.guided_accel_prob:
            return 5

        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(x)
            return int(torch.argmax(q, dim=1).item())

    def push_transition(self, s, a, r, s2, done) -> None:
        self.replay.add(s, a, r, s2, done)

    def train_step(self) -> Dict[str, float] | None:
        if len(self.replay) < self.cfg.min_replay_size:
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(batch.states, dtype=torch.float32, device=self.device)
        a = torch.tensor(batch.actions, dtype=torch.int64, device=self.device)
        r = torch.tensor(batch.rewards, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(batch.next_states, dtype=torch.float32, device=self.device)
        d = torch.tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: action selection by policy net, evaluation by target net.
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(s2), dim=1)
            next_q = self.target_net(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + self.cfg.gamma * (1.0 - d) * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), output_dir / "dddqn_policy.pt")

    def load(self, model_path: str | Path) -> None:
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
