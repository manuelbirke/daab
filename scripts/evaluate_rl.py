#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from driver_monitoring_rl.config import RLConfig
from driver_monitoring_rl.rl.agent import DDDQNAgent
from driver_monitoring_rl.rl.environment import DrowsyBrakingEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained DD-DQN agent")
    p.add_argument("--model-path", type=str, default="artifacts/rl/dddqn_policy.pt")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RLConfig()
    env = DrowsyBrakingEnv(dt=cfg.dt, max_steps=cfg.max_steps, drowsy_delay_s=cfg.drowsy_delay_seconds, seed=cfg.seed + 1)
    agent = DDDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, cfg=cfg, device=args.device)
    agent.load(args.model_path)
    agent.epsilon = 0.0

    rewards, successes, collisions = [], 0, 0

    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            a = agent.select_action(obs, episode=10_000)
            obs, r, done, info = env.step(a)
            ep_reward += r

        rewards.append(ep_reward)
        collision = int(info.get("collision", False))
        collisions += collision
        successes += int(not collision)

    print("Evaluation summary")
    print(f"  episodes: {args.episodes}")
    print(f"  success_rate: {successes / args.episodes:.4f}")
    print(f"  collision_rate: {collisions / args.episodes:.4f}")
    print(f"  mean_reward: {np.mean(rewards):.2f}")
    print(f"  median_reward: {np.median(rewards):.2f}")


if __name__ == "__main__":
    main()
