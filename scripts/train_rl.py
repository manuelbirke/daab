#!/usr/bin/env python3
from __future__ import annotations

"""Train the drowsiness-aware DD-DQN braking agent.

Follow-up note:
This script now supports an **optional environment factory** to make swapping
simulation backends easier (e.g., baseline simulator vs CARLA adapter) without
changing the training loop.
"""

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from driver_monitoring_rl.config import RLConfig
from driver_monitoring_rl.rl.agent import DDDQNAgent
from driver_monitoring_rl.rl.environment import DrowsyBrakingEnv
from driver_monitoring_rl.utils.seed import set_global_seed

EnvFactory = Callable[[RLConfig], Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train drowsiness-aware DD-DQN braking agent")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--out-dir", type=str, default="artifacts/rl")
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument(
        "--env-backend",
        type=str,
        default="sim",
        choices=["sim", "carla"],
        help="Environment backend to use. 'sim' uses built-in DrowsyBrakingEnv; 'carla' uses CARLABrakingEnv.",
    )
    p.add_argument("--carla-host", type=str, default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--carla-town", type=str, default="")
    p.add_argument("--carla-timeout-s", type=float, default=10.0)
    return p.parse_args()


def default_env_factory(cfg: RLConfig, args: argparse.Namespace) -> Any:
    """Create an environment from CLI args.

    This function is intentionally separate so callers can supply their own
    factory (e.g., for tests, custom wrappers, or richer CARLA scenarios).
    """
    if args.env_backend == "sim":
        return DrowsyBrakingEnv(dt=cfg.dt, max_steps=cfg.max_steps, drowsy_delay_s=cfg.drowsy_delay_seconds, seed=cfg.seed)

    try:
        from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv
    except Exception as exc:  # pragma: no cover - depends on optional CARLA install
        raise RuntimeError(
            "CARLA backend requested but CARLA adapter is unavailable. "
            "Install CARLA Python API and verify setup (see docs/CARLA_INTEGRATION.md)."
        ) from exc

    return CARLABrakingEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        drowsy_delay_s=cfg.drowsy_delay_seconds,
        seed=cfg.seed,
        host=args.carla_host,
        port=args.carla_port,
        timeout_s=args.carla_timeout_s,
        town=args.carla_town or None,
    )


def main(env_factory: Optional[EnvFactory] = None) -> None:
    args = parse_args()
    cfg = RLConfig(episodes=args.episodes)
    set_global_seed(cfg.seed)

    env = env_factory(cfg) if env_factory is not None else default_env_factory(cfg, args)
    agent = DDDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, cfg=cfg, device=args.device)

    rewards, collisions = [], []

    try:
        for ep in range(cfg.episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_collision = 0

            while not done:
                action = agent.select_action(obs, episode=ep)
                next_obs, reward, done, info = env.step(action)
                agent.push_transition(obs, action, reward, next_obs, done)
                agent.train_step()

                obs = next_obs
                ep_reward += reward
                ep_collision += int(info.get("collision", False))

            rewards.append(ep_reward)
            collisions.append(ep_collision)

            if (ep + 1) % 20 == 0:
                avg = np.mean(rewards[-20:])
                col = np.sum(collisions[-20:])
                print(f"Episode {ep+1:4d} | avg_reward(last20)={avg:8.2f} | collisions(last20)={col}")
    finally:
        if hasattr(env, "close"):
            env.close()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_rewards.npy", np.array(rewards, dtype=np.float32))
    np.save(out / "train_collisions.npy", np.array(collisions, dtype=np.int32))
    agent.save(out)

    print(f"Training completed. Saved artifacts to {out}")


if __name__ == "__main__":
    main()
