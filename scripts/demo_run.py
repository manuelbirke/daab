#!/usr/bin/env python3
from __future__ import annotations

import argparse

from driver_monitoring_rl.config import RLConfig
from driver_monitoring_rl.rl.agent import DDDQNAgent
from driver_monitoring_rl.rl.environment import DrowsyBrakingEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one interactive-ish demo episode")
    p.add_argument("--model-path", type=str, default="artifacts/rl/dddqn_policy.pt")
    p.add_argument("--drowsy", type=int, choices=[0, 1], default=1)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RLConfig()
    env = DrowsyBrakingEnv(dt=cfg.dt, max_steps=cfg.max_steps, drowsy_delay_s=cfg.drowsy_delay_seconds, seed=cfg.seed + 3)
    agent = DDDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, cfg=cfg, device=args.device)

    try:
        agent.load(args.model_path)
        agent.epsilon = 0.0
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print("Model not found; running random/guided policy demo.")

    obs = env.reset(drowsy=args.drowsy)
    done = False
    t = 0
    total = 0.0

    while not done:
        a = agent.select_action(obs, episode=10_000)
        obs, r, done, info = env.step(a)
        total += r
        t += 1
        if t % 20 == 0 or done:
            print(
                f"step={t:03d} v_ego={obs[0]:5.2f} d_rel={obs[2]:6.2f} v_rel={obs[3]:6.2f} "
                f"drowsy={int(obs[4])} action={a} reward={r:6.2f}"
            )

    print(f"Episode done. collision={info.get('collision', False)} total_reward={total:.2f}")


if __name__ == "__main__":
    main()
