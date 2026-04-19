import numpy as np

from driver_monitoring_rl.config import CapsuleConfig, RLConfig
from driver_monitoring_rl.data.dddb_loader import DDDBLoader
from driver_monitoring_rl.data.pipeline import build_drowsiness_dataset
from driver_monitoring_rl.rl.agent import DDDQNAgent
from driver_monitoring_rl.rl.environment import DrowsyBrakingEnv


def test_data_pipeline_smoke():
    recs = DDDBLoader.generate_synthetic_recordings(n_subjects=2, duration_seconds=900, fs=128, seed=7)
    bundle = build_drowsiness_dataset(recs, capsule_config=CapsuleConfig(C=6400, N=6, M=0.72), window_seconds=120)
    assert bundle.x.ndim == 3
    assert bundle.y.ndim == 1
    assert len(bundle.x) == len(bundle.y)


def test_rl_smoke_step_and_train():
    cfg = RLConfig(episodes=2, min_replay_size=64, batch_size=32, replay_size=5000)
    env = DrowsyBrakingEnv(dt=cfg.dt, max_steps=60, seed=9)
    agent = DDDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, cfg=cfg)

    for ep in range(3):
        obs = env.reset()
        done = False
        while not done:
            a = agent.select_action(obs, ep)
            nobs, r, done, _ = env.step(a)
            assert np.isfinite(r)
            agent.push_transition(obs, a, r, nobs, done)
            agent.train_step()
            obs = nobs
