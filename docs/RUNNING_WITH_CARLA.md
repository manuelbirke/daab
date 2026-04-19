# Running Driver Monitoring RL with CARLA (0.9.x)

This guide is a complete, end-to-end reference for running this project with the CARLA backend.

It is written so you can go from a fresh machine to:

1. installing CARLA,
2. connecting the CARLA Python API,
3. verifying the integration,
4. training and evaluating RL policies,
5. tuning for performance and stability.

> **Version scope:** This project targets **CARLA 0.9.x** (recommended: latest stable in 0.9.x line) with explicit compatibility notes where version differences matter.

---

### Table of contents

- [1) What this integration does](#1-what-this-integration-does)
- [2) Prerequisites](#2-prerequisites)
- [3) CARLA installation (Linux)](#3-carla-installation-linux)
- [4) CARLA installation (Windows)](#4-carla-installation-windows)
- [5) Project + Python API setup](#5-project--python-api-setup)
- [6) Quick start verification](#6-quick-start-verification)
- [7) Training examples with CARLA backend](#7-training-examples-with-carla-backend)
- [8) Evaluation examples](#8-evaluation-examples)
- [9) Full configuration options reference](#9-full-configuration-options-reference)
- [10) Troubleshooting](#10-troubleshooting)
- [11) Advanced usage examples](#11-advanced-usage-examples)
- [12) Performance benchmarks (procedure + template)](#12-performance-benchmarks-procedure--template)
- [13) Recommended settings: development vs production](#13-recommended-settings-development-vs-production)
- [14) Minimal runbook (copy/paste)](#14-minimal-runbook-copypaste)

---

### 1) What this integration does

This repository includes `CARLABrakingEnv` at:

- `src/driver_monitoring_rl/rl/carla_adapter.py`

It is a **drop-in replacement** for the baseline `DrowsyBrakingEnv` and preserves:

- state shape: `[v_ego, action_idx, d_rel, v_rel, drowsy]`
- action IDs `0..5`
- standard `reset()` / `step()` training loop interface

Internally, the CARLA adapter:

- connects to a CARLA 0.9.x server,
- spawns ego and lead vehicles,
- uses front radar + DBSCAN clustering to estimate front object distance/speed,
- injects drowsiness and delayed-action dynamics compatible with the RL formulation.

---

### 2) Prerequisites

#### OS and hardware

- Linux (Ubuntu recommended) or Windows 10/11
- Dedicated GPU strongly recommended for smooth CARLA simulation
- At least 16 GB RAM recommended (32 GB preferred for heavy runs)

#### Software

- Python **3.10+** (project requirement)
- `pip` + virtual environment tooling
- CARLA simulator binary (0.9.x)
- Matching CARLA Python API package for your Python version/platform

#### Version compatibility rules (important)

1. CARLA simulator and CARLA Python API must be from the **same 0.9.x family**.
2. Python ABI tag of the CARLA wheel must match your interpreter (e.g., cp310 for Python 3.10).
3. If CARLA API import fails, treat it as a version/environment mismatch first.

---

### 3) CARLA installation (Linux)

#### Option A (recommended): official prebuilt CARLA release

1. Download a CARLA 0.9.x Linux package from CARLA releases.
2. Extract it:

```bash
mkdir -p ~/carla
cd ~/carla
tar -xvf ~/Downloads/CARLA_0.9.x.tar.gz
```

3. Launch server:

```bash
cd ~/carla/CARLA_0.9.x
./CarlaUE4.sh
```

Optional headless/offscreen (for remote boxes):

```bash
./CarlaUE4.sh -RenderOffScreen
```

#### Option B: source build

If you build CARLA from source, ensure the generated Python API package is installed into the same venv used by this project.

#### Linux sanity checks

- CARLA server reachable at `127.0.0.1:2000`
- No firewall/process conflict on CARLA port
- GPU drivers properly installed if rendering enabled

---

### 4) CARLA installation (Windows)

1. Download CARLA 0.9.x Windows release ZIP.
2. Extract, e.g. `C:\CARLA\CARLA_0.9.x`.
3. Start simulator:

```powershell
cd C:\CARLA\CARLA_0.9.x
.\CarlaUE4.exe
```

Optional lower rendering load:

```powershell
.\CarlaUE4.exe -quality-level=Low
```

If you run project code in WSL while CARLA runs on Windows host, use host IP instead of `127.0.0.1` from WSL context.

---

### 5) Project + Python API setup

From project root (`/home/ubuntu/driver_monitoring_rl` on Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Install CARLA Python API (choose one approach that matches your release):

#### Install from wheel

```bash
pip install /path/to/carla-0.9.x-<python_abi>-<platform>.whl
```

#### Install from packaged egg

```bash
pip install /path/to/carla-0.9.x-py3.x-linux-x86_64.egg
```

Quick import check:

```bash
python -c "import carla; print('CARLA API OK')"
```

If this fails, jump to [Troubleshooting](#10-troubleshooting).

---

### 6) Quick start verification

#### Step 1: start CARLA server

Linux:

```bash
~/carla/CARLA_0.9.x/CarlaUE4.sh
```

Windows (PowerShell):

```powershell
C:\CARLA\CARLA_0.9.x\CarlaUE4.exe
```

#### Step 2: run connectivity probe

```bash
python - <<'PY'
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv
ok = CARLABrakingEnv.check_connection(host="127.0.0.1", port=2000)
print("CARLA reachable:", ok)
PY
```

Expected output:

- `CARLA reachable: True`

#### Step 3: smoke test one reset/step

```bash
python - <<'PY'
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv

env = CARLABrakingEnv(host="127.0.0.1", port=2000, synchronous_mode=True)
obs = env.reset(drowsy=0)
print("reset obs:", obs)
obs2, r, done, info = env.step(4)
print("step reward:", r, "done:", done, "collision:", info.get("collision"))
env.close()
PY
```

---

### 7) Training examples with CARLA backend

The training entrypoint already supports CARLA via `--env-backend carla`.

#### Basic training

```bash
python scripts/train_rl.py \
  --env-backend carla \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --carla-timeout-s 10 \
  --episodes 300 \
  --out-dir artifacts/rl_carla
```

#### Specify map/town explicitly

```bash
python scripts/train_rl.py \
  --env-backend carla \
  --carla-town Town04 \
  --episodes 500 \
  --out-dir artifacts/rl_carla_town04
```

#### Notes on training behavior

- RL hyperparameters come from `RLConfig` in `src/driver_monitoring_rl/config.py`.
- `scripts/train_rl.py` currently exposes only selected env controls as CLI args.
- Advanced radar/DBSCAN/drowsiness controls are configurable when constructing `CARLABrakingEnv` directly (see section 9).

---

### 8) Evaluation examples

`scripts/evaluate_rl.py` evaluates with the simple simulator only.
For CARLA-based evaluation, use a Python evaluation loop.

#### A) Evaluate in simple simulator (existing script)

```bash
python scripts/evaluate_rl.py \
  --model-path artifacts/rl/dddqn_policy.pt \
  --episodes 200
```

#### B) Evaluate with CARLA backend (custom loop)

```bash
python - <<'PY'
import numpy as np
from driver_monitoring_rl.config import RLConfig
from driver_monitoring_rl.rl.agent import DDDQNAgent
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv

cfg = RLConfig()
env = CARLABrakingEnv(
    dt=cfg.dt,
    max_steps=cfg.max_steps,
    drowsy_delay_s=cfg.drowsy_delay_seconds,
    seed=cfg.seed + 1,
    host="127.0.0.1",
    port=2000,
    synchronous_mode=True,
)
agent = DDDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, cfg=cfg, device="cpu")
agent.load("artifacts/rl_carla/dddqn_policy.pt")
agent.epsilon = 0.0

episodes = 50
rewards, collisions = [], 0
for _ in range(episodes):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    info = {}
    while not done:
        a = agent.select_action(obs, episode=10_000)
        obs, r, done, info = env.step(a)
        ep_reward += r
    rewards.append(ep_reward)
    collisions += int(info.get("collision", False))

env.close()

print("CARLA evaluation summary")
print("  episodes:", episodes)
print("  collision_rate:", collisions / episodes)
print("  mean_reward:", float(np.mean(rewards)))
print("  median_reward:", float(np.median(rewards)))
PY
```

---

### 9) Full configuration options reference

This section consolidates relevant knobs from training CLI, RL config, and `CARLABrakingEnv` constructor.

#### A) `scripts/train_rl.py` CLI options

| Argument | Default | Description |
|---|---:|---|
| `--episodes` | `500` | Number of training episodes |
| `--out-dir` | `artifacts/rl` | Output directory for rewards/collisions/model |
| `--device` | `cpu` | Torch device (`cpu`, `cuda`, etc.) |
| `--env-backend` | `sim` | `sim` or `carla` |
| `--carla-host` | `127.0.0.1` | CARLA server host |
| `--carla-port` | `2000` | CARLA server port |
| `--carla-town` | `""` | Optional map/town to load |
| `--carla-timeout-s` | `10.0` | CARLA client timeout |

#### B) RL hyperparameters (`RLConfig`)

| Field | Default |
|---|---:|
| `seed` | `42` |
| `episodes` | `500` |
| `max_steps` | `300` |
| `dt` | `0.1` |
| `gamma` | `0.90` |
| `lr` | `1e-3` |
| `batch_size` | `128` |
| `replay_size` | `1_000_000` |
| `min_replay_size` | `5_000` |
| `target_update_freq` | `500` |
| `epsilon_start` | `1.0` |
| `epsilon_end` | `0.05` |
| `epsilon_decay` | `0.995` |
| `guided_exploration_episodes` | `50` |
| `guided_accel_prob` | `0.8` |
| `drowsy_delay_seconds` | `0.5` |
| `hidden_sizes` | `[128, 256, 128]` |
| `value_stream_sizes` | `[128, 64]` |
| `advantage_stream_sizes` | `[128, 64]` |

#### C) `CARLABrakingEnv(...)` constructor options

##### Simulation core

| Parameter | Default | Meaning |
|---|---:|---|
| `dt` | `0.1` | Fixed simulation timestep target |
| `max_steps` | `300` | Episode horizon |
| `drowsy_delay_s` | `0.5` | Delay horizon for drowsiness-induced action lag |
| `seed` | `42` | RNG seed |
| `synchronous_mode` | `True` | Tick world in lockstep (`world.tick`) |

##### CARLA connection / world

| Parameter | Default | Meaning |
|---|---:|---|
| `host` | `127.0.0.1` | CARLA host |
| `port` | `2000` | CARLA RPC port |
| `timeout_s` | `10.0` | Client timeout |
| `town` | `None` | Optional world/map load |
| `traffic_manager_port` | `8000` | Traffic manager port |

##### Radar sensor

| Parameter | Default | Meaning |
|---|---:|---|
| `radar_range_m` | `80.0` | Radar max range |
| `radar_hfov_deg` | `25.0` | Horizontal FOV |
| `radar_vfov_deg` | `5.0` | Vertical FOV |
| `radar_points_per_second` | `1400` | Radar sample density |
| `radar_lateral_limit_m` | `3.0` | Lateral crop for frontal filtering |
| `radar_vertical_limit_m` | `2.5` | Vertical crop for frontal filtering |

##### Radar clustering / velocity interpretation

| Parameter | Default | Meaning |
|---|---:|---|
| `dbscan_eps_m` | `1.75` | DBSCAN neighborhood radius |
| `dbscan_min_samples` | `3` | Minimum samples per cluster |
| `radar_velocity_sign` | `1.0` | Velocity sign correction (`-1.0` if needed) |

##### Scenario dynamics / drowsiness

| Parameter | Default | Meaning |
|---|---:|---|
| `lead_spawn_distance_m` | `25.0` | Initial lead vehicle spacing |
| `random_drowsy_flip_prob` | `0.015` | Probability of drowsiness state flip each step |
| `drowsy_injector` | `None` | Callback `(step, state) -> Optional[0/1]` |

---

### 10) Troubleshooting

#### A) Connection issues

**Symptom:** `Failed to connect to CARLA server at host:port` or `check_connection=False`

Checklist:

1. CARLA simulator process is actually running.
2. Host/port are correct (`127.0.0.1:2000` by default).
3. Port not blocked/in use by another process.
4. If using Docker/WSL/remote host, verify reachable network path.

Quick probe:

```bash
python - <<'PY'
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv
print(CARLABrakingEnv.check_connection(host="127.0.0.1", port=2000, timeout_s=3.0))
PY
```

#### B) Import/API mismatch issues

**Symptom:** `ImportError: No module named carla`, egg/wheel install problems, ABI mismatch.

Fix path:

1. Confirm active venv: `which python` and `pip -V`.
2. Reinstall CARLA API package matching your Python ABI.
3. Ensure CARLA simulator + Python API are same 0.9.x release family.
4. Validate with `python -c "import carla"`.

#### C) Performance issues (low FPS, unstable stepping)

Mitigations:

- Keep `synchronous_mode=True` for RL determinism.
- Reduce render quality in CARLA launcher.
- Lower radar density (`radar_points_per_second`).
- Use simpler map/town for early experiments.
- Run fewer parallel heavy processes on same machine.

#### D) Memory growth / resource leaks

**Symptom:** memory keeps increasing across episodes.

Mitigations:

- Always call `env.close()` in `finally` blocks.
- Ensure training script terminates cleanly.
- Restart CARLA server periodically for long sessions.
- Monitor actor cleanup if extending adapter internals.

#### E) Radar sign convention issues

**Symptom:** Relative velocity appears inverted.

Fix:

- Set `radar_velocity_sign=-1.0` when constructing `CARLABrakingEnv`.
- Verify by printing `v_rel` while approaching/receding lead vehicle.

---

### 11) Advanced usage examples

#### A) Custom drowsiness injection schedule

```python
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv

def custom_injector(step, state):
    # Alert first 100 steps, then force drowsy for 80 steps, then recover.
    if step < 100:
        return 0
    if step < 180:
        return 1
    return 0

env = CARLABrakingEnv(drowsy_injector=custom_injector)
```

#### B) Runtime override during episode

```python
env.set_drowsiness_state(1)         # force drowsy
# ... run steps ...
env.clear_drowsiness_override()     # return to stochastic/callback logic
```

#### C) Multi-agent/parallel training pattern

Current adapter is single-ego per environment. For multi-agent throughput, run **multiple training workers in parallel**, each connected to a distinct CARLA server/port (or distinct machine):

- Worker 1 -> `hostA:2000`
- Worker 2 -> `hostB:2000`
- Worker 3 -> `hostC:2000`

Aggregate checkpoints/metrics asynchronously (e.g., periodic policy averaging or best-checkpoint selection).

This is generally more stable than trying to force many ego agents into one instance with shared stepping.

#### D) CARLA recording / replay hooks

If you want scenario recordings for debugging:

```python
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv

env = CARLABrakingEnv()
# Access underlying client intentionally for advanced workflows:
env._client.start_recorder("carla_episode.log")
obs = env.reset()
# ... rollout ...
env._client.stop_recorder()
env.close()
```

> Note: `_client` is an internal attribute. Prefer wrapping this in your own utility if used regularly.

---

### 12) Performance benchmarks (procedure + template)

This section provides a **reproducible benchmark procedure** and a template table you can fill on your own hardware.

#### A) Benchmark goals

Compare:

1. baseline simple simulator (`DrowsyBrakingEnv`), vs
2. CARLA backend (`CARLABrakingEnv`)

on consistent RL settings.

#### B) Controlled setup

Keep constant across runs:

- same code commit
- same Python/Torch versions
- same RLConfig (except backend-specific required fields)
- same episode count (e.g., 100)
- same seed(s)

Record environment details:

- CPU model
- GPU model/driver
- RAM
- OS
- CARLA version/map/settings

#### C) Metrics to collect

- wall-clock training time
- steps/sec (effective)
- mean episode reward (last N episodes)
- collision rate
- peak RAM (process)
- peak VRAM (if applicable)

#### D) Suggested command set

Simple simulator:

```bash
/usr/bin/time -v python scripts/train_rl.py \
  --env-backend sim \
  --episodes 100 \
  --out-dir artifacts/bench_sim
```

CARLA backend:

```bash
/usr/bin/time -v python scripts/train_rl.py \
  --env-backend carla \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --episodes 100 \
  --out-dir artifacts/bench_carla
```

#### E) Template benchmark table

| Backend | Episodes | Total time (s) | Steps/sec | Mean reward (last20) | Collision rate | Peak RAM (GB) | Peak VRAM (GB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Sim | 100 |  |  |  |  |  |  |  |
| CARLA | 100 |  |  |  |  |  |  |  |

#### F) Interpretation guidance

- Expect CARLA to run slower and use more memory than the simple simulator.
- Focus on policy transfer realism vs throughput trade-off.
- For rapid iteration use sim; for fidelity checks and transfer confidence use CARLA.

---

### 13) Recommended settings: development vs production

#### Development profile (fast iteration)

- `--env-backend sim` for most RL prototyping
- Shorter runs (`50-200` episodes)
- CPU training acceptable
- CARLA only for periodic spot checks

CARLA dev spot-check suggestion:

- `synchronous_mode=True`
- `radar_points_per_second=800-1200`
- moderate map complexity

#### Production/research profile (high-fidelity validation)

- Full CARLA backend for training/eval sweeps
- Longer runs (`500+` episodes)
- Strong GPU + memory budget
- controlled seed matrix and logging
- fixed CARLA version, map, and settings per experiment batch

CARLA production suggestion:

- `synchronous_mode=True`
- tune radar + DBSCAN for your scenario distribution
- explicit actor cleanup and recorder logs for failed episodes

---

### 14) Minimal runbook (copy/paste)

```bash
# 1) Activate project env
cd /home/ubuntu/driver_monitoring_rl
source .venv/bin/activate

# 2) Verify CARLA API import
python -c "import carla; print('CARLA import OK')"

# 3) Verify CARLA server connectivity
python - <<'PY'
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv
print('CARLA reachable:', CARLABrakingEnv.check_connection('127.0.0.1', 2000))
PY

# 4) Train with CARLA backend
python scripts/train_rl.py --env-backend carla --episodes 300 --out-dir artifacts/rl_carla

# 5) Evaluate with CARLA custom loop (see section 8B)
```

You are now set up to run end-to-end CARLA-backed RL experiments for this project.
