# CARLA Integration Guide

This guide explains how to run the RL braking environment with CARLA using:

- `driver_monitoring_rl.rl.carla_adapter.CARLABrakingEnv`

The adapter is designed as a **drop-in replacement** for `DrowsyBrakingEnv` and keeps the same:

- state shape: `[v_ego, action_idx, d_rel, v_rel, drowsy]`
- action IDs: `0..5`
- `reset(...)` / `step(...)` interface

---

## 1) Install CARLA (0.9.x)

> You need both the CARLA simulator server and Python API for the same version.

### Option A: Prebuilt CARLA package

1. Download CARLA 0.9.x from the official CARLA releases.
2. Extract and run the simulator server (example):

```bash
./CarlaUE4.sh
```

3. Install the matching Python API package (wheel/egg) into your environment.

Example (adjust path/version for your install):

```bash
pip install /path/to/carla-0.9.x-cp38-cp38-linux_x86_64.whl
```

### Option B: Debian package / custom build

If you build CARLA from source, ensure the generated Python API package is installed into the same Python environment where this project runs.

---

## 2) Project setup

From project root:

```bash
cd /home/ubuntu/driver_monitoring_rl
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

---

## 3) Basic usage (drop-in replacement)

Replace:

```python
from driver_monitoring_rl.rl.environment import DrowsyBrakingEnv
env = DrowsyBrakingEnv(...)
```

With:

```python
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv

env = CARLABrakingEnv(
    dt=0.1,
    max_steps=300,
    drowsy_delay_s=0.5,
    host="127.0.0.1",
    port=2000,
    synchronous_mode=True,
)
```

Then use exactly the same training/evaluation loop:

```python
obs = env.reset(drowsy=0)
done = False
while not done:
    action = 4  # example: neutral
    obs, reward, done, info = env.step(action)

env.close()
```

---

## 4) Radar + DBSCAN behavior

`CARLABrakingEnv` attaches a front-facing radar to ego vehicle and:

1. Converts detections from polar to Cartesian coordinates.
2. Filters frontal detections (`x > 0`) and lateral/vertical bounds.
3. Clusters points with DBSCAN.
4. Selects the nearest cluster as lead object estimate.
5. Maps cluster to:
   - `d_rel`: estimated lead distance
   - `v_rel`: estimated relative/closing speed

If clustering fails (sparse/noisy frame), it falls back to nearest valid point or last known estimate.

---

## 5) Drowsiness injection options

The adapter supports multiple ways to control drowsiness state:

1. **At reset**
   - `env.reset(drowsy=0)` or `env.reset(drowsy=1)`
2. **Runtime override**
   - `env.set_drowsiness_state(1)`
   - `env.clear_drowsiness_override()`
3. **Custom callback**
   - `drowsy_injector(step, state) -> Optional[int]`

Example callback:

```python
def injector(step, state):
    if step > 50:
        return 1
    return 0

env = CARLABrakingEnv(drowsy_injector=injector)
```

---

## 6) Configuration reference

Main parameters in `CARLABrakingEnv`:

- Simulation:
  - `dt`, `max_steps`, `drowsy_delay_s`, `synchronous_mode`
- CARLA connection:
  - `host`, `port`, `timeout_s`, `town`, `traffic_manager_port`
- Radar:
  - `radar_range_m`, `radar_hfov_deg`, `radar_vfov_deg`, `radar_points_per_second`
  - `radar_lateral_limit_m`, `radar_vertical_limit_m`
- Clustering:
  - `dbscan_eps_m`, `dbscan_min_samples`
- Relative velocity sign tuning:
  - `radar_velocity_sign` (set to `-1.0` if your CARLA build’s radar sign convention appears inverted)
- Scenario behavior:
  - `lead_spawn_distance_m`, `random_drowsy_flip_prob`

---

## 7) Connection checks and error handling

Quick check before training:

```python
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv

ok = CARLABrakingEnv.check_connection(host="127.0.0.1", port=2000)
print("CARLA reachable:", ok)
```

If connection/setup fails, adapter raises informative exceptions:

- `CarlaConnectionError` for server/world setup issues
- `CarlaAdapterError` for runtime/setup misuse

---

## 8) Practical notes

- Keep CARLA and Python API versions aligned (same 0.9.x build family).
- Prefer synchronous mode for stable RL stepping.
- Always call `env.close()` to destroy actors and restore CARLA world settings.
- Start with lower radar noise and tighter DBSCAN settings for easier early training.

---

## 9) Minimal training integration pattern

```python
from driver_monitoring_rl.rl.agent import DddqnAgent
from driver_monitoring_rl.rl.carla_adapter import CARLABrakingEnv

env = CARLABrakingEnv(dt=0.1, max_steps=300, drowsy_delay_s=0.5)
agent = DddqnAgent(state_dim=env.state_dim, action_dim=env.action_dim)

for _ in range(10):
    s = env.reset()
    done = False
    while not done:
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        agent.push_transition(s, a, r, s2, done)
        agent.train_step()
        s = s2

env.close()
```

This preserves your existing DD-DQN interface while swapping simulation backend.
