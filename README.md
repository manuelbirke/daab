# Drowsiness-Aware Adaptive Braking (ECG + DD-DQN)

Production-style reference implementation inspired by:
**Hafidi et al., “Drowsiness-Aware Adaptive Autonomous Braking System based on Deep Reinforcement Learning for Enhanced Road Safety” (arXiv:2604.13878, 2026).**

This repository reproduces the key components described in the paper:

- ECG-based drowsiness detection via HRV features
- Capsule-shifting temporal segmentation (CNM style: C, N, M)
- Recurrent neural network (RNN) classifier for drowsiness status
- Deep RL braking policy using **Double + Dueling DQN (DD-DQN)**
- Drowsiness-induced action delay in environment dynamics
- Training and evaluation scripts, demo run, docs, and tests

---

## Project structure

```text
src/driver_monitoring_rl/
  config.py
  data/
    dddb_loader.py           # DD-DB loader + synthetic placeholder generator
    preprocessing.py         # ECG filtering, R-peak, RR intervals
    pipeline.py              # DEW/NSRW extraction + balanced dataset build
  features/
    capsule_shift.py         # CNM config utilities + capsule segmentation
    hrv_features.py          # time/frequency-domain HRV features
  models/
    drowsiness_rnn.py        # RNN model + CV trainer
  rl/
    environment.py           # drowsy-aware longitudinal braking env
    networks.py              # dueling Q network
    replay_buffer.py
    agent.py                 # DD-DQN training logic
  utils/
    seed.py
scripts/
  train_drowsiness.py
  train_rl.py
  evaluate_rl.py
  demo_run.py
tests/
  test_smoke.py
```

---

## Methodology mapping to paper

### 1) Physiological pipeline

- ECG is band-pass filtered.
- R-peaks are detected.
- RR intervals are computed.
- HRV features are extracted (time + frequency domains):
  - mean RR, SDNN, RMSSD, pNN50, mean HR, LF power, HF power, LF/HF ratio.

### 2) DEW/NSRW windows + capsule shifting

- Drowsy Event Windows (DEW) and Normal Sinus Rhythm Windows (NSRW) are extracted.
- CNM capsule configuration `C6400 N6 M72` is used by default (`fs=128Hz`), matching the paper’s selected RNN setup.
- Each window becomes an `N`-length sequence of HRV feature vectors.
- Classes are balanced 1:1 by downsampling majority class.

### 3) Drowsiness model

- RNN classifier with 3 recurrent layers, hidden size 40, dropout 0.25.
- Cross-validation training with early stopping.

### 4) RL braking model

- State: `[v_t, a_t, d_rel_t, v_rel_t, theta_t]`
- Discrete action space (6 actions): `{brake100, brake70, brake40, brake20, neutral, accel100}`
- DD-DQN:
  - **Double-DQN** target formulation (decoupled action selection/evaluation)
  - **Dueling** network heads for value and advantage
- Reward shaping reflects paper’s finite-state reward logic and safe distance behavior.
- Drowsiness delay: when `theta_t=1`, executed action is delayed (default 0.5s).

---

## Installation

```bash
cd /home/ubuntu/driver_monitoring_rl
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### A) Train drowsiness model (synthetic placeholder data)

```bash
python scripts/train_drowsiness.py --out-dir artifacts/drowsiness
```

### B) Train DD-DQN braking agent

```bash
python scripts/train_rl.py --episodes 300 --out-dir artifacts/rl
```

### C) Evaluate trained agent

```bash
python scripts/evaluate_rl.py --model-path artifacts/rl/dddqn_policy.pt --episodes 200
```

### D) Demo run

```bash
python scripts/demo_run.py --model-path artifacts/rl/dddqn_policy.pt --drowsy 1
```

---

## Real DD-DB integration

To use real data, place per-subject CSV files in a folder and include at least:

- `ecg` column (required)
- `drowsy_event` column (optional 0/1 markers per sample)

Then run:

```bash
python scripts/train_drowsiness.py --data-dir /path/to/dddb_csv --out-dir artifacts/drowsiness_real
```

---

## Assumptions and reproducibility notes

Because arXiv papers often omit some low-level implementation details, this code follows faithful defaults aligned with the paper’s method:

- Uses selected paper configuration `C6400 N6 M72`
- Uses DD-DQN (Double + Dueling) and guided exploration in early episodes
- Uses reward design equivalent to paper’s finite-state reward logic

The environment is a controllable longitudinal simulator (not CARLA), designed as a reproducible baseline for method reproduction. You can swap in CARLA sensor streams by adapting `rl/environment.py` observation and transition logic.

---

## Testing

```bash
pytest -q
```

---

## Next extension points

- Replace synthetic ECG with real DD-DB or on-vehicle ECG stream.
- Add richer HRV features and feature-importance benchmarking (RFE/MI/RF/PI).
- Integrate CARLA for high-fidelity traffic and radar detections.
- Add model checkpointing and experiment tracking (MLflow/W&B).
