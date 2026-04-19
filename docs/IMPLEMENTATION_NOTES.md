# Implementation Notes

## Paper summary (arXiv:2604.13878)

The paper proposes a modular driver-assistance system that combines:

1. ECG-based drowsiness detection (HRV features + temporal model)
2. Traffic context from front sensing (radar-derived relative distance/velocity)
3. DD-DQN braking policy that adapts to drowsiness-induced reaction delay

Key elements implemented in this repo:

- MDP state: `[v_t, a_t, d_rel,t, v_rel,t, theta_t]`
- Action set of 6 discrete controls
- Reward components for collision avoidance, safe distance, and smooth control
- Delay injection when drowsiness is active
- RNN drowsiness classifier with capsule sequence inputs

## What is intentionally simplified

- CARLA integration is replaced by a deterministic+stochastic longitudinal simulator.
- HRV extraction uses robust SciPy baseline instead of full NeuroKit workflow.
- Public synthetic data generator is provided for reproducibility when DD-DB is unavailable.

These simplifications keep the implementation complete and runnable while preserving the method’s core logic.
