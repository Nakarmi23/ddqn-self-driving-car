# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-driving car simulation trained with a Double Deep Q-Network (DDQN). A pygame car drives a fixed race track, casting rays to sense walls, and a Keras/TensorFlow agent learns to steer/accelerate through checkpoints via reinforcement learning. There is no manual-play mode or test suite — `main.py` starts continuous training and runs forever until the window is closed.

## Running

```bash
# Windows venv is already set up in .venv/
.venv\Scripts\activate
python main.py
```

There are no lint/test/build commands configured in this repo (no pytest, no linter config). Training progress prints to stdout per step/episode.

## Architecture

**Training loop lives in `game/__init__.py` (`Game` class), not `main.py`.** `main.py` just does `Game().run()`. `Game.run()` is an infinite outer loop over episodes, with an inner loop that steps the environment until collision, `MAX_STEPS_PER_EPISODE`, or 1000 steps without a checkpoint. Look here first for anything about reward shaping, episode/epsilon bookkeeping, or save/load.

**Agent lives in `ai/ddqn_agent.py` (`DDQNAgent`).** Owns the online/target Keras networks, epsilon-greedy action selection, the (unbounded-list-based) prioritized replay buffer, and the DDQN learning step (online net picks the action, target net evaluates it). `Game` calls `choose_action`, `store_experience`, and `learn` each step; `Game` also reaches into agent internals directly for logging/saving (`agent.online_net`, `agent.memory`, `agent.step_count`, `agent.epsilon`).

**Game state/physics live under `game/`:**
- `game_car.py` (`Car`) — physics (drift/traction/acceleration), 12-ray raycasting for wall sensing, hitbox-edge collision and checkpoint-crossing detection via line-segment intersection. `Car.get_state()` returns the car's portion of the RL state vector (normalized ray distances + velocity + traction + drift angle).
- `game_track.py` (`Track`) — thin wrapper that loads a track definition from `game/track_templates/` and renders it.
- `game_grid.py` (`SpatialGrid`) — spatial partitioning so collision checks only test nearby walls, not the whole wall list.
- `game_checkpoint.py`, `game_wall.py`, `game_point.py`, `game_intersection.py` — small geometry/data classes used by the above.
- `game/track_templates/template1.py` — the actual track layout (walls, checkpoints, starting position/rotation). Add new tracks here.

**RL state vector** = `Car.get_state()` (rays + velocity + traction + drift angle) + `[current_checkpoint_index / num_checkpoints]` + `[steps_since_checkpoint / 250]`, assembled in `Game._get_state()`. If you change the ray count or append/remove features in either `Car.get_state()` or `Game._get_state()`, the two must stay in sync since `DDQNAgent`'s input layer size is derived from `len(car.get_state()) + 2` at construction time in `Game.__init__`.

**Action space** is 9 discrete actions (0=noop, 1/2=accel/reverse, 3/4=turn left/right, 5-8=combined accel+turn), mapped in `Game._apply_action`.

**Reward shaping** is spread across `Game._compute_reward` (action bias, velocity reward, wall-proximity penalty, checkpoint-alignment reward) and `Game.run` (progress-toward-checkpoint delta, +100 on checkpoint pass, crash penalty on collision/timeout). Changes to one should be considered against the others since they're summed into a single scalar per step.

**Persistence**: every `SAVE_EVERY` (10) episodes, `Game.save_model()`/`load_model()` write/read the online and target networks, epsilon, episode count, replay memory, top/recent reward history, and step count under `models/`. These files (`.keras`, `.npy`, `.pickle`) are checked into the repo working tree and are expected to change on every training run — that's normal, not a sign of an unrelated diff. `models.bak/` is a manual backup snapshot, not code the agent touches.

**Logging**: `logs/episode_rewards.csv` (per episode) and `logs/training_metrics.csv` (per learning step) are appended to directly by `Game` and `DDQNAgent` respectively (plain `open(...).write`, not a logging library). `logs/q_value_metrics.csv` is written every 50 episodes from a sample of replay memory to track online-vs-target Q divergence.
