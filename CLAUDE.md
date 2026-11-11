# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-driving car simulation trained with a Double Deep Q-Network (DDQN). A pygame car drives a fixed race track, casting rays to sense walls, and a PyTorch agent learns to steer/accelerate through checkpoints via reinforcement learning. There is no manual-play mode or test suite — `main.py` starts continuous training and runs forever until the window is closed (or Ctrl+C in headless mode).

## Running

```bash
# Windows venv is already set up in .venv/
.venv\Scripts\activate
python main.py
# HEADLESS=1 (no window, no frame cap) or RENDER_EVERY=N (draw every Nth step, no cap) for fast training
```

There are no lint/test/build commands configured in this repo (no pytest, no linter config). Training progress prints to stdout per step/episode, including a `[DDQNAgent] Using device: ...` line at startup showing whether it's running on CUDA, MPS, or CPU — see README for installing a GPU-enabled `torch` build per platform.

## Architecture

**Training loop lives in `game/__init__.py` (`Game` class), not `main.py`.** `main.py` just does `Game().run()`. `Game.run()` is an infinite outer loop over episodes, with an inner loop that steps the environment until collision, `MAX_STEPS_PER_EPISODE`, or `STALL_LIMIT` (1000) steps without a checkpoint. Crash and stall timeout are failures (crash penalty, stored `done=True`); hitting `MAX_STEPS_PER_EPISODE` is truncation (no penalty, `done=False`). Look here first for anything about reward shaping, episode/epsilon bookkeeping, or save/load.

**Agent lives in `ai/ddqn_agent.py` (`DDQNAgent`, built on PyTorch).** Owns the online/target `QNetwork` (`nn.Module`) networks, epsilon-greedy action selection, the prioritized replay buffer (`ai/replay_buffer.py`, `PrioritizedReplayBuffer`: numpy ring buffers + sum-tree/max-tree, capacity `MEMORY_CAPACITY` = 100k, O(log n) add/sample/update; PER beta annealed 0.4 → 1.0 over `TOTAL_TRAIN_STEPS`), and the DDQN learning step (online net picks the action, target net evaluates it, soft target update with `TARGET_TAU` every learn step). Learning starts only after `WARMUP_TRANSITIONS` (5k) transitions are stored. `DDQNAgent.__init__` picks a device via `get_device()` (CUDA > MPS > CPU, in that order) and moves both networks to it; all tensors built in `choose_action`/`learn` are created on that device. `Game` calls `choose_action`, `store_experience`, and `learn` each step; `Game` also reaches into agent internals directly for logging/saving (`agent.online_net`, `agent.memory`, `agent.step_count`, `agent.epsilon`, `agent.device`) and uses `agent.predict_batch(net, states)` for the periodic Q-value divergence check rather than calling the network module directly.

**Game state/physics live under `game/`:**
- `game_car.py` (`Car`) — physics (drift/traction/acceleration), 12-ray raycasting for wall sensing, hitbox-edge collision and checkpoint-crossing detection via line-segment intersection. `Car.get_state()` returns the car's portion of the RL state vector (normalized ray distances + velocity + traction + drift angle).
- `game_track.py` (`Track`) — thin wrapper that loads a track definition from `game/track_templates/` and renders it.
- `game_grid.py` (`SpatialGrid`) — spatial partitioning so collision checks only test nearby walls, not the whole wall list.
- `game_checkpoint.py`, `game_wall.py`, `game_point.py`, `game_intersection.py` — small geometry/data classes used by the above.
- `game/track_templates/template1.py` — the actual track layout (walls, checkpoints, starting position/rotation). Add new tracks here.

**RL state vector** = `Car.get_state()` (rays + velocity + traction + drift angle) + `[current_checkpoint_index / num_checkpoints]` + `[min(steps_since_checkpoint / STALL_LIMIT, 1.0)]`, assembled in `Game._get_state()`. If you change the ray count or append/remove features in either `Car.get_state()` or `Game._get_state()`, the two must stay in sync since `DDQNAgent`'s input layer size is derived from `len(car.get_state()) + 2` at construction time in `Game.__init__`.

**Action space** is 9 discrete actions (0=noop, 1/2=accel/reverse, 3/4=turn left/right, 5-8=combined accel+turn), mapped in `Game._apply_action`.

**Reward shaping** is progress-led (see `docs/adr/0001-progress-led-reward.md`). `Game._run_episode` adds progress (distance closed toward the checkpoint / `max_vel`, clipped to ±1), `CHECKPOINT_BONUS` on a pass, and the crash penalty on collision/stall. `Game._compute_reward` adds small shaping terms: `SPEED_WEIGHT * speed toward checkpoint + ALIGNMENT_WEIGHT * alignment`, minus `TIME_COST` (derived as `ALIGNMENT_WEIGHT * 0.5`), plus the wall-proximity penalty. Shaping is tuned to net ~0 for aimless driving, so only progress pays. Changes to one term should be considered against the others since they're summed into a single scalar per step. `Game._angle_to_checkpoint` returns radians; the car's heading in screen space is `car.angle + π/2`.

**Persistence**: every `SAVE_EVERY` (10) episodes, `Game.save_model()` bundles everything needed to resume — online/target `state_dict`s, epsilon, step count, replay buffer arrays + priorities (via `DDQNAgent.export_state()`; `import_state()` also converts the older list-of-tuples `memory`/`priorities` format), plus episode count and top/recent reward history — into one dict and writes it with `torch.save` to `models/checkpoint_latest.pt` (always overwritten, read back on startup by `load_model()`/`DDQNAgent.import_state()`, `map_location=agent.device` so a checkpoint saved on one device loads on another). The same dict is also written to `models/history/checkpoint_ep{N:06d}.pt`; `Game._prune_history()` deletes all but the newest `KEEP_HISTORY_COUNT` (5) history files, so there's always a rollback point if a run degrades. These files are checked into the repo working tree and are expected to change on every training run — that's normal, not a sign of an unrelated diff. `models.bak/` is a manual backup snapshot, not code the agent touches.

**Logging**: `utils/csv_logger.py` (`CsvLogger`) is a small shared helper — writes a CSV header once per file, keeps the file open, and flushes every `flush_every` rows (the per-step training log uses 1000), on `flush()` (called from `save_model`) and on `close()`. `Game` owns `self.episode_log` (`logs/episode_rewards.csv`, per episode) and `self.q_value_log` (`logs/q_value_metrics.csv`, every 50 episodes from a sample of replay memory, to track online-vs-target Q divergence); `DDQNAgent` owns `self.training_log` (`logs/training_metrics.csv`, per learning step). Add a new metric by instantiating another `CsvLogger` rather than hand-rolling `open(...).write`.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (Nakarmi23/ddqn-self-driving-car) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` plus `docs/adr/` (both created lazily). See `docs/agents/domain.md`.
