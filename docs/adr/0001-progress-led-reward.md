# 0001: Progress-led reward

Date: 2026-11-10. Issue: #5.

## Context

The per-step reward paid the agent for staying alive rather than for getting round the track:

| Term | Per step |
|---|---|
| Any accelerate action | +1 |
| Velocity `vel / max_vel` | 0 to +1 |
| Alignment to checkpoint | 0 to +1 (and mostly noise, because it mixed degrees with radians) |
| Wall-proximity baseline | +0.03 |
| Progress `prev_dist - dist`, clipped to ±1 | ±1 |
| Checkpoint passed | +100 |
| Crash or stall | −50 − \|vel\| (at most −65) |

- About +3 per step just for driving. With γ = 0.99 that stream is worth about 300, so a crash mostly cost the lost stream rather than the penalty, and circling until the stall limit earned about 3000.
- The progress clip saturated at 1 px of movement per step. The car moves up to 15 px per step and checkpoints are about 78 px apart on `template1`, so progress paid for time spent near the right direction, not distance covered. Driving slowly earned more.

## Decision

Progress is the main dense signal. Shaping is small and close to zero on average for aimless driving.

- **Progress:** `(prev_dist - dist) / max_vel`, clipped to ±1. Full speed straight at the checkpoint earns about +1 per step. Summed over a segment, it equals the distance covered divided by `max_vel`, about 5 per checkpoint on `template1`, whatever the speed.
- **Removed:** the per-action bias (+1 for accelerating) and the +0.03 wall-proximity baseline.
- **Shaping:** `SHAPING_WEIGHT * (speed_toward + alignment)` with `SHAPING_WEIGHT = 0.1`.
  - `speed_toward = (vel / max_vel) * cos(angle_diff)` is positive when driving at the checkpoint, negative when driving away, and about 0 when circling.
  - `alignment = 1 - |angle_diff| / π` is 1 when pointing at the checkpoint and 0 when pointing away.
- **Time cost:** `TIME_COST = 0.05` per step. That equals the mean alignment shaping for a random heading (0.1 × 0.5), so aimless driving nets about 0 per step instead of a survival bonus.
- **Unchanged:** the wall-proximity penalty below 20 px, `CHECKPOINT_BONUS = 100`, and the crash/stall penalty `CRASH_PENALTY_BASE + |vel|` (50 to 65).

## Consequences

- Circling until a stall now nets roughly 0 from shaping, minus the stall penalty, so it is strictly worse than reaching checkpoints.
- A checkpoint segment is worth about +5 progress plus +100 bonus. The bonus dominates, and the progress and shaping terms guide the car between checkpoints.
- Episode rewards are on a much smaller scale than before, so they can't be compared with logs from earlier runs. Weights trained on the old reward should not be resumed.
- All constants are class attributes on `Game`, so they can be tuned in one place.
