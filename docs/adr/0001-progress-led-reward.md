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
- The progress clip saturated at 1 px of movement per step. The car moves up to 15 px per step and checkpoints are about 67 px apart on the default track `template5` (78 px on `template1`), so progress paid for time spent near the right direction, not distance covered. Driving slowly earned more.

## Decision

Progress is the main dense signal. Shaping is small and close to zero on average for aimless driving.

- **Progress:** `(prev_dist - dist) / max_vel`, clipped to ±1. Full speed straight at the checkpoint earns about +1 per step. Summed over a segment, it equals the distance covered divided by `max_vel`, about 4.5 per checkpoint on `template5`, whatever the speed.
- **Removed:** the per-action bias (+1 for accelerating) and the +0.03 wall-proximity baseline.
- **Shaping:** `SPEED_WEIGHT * speed_toward + ALIGNMENT_WEIGHT * alignment`, with `SPEED_WEIGHT = 0.1` and `ALIGNMENT_WEIGHT = 0.04`.
  - `speed_toward = (vel / max_vel) * cos(angle_diff)` is positive when driving at the checkpoint, negative when driving away, and about 0 when circling.
  - `alignment = 1 - |angle_diff| / π` is 1 when pointing at the checkpoint and 0 when pointing away.
- **Time cost:** `TIME_COST = ALIGNMENT_WEIGHT × 0.5 = 0.02` per step. That equals the mean alignment shaping for a random heading, so aimless driving nets about 0 per step instead of a survival bonus.
- **Unchanged:** the wall-proximity penalty below 20 px, `CHECKPOINT_BONUS = 100`, and the crash/stall penalty `CRASH_PENALTY_BASE + |vel|` (50 to 65).

## Consequences

- Circling until a stall now nets roughly 0 from shaping, minus the stall penalty, so it is strictly worse than reaching checkpoints.
- A stall costs about −70 (1000 × 0.02 time cost plus −50), close to a crash (−50 to −65). A first version used `SHAPING_WEIGHT = 0.1` for both terms and `TIME_COST = 0.05`. In a 10-minute headless run that made a stall cost about −100, twice a crash, so once the car was stuck, crashing was the cheaper way out. The alignment weight and time cost were lowered together to keep the two outcomes comparable while aimless driving still nets about 0.
- A checkpoint segment is worth about +4.5 progress plus +100 bonus. The bonus dominates, and the progress and shaping terms guide the car between checkpoints.
- Episode rewards are on a much smaller scale than before, so they can't be compared with logs from earlier runs. Weights trained on the old reward should not be resumed.
- All constants are class attributes on `Game`, so they can be tuned in one place.
