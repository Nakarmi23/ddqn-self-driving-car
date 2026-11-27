# Reward-design comparison

A DDQN car learns to drive a wall-bounded loop track. The thesis compares reward designs by how they learn and how the trained car drives.

## Designs and runs

**Reward design**:
One of the candidate reward functions under comparison (D1–D6), identified by its ID. A design fixes the form of the reward, including its terminal penalty; its constants come from a config. D1 is the baseline: the form of the progress-led reward of ADR 0001.
_Avoid_: reward, reward function (when the candidate is meant), variant

**Reward component**:
One named term of a reward design, such as progress, checkpoint bonus or crash penalty, whose value is recorded separately from the summed reward.
_Avoid_: shaping term (only some components are shaping)

**Config**:
One choice of constants for a reward design. A design has several configs during tuning and one selected config afterwards.
_Avoid_: hyperparameters (reserved for the learner's own settings, shared by all designs)

**Run**:
One training of one config of one reward design from one seed, for a fixed budget, with its own records.
_Avoid_: experiment, trial, session

**Seed**:
The number that fixes every source of randomness in a run, so the run can be reproduced.

**Tuning sweep**:
The same small, short search over configs given to every reward design before its full runs, so no design benefits from extra tuning effort.
_Avoid_: hyperparameter search, grid search

## Track geometry

**Centreline**:
The closed line through the midpoints of the checkpoints, in lap order.
_Avoid_: racing line, path

**Track progress**:
How far the car has moved along the centreline, counted cumulatively across laps.
_Avoid_: distance, checkpoint progress (D1's distance toward the next checkpoint)

**Lane offset**:
The car's signed distance from the centreline as a fraction of the local half-width of the track: 0 at the centre, about ±1 at a wall.
_Avoid_: cross-track error, lateral error

## Evaluation

**Evaluation round**:
A set of greedy episodes (no exploration, no learning) run from the fixed set of evaluation starts and scored together.
_Avoid_: test, validation

**Evaluation episode**:
A greedy episode that ends on success (three laps completed), crash, stall or the step limit, so every design faces the same distance.

**Final policy**:
The policy at the end of a run's budget; its driving outcome is the run's result. The best evaluation round is a secondary figure.
_Avoid_: best model (as the run's result)

**Evaluation start**:
One of a fixed set of slightly perturbed spawn poses, shared by every run, that makes evaluation episodes differ.
_Avoid_: random spawn

**Learning performance**:
How fast and how reliably a reward design learns, for example steps until an evaluation round first completes two laps, and the spread across seeds.
_Avoid_: performance (unqualified)

**Driving outcome**:
How a trained policy drives, measured without any reward design's own reward: lap time, termination cause, completion, speed, wall clearance and comfort.
_Avoid_: results, score, reward (as a comparison measure)

**Lap time**:
Steps taken from one start-line pass to the next.

**Termination cause**:
Why an episode ended: crash, stall, or survival to truncation.

**Crash rate**:
The fraction of evaluation episodes ending in a crash.

**Comfort**:
How smoothly the car is driven, measured from changes between consecutive actions: action changes, steering reversals and throttle reversals.
_Avoid_: jerk (undefined for discrete actions)
