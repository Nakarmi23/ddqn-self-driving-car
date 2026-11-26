"""Scoring for greedy evaluation episodes (issue #10)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    """Aggregate of one evaluation round of greedy episodes."""
    mean_reward: float
    mean_checkpoints: float
    min_checkpoints: int
    mean_laps: float
    min_laps: int


def laps_completed(checkpoints_passed, num_checkpoints):
    """Full laps in an episode that passed `checkpoints_passed` checkpoints.

    The spawn sits behind checkpoint 0 (see game/game_spawn.py), so the
    first pass only crosses the start line; each lap after it is
    `num_checkpoints` more passes, ending back on checkpoint 0.
    """
    return max(checkpoints_passed - 1, 0) // num_checkpoints


def summarize(episodes, num_checkpoints):
    """EvalResult for `episodes`, a non-empty list of
    (reward, checkpoints_passed) pairs."""
    rewards = [reward for reward, _ in episodes]
    checkpoints = [passed for _, passed in episodes]
    laps = [laps_completed(passed, num_checkpoints) for passed in checkpoints]
    n = len(episodes)
    return EvalResult(
        mean_reward=sum(rewards) / n,
        mean_checkpoints=sum(checkpoints) / n,
        min_checkpoints=min(checkpoints),
        mean_laps=sum(laps) / n,
        min_laps=min(laps))


def is_better(result, best):
    """Whether `result` beats `best` (None: nothing saved yet).

    Judged by checkpoints passed, the mean first and then the worst
    episode; laps follow from checkpoints. Shaped reward is ignored, since
    its scale changes whenever reward shaping is retuned.
    """
    if best is None:
        return True
    return ((result.mean_checkpoints, result.min_checkpoints)
            > (best.mean_checkpoints, best.min_checkpoints))


def meets_lap_target(result, laps):
    """Whether every evaluation episode completed at least `laps` laps."""
    return result.min_laps >= laps
