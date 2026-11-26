import unittest

from game.game_eval import (EvalResult, is_better, laps_completed, meets_lap_target,
                            summarize)


class LapsCompletedTest(unittest.TestCase):
    # The spawn sits behind checkpoint 0, so the first pass is the start
    # line, and a lap is every checkpoint after it including cp0 again.
    def test_crossing_the_start_line_is_not_a_lap(self):
        self.assertEqual(laps_completed(1, num_checkpoints=8), 0)

    def test_lap_counts_once_back_over_the_start_line(self):
        self.assertEqual(laps_completed(8, num_checkpoints=8), 0)
        self.assertEqual(laps_completed(9, num_checkpoints=8), 1)
        self.assertEqual(laps_completed(17, num_checkpoints=8), 2)

    def test_no_checkpoints_is_no_laps(self):
        self.assertEqual(laps_completed(0, num_checkpoints=8), 0)


class SummarizeTest(unittest.TestCase):
    def test_summary_of_greedy_episodes(self):
        # (reward, checkpoints passed) per episode, on an 8-checkpoint track.
        result = summarize([(120.0, 9), (40.0, 3), (200.0, 17)],
                           num_checkpoints=8)

        self.assertEqual(result, EvalResult(
            mean_reward=120.0, mean_checkpoints=29 / 3, min_checkpoints=3,
            mean_laps=1.0, min_laps=0))


def result(mean_checkpoints, min_checkpoints, mean_reward=0.0):
    return EvalResult(mean_reward=mean_reward, mean_checkpoints=mean_checkpoints,
                      min_checkpoints=min_checkpoints, mean_laps=0.0, min_laps=0)


class IsBetterTest(unittest.TestCase):
    def test_first_result_is_always_the_best_so_far(self):
        self.assertTrue(is_better(result(0, 0), None))

    def test_more_checkpoints_on_average_wins(self):
        self.assertTrue(is_better(result(6.0, 2), result(5.0, 4)))
        self.assertFalse(is_better(result(5.0, 4), result(6.0, 2)))

    def test_equal_average_falls_back_to_the_worst_episode(self):
        self.assertTrue(is_better(result(5.0, 4), result(5.0, 3)))

    def test_shaped_reward_does_not_count(self):
        self.assertFalse(is_better(result(5.0, 3, mean_reward=999.0),
                                   result(5.0, 3, mean_reward=1.0)))


class MeetsLapTargetTest(unittest.TestCase):
    def test_every_episode_must_reach_the_lap_target(self):
        all_two = summarize([(0.0, 17)] * 5, num_checkpoints=8)
        one_short = summarize([(0.0, 17)] * 4 + [(0.0, 16)], num_checkpoints=8)

        self.assertTrue(meets_lap_target(all_two, laps=2))
        self.assertFalse(meets_lap_target(one_short, laps=2))


if __name__ == "__main__":
    unittest.main()
