import csv
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

from game import Game
from game.game_eval import EvalResult


class HeadlessGameTestCase(unittest.TestCase):
    """Real headless Games, run in a scratch working directory so logs/ and
    models/ don't touch the repo's."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        env = mock.patch.dict(os.environ, {"HEADLESS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self._games = []

    def tearDown(self):
        for game in self._games:
            game.close()
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def make_game(self):
        game = Game()
        # Short episodes keep the test fast; an untrained greedy policy
        # would otherwise drive until the stall limit.
        game.MAX_STEPS_PER_EPISODE = 40
        game.EVAL_EPISODES = 2
        self._games.append(game)
        return game

    def eval_rows(self):
        with open(os.path.join("logs", "eval_metrics.csv"), newline="") as f:
            return list(csv.DictReader(f))


class GameEvaluateTest(HeadlessGameTestCase):
    """Greedy evaluation."""

    def test_evaluation_does_not_train(self):
        game = self.make_game()
        epsilon = game.agent.epsilon
        step_count = game.agent.step_count
        episode = game.episode

        result = game.evaluate()

        self.assertIsInstance(result, EvalResult)
        self.assertEqual(len(game.agent.memory), 0)
        self.assertEqual(game.agent.step_count, step_count)
        self.assertEqual(game.agent.epsilon, epsilon)
        self.assertEqual(game.episode, episode)

    def test_evaluation_is_logged(self):
        game = self.make_game()

        result = game.evaluate()
        game.evaluate()

        rows = self.eval_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["mean_checkpoints"]),
                         result.mean_checkpoints)
        self.assertEqual(int(rows[0]["min_laps"]), result.min_laps)

    def test_first_evaluation_is_saved_as_best(self):
        game = self.make_game()

        game.evaluate()

        self.assertTrue(os.path.exists(Game.BEST_MODEL_PATH))

    def test_best_model_is_kept_across_restarts_unless_beaten(self):
        game = self.make_game()
        game.evaluate()
        saved = os.stat(Game.BEST_MODEL_PATH).st_mtime_ns

        # Same weights (resumed from best.pt), deterministic greedy policy:
        # an equal result, which must not replace the saved best, even from
        # a fresh process.
        shutil.copy(Game.BEST_MODEL_PATH, Game.CHECKPOINT_LATEST_PATH)
        restarted = self.make_game()
        restarted.evaluate()

        self.assertEqual(os.stat(Game.BEST_MODEL_PATH).st_mtime_ns, saved)

    def test_best_model_from_another_track_is_not_the_bar(self):
        self.make_game().evaluate()
        saved = os.stat(Game.BEST_MODEL_PATH).st_mtime_ns

        with mock.patch.dict(os.environ, {"TRACK_TEMPLATE": "template1"}):
            other_track = self.make_game()
        other_track.evaluate()

        self.assertNotEqual(os.stat(Game.BEST_MODEL_PATH).st_mtime_ns, saved)

    def test_training_can_resume_from_the_best_model(self):
        game = self.make_game()
        game.episode = 123
        game.agent.step_count = 4567
        game.evaluate()
        shutil.copy(Game.BEST_MODEL_PATH, Game.CHECKPOINT_LATEST_PATH)

        resumed = self.make_game()

        self.assertEqual(resumed.episode, 123)
        self.assertEqual(resumed.agent.step_count, 4567)

    def test_best_model_from_another_checkpoint_layout_is_not_the_bar(self):
        game = self.make_game()
        game.evaluate()
        best = torch.load(Game.BEST_MODEL_PATH, weights_only=False)
        best["num_checkpoints"] += 1
        torch.save(best, Game.BEST_MODEL_PATH)

        self.assertIsNone(self.make_game().best_eval)

    def test_unreadable_best_eval_is_ignored(self):
        game = self.make_game()
        game.evaluate()
        best = torch.load(Game.BEST_MODEL_PATH, weights_only=False)
        best["eval"]["retired_field"] = 0
        torch.save(best, Game.BEST_MODEL_PATH)

        with mock.patch("builtins.print"):
            self.assertIsNone(self.make_game().best_eval)


class GameRunTest(HeadlessGameTestCase):
    """run() end to end with a tiny budget and warm-up."""

    BUDGET = 60

    def make_training_game(self):
        game = self.make_game()
        game.TOTAL_TRAIN_STEPS = self.BUDGET
        game.EVAL_EVERY = 1
        game.EARLY_STOP_LAPS = None
        game.agent.WARMUP_TRANSITIONS = game.agent.batch_size
        return game

    def episode_rows(self):
        with open(os.path.join("logs", "episode_rewards.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_run_stops_at_the_budget_and_scores_the_final_policy_once(self):
        game = self.make_training_game()

        with mock.patch("builtins.print"):
            game.run()

        self.assertEqual(game.agent.step_count, self.BUDGET)
        steps = [int(row["step"]) for row in self.eval_rows()]
        self.assertEqual(steps.count(self.BUDGET), 1)
        # The episode the budget cut short is not logged.
        logged = sum(int(row["steps"]) for row in self.episode_rows())
        warmup = game.agent.WARMUP_TRANSITIONS
        self.assertLess(logged, warmup + self.BUDGET)

    def test_run_with_spent_budget_does_nothing(self):
        game = self.make_training_game()
        with mock.patch("builtins.print"):
            game.run()
        rows = len(self.eval_rows())

        restarted = self.make_training_game()
        with mock.patch("builtins.print"):
            restarted.run()

        self.assertEqual(len(self.eval_rows()), rows)
        self.assertEqual(restarted.agent.step_count, self.BUDGET)

    def test_run_stops_early_once_the_lap_target_is_met(self):
        game = self.make_training_game()
        game.EARLY_STOP_LAPS = 0  # any round meets it

        with mock.patch("builtins.print"):
            game.run()

        self.assertEqual(len(self.eval_rows()), 1)
        self.assertLess(game.agent.step_count, self.BUDGET)


if __name__ == "__main__":
    unittest.main()
