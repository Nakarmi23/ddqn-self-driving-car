import csv
import os
import shutil
import tempfile
import unittest
from unittest import mock

from game import Game
from game.game_eval import EvalResult


class GameEvaluateTest(unittest.TestCase):
    """Greedy evaluation on a real headless Game, run in a scratch working
    directory so logs/ and models/ don't touch the repo's."""

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

        # Same weights, deterministic greedy policy: an equal result, which
        # must not replace the saved best, even from a fresh process.
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
        game.evaluate()
        shutil.copy(Game.BEST_MODEL_PATH, Game.CHECKPOINT_LATEST_PATH)

        resumed = self.make_game()

        self.assertEqual(resumed.episode, game.episode)
        self.assertEqual(resumed.agent.step_count, game.agent.step_count)


if __name__ == "__main__":
    unittest.main()
