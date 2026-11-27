import json
import os
from unittest import mock

from game import Game
from tests.test_game_evaluate import HeadlessGameTestCase

LOGS = ("training_metrics.csv", "episode_rewards.csv", "eval_metrics.csv",
        "q_value_metrics.csv")


class RunIsolationTest(HeadlessGameTestCase):
    """Seeded runs in their own run directory (issue #27)."""

    BUDGET = 60

    def make_run(self, run_dir, seed=None, **env):
        values = {"RUN_DIR": run_dir, **env}
        if seed is not None:
            values["SEED"] = str(seed)
        with mock.patch.dict(os.environ, values):
            game = self.make_game()
        game.TOTAL_TRAIN_STEPS = self.BUDGET
        game.EVAL_EVERY = 1
        game.EARLY_STOP_LAPS = None
        game.SAVE_EVERY = 1
        game.agent.WARMUP_TRANSITIONS = game.agent.batch_size
        return game

    def train(self, run_dir, seed=None, **env):
        game = self.make_run(run_dir, seed, **env)
        with mock.patch("builtins.print"):
            game.run()
        return game

    def logs(self, run_dir):
        contents = {}
        for name in LOGS:
            with open(os.path.join(run_dir, "logs", name), "rb") as f:
                contents[name] = f.read()
        return contents

    def manifest(self, run_dir):
        with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_same_seed_gives_identical_logs(self):
        self.train("run_a", seed=7)
        self.train("run_b", seed=7)

        a, b = self.logs("run_a"), self.logs("run_b")
        for name in LOGS:
            self.assertEqual(a[name], b[name], name)
        # Not trivially equal: the runs trained and evaluated.
        self.assertGreater(a["training_metrics.csv"].count(b"\n"), self.BUDGET)
        self.assertGreater(a["eval_metrics.csv"].count(b"\n"), 1)

    def test_different_seeds_give_different_logs(self):
        self.train("run_a", seed=7)
        self.train("run_b", seed=8)

        self.assertNotEqual(self.logs("run_a")["training_metrics.csv"],
                            self.logs("run_b")["training_metrics.csv"])

    def test_manifest_is_written_at_start_and_finished_at_the_budget(self):
        game = self.make_run("run", seed=3)
        started = {}

        def first_episode(*args, **kwargs):
            started.update(self.manifest("run"))
            raise KeyboardInterrupt

        with mock.patch.object(game, "_run_episode", first_episode), \
                mock.patch("builtins.print"):
            game.run()
        self.assertEqual(started["status"], "running")
        self.assertEqual(started["seed"], 3)
        self.assertEqual(started["budget"], self.BUDGET)
        self.assertEqual(started["track"], "template5")
        self.assertIn("git_commit", started)

        self.train("run")

        finished = self.manifest("run")
        self.assertEqual(finished["status"], "finished")
        self.assertEqual(finished["stop_reason"], "budget")
        self.assertEqual(finished["started_at"], started["started_at"])

    def test_early_stop_finishes_the_run(self):
        game = self.make_run("run", seed=3)
        game.EARLY_STOP_LAPS = 0  # any round meets it

        with mock.patch("builtins.print"):
            game.run()

        self.assertEqual(self.manifest("run")["status"], "finished")
        self.assertEqual(self.manifest("run")["stop_reason"], "early_stop")

    def test_interrupted_run_stays_running_and_resumes_in_its_directory(self):
        game = self.make_run("run", seed=3)
        learn = game.agent.learn

        def interrupt_halfway():
            learn()
            if game.agent.step_count >= self.BUDGET // 2:
                raise KeyboardInterrupt

        with mock.patch.object(game.agent, "learn", interrupt_halfway), \
                mock.patch("builtins.print"):
            game.run()
        self.assertEqual(self.manifest("run")["status"], "running")

        # No SEED this time: the run directory remembers it.
        resumed = self.make_run("run")
        self.assertEqual(resumed.seed, 3)
        self.assertEqual(resumed.agent.step_count, self.BUDGET // 2)
        with mock.patch("builtins.print"):
            resumed.run()

        self.assertEqual(resumed.agent.step_count, self.BUDGET)
        self.assertEqual(self.manifest("run")["status"], "finished")
        self.assertTrue(os.path.exists(
            os.path.join("run", "models", "checkpoint_latest.pt")))
        self.assertTrue(os.path.exists(os.path.join("run", "models", "best.pt")))
        self.assertFalse(os.path.exists("logs"))
        self.assertFalse(os.path.exists("models"))

    def test_resuming_with_another_seed_or_track_is_refused(self):
        self.train("run", seed=3)

        with self.assertRaises(SystemExit):
            self.make_run("run", seed=4)
        with self.assertRaises(SystemExit):
            self.make_run("run", TRACK_TEMPLATE="template1")

    def test_without_a_run_directory_paths_are_unchanged(self):
        with mock.patch.dict(os.environ, {"SEED": "5"}):
            game = self.make_game()
        game.evaluate()
        game.save_model()

        self.assertTrue(os.path.exists(Game.CHECKPOINT_LATEST_PATH))
        self.assertTrue(os.path.exists(Game.BEST_MODEL_PATH))
        self.assertTrue(os.path.exists(os.path.join("logs", "eval_metrics.csv")))
        self.assertFalse(os.path.exists("manifest.json"))


if __name__ == "__main__":
    import unittest
    unittest.main()
