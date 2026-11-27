import json
import os
import random
from unittest import mock

import numpy as np
import torch

from game import Game
from tests.test_game_evaluate import HeadlessGameTestCase

LOGS = ("training_metrics.csv", "episode_rewards.csv", "eval_metrics.csv",
        "q_value_metrics.csv")


class RunIsolationTest(HeadlessGameTestCase):
    """Seeded runs in their own run directory (issue #27)."""

    BUDGET = 60

    def setUp(self):
        super().setUp()
        # seed_everything changes process-wide state; put it back so later
        # tests don't run seeded or with deterministic kernels.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn = (torch.backends.cudnn.deterministic,
                 torch.backends.cudnn.benchmark)

        def restore():
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            torch.use_deterministic_algorithms(deterministic,
                                               warn_only=warn_only)
            (torch.backends.cudnn.deterministic,
             torch.backends.cudnn.benchmark) = cudnn

        self.addCleanup(restore)

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

    def test_finished_early_stop_run_is_not_trained_again(self):
        game = self.make_run("run", seed=3)
        game.EARLY_STOP_LAPS = 0
        with mock.patch("builtins.print"):
            game.run()
        finished = self.manifest("run")
        self.assertLess(game.agent.step_count, self.BUDGET)

        rerun = self.make_run("run")
        step_count = rerun.agent.step_count
        with mock.patch.object(rerun, "_run_episode") as run_episode,                 mock.patch.object(rerun, "evaluate") as evaluate,                 mock.patch("builtins.print"):
            rerun.run()

        run_episode.assert_not_called()
        evaluate.assert_not_called()
        self.assertEqual(rerun.agent.step_count, step_count)
        self.assertEqual(self.manifest("run"), finished)

    def test_manifest_is_marked_finished_only_after_the_final_save(self):
        game = self.make_run("run", seed=3)
        save_model = game.save_model
        status_at_save = []

        def record_status(*args, **kwargs):
            status_at_save.append(self.manifest("run")["status"])
            save_model(*args, **kwargs)

        with mock.patch.object(game, "save_model", record_status),                 mock.patch("builtins.print"):
            game.run()

        self.assertEqual(status_at_save[-1], "running")
        self.assertEqual(self.manifest("run")["status"], "finished")

    def test_a_deleted_manifest_is_rewritten_when_the_run_finishes(self):
        game = self.make_run("run", seed=3)
        run_episode = game._run_episode

        def delete_manifest(*args, **kwargs):
            if os.path.exists(os.path.join("run", "manifest.json")):
                os.remove(os.path.join("run", "manifest.json"))
            return run_episode(*args, **kwargs)

        with mock.patch.object(game, "_run_episode", delete_manifest),                 mock.patch("builtins.print"):
            game.run()

        manifest = self.manifest("run")
        self.assertEqual(manifest["status"], "finished")
        self.assertEqual(manifest["seed"], 3)
        self.assertTrue(os.path.exists(
            os.path.join("run", "models", "checkpoint_latest.pt")))

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

    def test_without_a_run_directory_another_seed_cannot_resume(self):
        with mock.patch.dict(os.environ, {"SEED": "1"}):
            self.make_game().save_model()

        for seed in ("2", None):
            env = {"SEED": seed} if seed else {}
            with mock.patch.dict(os.environ, env),                     self.assertRaises(SystemExit):
                if seed is None:
                    os.environ.pop("SEED", None)
                self.make_game()
        with mock.patch.dict(os.environ, {"SEED": "1"}):
            self.assertEqual(self.make_game().seed, 1)


if __name__ == "__main__":
    import unittest
    unittest.main()
