import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from game.game_car import Car  # noqa: E402
from game.game_checkpoint import Checkpoint  # noqa: E402
from game.game_point import Point  # noqa: E402
from game.game_track import Track  # noqa: E402
from scripts.check_spawns import template_names  # noqa: E402


class CrossedForwardTest(unittest.TestCase):
    """Checkpoint.crossed_forward: a vertical line at x = 10, y in [0, 20],
    with the previous checkpoint on the x < 10 side."""

    def setUp(self):
        self.cp = Checkpoint(10, 0, 10, 20)
        self.behind = Point(0, 10)

    def crossed(self, start, end):
        return self.cp.crossed_forward(Point(*start), Point(*end), self.behind)

    def test_forward_crossing_counts(self):
        self.assertTrue(self.crossed((5, 10), (15, 10)))

    def test_wrong_way_crossing_does_not_count(self):
        self.assertFalse(self.crossed((15, 10), (5, 10)))

    def test_staying_on_one_side_does_not_count(self):
        self.assertFalse(self.crossed((2, 10), (8, 10)))
        self.assertFalse(self.crossed((12, 10), (18, 10)))

    def test_landing_on_the_line_counts_once(self):
        self.assertTrue(self.crossed((5, 10), (10, 10)))
        self.assertFalse(self.crossed((10, 10), (15, 10)))

    def test_leaving_the_line_backwards_does_not_count(self):
        self.assertFalse(self.crossed((10, 10), (5, 10)))

    def test_moving_along_the_line_does_not_count(self):
        self.assertFalse(self.crossed((10, 2), (10, 18)))

    def test_crossing_past_the_end_of_the_line_does_not_count(self):
        self.assertFalse(self.crossed((5, 25), (15, 25)))
        self.assertFalse(self.crossed((5, -1), (15, -1)))

    def test_crossing_at_an_endpoint_counts(self):
        self.assertTrue(self.crossed((5, 20), (15, 20)))


class TrackTemplateCrossingTest(unittest.TestCase):
    """Real cars on every track template."""

    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.surface = pygame.display.set_mode((1000, 600))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def spawn(self, name):
        track = Track(self.surface, name)
        car = Car(*track.starting_position, track, track.starting_rotation)
        return track, car

    def crossings(self, track, car):
        """Indices of the checkpoints the car's last update() crossed forward."""
        cps = track.checkpoints
        return [i for i, cp in enumerate(cps)
                if car.check_checkpoint_crossing(cp, cps[i - 1].center)]

    def test_driving_straight_from_spawn_passes_checkpoint_0_first(self):
        for name in template_names():
            with self.subTest(track=name):
                track, car = self.spawn(name)
                for _ in range(200):
                    car.accelerate(1)
                    car.update()
                    self.assertFalse(car.check_collisions(track.walls_grid),
                                     "hit a wall before checkpoint 0")
                    crossed = self.crossings(track, car)
                    if crossed:
                        self.assertEqual(crossed, [0])
                        break
                else:
                    self.fail("never crossed checkpoint 0")

    def test_reversing_from_spawn_passes_nothing(self):
        # Backing away from checkpoint 0 crosses no checkpoint forward, not
        # even the last one the car backs over or into.
        for name in template_names():
            with self.subTest(track=name):
                track, car = self.spawn(name)
                for _ in range(60):
                    car.accelerate(-1)
                    car.update()
                    self.assertEqual(self.crossings(track, car), [])
                    if car.check_collisions(track.walls_grid):
                        break

    def test_standing_on_checkpoint_0_is_not_a_pass(self):
        # The #8 bug: a hitbox overlapping the line used to count the pass
        # on step 1 without the car moving.
        for name in template_names():
            with self.subTest(track=name):
                track, car = self.spawn(name)
                cp0 = track.checkpoints[0].center
                car.x, car.y = cp0.x, cp0.y
                car._update_hitbox_points()
                car.prev_front = car.front
                car.update()  # vel == 0: the car stays put
                self.assertEqual(self.crossings(track, car), [])


if __name__ == "__main__":
    unittest.main()
