import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from game.game_car import Car  # noqa: E402
from game.game_checkpoint import Checkpoint, count_passes  # noqa: E402
from game.game_point import Point  # noqa: E402
from game.game_spawn import spawn_problems  # noqa: E402
from game.game_track import Track  # noqa: E402
from game.game_wall import Wall  # noqa: E402
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


class CountPassesTest(unittest.TestCase):
    """count_passes: a fan of lines sharing the end (10, 0), like template1's
    hairpins, swept clockwise from x < 10 round to y > 0."""

    def setUp(self):
        # Previous checkpoint far behind, then three lines through (10, 0).
        self.cps = [Checkpoint(-50, -10, -50, 10),
                    Checkpoint(10, 0, 10, -20),
                    Checkpoint(10, 0, 30, -20),
                    Checkpoint(10, 0, 30, 0)]

    def passes(self, index, start, end):
        return count_passes(self.cps, index, Point(*start), Point(*end))

    def test_one_move_across_two_lines_counts_both(self):
        # Crosses x = 10 and the diagonal, stops short of y = 0.
        self.assertEqual(self.passes(1, (5, -10), (25, -2)), 2)

    def test_counting_stops_at_the_first_line_not_crossed(self):
        self.assertEqual(self.passes(1, (5, -10), (15, -12)), 1)
        self.assertEqual(self.passes(1, (12, -18), (15, -12)), 0)

    def test_landing_on_the_shared_end_counts_every_line_through_it(self):
        # The front point stops exactly on (10, 0), which lies on lines 1-3.
        self.assertEqual(self.passes(1, (5, -5), (10, 0)), 3)

    def test_wraps_round_to_checkpoint_0(self):
        # Moving left crosses the last checkpoint (x = -20), then cp0 (x = -40).
        cps = [Checkpoint(-40, -10, -40, 10), Checkpoint(-100, 100, 100, 100),
               Checkpoint(-20, -10, -20, 10)]
        self.assertEqual(
            count_passes(cps, 2, Point(-15, 0), Point(-45, 0)), 2)


class ExtendToWallsTest(unittest.TestCase):
    """A corridor between walls y = 0 and y = 100; the checkpoint at x = 50
    stops 30 px short of the bottom wall."""

    def setUp(self):
        self.walls = [Wall(0, 0, 200, 0), Wall(0, 100, 200, 100)]
        self.behind = Point(0, 50)

    def test_gap_between_end_and_wall_is_closed(self):
        cp = Checkpoint(50, 0, 50, 70)
        self.assertFalse(cp.crossed_forward(Point(45, 85), Point(55, 85),
                                            self.behind))
        cp.extend_to_walls(self.walls)
        self.assertEqual((cp.cross_x2, cp.cross_y2), (50, 100))
        self.assertTrue(cp.crossed_forward(Point(45, 85), Point(55, 85),
                                           self.behind))

    def test_end_already_beyond_a_wall_is_left_alone(self):
        # The top end pokes through y = 0; extending it would reach into
        # whatever lies beyond that wall.
        walls = self.walls + [Wall(0, -100, 200, -100)]
        cp = Checkpoint(50, -20, 50, 100)
        cp.extend_to_walls(walls)
        self.assertEqual((cp.cross_x1, cp.cross_y1), (50, -20))

    def test_drawn_line_is_unchanged(self):
        cp = Checkpoint(50, 0, 50, 70)
        cp.extend_to_walls(self.walls)
        self.assertEqual((cp.x1, cp.y1, cp.x2, cp.y2), (50, 0, 50, 70))


class MinimumCheckpointsTest(unittest.TestCase):
    def test_fewer_than_three_checkpoints_is_rejected(self):
        class FakeTrack:
            walls = ()
        for n in (1, 2):
            with self.subTest(checkpoints=n):
                track = FakeTrack()
                track.checkpoints = [Checkpoint(i * 10, 0, i * 10, 10)
                                     for i in range(n)]
                problems = spawn_problems(None, track)
                self.assertEqual(len(problems), 1)
                self.assertIn("at least 3", problems[0])


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
