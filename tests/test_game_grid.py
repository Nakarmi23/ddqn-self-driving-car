import math
import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from game.game_car import Car  # noqa: E402
from game.game_grid import QUERY_PAD, SpatialGrid  # noqa: E402
from game.game_track import Track  # noqa: E402
from game.game_wall import Wall  # noqa: E402
from scripts.check_spawns import template_names  # noqa: E402


def walls_in_cell(grid, cx, cy):
    """Walls a query confined to one cell's interior gets back."""
    s = grid.cell_size
    return grid.get_walls_in_rect(cx * s + 0.01, cy * s + 0.01,
                                  (cx + 1) * s - 0.01, (cy + 1) * s - 0.01)


def segment_meets_closed_box(x1, y1, x2, y2, min_x, min_y, max_x, max_y):
    """Reference: whether the segment has a point in the closed box."""
    t0, t1 = 0.0, 1.0
    dx, dy = x2 - x1, y2 - y1
    for p, q in ((-dx, x1 - min_x), (dx, max_x - x1),
                 (-dy, y1 - min_y), (dy, max_y - y1)):
        if p == 0:
            if q < 0:
                return False
        elif p < 0:
            t0 = max(t0, q / p)
        else:
            t1 = min(t1, q / p)
        if t0 > t1:
            return False
    return True


class SpatialGridTest(unittest.TestCase):
    def test_wall_found_in_a_cell_it_only_grazes(self):
        grid = SpatialGrid(cell_size=10)
        # y = 1 + 11x/12 enters cell (0, 1) at x ~ 9.82 and leaves it at x = 10.
        wall = Wall(0, 1, 12, 12)
        grid.add_wall(wall)

        self.assertIn(wall, walls_in_cell(grid, 0, 1))

    def test_wall_on_a_cell_border_is_found_from_both_sides(self):
        grid = SpatialGrid(cell_size=10)
        wall = Wall(10, 2, 10, 8)
        grid.add_wall(wall)

        # Boxes reaching the border from the left and from the right.
        self.assertIn(wall, grid.get_walls_in_rect(0.01, 0.01, 10, 9.99))
        self.assertIn(wall, grid.get_walls_in_rect(10, 0.01, 19.99, 9.99))

    def test_wall_on_a_cell_border_not_found_short_of_it(self):
        # #23: the box x in [0.01, 9.99] holds no point of the wall at x = 10.
        grid = SpatialGrid(cell_size=10)
        wall = Wall(10, 2, 10, 8)
        grid.add_wall(wall)

        self.assertEqual(walls_in_cell(grid, 0, 0), [])
        self.assertEqual(set(grid.grid), {(1, 0)})

    def test_wall_on_a_cell_border_found_by_a_box_a_rounding_error_short(self):
        # A ray end computed as 79.99999999999999 instead of 80 must still
        # see the wall on x = 80, which Car's closed hit test counts.
        grid = SpatialGrid(cell_size=80)
        wall = Wall(80, 10, 80, 70)
        grid.add_wall(wall)
        short = math.nextafter(80.0, 0.0)

        self.assertIn(wall, grid.get_walls_in_rect(5, 5, short, 75))
        self.assertIn(wall, grid.get_walls_in_rect(5, 5, 80 - QUERY_PAD / 2, 75))
        self.assertEqual(grid.get_walls_in_rect(5, 5, 79.99, 75), [])

    def test_wall_ending_on_a_cell_corner(self):
        grid = SpatialGrid(cell_size=10)
        wall = Wall(0, 0, 10, 10)
        grid.add_wall(wall)

        # (10, 10) is a point of the wall and lies in cell (1, 1) only.
        self.assertIn(wall, grid.get_walls_in_rect(10, 10, 15, 15))
        self.assertIn(wall, grid.get_walls_in_rect(5, 5, 10, 10))
        self.assertEqual(walls_in_cell(grid, 1, 0), [])
        self.assertEqual(walls_in_cell(grid, 0, 1), [])
        self.assertEqual(set(grid.grid), {(0, 0), (1, 1)})

    def test_wall_not_found_in_cells_it_does_not_touch(self):
        grid = SpatialGrid(cell_size=10)
        wall = Wall(0, 1, 12, 12)
        grid.add_wall(wall)

        self.assertEqual(walls_in_cell(grid, 1, 0), [])
        self.assertEqual(walls_in_cell(grid, 0, 2), [])

    def test_queries_are_exact_to_the_cell(self):
        # Integer coordinates on a cell size of 5 put many walls and box
        # edges on cell borders and corners.
        s = 5
        edge = 1e-9  # turns a cell's open max edges into closed ones
        rng = random.Random(23)

        def coord(lo, hi):
            return rng.randint(lo, hi) if rng.random() < 0.5 else rng.uniform(lo, hi)

        for _ in range(5000):
            grid = SpatialGrid(cell_size=s)
            x1, y1, x2, y2 = (rng.randint(-10, 20) for _ in range(4))
            wall = Wall(x1, y1, x2, y2)
            grid.add_wall(wall)
            msg = f"wall {x1, y1, x2, y2}"

            # Every registered cell holds a point of the wall, so the
            # overlay outlines no cell the wall only borders.
            for cx, cy in grid.grid:
                self.assertTrue(segment_meets_closed_box(
                    x1, y1, x2, y2, cx * s, cy * s,
                    (cx + 1) * s - edge, (cy + 1) * s - edge),
                    f"{msg} registered in cell {cx, cy}")

            bx0, bx1 = sorted(coord(-12, 22) for _ in range(2))
            by0, by1 = sorted(coord(-12, 22) for _ in range(2))
            found = wall in grid.get_walls_in_rect(bx0, by0, bx1, by1)
            box = f"{msg}, box {bx0, by0, bx1, by1}"
            # A box holding a point of the wall gets it back...
            if segment_meets_closed_box(x1, y1, x2, y2, bx0, by0, bx1, by1):
                self.assertTrue(found, box)
            # ...and a box gets it back only if one of the cells its
            # QUERY_PAD-widened self overlaps holds a point of it.
            if found:
                self.assertTrue(segment_meets_closed_box(
                    x1, y1, x2, y2,
                    math.floor((bx0 - QUERY_PAD) / s) * s,
                    math.floor((by0 - QUERY_PAD) / s) * s,
                    (math.floor((bx1 + QUERY_PAD) / s) + 1) * s - edge,
                    (math.floor((by1 + QUERY_PAD) / s) + 1) * s - edge), box)


class _AllWalls:
    """Stand-in grid that hands back every wall: a full scan."""

    def __init__(self, walls):
        self.walls = list(walls)

    def get_walls_around(self, points):
        return self.walls


class GridMatchesFullScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.surface = pygame.display.set_mode((1000, 600))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_rays_and_collisions_match_a_full_scan_on_every_track(self):
        rng = random.Random(23)
        width, height = self.surface.get_size()
        for name in template_names():
            with self.subTest(track=name):
                track = Track(self.surface, name)
                full = _AllWalls(track.walls)
                car = Car(*track.starting_position, track, track.starting_rotation)
                s = track.walls_grid.cell_size
                for i in range(1000):
                    car.x, car.y = rng.uniform(0, width), rng.uniform(0, height)
                    car.angle = rng.uniform(-math.pi, math.pi)
                    if i % 2:
                        # On cell borders, heading along an axis, so rays and
                        # hitbox edges run along or end on border walls.
                        car.x = round(car.x / s) * s
                        car.y = round(car.y / s) * s
                        car.angle = rng.randrange(-2, 2) * math.pi / 2
                    car._update_hitbox_points()

                    car.grid = track.walls_grid
                    car.cast_rays()
                    grid_rays = car.closest_ray_distances
                    car.grid = full
                    car.cast_rays()
                    self.assertEqual(grid_rays, car.closest_ray_distances)

                    self.assertEqual(car.check_collisions(track.walls_grid),
                                     car.check_collisions(full))


if __name__ == "__main__":
    unittest.main()
