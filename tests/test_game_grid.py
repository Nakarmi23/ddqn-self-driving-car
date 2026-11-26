import unittest

from game.game_grid import SpatialGrid
from game.game_wall import Wall


def walls_in_cell(grid, cx, cy):
    """Walls a query confined to one cell's interior gets back."""
    s = grid.cell_size
    return grid.get_walls_in_rect(cx * s + 0.01, cy * s + 0.01,
                                  (cx + 1) * s - 0.01, (cy + 1) * s - 0.01)


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

        self.assertIn(wall, walls_in_cell(grid, 0, 0))
        self.assertIn(wall, walls_in_cell(grid, 1, 0))

    def test_wall_not_found_in_cells_it_does_not_touch(self):
        grid = SpatialGrid(cell_size=10)
        wall = Wall(0, 1, 12, 12)
        grid.add_wall(wall)

        self.assertEqual(walls_in_cell(grid, 1, 0), [])
        self.assertEqual(walls_in_cell(grid, 0, 2), [])


if __name__ == "__main__":
    unittest.main()
