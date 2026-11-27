import pygame
from .game_wall import Wall


class SpatialGrid:
    def __init__(self, cell_size=100):
        self.cell_size = cell_size
        self.grid = {}

    def __get_cell(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def add_wall(self, wall: Wall):
        """Register the wall in every cell holding a point of it, so any
        query rect that contains a point of the wall gets it back without
        padding, and no query rect that contains none does (#23)."""
        for cell in self.cells_touched_by_segment(
                wall.x1, wall.y1, wall.x2, wall.y2):
            self.grid.setdefault(cell, []).append(wall)

    def cells_touched_by_segment(self, x1, y1, x2, y2):
        """Every cell holding a point of the segment. A cell is half-open,
        [cx*s, (cx+1)*s) x [cy*s, (cy+1)*s), the same way __get_cell maps a
        query point to one cell, so a segment on a cell border belongs only
        to the cells whose min edge it lies on. Grid construction is one-off,
        so this tests each cell of the segment's bounding box rather than
        walking the line."""
        cx0, cy0 = self.__get_cell(min(x1, x2), min(y1, y2))
        cx1, cy1 = self.__get_cell(max(x1, x2), max(y1, y2))
        s = self.cell_size
        return [(cx, cy)
                for cx in range(cx0, cx1 + 1)
                for cy in range(cy0, cy1 + 1)
                if _segment_touches_cell(x1, y1, x2, y2,
                                         cx * s, cy * s, (cx + 1) * s, (cy + 1) * s)]

    def get_walls_around(self, points):
        """Every wall that could cross a segment between any of these points
        (each with .x/.y): the walls in their bounding box's cells."""
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return self.get_walls_in_rect(min(xs), min(ys), max(xs), max(ys))

    def get_walls_in_rect(self, min_x, min_y, max_x, max_y):
        """Unique walls registered in any cell overlapping the rectangle."""
        cx0, cy0 = self.__get_cell(min_x, min_y)
        cx1, cy1 = self.__get_cell(max_x, max_y)
        walls = {}
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for wall in self.grid.get((cx, cy), ()):
                    walls[id(wall)] = wall
        return list(walls.values())

    def draw(self, surface, color=(50, 50, 50)):
        for cell in self.grid:
            x = cell[0] * self.cell_size
            y = cell[1] * self.cell_size
            pygame.draw.rect(
                surface, color, (x, y, self.cell_size, self.cell_size), 1
            )


def _segment_touches_cell(x1, y1, x2, y2, min_x, min_y, max_x, max_y):
    """Whether segment (x1, y1)-(x2, y2) has a point in the half-open box
    [min_x, max_x) x [min_y, max_y) (Liang-Barsky clipping, strict on the
    max edges)."""
    lo, lo_strict = 0.0, False   # t >= lo (t > lo if strict)
    hi, hi_strict = 1.0, False   # t <= hi (t < hi if strict)
    dx, dy = x2 - x1, y2 - y1
    # Each constraint is p * t <= q, or p * t < q when strict.
    for p, q, strict in ((-dx, x1 - min_x, False), (dx, max_x - x1, True),
                         (-dy, y1 - min_y, False), (dy, max_y - y1, True)):
        if p == 0:
            if q < 0 or (strict and q == 0):
                return False
            continue
        r = q / p
        if p < 0:   # t >= r
            if r > lo:
                lo, lo_strict = r, strict
            elif r == lo:
                lo_strict = lo_strict or strict
        else:       # t <= r
            if r < hi:
                hi, hi_strict = r, strict
            elif r == hi:
                hi_strict = hi_strict or strict
        if lo > hi or (lo == hi and (lo_strict or hi_strict)):
            return False
    return True
