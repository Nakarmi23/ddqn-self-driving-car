import pygame
from .game_wall import Wall


class SpatialGrid:
    def __init__(self, cell_size=100):
        self.cell_size = cell_size
        self.grid = {}

    def __get_cell(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def add_wall(self, wall: Wall):
        """Register the wall in every cell it touches, so any query rect
        that contains a point of the wall gets it back without padding."""
        for cell in self.cells_touched_by_segment(
                wall.x1, wall.y1, wall.x2, wall.y2):
            self.grid.setdefault(cell, []).append(wall)

    def cells_touched_by_segment(self, x1, y1, x2, y2):
        """Supercover: every cell whose closed square the segment touches,
        including cells it only grazes and both sides of a cell border it
        lies on. Grid construction is one-off, so this tests each cell of
        the segment's bounding box rather than walking the line."""
        cx0, cy0 = self.__get_cell(min(x1, x2), min(y1, y2))
        cx1, cy1 = self.__get_cell(max(x1, x2), max(y1, y2))
        # A segment on a border belongs to the cells either side of it.
        if min(x1, x2) % self.cell_size == 0:
            cx0 -= 1
        if min(y1, y2) % self.cell_size == 0:
            cy0 -= 1
        s = self.cell_size
        return [(cx, cy)
                for cx in range(cx0, cx1 + 1)
                for cy in range(cy0, cy1 + 1)
                if _segment_touches_box(x1, y1, x2, y2,
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


def _segment_touches_box(x1, y1, x2, y2, min_x, min_y, max_x, max_y):
    """Whether segment (x1, y1)-(x2, y2) has a point in the closed box
    (Liang-Barsky clipping)."""
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
