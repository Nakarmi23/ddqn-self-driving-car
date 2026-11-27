import pygame
import math
from .game_point import Point


def _ray_hit(px, py, dx, dy, wall):
    """Parameter s >= 0 where the ray (px, py) + s * (dx, dy) meets `wall`,
    or None if it misses (or runs parallel to it)."""
    ex, ey = wall.x2 - wall.x1, wall.y2 - wall.y1
    den = dx * ey - dy * ex
    if den == 0:
        return None
    s = ((wall.x1 - px) * ey - (wall.y1 - py) * ex) / den
    u = ((wall.x1 - px) * dy - (wall.y1 - py) * dx) / den
    return s if s >= 0 and 0 <= u <= 1 else None


def count_passes(checkpoints, index, start, end):
    """How many checkpoints, from `checkpoints[index]` on, the move from
    point `start` to `end` crosses forward in order.

    One move can cross several lines where consecutive checkpoints are close
    or share an end; each later line is tested against the same move, so a
    pass is never lost because an earlier one in the same step took it.
    Capped at one lap.
    """
    n = len(checkpoints)
    passes = 0
    while passes < n:
        i = (index + passes) % n
        if not checkpoints[i].crossed_forward(start, end, checkpoints[i - 1].center):
            break
        passes += 1
    return passes


class Checkpoint:
    def __init__(self, x1, y1, x2, y2, base_reward=100, difficulty_factor=1.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.base_reward = base_reward
        self.difficulty_factor = difficulty_factor
        self.is_active = False  # Add activation state
        self.is_passed = False  # Track completion
        # Segment a pass is tested against: the checkpoint itself, or longer
        # once extend_to_walls() closes the gaps between its ends and the
        # walls (issue #39 review).
        self.cross_x1, self.cross_y1 = x1, y1
        self.cross_x2, self.cross_y2 = x2, y2

    @property
    def center(self):
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def __getStartCords(self):
        return (self.x1, self.y1)

    def __getEndCords(self):
        return (self.x2, self.y2)

    def side_of(self, x, y):
        """Signed side of the checkpoint line (x, y) is on: positive on one
        side, negative on the other, 0 on the line itself."""
        return ((self.x2 - self.x1) * (y - self.y1)
                - (self.y2 - self.y1) * (x - self.x1))

    def crossed_forward(self, start, end, behind):
        """True if moving from point `start` to point `end` crosses this
        checkpoint from the side `behind` is on to the other side.

        Half-open: `start` must be strictly on the behind side and `end` on
        the line or past it, so a point landing exactly on the line counts
        once, and touching or leaving the line from the far side never counts.
        """
        behind_side = self.side_of(behind.x, behind.y)
        s0 = self.side_of(start.x, start.y) * behind_side
        s1 = self.side_of(end.x, end.y) * behind_side
        if not (s0 > 0 and s1 <= 0):
            return False
        # s0 != s1, so the move is not parallel to the line. Check the
        # crossing point lies within the segment, not on its extension.
        t = s0 / (s0 - s1)
        px = start.x + t * (end.x - start.x)
        py = start.y + t * (end.y - start.y)
        dx, dy = self.cross_x2 - self.cross_x1, self.cross_y2 - self.cross_y1
        u = (((px - self.cross_x1) * dx + (py - self.cross_y1) * dy)
             / (dx * dx + dy * dy))
        return 0 <= u <= 1

    def extend_to_walls(self, walls):
        """Stretch the crossing segment so a car can't drive around the line.

        An end that the centre reaches without crossing a wall lies inside the
        driveable corridor, so a car could pass between it and the wall
        uncounted; that end is pushed outward along the line to the first
        wall. An end already beyond a wall is left alone, so the segment never
        reaches into another stretch of track. Only crossed_forward() uses the
        result; drawing and distances still use the checkpoint as defined.
        """
        if (self.x1, self.y1) == (self.x2, self.y2):
            return
        c = self.center
        ends = []
        for (ex, ey), (ox, oy) in (((self.x1, self.y1), (self.x2, self.y2)),
                                   ((self.x2, self.y2), (self.x1, self.y1))):
            blocked = any(
                (s := _ray_hit(c.x, c.y, ex - c.x, ey - c.y, w)) is not None
                and s <= 1 for w in walls)
            if not blocked:
                hits = [s for w in walls
                        if (s := _ray_hit(ex, ey, ex - ox, ey - oy, w)) is not None]
                if hits:
                    s = min(hits)
                    ex, ey = ex + s * (ex - ox), ey + s * (ey - oy)
            ends.append((ex, ey))
        (self.cross_x1, self.cross_y1), (self.cross_x2, self.cross_y2) = ends

    def distance_to(self, x, y):
        """Calculate shortest distance from point (x,y) to this checkpoint"""
        x1, y1 = self.x1, self.y1
        x2, y2 = self.x2, self.y2

        # Handle zero-length checkpoints
        if (x1 == x2) and (y1 == y2):
            return math.hypot(x - x1, y - y1)

        # Vector math for line segment distance
        dx = x2 - x1
        dy = y2 - y1
        t = ((x - x1) * dx + (y - y1) * dy) / (dx**2 + dy**2 + 1e-8)
        t = max(0, min(1, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.hypot(x - closest_x, y - closest_y)

    def draw(
            self,
            surface: pygame.Surface,
            start_line_color=(255, 0, 0),
            inactive_color=(100, 100, 100),
            thickness=2
    ):
        color = start_line_color if self.is_active else inactive_color
        pygame.draw.line(
            surface,
            color,
            self.__getStartCords(),
            self.__getEndCords(),
            thickness
        )
