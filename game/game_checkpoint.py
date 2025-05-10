import pygame
import math
from .game_point import Point


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

    @property
    def center(self):
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def __getStartCords(self):
        return (self.x1, self.y1)

    def __getEndCords(self):
        return (self.x2, self.y2)

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
