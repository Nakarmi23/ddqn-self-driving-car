from .track_templates.template1 import get_track
import pygame


class Track():
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self.starting_position, self.starting_rotation, self.walls, self.walls_grid, self.checkpoints = get_track()

    def render(self):
        self.walls_grid.draw(self.surface)

        for wall in self.walls:
            wall.draw(self.surface)

        for checkpoint in self.checkpoints:
            checkpoint.draw(self.surface)
