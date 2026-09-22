import importlib
import pygame

DEFAULT_TRACK = "template1"


class Track():
    def __init__(self, surface: pygame.Surface, track_name: str = DEFAULT_TRACK):
        self.surface = surface
        get_track = importlib.import_module(
            f".track_templates.{track_name}", package="game").get_track
        self.starting_position, self.starting_rotation, self.walls, self.walls_grid, self.checkpoints = get_track()

    def render(self):
        self.walls_grid.draw(self.surface)

        for wall in self.walls:
            wall.draw(self.surface)

        for checkpoint in self.checkpoints:
            checkpoint.draw(self.surface)
