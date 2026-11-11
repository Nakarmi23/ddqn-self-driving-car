"""Check that no track template spawns the car touching a checkpoint or wall.

A spawn hitbox that overlaps a checkpoint counts it as passed on step 1 of
every episode without the car moving (issue #8). Run from the repo root:

    python -m scripts.check_spawns

Prints one row per template and exits with status 1 if any spawn overlaps.
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame  # noqa: E402

from game.game_car import Car  # noqa: E402
from game.game_track import Track  # noqa: E402

TEMPLATE_DIR = os.path.join("game", "track_templates")


def template_names():
    return sorted(f[:-3] for f in os.listdir(TEMPLATE_DIR)
                  if f.startswith("template") and f.endswith(".py"))


def main():
    pygame.init()
    surface = pygame.display.set_mode((1000, 600))
    failed = False
    print(f"{'Template':<10} {'Start':<12} {'Overlapping checkpoints':<24} "
          f"{'Wall':<5} Total")
    for name in template_names():
        track = Track(surface, name)
        car = Car(*track.starting_position, track, track.starting_rotation)
        overlaps = car.overlapping_checkpoints()
        wall = car.check_collisions(track.walls_grid)
        failed |= bool(overlaps) or wall
        start = "({}, {})".format(*track.starting_position)
        print(f"{name:<10} {start:<12} "
              f"{', '.join(map(str, overlaps)) or 'none':<24} "
              f"{'yes' if wall else 'no':<5} {len(track.checkpoints)}")
    pygame.quit()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
