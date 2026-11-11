"""Check that every track template has a valid spawn (see game/game_spawn.py):
between the last checkpoint and checkpoint 0, facing checkpoint 0, at least
SPAWN_MARGIN px from every checkpoint and clear of walls. A spawn on a
checkpoint counts the pass on step 1 without the car moving (issue #8).

    python -m scripts.check_spawns      # or: python scripts/check_spawns.py

Prints one row per template and exits with status 1 if any spawn is bad.
"""
import os
import pkgutil
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pygame  # noqa: E402

import game.track_templates  # noqa: E402
from game.game_car import Car  # noqa: E402
from game.game_spawn import hitbox_distance, spawn_problems  # noqa: E402
from game.game_track import Track  # noqa: E402


def template_names():
    return sorted(m.name for m in pkgutil.iter_modules(game.track_templates.__path__)
                  if m.name.startswith("template"))


def main():
    pygame.init()
    surface = pygame.display.set_mode((1000, 600))
    failed = False
    print(f"{'Template':<10} {'Start':<12} {'Checkpoint gap':<15} Problems")
    for name in template_names():
        track = Track(surface, name)
        car = Car(*track.starting_position, track, track.starting_rotation)
        problems = spawn_problems(car, track)
        failed |= bool(problems)
        gap = min(hitbox_distance(car, cp.x1, cp.y1, cp.x2, cp.y2)
                  for cp in track.checkpoints)
        start = "({}, {})".format(*track.starting_position)
        print(f"{name:<10} {start:<12} {gap:>6.1f} px      "
              f"{'; '.join(problems) or 'none'}")
    pygame.quit()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
