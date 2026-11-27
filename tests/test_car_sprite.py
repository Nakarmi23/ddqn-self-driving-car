import os
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from game.game_car import Car  # noqa: E402
from game.game_track import Track  # noqa: E402


class CarSpriteCacheTest(unittest.TestCase):
    """The class-level sprite cache must not outlive a pygame session (#22)."""

    def tearDown(self):
        pygame.quit()

    def start(self):
        pygame.init()
        surface = pygame.display.set_mode((1000, 600))
        track = Track(surface, "template5")
        car = Car(*track.starting_position, track, track.starting_rotation)
        return surface, car

    def test_sprite_reloads_after_pygame_restart(self):
        with mock.patch("pygame.image.load", wraps=pygame.image.load) as load:
            surface, car = self.start()
            car.draw(surface)
            car.draw(surface)
            first = Car._sprite()
            self.assertEqual(load.call_count, 1)  # cached within a session

            pygame.quit()
            self.assertIsNone(Car._original_image)

            surface, car = self.start()
            car.draw(surface)  # must not raise pygame.error
            car.draw(surface)
            self.assertIsNot(Car._sprite(), first)
            self.assertEqual(load.call_count, 2)  # once per session


if __name__ == "__main__":
    unittest.main()
