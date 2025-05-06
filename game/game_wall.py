import pygame


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __getStartCords(self):
        return (self.x1, self.y1)

    def __getEndCords(self):
        return (self.x2, self.y2)

    def draw(
            self,
            surface: pygame.Surface,
            color=(255, 255, 255),
            thickness=2
    ):
        pygame.draw.line(
            surface,
            color,
            self.__getStartCords(),
            self.__getEndCords(),
            thickness
        )
