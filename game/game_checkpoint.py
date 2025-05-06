import pygame


class Checkpoint:
    def __init__(self, x1, y1, x2, y2, isStart=False):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.isStart = isStart

    def __getStartCords(self):
        return (self.x1, self.y1)

    def __getEndCords(self):
        return (self.x2, self.y2)

    def draw(
            self,
            surface: pygame.Surface,
            start_line_color=(255, 0, 0),
            inactive_color=(100, 100, 100),
            thickness=2
    ):
        color = start_line_color if self.isStart else inactive_color
        pygame.draw.line(
            surface,
            color,
            self.__getStartCords(),
            self.__getEndCords(),
            thickness
        )
