from .game_point import Point


class Intersection:
    def __init__(self, distance: int, point: Point):
        self.distance = distance
        self.point = point
