import pygame
from .game_wall import Wall


class SpatialGrid:
    def __init__(self, cell_size=100):
        self.cell_size = cell_size
        self.grid = {}

    def __get_cell(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def add_wall(self, wall: Wall):
        # Simple approach: add wall to all cells along its length
        cells = self.__get_cells_along_wall_line(
            wall.x1, wall.y1, wall.x2, wall.y2)

        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(wall)

    def __get_cells_along_wall_line(self, x1, y1, x2, y2):
        """Bresenham's line algorithm to get all cells a line passes through"""
        cells = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                cells.append(self.__get_cell(x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                cells.append(self.__get_cell(x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        cells.append(self.__get_cell(x, y))
        return list(set(cells))  # Remove duplicates

    def draw(self, surface, color=(50, 50, 50)):
        for cell in self.grid:
            x = cell[0] * self.cell_size
            y = cell[1] * self.cell_size
            pygame.draw.rect(
                surface, color, (x, y, self.cell_size, self.cell_size), 1
            )
