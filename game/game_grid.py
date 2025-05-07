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
        cells = self.get_cells_along_wall_line(
            wall.x1, wall.y1, wall.x2, wall.y2)

        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(wall)

    def get_cells_along_wall_line(self, x1, y1, x2, y2):
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

    def __get_current_cell(self, x, y):
        x_cell, y_cell = int(x // self.cell_size), int(y // self.cell_size)
        return x_cell, y_cell

    def get_nearby_cell_wall(self, x, y):
        x_cell, y_cell = self.__get_current_cell(x, y)

        nearby_cells = [
            (x_cell-1, y_cell-1), (x_cell, y_cell-1), (x_cell+1, y_cell-1),
            (x_cell-1, y_cell),   (x_cell, y_cell),   (x_cell+1, y_cell),
            (x_cell-1, y_cell+1), (x_cell, y_cell+1), (x_cell+1, y_cell+1)
        ]

        wall = []
        for cell in nearby_cells:
            if cell in self.grid:
                wall.extend(self.grid[cell])
        return wall

    def draw(self, surface, color=(50, 50, 50)):
        for cell in self.grid:
            x = cell[0] * self.cell_size
            y = cell[1] * self.cell_size
            pygame.draw.rect(
                surface, color, (x, y, self.cell_size, self.cell_size), 1
            )
