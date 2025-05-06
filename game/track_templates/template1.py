from ..game_wall import Wall
from ..game_grid import SpatialGrid
from ..game_checkpoint import Checkpoint
from typing import Tuple

__all_walls = (
    # Outer walls (clockwise)
    Wall(12, 451, 15, 130),
    Wall(15, 130, 61, 58),
    Wall(61, 58, 149, 14),
    Wall(149, 14, 382, 20),
    Wall(382, 20, 549, 31),
    Wall(549, 31, 636, 58),
    Wall(636, 58, 678, 102),
    Wall(678, 102, 669, 167),
    Wall(669, 167, 600, 206),
    Wall(600, 206, 507, 214),
    Wall(507, 214, 422, 232),
    Wall(422, 232, 375, 263),
    Wall(375, 263, 379, 283),
    Wall(379, 283, 454, 299),
    Wall(454, 299, 613, 286),
    Wall(613, 286, 684, 238),
    Wall(684, 238, 752, 180),
    Wall(752, 180, 862, 185),
    Wall(862, 185, 958, 279),
    Wall(958, 279, 953, 410),
    Wall(953, 410, 925, 505),
    Wall(925, 505, 804, 566),
    Wall(804, 566, 150, 570),
    Wall(150, 570, 46, 529),
    Wall(46, 529, 12, 451),

    # Inner walls (counter-clockwise)
    Wall(104, 436, 96, 161),
    Wall(96, 161, 122, 122),
    Wall(122, 122, 199, 91),
    Wall(199, 91, 376, 94),
    Wall(376, 94, 469, 100),
    Wall(469, 100, 539, 102),
    Wall(539, 102, 585, 121),
    Wall(585, 121, 585, 139),
    Wall(585, 139, 454, 158),
    Wall(454, 158, 352, 183),
    Wall(352, 183, 293, 239),
    Wall(293, 239, 294, 318),
    Wall(294, 318, 361, 357),
    Wall(361, 357, 490, 373),
    Wall(490, 373, 671, 359),
    Wall(671, 359, 752, 300),
    Wall(752, 300, 812, 310),
    Wall(812, 310, 854, 369),
    Wall(854, 369, 854, 429),
    Wall(854, 429, 754, 483),
    Wall(754, 483, 192, 489),
    Wall(192, 489, 104, 436)
)

__all_checkpoints = (
    # First checkpoint (starting line) marked explicitly
    Checkpoint(0, 200, 120, 200, isStart=True),

    # All subsequent checkpoints (isStart=False by default)
    Checkpoint(0, 100, 120, 150),
    Checkpoint(0, 0, 150, 130),
    Checkpoint(120, 0, 170, 120),
    Checkpoint(200, 0, 200, 120),
    Checkpoint(270, 0, 270, 110),
    Checkpoint(350, 0, 350, 110),
    Checkpoint(450, 0, 450, 110),
    Checkpoint(525, 0, 525, 110),
    Checkpoint(600, 0, 550, 130),
    Checkpoint(550, 130, 700, 60),
    Checkpoint(550, 130, 700, 130),
    Checkpoint(550, 130, 650, 200),
    Checkpoint(550, 130, 570, 240),
    Checkpoint(410, 130, 430, 260),
    Checkpoint(430, 260, 280, 180),
    Checkpoint(430, 260, 260, 260),
    Checkpoint(430, 260, 300, 350),
    Checkpoint(430, 260, 400, 400),
    Checkpoint(550, 260, 570, 400),
    Checkpoint(750, 400, 650, 200),
    Checkpoint(750, 400, 800, 160),
    Checkpoint(750, 400, 950, 240),
    Checkpoint(750, 400, 980, 440),
    Checkpoint(750, 400, 900, 600),
    Checkpoint(750, 460, 750, 600),
    Checkpoint(670, 460, 670, 600),
    Checkpoint(590, 460, 590, 600),
    Checkpoint(510, 460, 510, 600),
    Checkpoint(430, 460, 430, 600),
    Checkpoint(350, 460, 350, 600),
    Checkpoint(280, 460, 278, 600),
    Checkpoint(210, 460, 190, 600),
    Checkpoint(80, 600, 175, 440),
    Checkpoint(150, 420, 0, 570),
    Checkpoint(0, 450, 130, 400),
    Checkpoint(0, 380, 130, 380)
)


def get_track() -> Tuple[Tuple[int, int], int, Tuple[Wall, ...], SpatialGrid, Tuple[Checkpoint, ...]]:
    """Returns track definition including starting position, rotation, walls, grid, and checkpoints"""

    starting_position = (50, 300)
    starting_rotation = 180
    # Initialize spatial grid
    grid = SpatialGrid(cell_size=80)
    for wall in __all_walls:
        grid.add_wall(wall)

    return starting_position, starting_rotation, __all_walls, grid, __all_checkpoints
