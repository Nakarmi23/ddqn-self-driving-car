import pygame
import math
from .game_point import Point
from .game_grid import SpatialGrid
from .game_wall import Wall


class Car:
    def __init__(self, x, y, rotation=0):
        # Physics properties
        self.x = x
        self.y = y
        self.width = 30
        self.height = 14
        self.vel = 0
        self.max_vel = 15
        self.dvel = 0.2
        self.angle = math.radians(rotation)  # Physics angle in radians
        self.soll_angle = self.angle

        # Drag system
        self.drag = 0.985
        self.free_decel = 0.97
        self.accelerating = False

        # Image handling
        self.original_image = pygame.image.load(
            "./game/assets/car.png").convert_alpha()
        self.original_image = pygame.transform.flip(
            self.original_image, True, False)
        self.image = pygame.transform.rotate(
            self.original_image,
            90 - math.degrees(self.angle)  # Compensate for physics angle
        )
        self.rect = self.image.get_rect(center=(self.x, self.y))

        # Ray casting system
        self.ray_length = 65
        self.closest_rays = []

        # Hitbox points (will be calculated in _update_hitbox_points)
        self.pt1 = None  # Front-left
        self.pt2 = None  # Front-right
        self.pt3 = None  # Rear-right
        self.pt4 = None  # Rear-left
        self.front_center = None
        self.mid_left = None
        self.mid_right = None
        self.back_center = None

        self._update_hitbox_points()
        self.cast_rays()

    def reset(self):
        self.x = 0
        self.y = 0
        self.vel = 0

    def accelerate(self, direction):
        """Direction: 1=forward, -1=reverse"""
        if direction != 0:
            self.accelerating = True
            self.vel += direction * self.dvel
            self.vel = max(-self.max_vel, min(self.vel, self.max_vel))

    def turn(self, direction):
        """Direction: 1=right, -1=left"""
        self.soll_angle += direction * math.radians(5)

    def _update_hitbox_points(self):
        """Calculate all hitbox points with perfect visual alignment"""
        half_w = self.width / 2
        half_h = self.height / 2

        # Base points (relative to center, car facing RIGHT in image space)
        base_pt1 = Point(-half_w, -half_h)  # Front-left
        base_pt2 = Point(half_w, -half_h)   # Front-right
        base_pt3 = Point(half_w, half_h)    # Rear-right
        base_pt4 = Point(-half_w, half_h)   # Rear-left

        # Rotate points using visual angle (physics angle - 90°)
        visual_angle = self.angle - math.pi/2
        self.pt1 = self._rotate_point(base_pt1, visual_angle)
        self.pt2 = self._rotate_point(base_pt2, visual_angle)
        self.pt3 = self._rotate_point(base_pt3, visual_angle)
        self.pt4 = self._rotate_point(base_pt4, visual_angle)

        # Convert to world coordinates
        for p in [self.pt1, self.pt2, self.pt3, self.pt4]:
            p.x += self.x
            p.y += self.y

        # Calculate strategic points
        self.front_center = Point(
            (self.pt1.x + self.pt2.x)/2,
            (self.pt1.y + self.pt2.y)/2
        )
        self.mid_left = Point(
            (self.pt1.x + self.pt4.x)/2,
            (self.pt1.y + self.pt4.y)/2
        )
        self.mid_right = Point(
            (self.pt2.x + self.pt3.x)/2,
            (self.pt2.y + self.pt3.y)/2
        )
        self.back_center = Point(
            (self.pt3.x + self.pt4.x)/2,
            (self.pt3.y + self.pt4.y)/2
        )

    def _rotate_point(self, point, angle):
        """Rotate point using standard rotation matrix"""
        x = point.x * math.cos(angle) - point.y * math.sin(angle)
        y = point.x * math.sin(angle) + point.y * math.cos(angle)
        return Point(x, y)

    def cast_rays(self):
        """Cast rays from precise hitbox points with visual alignment"""
        self.closest_rays = []

        # Front center (0°)
        end = self._calculate_ray_end(self.front_center, 0)
        self.closest_rays.append(
            (self.front_center.x, self.front_center.y, end.x, end.y))

        # Front-left (-30° from pt1)
        end = self._calculate_ray_end(self.pt1, -30)
        self.closest_rays.append((self.pt1.x, self.pt1.y, end.x, end.y))

        # Front-right (+30° from pt2)
        end = self._calculate_ray_end(self.pt2, 30)
        self.closest_rays.append((self.pt2.x, self.pt2.y, end.x, end.y))

        # Left (-90° from mid-left)
        end = self._calculate_ray_end(self.mid_left, -90)
        self.closest_rays.append(
            (self.mid_left.x, self.mid_left.y, end.x, end.y))

        # Right (+90° from mid-right)
        end = self._calculate_ray_end(self.mid_right, 90)
        self.closest_rays.append(
            (self.mid_right.x, self.mid_right.y, end.x, end.y))

        # Rear-left (-135° from pt4)
        end = self._calculate_ray_end(self.pt4, -135)
        self.closest_rays.append((self.pt4.x, self.pt4.y, end.x, end.y))

        # Rear-right (+135° from pt3)
        end = self._calculate_ray_end(self.pt3, 135)
        self.closest_rays.append((self.pt3.x, self.pt3.y, end.x, end.y))

        # Back (180° from back center)
        end = self._calculate_ray_end(self.back_center, 180)
        self.closest_rays.append(
            (self.back_center.x, self.back_center.y, end.x, end.y))

    def _calculate_ray_end(self, origin, angle_offset):
        """Calculate ray end point with visual angle compensation"""
        visual_angle = self.angle - math.pi/2
        ray_angle = visual_angle + math.radians(angle_offset)
        dx = math.sin(ray_angle) * self.ray_length
        # Negative for pygame's Y axis
        dy = -math.cos(ray_angle) * self.ray_length
        return Point(origin.x + dx, origin.y + dy)

    def update(self):
        # Apply drag
        self.vel *= self.drag
        if not self.accelerating:
            self.vel *= self.free_decel

        # Reset acceleration state
        self.accelerating = False

        # Clamp near-zero velocity
        if abs(self.vel) < 0.1:
            self.vel = 0

        # Smooth angle interpolation
        angle_diff = self.soll_angle - self.angle
        self.angle += angle_diff * 0.2

        # Update position (using physics angle)
        vec = self._rotate_point(Point(0, self.vel), self.angle)
        self.x += vec.x
        self.y += vec.y

        # Update hitbox and rays
        self._update_hitbox_points()
        self.cast_rays()

        # Update image (with visual rotation)
        self.image = pygame.transform.rotate(
            self.original_image,
            90 - math.degrees(self.angle)
        )
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def check_collisions(self, grid: SpatialGrid):
        nearby_walls = grid.get_nearby_cell_wall(self.x, self.y)

        for wall in nearby_walls:
            if self.__check_collision(wall):
                return True

        return False

    def __check_collision(self, wall: Wall):
        edge = [
            (self.pt1, self.pt2),
            (self.pt2, self.pt3),
            (self.pt3, self.pt4),
            (self.pt4, self.pt1)
        ]

        for line in edge:
            if self.__check_line_intersection(line[0], line[1], wall):
                return True

        return False

    def __check_line_intersection(self, p1: Point, p2: Point, wall: Wall):
        x1, y1 = wall.x1, wall.y1
        x2, y2 = wall.x2, wall.y2
        x3, y3 = p1.x, p1.y
        x4, y4 = p2.x, p2.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return False

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        return 0 <= t <= 1 and 0 <= u <= 1

    def draw(self, surface):
        # Draw car
        surface.blit(self.image, self.rect)

        # Draw rays (cyan)
        for ray in self.closest_rays:
            pygame.draw.line(
                surface, (0, 255, 255),
                (ray[0], ray[1]), (ray[2], ray[3]), 1
            )

        # Draw hitbox points (red)
        for p in [self.pt1, self.pt2, self.pt3, self.pt4,
                  self.front_center, self.mid_left,
                  self.mid_right, self.back_center]:
            pygame.draw.circle(surface, (255, 0, 0), (int(p.x), int(p.y)), 3)
