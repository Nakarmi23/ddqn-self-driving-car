import pygame
import math
import numpy as np
from .game_point import Point
from .game_grid import SpatialGrid
from .game_wall import Wall
from .game_checkpoint import Checkpoint
from .game_intersection import Intersection
from .game_track import Track


class Car:
    def __init__(self, x, y, track: Track, rotation=0,):
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
        
        self.checkpoints = track.checkpoints
        self.walls = track.walls
        # Grid properties
        self.grid = track.walls_grid

        # Add these drift properties
        self.traction = 1.0  # 1.0 = full grip, 0.0 = full drift
        self.drift_angle = 0.0
        self.drift_threshold = 3  # Min speed to drift (m/s)
        self.drift_torque = 0.02  # Rotational force during drift

        # Drag system
        self.drag = 0.985
        self.free_decel = 0.97
        self.accelerating = False

        # Image handling
        self.original_image = pygame.image.load(
            "./game/assets/car.png")
        self.original_image = pygame.transform.flip(
            self.original_image, True, False)
        self.image = pygame.transform.rotate(
            self.original_image,
            90 - math.degrees(self.angle)  # Compensate for physics angle
        )
        self.rect = self.image.get_rect(center=(self.x, self.y))

        # Ray casting system
        self.ray_length = 170
        self.closest_rays = []
        self.closest_ray_distances = []

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

    def get_state(self):
        normalized_ray_distances = [
            d / self.ray_length for d in self.closest_ray_distances]

        normalized_vel = self.vel / self.max_vel

        traction = self.traction

        normalized_drift_angle = self.drift_angle / math.pi

        return normalized_ray_distances + [normalized_vel, traction, normalized_drift_angle]

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
        ray_definition = [
            (self.front_center, 0),
            (self.pt1, -30),
            (self.pt2, 30),
            (self.mid_left, -90),
            (self.mid_right, 90),
            (self.pt3, 135),
            (self.pt4, -135),
            (self.back_center, 180)
        ]

        self.closest_rays = []
        self.closest_ray_distances = []

        for origin, angle_offset in ray_definition:
            end_point, distance = self._calculate_ray_end(origin, angle_offset)
            self.closest_ray_distances.append(distance)

            self.closest_rays.append(
                (origin.x, origin.y, end_point.x, end_point.y))

    def _calculate_ray_end(self, origin, angle_offset):
        """Calculate ray end point with visual angle compensation"""
        visual_angle = self.angle - math.pi/2
        ray_angle = visual_angle + math.radians(angle_offset)
        dx = math.sin(ray_angle) * self.ray_length
        # Negative for pygame's Y axis
        dy = -math.cos(ray_angle) * self.ray_length
        collision_point = Point(origin.x + dx, origin.y + dy)
        closest_distance = self.ray_length

        for wall in self.walls:
            intersect = self._ray_wall_intersection(
                origin, collision_point, wall)
            if intersect and intersect.distance < closest_distance:
                closest_distance = intersect.distance
                collision_point = intersect.point

        return collision_point, closest_distance

    def _ray_wall_intersection(self, start: Point, end: Point, wall: Wall):
        x1, y1 = wall.x1, wall.y1
        x2, y2 = wall.x2, wall.y2
        x3, y3 = start.x, start.y
        x4, y4 = end.x, end.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 <= t <= 1 and 0 <= u <= 1:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            distance = math.hypot(px - x3, py - y3)
            return Intersection(distance, Point(px, py))

        return None

    def update(self):
        # Apply drag with drift modifier
        self.vel *= self.drag * (0.97 if self.traction < 0.6 else 1.0)

        if not self.accelerating:
            self.vel *= self.free_decel * (0.9 + 0.1 * self.traction)

        # Reset acceleration state
        self.accelerating = False

        # Clamp near-zero velocity
        if abs(self.vel) < 0.1:
            self.vel = 0

        # Update drift state
        self._update_traction()

        # Smooth angle interpolation with traction
        angle_diff = self.soll_angle - self.angle
        self.angle += angle_diff * 0.2 * self.traction

        # Apply drift physics
        if abs(self.vel) > self.drift_threshold and self.traction < 0.8:
            # Add angular momentum
            self.drift_angle += angle_diff * 0.1 * (1 - self.traction)
            self.drift_angle *= 0.95  # Natural decay
            self.angle += self.drift_angle

            # Calculate drift vector
            drift_vec = self._rotate_point(Point(0, self.vel * 0.4),
                                           self.angle + math.pi/2 * (1 if self.vel > 0 else -1))

            # Update position with traction blend
            base_vec = self._rotate_point(Point(0, self.vel), self.angle)
            self.x += base_vec.x * self.traction + \
                drift_vec.x * (1 - self.traction)
            self.y += base_vec.y * self.traction + \
                drift_vec.y * (1 - self.traction)
        else:
            # Normal movement
            vec = self._rotate_point(Point(0, self.vel), self.angle)
            self.x += vec.x
            self.y += vec.y

        # Update hitbox and rays
        self._update_hitbox_points()
        self.cast_rays()

        # Rotate image with slight drift visualization
        visual_angle = self.angle + self.drift_angle * 0.3
        self.image = pygame.transform.rotate(
            self.original_image,
            90 - math.degrees(visual_angle)
        )
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def _update_traction(self):
        # Auto-manage traction based on drift conditions
        steering_angle = abs(self.soll_angle - self.angle)
        speed_factor = abs(self.vel) / self.max_vel

        if speed_factor > 0.4 and steering_angle > math.radians(15):
            self.traction = max(0.4, self.traction - 0.04)
        else:
            self.traction = min(1.0, self.traction + 0.02)

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

    def check_checkpoint_collision(self, checkpoint: Checkpoint):
        if not hasattr(self, 'prev_front_center'):
            return False

        line_start = self.prev_front_center
        line_end = self.front_center

        return self._check_checkpoint_intersection(line_start, line_end, checkpoint)

    def _check_checkpoint_intersection(self, p1: Point, p2: Point, checkpoint: Checkpoint):
        x1, y1 = checkpoint.x1, checkpoint.y1
        x2, y2 = checkpoint.x2, checkpoint.y2
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
