from .game_track import Track
from .game_car import Car
from .game_point import Point
import pygame


class Game:
    def __init__(self):
        pygame.init()

        self.width = 1000
        self.height = 600

        self.running = True

        self.clock = pygame.time.Clock()

        self.surface = pygame.display.set_mode((self.width, self.height))

        self.track = Track(self.surface)

        self.car = Car(*self.track.starting_position,
                       self.track.starting_rotation)

        self.current_checkpoint_index = 0
        self.score = 0
        self.checkpoints = self.track.checkpoints
        self.checkpoints[0].is_active = True  # Activate first checkpoint

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def handle_input(self):
        keys = pygame.key.get_pressed()

        # Acceleration controls
        if keys[pygame.K_UP]:
            self.car.accelerate(1)  # Forward
        if keys[pygame.K_DOWN]:
            self.car.accelerate(-1)  # Reverse

        # Steering controls
        if keys[pygame.K_LEFT]:
            self.car.turn(-1)  # Left
        if keys[pygame.K_RIGHT]:
            self.car.turn(1)   # Right

    def run(self):
        while self.running:
            self.handle_events()
            self.handle_input()

            self.car.prev_front_center = Point(
                self.car.front_center.x, self.car.front_center.y)

            # Update car physics
            self.car.update()

            # Check checkpoint collision
            current_checkpoint = self.checkpoints[self.current_checkpoint_index]
            if current_checkpoint.is_active:
                if self.car.check_checkpoint_collision(current_checkpoint):
                    self._handle_checkpoint_passed()

            if self.car.check_collisions(self.track.walls_grid):
                self.handle_collision()

            # Render everything
            self.render()

            # Maintain 60 FPS
            self.clock.tick(60)

        pygame.quit()

    def _handle_checkpoint_passed(self):
        # Update score and checkpoint state
        self.score += 100
        self.checkpoints[self.current_checkpoint_index].is_active = False
        self.checkpoints[self.current_checkpoint_index].is_passed = True

        # Activate next checkpoint
        self.current_checkpoint_index = (
            self.current_checkpoint_index + 1) % len(self.checkpoints)
        self.checkpoints[self.current_checkpoint_index].is_active = True

    def handle_collision(self):
        """Handle collision consequences"""
        self.screen_flash()

        for checkpoint in self.checkpoints:
            checkpoint.is_active = False
            checkpoint.is_passed = False

        self.checkpoints[0].is_active = True
        self.current_checkpoint_index = 0
        self.score = 0
        self.car = Car(*self.track.starting_position,
                       self.track.starting_rotation)

    def screen_flash(self):
        """Visual feedback for collision"""
        flash = pygame.Surface((self.width, self.height))
        flash.fill((255, 255, 255))
        self.surface.blit(flash, (0, 0))
        pygame.display.flip()
        pygame.time.delay(10)  # 10ms flash

    def render(self):
        self.surface.fill("black")

        font = pygame.font.Font(None, 36)
        text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.surface.blit(text, (10, 10))

        self.track.render()

        self.car.draw(self.surface)

        pygame.display.flip()
