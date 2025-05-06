from .game_track import Track
from .game_car import Car
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

            # Update car physics
            self.car.update()

            if self.car.check_collisions(self.track.walls_grid):
                self.handle_collision()

            # Render everything
            self.render()

            # Maintain 60 FPS
            self.clock.tick(60)

        pygame.quit()

    def handle_collision(self):
        """Handle collision consequences"""
        print("Collision detected!")
        # Implement your collision response:
        # - Reset position
        # - Apply penalty
        # - End episode
        # self.car.vel = 0  # Immediate stop
        # self.car.vel *= -0.5  # Bounce effect

        self.screen_flash()
        self.car = Car(*self.track.starting_position,
                       self.track.starting_rotation)

    def screen_flash(self):
        """Visual feedback for collision"""
        flash = pygame.Surface((self.width, self.height))
        flash.fill((255, 255, 255))
        self.surface.blit(flash, (0, 0))
        pygame.display.flip()
        pygame.time.delay(10)  # 100ms flash

    def render(self):
        self.surface.fill("black")

        self.track.render()

        self.car.draw(self.surface)

        pygame.display.flip()
