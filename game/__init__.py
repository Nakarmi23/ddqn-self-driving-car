from .game_track import Track
from .game_car import Car
from .game_point import Point
import pygame
import math
import numpy as np
from ai.ddqn_agent import DDQNAgent


class Game:
    def __init__(self):
        pygame.init()

        # Existing initialization

        self.width = 1000
        self.height = 600

        self.running = True

        self.clock = pygame.time.Clock()

        self.surface = pygame.display.set_mode((self.width, self.height))

        self.track = Track(self.surface)

        self.car = Car(*self.track.starting_position,
                       self.track, self.track.starting_rotation)

        self.consecutive_checkpoints = 0
        self.combo_multiplier = 1.0
        self.current_velocity = 0.0
        self.max_combo = 1
        self.highest_speed = 0.0

        self.current_checkpoint_index = 0
        self.score = 0
        self.checkpoints = self.track.checkpoints
        self.checkpoints[0].is_active = True  # Activate first checkpoint

        state_size = len(self.car.get_state())+1
        action_size = 9  # Modify based on your action space
        self.agent = DDQNAgent(state_size, action_size)
        self.episode = 0
        self.total_steps = 0

    def _calculate_reward(self, checkpoint):
        # Normalize velocity (0-1 range based on car's max speed)
        velocity_ratio = min(
            1.0, abs(self.current_velocity) / self.car.max_vel)

        # Difficulty curve (exponential reward for high speeds)
        speed_bonus = 1.0 + (velocity_ratio ** 2)  # 1-2 range

        # Combo system (increases with consecutive checkpoints)
        combo_bonus = 1.0 + (self.consecutive_checkpoints * 0.15)

        # Calculate final reward
        reward = (
            checkpoint.base_reward
            * checkpoint.difficulty_factor
            * speed_bonus
            * combo_bonus
            * self.combo_multiplier
        )

        return int(reward)

    def _update_combo_multiplier(self):
        # Combo grows with speed and consecutive checkpoints
        speed_component = min(
            2.0, 1.0 + (abs(self.current_velocity) / self.car.max_vel))
        streak_component = 1.0 + (self.consecutive_checkpoints * 0.05)
        self.combo_multiplier = min(3.0, speed_component * streak_component)
        self.max_combo = max(self.max_combo, self.combo_multiplier)

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
        state = np.array(self.car.get_state() +
                         [self.current_checkpoint_index], dtype=np.float32)
        while self.running:

            self.handle_events()
            action = self.agent.choose_action(state)
            self._apply_action(action)

            self.car.prev_front_center = Point(
                self.car.front_center.x, self.car.front_center.y)

            # Update car physics
            self.car.update()

            # Check checkpoint collision
            current_checkpoint = self.checkpoints[self.current_checkpoint_index]
            if current_checkpoint.is_active:
                if self.car.check_checkpoint_collision(current_checkpoint):
                    self._handle_checkpoint_passed()
            has_collision = self.car.check_collisions(self.track.walls_grid)
            if has_collision:
                self.handle_collision()

            # Render everything
            self.render()

            reward = self.score / 100.0

            next_state = np.array(self.car.get_state(
            ) + [self.current_checkpoint_index], dtype=np.float32)
            self.agent.store_experience(
                state, action, reward, next_state, not self.running)
            self.agent.learn()

            state = next_state
            self.total_steps += 1
            print(self.total_steps)
            # Maintain 60 FPS
            self.clock.tick(60)

        pygame.quit()

    def _handle_checkpoint_passed(self):

        current = self.checkpoints[self.current_checkpoint_index]

        # Store velocity at moment of passing
        self.current_velocity = self.car.vel
        self.highest_speed = max(
            self.highest_speed, abs(self.current_velocity))

        # Calculate and award reward
        reward = self._calculate_reward(current)
        self.score += reward

        # Update streak
        self.consecutive_checkpoints += 1

        # Visual feedback
        self._show_speed_bonus(current, reward)
        self._update_combo_multiplier()

        # Progress to next checkpoint
        current.is_active = False
        current.is_passed = True
        self.current_checkpoint_index = (
            self.current_checkpoint_index + 1) % len(self.checkpoints)
        self.checkpoints[self.current_checkpoint_index].is_active = True

    def handle_collision(self):
        """Handle collision consequences"""
        self.screen_flash()

        speed_penalty = int(abs(self.car.vel)) * 2
        streak_penalty = self.consecutive_checkpoints * 10
        combo_penalty = int(self.combo_multiplier * 50)

        total_penalty = speed_penalty + streak_penalty + combo_penalty
        self.score = max(0, self.score - total_penalty)

        # Reset counters
        self.consecutive_checkpoints = 0
        self.combo_multiplier = 1.0

        for checkpoint in self.checkpoints:
            checkpoint.is_active = False
            checkpoint.is_passed = False

        self.checkpoints[0].is_active = True
        self.current_checkpoint_index = 0
        self.score = 0

        self.car = Car(*self.track.starting_position,
                       self.track, self.track.starting_rotation)

    def screen_flash(self):
        """Visual feedback for collision"""
        flash = pygame.Surface((self.width, self.height))
        flash.fill((255, 255, 255))
        self.surface.blit(flash, (0, 0))
        pygame.display.flip()
        pygame.time.delay(10)  # 10ms flash

    def _show_speed_bonus(self, checkpoint, reward):
        # Speed indicator colors
        speed_ratio = abs(self.current_velocity) / self.car.max_vel
        color = (
            int(255 * speed_ratio),
            int(255 * (1 - speed_ratio)),
            0
        )

        # Reward text
        font = pygame.font.Font(None, 28)
        texts = [
            f"+{reward}",
            f"Speed: {abs(self.current_velocity):.1f} px/s",
            f"Combo: x{self.combo_multiplier:.1f}"
        ]

        # Position above checkpoint
        cx = (checkpoint.x1 + checkpoint.x2) // 2
        cy = (checkpoint.y1 + checkpoint.y2) // 2

        for i, text in enumerate(texts):
            text_surface = font.render(
                text, True, color if i == 0 else (200, 200, 200))
            self.surface.blit(text_surface, (cx - 50, cy - 30 - i*20))

    def render(self):
        self.surface.fill("black")

        font = pygame.font.Font(None, 36)
        text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.surface.blit(text, (10, 10))

        self.track.render()

        self.car.draw(self.surface)

        # Speed gauge
        self._draw_speedometer()

        # Combo display
        self._draw_combo_meter()

        pygame.display.flip()

    def _draw_speedometer(self):
        speed = abs(self.car.vel)
        max_speed = self.car.max_vel
        ratio = speed / max_speed

        # Draw circular gauge
        center = (self.width - 80, self.height - 80)
        radius = 30
        pygame.draw.circle(self.surface, (50, 50, 50), center, radius, 3)
        pygame.draw.arc(self.surface, (0, 255, 0),
                        (center[0]-radius, center[1] -
                         radius, radius*2, radius*2),
                        -math.pi/2, -math.pi/2 + 2*math.pi*ratio, 5)

        # Speed text
        font = pygame.font.Font(None, 24)
        text = font.render(f"{speed:.1f}", True, (255, 255, 255))
        self.surface.blit(text, (center[0]-15, center[1]-10))

    def _draw_combo_meter(self):
        # Combo bar
        combo_ratio = self.combo_multiplier / 3.0
        pygame.draw.rect(self.surface, (50, 50, 50),
                         (20, self.height-40, 200, 20))
        pygame.draw.rect(self.surface, (255, 215, 0),
                         (20, self.height-40, 200 * combo_ratio, 20))

        # Combo text
        font = pygame.font.Font(None, 24)
        text = font.render(
            f"COMBO x{self.combo_multiplier:.1f}", True, (255, 255, 255))
        self.surface.blit(text, (20, self.height-60))

    def _apply_action(self, action):
        """Map action index to car controls"""
        # Example mapping:
        if action == 0:  # Coast
            pass
        elif action == 1:  # Accelerate
            self.car.accelerate(1)
        elif action == 2:  # Brake
            self.car.accelerate(-1)
        elif action == 3:  # Left
            self.car.turn(-1)
        elif action == 4:  # Right
            self.car.turn(1)
        elif action == 5:  # Accelerate and left
            self.car.accelerate(1)
            self.car.turn(-1)
        elif action == 6:  # Accelerate and right
            self.car.accelerate(1)
            self.car.turn(1)
        elif action == 7:  # Brake and left
            self.car.accelerate(-1)
            self.car.turn(-1)
        elif action == 8:  # Brake and right
            self.car.accelerate(-1)
            self.car.turn(1)
