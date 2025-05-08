from .game_track import Track
from .game_car import Car
from .game_point import Point
import pygame
import math
import numpy as np
from ai.ddqn_agent import DDQNAgent
import os


class Game:
    MIN_SAFE_DISTANCE = 20
    COLLISION_PROXIMITY = 10
    TIME_PENALTY = 0.01
    VELOCITY_BONUS_FACTOR = 0.2
    EDGE_PENALTY_FACTOR = 0.5
    CRASH_PENALTY_BASE = 30
    SAVE_EVERY = 10
    MAX_EPISODES = 500
    MAX_STEPS_PER_EPISODE = 3000
    SAVE_PATH = "models/ddqn_agent.keras"
    ACCELERATION_PRIORITIZATION_BONUS = 0.5
    LEARN_EVERY = 30
    SYMMETRY_REWARD_FACTOR = 0.2
    ALIGNMENT_REWARD_FACTOR = 0.4
    LOW_VELOCITY_PENALTY_FACTOR = 0.1

    # Acceleration prioritization
    ACCEL_ACTIONS = {1, 5, 6}       # Forward
    DECEL_ACTIONS = {2, 7, 8}       # Reverse
    NO_ACCEL_ACTIONS = {0}          # Idle or do nothing

    def __init__(self):
        pygame.init()
        self.width, self.height = 1000, 600
        self.surface = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.steps_since_checkpoint = 0

        self.track = Track(self.surface)
        self.car = Car(*self.track.starting_position,
                       self.track, self.track.starting_rotation)

        self._reset_game_state()
        self.agent = DDQNAgent(len(self.car.get_state()) + 2, 8)
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
        if os.path.exists(self.SAVE_PATH):
            self.agent.load_model(self.SAVE_PATH)
        else:
            print(f"No saved model found at {self.SAVE_PATH}, starting fresh.")

    def run(self):
        for self.episode in range(1, self.MAX_EPISODES + 1):
            state = self._get_state()
            episode_reward = 0
            steps = 0
            self.running = True

            while self.running and steps < self.MAX_STEPS_PER_EPISODE:
                self.handle_events()
                action = self.agent.choose_action(state)
                self._apply_action(action)

                self.car.prev_front_center = Point(
                    self.car.right_center.x, self.car.right_center.y)
                self.car.update()

                current_checkpoint = self.checkpoints[self.current_checkpoint_index]
                has_passed = self.car.check_checkpoint_collision(
                    current_checkpoint)
                if has_passed:
                    self.steps_since_checkpoint = 0
                    self._handle_checkpoint_passed(current_checkpoint)
                else:
                    self.steps_since_checkpoint += 1

                reward = self._compute_reward(state, action)
                done = self.car.check_collisions(self.track.walls_grid)
                if done or self.steps_since_checkpoint > 250:
                    reward -= self.CRASH_PENALTY_BASE + abs(self.car.vel)
                    self.steps_since_checkpoint = 0
                    done = True
                    self.handle_collision()

                next_state = self._get_state()
                self.agent.store_experience(
                    state, action, reward, next_state, done)

                self.agent.learn()

                state = next_state
                episode_reward += reward
                steps += 1
                self.total_steps += 1

                self.render(reward)
                self.clock.tick(60)  # now time penalty used elsewhere

            print(
                f"Episode {self.episode} | Total Reward: {episode_reward:.2f} | Steps: {steps}")
            if self.episode % self.SAVE_EVERY == 0:
                self.save_model()

        pygame.quit()

    def _reset_game_state(self):
        self.car = Car(*self.track.starting_position,
                       self.track, self.track.starting_rotation)
        self.score = 0
        self.current_checkpoint_index = 0
        self.checkpoints = self.track.checkpoints
        for cp in self.checkpoints:
            cp.is_active = cp.is_passed = False
        self.checkpoints[0].is_active = True

        self.consecutive_checkpoints = 0
        self.combo_multiplier = 1.0
        self.current_velocity = 0.0
        self.highest_speed = 0.0
        self.total_steps = 0
        self.max_combo = 1.0

    def _get_state(self):
        return np.array(self.car.get_state() + [self.current_checkpoint_index / len(self.track.checkpoints)] + [self.steps_since_checkpoint / 250], dtype=np.float32)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _compute_reward(self, state, action):
        # Base reward: progress through checkpoints
        reward = self.score/50

        # Example: use mid-left and mid-right ray distances
        # left = self.car.closest_ray_distances[0]  # Adjust index to match cast_rays()
        # right = self.car.closest_ray_distances[-1]
        #
        # # Symmetry reward (max at center)
        # symmetry = 1 - abs(left - right) / self.car.ray_length
        # reward += symmetry * self.SYMMETRY_REWARD_FACTOR

        # if action in self.ACCEL_ACTIONS:
        #     reward += self.ACCELERATION_PRIORITIZATION_BONUS
        # elif action in self.DECEL_ACTIONS or action in self.NO_ACCEL_ACTIONS:
        #     reward -= self.ACCELERATION_PRIORITIZATION_BONUS * 0.5  # Mild penalty for non-acceleration

        # # Smooth driving: penalize steering changes
        # prev_angle = self.car.prev_angle if hasattr(self.car, 'prev_angle') else self.car.angle
        # angle_diff = abs(self.car.soll_angle - prev_angle)
        # reward -= angle_diff * 0.01
        # self.car.prev_angle = self.car.angle

        # Time penalty
        reward -= self.TIME_PENALTY

        if self.steps_since_checkpoint > 50:
            reward -= self.TIME_PENALTY

        #  Velocity bonus

        # if self.car.vel < 0.1:
        #     reward -= self.LOW_VELOCITY_PENALTY_FACTOR
        # else:
        #     vel_ratio = abs(self.car.vel) / self.car.max_vel
        #     reward += vel_ratio * self.VELOCITY_BONUS_FACTOR

        # Proximity penalty: smooth based
        # min_dist = min(self.car.closest_ray_distances)
        # if min_dist < self.MIN_SAFE_DISTANCE:
        #     reward -= (self.MIN_SAFE_DISTANCE - min_dist) * \
        #         self.EDGE_PENALTY_FACTOR

        # Progress reward: inversely proportional to distance (non-linear scaling)
        next_cp = self.checkpoints[self.current_checkpoint_index]
        car_pos = np.array([self.car.x, self.car.y])
        cp_center = np.array(
            [(next_cp.x1 + next_cp.x2) / 2, (next_cp.y1 + next_cp.y2) / 2])

        dist = np.linalg.norm(car_pos - cp_center)
        normalized_dist = dist / np.hypot(self.width, self.height)
        #
        # # Nonlinear shaping: steeper reward near the checkpoint
        # # Max ~2.0 near checkpoint, drops off faster
        progress_reward = (1 - normalized_dist) ** 2 * 2.0
        if self.car.vel > 0.2:
            reward += progress_reward

        # Steering smoothness
        # alignment = math.cos(self.car.soll_angle - self.car.angle)
        # reward += alignment * 0.5

        return reward

    def _apply_action(self, action):
        mapping = {
            0: (self.car.accelerate, 1),
            1: (self.car.accelerate, -1),
            2: (self.car.turn, -1),
            3: (self.car.turn, 1),
            4: ((self.car.accelerate, self.car.turn), (1, -1)),
            5: ((self.car.accelerate, self.car.turn), (1, 1)),
            6: ((self.car.accelerate, self.car.turn), (-1, -1)),
            7: ((self.car.accelerate, self.car.turn), (-1, 1)),
        }
        if action in mapping:
            funcs, args = mapping[action]
            if isinstance(funcs, tuple):
                for f, a in zip(funcs, args):
                    f(a)
            else:
                funcs(args)

    def _handle_checkpoint_passed(self, checkpoint):
        self.current_velocity = self.car.vel
        self.highest_speed = max(
            self.highest_speed, abs(self.current_velocity))

        cp_reward = self._calculate_checkpoint_reward(checkpoint)
        self.score += cp_reward
        self.consecutive_checkpoints += 1
        self._update_combo_multiplier()

        checkpoint.is_active = False
        checkpoint.is_passed = True
        self.current_checkpoint_index = (
            self.current_checkpoint_index + 1) % len(self.checkpoints)
        self.checkpoints[self.current_checkpoint_index].is_active = True

    def _calculate_checkpoint_reward(self, checkpoint):
        vel_ratio = min(1.0, abs(self.current_velocity) / self.car.max_vel)
        speed_bonus = 1.0 + vel_ratio**2
        combo_bonus = 1.0 + (self.consecutive_checkpoints * 0.15)
        return checkpoint.base_reward * checkpoint.difficulty_factor * speed_bonus * combo_bonus * self.combo_multiplier

    def _update_combo_multiplier(self):
        speed_factor = min(2.0, 1.0 + abs(self.car.vel)/self.car.max_vel)
        streak_factor = 1.0 + (self.consecutive_checkpoints * 0.05)
        self.combo_multiplier = min(3.0, speed_factor * streak_factor)
        self.max_combo = max(self.max_combo, self.combo_multiplier)

    def handle_collision(self):
        self.screen_flash()
        self._reset_game_state()

    def screen_flash(self):
        flash = pygame.Surface((self.width, self.height))
        flash.fill((255, 255, 255))
        self.surface.blit(flash, (0, 0))
        pygame.display.flip()
        pygame.time.delay(10)

    def render(self, reward):
        self.surface.fill('black')
        self._draw_text(f"Reward: {reward}", (10, 10), 36)
        self.track.render()
        self.car.draw(self.surface)
        self._draw_speedometer()
        self._draw_combo_meter()
        pygame.display.flip()

    def save_model(self):
        self.agent.save_model(self.SAVE_PATH)
        print(f"Model saved at episode {self.episode} -> {self.SAVE_PATH}")

    def _draw_text(self, text, pos, size, color=(255, 255, 255)):
        font = pygame.font.Font(None, size)
        surf = font.render(text, True, color)
        self.surface.blit(surf, pos)

    def _draw_speedometer(self):
        speed = abs(self.car.vel)
        ratio = speed/self.car.max_vel
        c = (self.width-80, self.height-80)
        r = 30
        pygame.draw.circle(self.surface, (50, 50, 50), c, r, 3)
        pygame.draw.arc(self.surface, (0, 255, 0),
                        (c[0]-r, c[1]-r, r*2, r*2), -math.pi/2, -math.pi/2+2*math.pi*ratio, 5)
        self._draw_text(f"{speed:.1f}", (c[0]-15, c[1]-10), 24)

    def _draw_combo_meter(self):
        ratio = self.combo_multiplier/3.0
        pygame.draw.rect(self.surface, (50, 50, 50),
                         (20, self.height-40, 200, 20))
        pygame.draw.rect(self.surface, (255, 215, 0),
                         (20, self.height-40, 200*ratio, 20))
        self._draw_text(
            f"COMBO x{self.combo_multiplier:.1f}", (20, self.height-60), 24)
