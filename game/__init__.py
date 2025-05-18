from .game_track import Track
from .game_car import Car
from .game_point import Point
import pygame
import math
import numpy as np
from ai.ddqn_agent import DDQNAgent
import os
import pickle
import random


class Game:
    CRASH_PENALTY_BASE = 50
    SAVE_EVERY = 10
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 3000
    ORGINAL_SAVE_PATH = "models/ddqn_agent_original.keras"
    TARGET_SAVE_PATH = "models/ddqn_agent_target.keras"
    EPSILON_SAVE_PATH = "models/ddqn_agent_epsilon.npy"
    EPISODES_SAVE_PATH = "models/ddqn_agent_episodes.npy"
    TOP_REWARD_SAVE_PATH = "models/ddqn_agent_top_rewards.npy"
    RECENT_REWARD_SAVE_PATH = "models/ddqn_agent_recent_rewards.npy"
    MEMORY_SAVE_PATH = "models/ddqn_agent_memory.pickle"
    STEPS_SAVE_PATH = "models/ddqn_agent_steps.pickle"
    TOP_N = 20
    RECENT_N = 50
    EPSILON_DROP_THRESHOLD = 0.7
    RESET_EPSILON_VALUE = 0.1
    EPSILON_BUMP_RATE = 0.005

    def __init__(self):
        pygame.init()
        self.width, self.height = 1000, 600
        self.surface = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.steps_since_checkpoint = 0
        self.prev_checkpoint_dist = None
        self.episode = 0
        self.top_rewards = [0]
        self.recent_rewards = [0]

        self.track = Track(self.surface)
        self.car = Car(*self.track.starting_position,
                       self.track, self.track.starting_rotation)

        self._reset_game_state()
        self.agent = DDQNAgent(len(self.car.get_state()) + 2, 9)

        # Create directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        self.load_model()

    def run(self):

        while True:
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

                car_pos = self.car.front
                dist_to_cp = current_checkpoint.distance_to(
                    car_pos.x, car_pos.y)

                if self.prev_checkpoint_dist is None:
                    self.prev_checkpoint_dist = dist_to_cp

                progess_reward = self.prev_checkpoint_dist - dist_to_cp
                self.prev_checkpoint_dist = dist_to_cp

                progess_reward = max(min(progess_reward, 1.0), -1.0)

                reward = self._compute_reward(state, action)

                reward += progess_reward

                if has_passed:
                    reward += 100

                next_state = self._get_state()
                done = self.car.check_collisions(self.track.walls_grid)

                episode_reward += reward

                if done or self.steps_since_checkpoint == 1000:
                    reward -= self.CRASH_PENALTY_BASE + abs(self.car.vel)
                    self.running = False
                    self.steps_since_checkpoint = 0

                self.agent.store_experience(
                    state, action, reward, next_state, done)

                self.render(reward)
                if done or self.steps_since_checkpoint == 1000:
                    self.handle_collision()

                next_state = self._get_state()
                state = next_state
                steps += 1

                shouldComputeNewEplison = steps == self.MAX_STEPS_PER_EPISODE or done or self.steps_since_checkpoint == 250
                self.agent.learn(shouldComputeEplison=shouldComputeNewEplison)

                self.clock.tick(60)  # now time penalty used elsewhere

            print(
                f"Episode {self.episode} | Total Reward: {episode_reward:.2f} | Steps: {steps}")

            self.recent_rewards.append(episode_reward)
            if len(self.recent_rewards) > self.TOP_N:
                self.recent_rewards = self.recent_rewards[-self.RECENT_N:]

            self.top_rewards.append(episode_reward)
            self.top_rewards = sorted(self.top_rewards, reverse=True)[
                :self.TOP_N]

            # combined_rewards = self.recent_rewards + self.top_rewards

            # avg_combined = sum(combined_rewards) / len(combined_rewards)

            avg_recent = sum(self.recent_rewards) / len(self.recent_rewards)
            avg_top = sum(self.top_rewards) / len(self.top_rewards)

            hybrid_avg = 0.60 * avg_recent + 0.40 * avg_top

            if episode_reward < self.EPSILON_DROP_THRESHOLD * hybrid_avg and self.agent.epsilon < 0.7:
                old_eps = self.agent.epsilon
                self.agent.epsilon = min(max(
                    self.RESET_EPSILON_VALUE, self.agent.epsilon + self.EPSILON_BUMP_RATE), 0.7)

                print(
                    f"[Epsilon Adjusted] Episode reward ({episode_reward:.2f}) < {self.EPSILON_DROP_THRESHOLD*100:.0f}% of hybrid average ({hybrid_avg:.2f}).")
                print(
                    f"  → Epsilon increased from {old_eps:.3f} → {self.agent.epsilon:.3f}")

            if self.episode % self.SAVE_EVERY == 0:
                self.save_model()

            with open("logs/episode_rewards.csv", "a") as f:
                f.write(
                    f"{self.episode},{episode_reward},{steps},{self.agent.epsilon}\n")

            if self.episode % 50 == 0 and len(self.agent.memory) >= 50:
                sampled = random.sample(self.agent.memory, 50)
                states = np.array([s[0] for s in sampled])

                online_q = self.agent.online_net.predict(states, verbose=0)
                target_q = self.agent.target_net.predict(states, verbose=0)
                delta_q = np.abs(online_q - target_q)

                mean_online_q = np.mean(online_q)
                std_online_q = np.std(online_q)
                mean_delta_q = np.mean(delta_q)
                max_delta_q = np.max(delta_q)

                # Save to file
                with open("logs/q_value_metrics.csv", "a") as f:
                    f.write(
                        f"{self.episode},{mean_online_q:.4f},{std_online_q:.4f},{mean_delta_q:.4f},{max_delta_q:.4f}\n")

            self.episode += 1

        pygame.quit()

    def _angle_to_checkpoint(self):
        checkpoint = self.checkpoints[self.current_checkpoint_index]
        car_angle = self.car.angle
        dx = checkpoint.center.x - self.car.front.x
        dy = checkpoint.center.y - self.car.front.y
        desired_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (desired_angle - car_angle) % 360 - 180

        return angle_diff

    def _reset_game_state(self):
        self.car = Car(*self.track.starting_position,
                       self.track, self.track.starting_rotation)
        self.score = 0
        self.current_checkpoint_index = 0
        self.checkpoints = self.track.checkpoints
        for cp in self.checkpoints:
            cp.is_active = cp.is_passed = False
        self.checkpoints[0].is_active = True
        self.prev_checkpoint_dist = None

        self.consecutive_checkpoints = 0
        self.combo_multiplier = 1.0
        self.current_velocity = 0.0
        self.highest_speed = 0.0
        self.max_combo = 1.0

    def _get_state(self):
        return np.array(self.car.get_state() + [self.current_checkpoint_index / len(self.track.checkpoints)] + [self.steps_since_checkpoint / 250], dtype=np.float32)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _compute_reward(self, state, action):
        reward = 0

        if (action == 5 or action == 6 or action == 1):
            reward += 1
        elif (action == 7 or action == 8 or action == 2):
            reward += 0.03
        elif (action == 3 or action == 4):
            reward += 0.001
        else:
            reward += 0.001

        # Velocity Reward
        reward += min(self.car.vel / self.car.max_vel, 1.0)

        reward += self._calculate_wall_proximity_penalty()

        angle_diff = self._angle_to_checkpoint()
        alignment_reward = max(0, 1 - angle_diff / 180)
        reward += alignment_reward

        return reward

    def _calculate_wall_proximity_penalty(self):
        penalty = 0.03
        threshold = 20  # pixels

        for dist, length in self.car.closest_ray_distances:
            norm_dist = dist / length
            if norm_dist < threshold / length:
                # stronger penalty as it gets closer
                penalty -= (1.0 - norm_dist) * 0.5
        return penalty

    def _apply_action(self, action):
        if action == 0:
            print("No action")
            return
        mapping = {
            1: (self.car.accelerate, 1),
            2: (self.car.accelerate, -1),
            3: (self.car.turn, -1),
            4: (self.car.turn, 1),
            5: ((self.car.accelerate, self.car.turn), (1, -1)),
            6: ((self.car.accelerate, self.car.turn), (1, 1)),
            7: ((self.car.accelerate, self.car.turn), (-1, -1)),
            8: ((self.car.accelerate, self.car.turn), (-1, 1)),
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
        self.prev_checkpoint_dist = None

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
        self.agent.online_net.save(self.ORGINAL_SAVE_PATH)
        self.agent.target_net.save(self.TARGET_SAVE_PATH)
        np.save(self.EPSILON_SAVE_PATH, self.agent.epsilon)
        np.save(self.EPISODES_SAVE_PATH, self.episode)
        self.agent.save_memory(self.MEMORY_SAVE_PATH)
        with open(self.TOP_REWARD_SAVE_PATH, 'wb') as f:
            pickle.dump(self.top_rewards, f)
        with open(self.RECENT_REWARD_SAVE_PATH, 'wb') as f:
            pickle.dump(self.recent_rewards, f)
        with open(self.STEPS_SAVE_PATH, 'wb') as f:
            pickle.dump(self.agent.step_count, f)

    def load_model(self):
        if os.path.exists(self.ORGINAL_SAVE_PATH):
            self.agent.online_net.load_weights(self.ORGINAL_SAVE_PATH)
        if os.path.exists(self.TARGET_SAVE_PATH):
            self.agent.target_net.load_weights(self.TARGET_SAVE_PATH)
        if os.path.exists(self.EPSILON_SAVE_PATH):
            self.agent.epsilon = np.load(self.EPSILON_SAVE_PATH)
        if os.path.exists(self.EPISODES_SAVE_PATH):
            self.episode = int(np.load(self.EPISODES_SAVE_PATH))
        if os.path.exists(self.MEMORY_SAVE_PATH):
            self.agent.load_memory(self.MEMORY_SAVE_PATH)
        if os.path.exists(self.TOP_REWARD_SAVE_PATH):
            with open(self.TOP_REWARD_SAVE_PATH, 'rb') as f:
                self.top_rewards = pickle.load(f)
        if os.path.exists(self.RECENT_REWARD_SAVE_PATH):
            with open(self.RECENT_REWARD_SAVE_PATH, 'rb') as f:
                self.recent_rewards = pickle.load(f)
        if os.path.exists(self.STEPS_SAVE_PATH):
            with open(self.STEPS_SAVE_PATH, 'rb') as f:
                self.agent.step_count = pickle.load(f)

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
