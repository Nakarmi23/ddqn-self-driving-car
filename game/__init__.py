from .game_track import Track
from .game_car import Car
from .game_point import Point
import pygame
import math
import numpy as np
import torch
from ai.ddqn_agent import DDQNAgent
from utils.csv_logger import CsvLogger
import glob
import os


def _env_flag(name):
    """Boolean env var: 1/true/yes/on (any case) is true; unset, empty or
    0/false/no/off is false; anything else is an error."""
    value = os.environ.get(name, "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise SystemExit(f"{name}={os.environ[name]!r}: expected 1/0, true/false, yes/no or on/off")


def _env_positive_int(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise SystemExit(f"{name}={os.environ[name]!r}: expected a positive integer")
    return number


class Game:
    # Reward scale, see docs/adr/0001-progress-led-reward.md.
    CRASH_PENALTY_BASE = 50
    CHECKPOINT_BONUS = 100
    SPEED_WEIGHT = 0.1  # weight on speed toward the checkpoint (zero-mean)
    ALIGNMENT_WEIGHT = 0.04  # weight on alignment (mean 0.5 for random heading)
    # Per-step cost. Equals the mean alignment shaping for a random heading,
    # so aimless driving nets ~0 instead of a survival bonus. Kept small so a
    # stall (STALL_LIMIT * TIME_COST + crash penalty, ~-70) costs about the
    # same as crashing (-50..-65); a larger cost made crashing the cheaper
    # way out once stuck.
    TIME_COST = ALIGNMENT_WEIGHT * 0.5
    SAVE_EVERY = 10
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 3000
    STALL_LIMIT = 1000  # steps without passing a checkpoint before giving up
    CHECKPOINT_LATEST_PATH = "models/checkpoint_latest.pt"
    CHECKPOINT_HISTORY_DIR = "models/history"
    KEEP_HISTORY_COUNT = 5
    TOP_N = 20
    RECENT_N = 50
    # Training budget in learning steps (agent.step_count). Stopping on it
    # is tracked in issue #10; for now it only sets the epsilon and PER beta
    # schedules.
    TOTAL_TRAIN_STEPS = 1_000_000
    # Fraction of the budget over which epsilon decays linearly 1.0 -> min.
    EPSILON_DECAY_FRACTION = 0.4
    FPS_CAP = 60  # only applied when drawing every step (watch mode)

    def __init__(self):
        # Display modes, from env vars:
        # - default: draw every step, capped at FPS_CAP (watchable).
        # - RENDER_EVERY=N (N > 1): draw every Nth step, no frame cap.
        # - HEADLESS=1: no window, never draw, no frame cap. Fastest.
        self.headless = _env_flag("HEADLESS")
        self.render_every = _env_positive_int("RENDER_EVERY", 1)
        if self.headless:
            # Override, not setdefault: a pre-set real driver would still
            # open a (never drawn, unresponsive) window.
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.frame_cap = (None if self.headless or self.render_every > 1
                          else self.FPS_CAP)
        self.total_steps = 0  # environment steps this process, for render_every

        pygame.init()
        self.width, self.height = 1000, 600
        self.surface = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.steps_since_checkpoint = 0
        self.prev_checkpoint_dist = None
        self.episode = 0
        self.top_rewards = [0]
        self.recent_rewards = [0]

        self.track = Track(self.surface, os.environ.get(
            "TRACK_TEMPLATE", "template5"))
        self.car = Car(*self.track.starting_position,
                       self.track, self.track.starting_rotation)

        self._reset_game_state()
        self._check_spawn()
        self.agent = DDQNAgent(
            len(self.car.get_state()) + 2, 9,
            epsilon_decay_steps=int(self.TOTAL_TRAIN_STEPS * self.EPSILON_DECAY_FRACTION),
            beta_anneal_steps=self.TOTAL_TRAIN_STEPS)

        # Create directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs(self.CHECKPOINT_HISTORY_DIR, exist_ok=True)

        self.episode_log = CsvLogger(
            "logs/episode_rewards.csv", ["episode", "reward", "steps", "epsilon"])
        self.q_value_log = CsvLogger("logs/q_value_metrics.csv", [
            "episode", "mean_online_q", "std_online_q", "mean_delta_q", "max_delta_q"])

        self.load_model()

    def _should_render(self):
        return not self.headless and self.total_steps % self.render_every == 0

    def _logs(self):
        return (self.episode_log, self.q_value_log, self.agent.training_log)

    def run(self):
        try:
            while True:
                episode_reward, steps, quit_requested = self._run_episode()
                if quit_requested:
                    break
                self._finish_episode(episode_reward, steps)
        except KeyboardInterrupt:
            # Ctrl+C (how headless runs are stopped) is handled like closing
            # the window. save_model writes atomically, so an interrupt that
            # lands mid-save still leaves the previous checkpoint intact.
            print("\nInterrupted, saving checkpoint...")

        # The interrupted episode isn't logged; save so the next run resumes
        # at this same (unfinished) episode number.
        self.save_model(write_history=False)
        for log in self._logs():
            log.close()
        pygame.quit()

    @staticmethod
    def _atomic_torch_save(obj, path):
        """Write to a temp file then rename over `path`, so an interrupt or
        crash mid-write never leaves a truncated checkpoint behind."""
        tmp_path = path + ".tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    def _run_episode(self):
        """Step the environment until the episode ends.

        Returns (episode_reward, steps, quit_requested). Every way an episode
        can end (crash, stall timeout, max steps) falls through to the single
        _reset_game_state() call at the bottom, so the next episode always
        starts from a clean reset.
        """
        state = self._get_state()
        episode_reward = 0
        steps = 0

        while steps < self.MAX_STEPS_PER_EPISODE:
            if self.handle_events():
                return episode_reward, steps, True

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

            # Distance closed this step, scaled by top speed so full speed
            # straight at the checkpoint earns ~+1. Summed over a segment it
            # equals the distance covered, independent of how slowly the car
            # drove it. See docs/adr/0001-progress-led-reward.md.
            progress_reward = (self.prev_checkpoint_dist - dist_to_cp) / self.car.max_vel
            self.prev_checkpoint_dist = dist_to_cp

            progress_reward = max(min(progress_reward, 1.0), -1.0)

            reward = self._compute_reward()

            reward += progress_reward

            if has_passed:
                reward += self.CHECKPOINT_BONUS

            next_state = self._get_state()
            crashed = self.car.check_collisions(self.track.walls_grid)
            stalled = self.steps_since_checkpoint >= self.STALL_LIMIT
            episode_over = crashed or stalled

            # Terminal semantics:
            # - crash and stall timeout are both failures: penalised and
            #   stored with done=True. The stall is a true terminal state
            #   because steps_since_checkpoint / STALL_LIMIT is part of the
            #   observation, so the agent can see the deadline coming.
            # - hitting MAX_STEPS_PER_EPISODE is truncation: no penalty and
            #   done=False, so the target still bootstraps from next_state.
            if episode_over:
                reward -= self.CRASH_PENALTY_BASE + abs(self.car.vel)

            episode_reward += reward

            self.agent.store_experience(
                state, action, reward, next_state, episode_over)

            rendering = self._should_render()
            if rendering:
                self.render(episode_reward)

            state = next_state
            steps += 1
            self.total_steps += 1

            self.agent.learn()

            if self.frame_cap:
                self.clock.tick(self.frame_cap)

            if episode_over:
                if rendering:
                    self.screen_flash()
                break

        # Single reset path for every way an episode can end.
        self._reset_game_state()
        return episode_reward, steps, False

    def _finish_episode(self, episode_reward, steps):
        """Bookkeeping for a completed episode: stats, logs, saving."""
        print(
            f"Episode {self.episode} | Total Reward: {episode_reward:.2f} | Steps: {steps}")

        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > self.TOP_N:
            self.recent_rewards = self.recent_rewards[-self.RECENT_N:]

        self.top_rewards.append(episode_reward)
        self.top_rewards = sorted(self.top_rewards, reverse=True)[
            :self.TOP_N]

        self.episode_log.log(
            self.episode, f"{episode_reward:.2f}", steps, f"{self.agent.epsilon:.4f}")

        if self.episode % 50 == 0 and len(self.agent.memory) >= 50:
            sample = np.random.choice(
                len(self.agent.memory), 50, replace=False)
            states = self.agent.memory.states[sample]

            online_q = self.agent.predict_batch(
                self.agent.online_net, states)
            target_q = self.agent.predict_batch(
                self.agent.target_net, states)
            delta_q = np.abs(online_q - target_q)

            mean_online_q = np.mean(online_q)
            std_online_q = np.std(online_q)
            mean_delta_q = np.mean(delta_q)
            max_delta_q = np.max(delta_q)

            self.q_value_log.log(
                self.episode, f"{mean_online_q:.4f}", f"{std_online_q:.4f}",
                f"{mean_delta_q:.4f}", f"{max_delta_q:.4f}")

        completed_episode = self.episode
        self.episode += 1

        # Saved after incrementing so the checkpoint records the next episode
        # to run, not the one just finished.
        if completed_episode % self.SAVE_EVERY == 0:
            self.save_model()

    def _angle_to_checkpoint(self):
        """Signed angle in radians, wrapped to [-pi, pi), between the car's
        heading and the direction from the car's front to the active
        checkpoint. 0 means the car points straight at it.

        Car.update() moves the car along rotate((0, vel), angle), i.e.
        direction (-sin a, cos a), whose screen-space atan2 is angle + pi/2.
        """
        checkpoint = self.checkpoints[self.current_checkpoint_index]
        heading = self.car.angle + math.pi / 2
        dx = checkpoint.center.x - self.car.front.x
        dy = checkpoint.center.y - self.car.front.y
        desired = math.atan2(dy, dx)
        return (desired - heading + math.pi) % (2 * math.pi) - math.pi

    def _check_spawn(self):
        """Refuse a track whose spawn hitbox already touches a checkpoint
        or wall: the pass would be counted on step 1 without the car moving
        (issue #8). `python -m scripts.check_spawns` checks every template."""
        overlaps = self.car.overlapping_checkpoints()
        if overlaps:
            raise ValueError(
                f"Spawn {self.track.starting_position} overlaps checkpoint(s) "
                f"{overlaps}; move starting_position back behind checkpoint 0.")
        if self.car.check_collisions(self.track.walls_grid):
            raise ValueError(
                f"Spawn {self.track.starting_position} overlaps a wall.")

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
        self.steps_since_checkpoint = 0

        self.consecutive_checkpoints = 0
        self.current_velocity = 0.0
        self.highest_speed = 0.0

    def _get_state(self):
        return np.array(self.car.get_state() + [self.current_checkpoint_index / len(self.track.checkpoints)] + [min(self.steps_since_checkpoint / self.STALL_LIMIT, 1.0)], dtype=np.float32)

    def handle_events(self):
        """Pump the pygame event queue. Returns True if the window was closed."""
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
        return quit_requested

    def _compute_reward(self):
        """Per-step shaping terms. Progress, checkpoint bonus and crash
        penalty are added in _run_episode. Shaping is kept small and roughly
        zero-mean for aimless driving so that only progress pays; see
        docs/adr/0001-progress-led-reward.md."""
        angle_diff = self._angle_to_checkpoint()

        # Speed toward the checkpoint: positive driving at it, negative
        # driving away, ~0 on average when circling.
        speed_toward = (self.car.vel / self.car.max_vel) * math.cos(angle_diff)

        # 1 when pointing at the checkpoint, 0 when pointing directly away.
        alignment = 1 - abs(angle_diff) / math.pi

        reward = self.SPEED_WEIGHT * speed_toward + self.ALIGNMENT_WEIGHT * alignment
        reward -= self.TIME_COST
        reward += self._calculate_wall_proximity_penalty()
        return reward

    def _calculate_wall_proximity_penalty(self):
        penalty = 0.0
        threshold = 20  # pixels

        for dist, length in self.car.closest_ray_distances:
            norm_dist = dist / length
            if norm_dist < threshold / length:
                # stronger penalty as it gets closer
                penalty -= (1.0 - norm_dist) * 0.5
        return penalty

    def _apply_action(self, action):
        if action == 0:
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

        self.consecutive_checkpoints += 1

        checkpoint.is_active = False
        checkpoint.is_passed = True
        self.current_checkpoint_index = (
            self.current_checkpoint_index + 1) % len(self.checkpoints)
        self.checkpoints[self.current_checkpoint_index].is_active = True
        self.prev_checkpoint_dist = None

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
        pygame.display.flip()

    def save_model(self, write_history=True):
        """Save a resumable checkpoint.

        `next_episode` is the episode number training should resume at.
        History files are named after the last completed episode.
        """
        run_state = {
            "next_episode": self.episode,
            "top_rewards": self.top_rewards,
            "recent_rewards": self.recent_rewards,
        }

        checkpoint = self.agent.export_state()
        checkpoint.update(run_state)
        self._atomic_torch_save(checkpoint, self.CHECKPOINT_LATEST_PATH)
        # Keep the CSV logs on disk in step with the checkpoint.
        for log in self._logs():
            log.flush()

        if write_history:
            # History is a rollback point for the weights and counters; the
            # replay buffer (~16 MB at 100k) is only kept in the latest
            # checkpoint; restoring a history file refills it via warm-up.
            history = self.agent.export_state(include_buffer=False)
            history.update(run_state)
            history_path = os.path.join(
                self.CHECKPOINT_HISTORY_DIR, f"checkpoint_ep{self.episode - 1:06d}.pt")
            self._atomic_torch_save(history, history_path)
            self._prune_history()

    def _prune_history(self):
        checkpoints = sorted(glob.glob(os.path.join(
            self.CHECKPOINT_HISTORY_DIR, "checkpoint_ep*.pt")))
        for stale_checkpoint in checkpoints[:-self.KEEP_HISTORY_COUNT]:
            os.remove(stale_checkpoint)

    def load_model(self):
        if not os.path.exists(self.CHECKPOINT_LATEST_PATH):
            return

        # weights_only=False: the checkpoint bundles the replay memory (numpy
        # arrays/scalars), which PyTorch >= 2.6's safe unpickler rejects. This
        # file is only ever written by save_model() locally, so it's trusted.
        checkpoint = torch.load(
            self.CHECKPOINT_LATEST_PATH, map_location=self.agent.device,
            weights_only=False)
        self.agent.import_state(checkpoint)
        if "next_episode" in checkpoint:
            self.episode = checkpoint["next_episode"]
        else:
            # Older checkpoints stored the last completed episode.
            self.episode = checkpoint["episode"] + 1
        self.top_rewards = checkpoint["top_rewards"]
        self.recent_rewards = checkpoint["recent_rewards"]

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
