import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai.replay_buffer import PrioritizedReplayBuffer
from utils.csv_logger import CsvLogger


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        return self.net(x)


class DDQNAgent:
    MEMORY_CAPACITY = 100_000
    # Transitions to collect (at epsilon 1.0) before the first learning step,
    # so early updates don't overfit a handful of correlated transitions.
    WARMUP_TRANSITIONS = 5_000
    # Soft (Polyak) target update rate, applied after every learning step.
    TARGET_TAU = 0.01
    LOG_EVERY = 1000  # learning steps between stdout progress lines

    def __init__(self, state_size, action_size, epsilon_decay_steps=400_000,
                 beta_anneal_steps=1_000_000):
        self.state_size = state_size
        self.action_size = action_size
        self.priority_alpha = 0.6
        # PER importance-sampling exponent, annealed linearly
        # priority_beta_start -> 1.0 over beta_anneal_steps learning steps
        # (Schaul et al. 2015), so the bias correction is full by the end.
        self.priority_beta_start = 0.4
        self.beta_anneal_steps = beta_anneal_steps
        self.memory = PrioritizedReplayBuffer(
            self.MEMORY_CAPACITY, state_size, alpha=self.priority_alpha)

        self.gamma = 0.99
        # Linear schedule: epsilon_start -> epsilon_min over
        # epsilon_decay_steps learning steps, then held at epsilon_min.
        # Derived from step_count, so it resumes correctly from a checkpoint.
        self.epsilon_start = 1.00
        self.epsilon_min = 0.05
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_priority = 1e-6

        self.batch_size = 128

        self.step_count = 0
        self.epsilon = self.scheduled_epsilon()
        self.priority_beta = self.scheduled_beta()

        self.device = get_device()
        print(f"[DDQNAgent] Using device: {self.device}")

        self.online_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=5e-5)
        self.huber_loss = nn.HuberLoss(reduction='none')

        # One row per learning step, so buffer writes instead of flushing
        # every row.
        self.training_log = CsvLogger(
            "logs/training_metrics.csv", ["step", "loss", "avg_td_error"],
            flush_every=1000)

        self._update_target_network(hard_copy=True)

    def _update_target_network(self, hard_copy=True, tau=TARGET_TAU):
        if hard_copy:
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            online_state = self.online_net.state_dict()
            target_state = self.target_net.state_dict()
            new_state = {
                k: tau * online_state[k] + (1 - tau) * target_state[k]
                for k in online_state
            }
            self.target_net.load_state_dict(new_state)

    def scheduled_epsilon(self):
        """Epsilon for the current step_count under the linear schedule."""
        progress = min(self.step_count / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_min - self.epsilon_start)

    def scheduled_beta(self):
        """PER beta for the current step_count: linear to 1.0, then held."""
        progress = min(self.step_count / self.beta_anneal_steps, 1.0)
        return self.priority_beta_start + progress * (1.0 - self.priority_beta_start)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(torch.argmax(q_values[0]).item())

    def learn(self):
        if len(self.memory) < max(self.batch_size, self.WARMUP_TRANSITIONS):
            return

        indices, probs = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.memory.batch(
            indices)

        states_t = torch.as_tensor(
            states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            online_next_q = self.online_net(next_states_t)
            best_actions = torch.argmax(online_next_q, dim=1)

            target_next_q = self.target_net(next_states_t)
            target_q = rewards_t + (1 - dones_t) * self.gamma * target_next_q.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

        q_values = self.online_net(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        td_errors = (target_q - current_q).detach().cpu().numpy()
        avg_td_error = np.mean(np.abs(td_errors))

        total = len(self.memory)
        weights = (total * probs) ** (-self.priority_beta)
        weights /= weights.max()
        weights_t = torch.as_tensor(
            weights, dtype=torch.float32, device=self.device)

        elementwise_loss = self.huber_loss(current_q, target_q)
        loss = (elementwise_loss * weights_t).mean()

        loss_value = loss.item()
        self.training_log.log(
            self.step_count, f"{loss_value:.4f}", f"{avg_td_error:.4f}")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        self.memory.update_priorities(
            indices, np.abs(td_errors) + self.epsilon_priority)

        self._update_target_network(hard_copy=False)

        self.epsilon = self.scheduled_epsilon()
        self.priority_beta = self.scheduled_beta()

        if self.step_count % self.LOG_EVERY == 0:
            print(
                f"[Step {self.step_count}] Loss: {loss_value:.4f}, "
                f"Epsilon: {self.epsilon:.4f}, Beta: {self.priority_beta:.3f}")

    def predict_batch(self, net, states):
        states_t = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = net(states_t)
        return q_values.cpu().numpy()

    def export_state(self):
        """Bundle everything needed to resume training into one dict."""
        return {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "replay_buffer": self.memory.export_state(),
        }

    def import_state(self, data):
        self.online_net.load_state_dict(data["online_state_dict"])
        self.target_net.load_state_dict(data["target_state_dict"])
        self.step_count = data["step_count"]
        # The saved "epsilon" is ignored: it's recomputed from step_count so
        # a changed schedule (or an old checkpoint) can't leave it stale.
        self.epsilon = self.scheduled_epsilon()
        self.priority_beta = self.scheduled_beta()
        if "replay_buffer" in data:
            buffer_state = data["replay_buffer"]
        else:
            # Older checkpoints stored a list of transition tuples.
            buffer_state = PrioritizedReplayBuffer.from_transition_list(
                data.get("memory", []), data.get("priorities", []))
        if buffer_state is not None:
            self.memory.import_state(buffer_state)
