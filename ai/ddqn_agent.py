import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os


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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.priorities = []
        self.priority_alpha = 0.6
        self.priority_beta = 0.4

        self.gamma = 0.99
        self.epsilon = 1.00
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.epsilon_priority = 1e-6

        self.batch_size = 128
        self.update_target_every = 1000

        self.epos = 1
        self.step_count = 0

        self.device = get_device()
        print(f"[DDQNAgent] Using device: {self.device}")

        self.online_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=5e-5)
        self.huber_loss = nn.HuberLoss(reduction='none')

        self._update_target_network(hard_copy=True)

    def _update_target_network(self, hard_copy=True, tau=0.01):
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

    def store_experience(self, state, action, reward, next_state, done):
        max_priorities = max(self.priorities, default=1.0)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priorities)

        if len(self.memory) > 10000:
            self.memory.pop(0)
            self.priorities.pop(0)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(torch.argmax(q_values[0]).item())

    def learn(self, shouldComputeEplison=False):
        if len(self.memory) < self.batch_size:
            return

        probs = np.array(self.priorities) ** self.priority_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        minibatch = [self.memory[i] for i in indices]

        # --- Extract tensors from minibatch ---
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*minibatch))

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
        self.last_td_erros = td_errors

        avg_td_error = np.mean(np.abs(td_errors))

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.priority_beta)
        weights /= weights.max()
        weights_t = torch.as_tensor(
            weights, dtype=torch.float32, device=self.device)

        elementwise_loss = self.huber_loss(current_q, target_q)
        loss = (elementwise_loss * weights_t).mean()

        with open("logs/training_metrics.csv", "a") as f:
            f.write(
                f"{self.step_count},{loss.item():.4f},{avg_td_error:.4f}\n")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        for i, error in zip(indices, td_errors):
            self.priorities[i] = abs(error) + self.epsilon_priority

        self._update_target_network(hard_copy=False)

        if shouldComputeEplison and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(
            f"[Step {self.step_count}] Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")

    def predict_batch(self, net, states):
        states_t = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = net(states_t)
        return q_values.cpu().numpy()

    def save_memory(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.memory)[-50_000:], f)

    def load_memory(self, filepath):
        if not os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.memory = pickle.load(f)
