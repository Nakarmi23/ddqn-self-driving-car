import numpy as np
from collections import deque
import random
import tensorflow as tf
import pickle
import os


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

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=5e-5, clipnorm=1.0)
        self.online_net = self._build_network()
        self.target_net = self._build_network()

        self._update_target_network(hard_copy=True)

    def _build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=self.optimizer)
        return model

    def _update_target_network(self, hard_copy=True, tau=0.01):
        if hard_copy:
            self.target_net.set_weights(self.online_net.get_weights())
        else:
            online_weights = self.online_net.get_weights()
            target_weights = self.target_net.get_weights()
            new_weights = [
                tau * ow + (1 - tau) * tw for ow, tw in zip(online_weights, target_weights)
            ]
            self.target_net.set_weights(new_weights)

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
        state = np.expand_dims(state, axis=0)
        q_values = self.online_net.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def learn(self, step_count, shouldComputeEplison=False):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        probs = np.array(self.priorities) ** self.priority_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        minibatch = [self.memory[i] for i in indices]

        # --- Extract tensors from minibatch ---
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*minibatch))

        online_next_q = self.online_net.predict(next_states, verbose=0)
        best_actions = np.argmax(online_next_q, axis=1)

        target_next_q = self.target_net.predict(next_states, verbose=0)
        target_q = rewards + (1 - dones) * self.gamma * target_next_q[
            np.arange(self.batch_size), best_actions
        ]

        with tf.GradientTape() as tape:
            q_values = self.online_net(states, training=True)
            current_q = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_size), axis=1)

            td_errors = target_q - current_q.numpy()
            self.last_td_erros = td_errors

            avg_td_error = np.mean(np.abs(td_errors))

            total = len(self.memory)
            weights = (total * probs[indices]) ** (-self.priority_beta)
            weights /= weights.max()
            weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)

            huder_loss = tf.keras.losses.Huber(
                reduction=tf.keras.losses.Reduction.NONE)
            loss = tf.reduce_mean(huder_loss(
                target_q, current_q) * weights_tensor)

            with open("logs/training_metrics.csv", "a") as f:
                f.write(
                    f"{self.step_count},{loss.numpy():.4f},{avg_td_error:.4f}\n")

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.online_net.trainable_variables))

        self.step_count += 1
        for i, error in zip(indices, td_errors):
            self.priorities[i] = abs(error) + self.epsilon_priority

        # if self.step_count % self.update_target_every == 0:
        self._update_target_network(hard_copy=False)

        if shouldComputeEplison and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(
            f"[Step {self.step_count}] Loss: {loss.numpy():.4f}, Epsilon: {self.epsilon:.4f}")

    def save_memory(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.memory)[-50_000:], f)

    def load_memory(self, filepath):
        if not os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.memory = pickle.load(f)
