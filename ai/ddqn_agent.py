import numpy as np
from collections import deque
import random
import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter


def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
    data_adapter._is_distributed_dataset = _is_distributed_dataset


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Should match your Car.get_state() output
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Experience replay buffer
        self.gamma = 0.99          # Discount factor
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.update_target_every = 1000  # Steps between target network updates
        self.step_count = 0
        # Online and target networks
        self.online_net = self._build_network()
        self.target_net = self._build_network()
        self._update_target_network()

    def _build_network(self):
        """Network architecture matching your state size"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu',
                                  input_dim=self.state_size),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(
                self.action_size, activation='linear')  # Q-values
        ])

        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        return model

    def _update_target_network(self):
        """Sync target network with online network"""
        self.target_net.set_weights(self.online_net.get_weights())

    def store_experience(self, state, action, reward, next_state, done):
        """Save experience to replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        state = np.expand_dims(state, axis=0)
        q_values = self.online_net.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def learn(self):
        """Train on a batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # Convert to numpy arrays explicitly
        states = np.array([x[0] for x in minibatch], dtype=np.float32)
        actions = np.array([x[1] for x in minibatch], dtype=np.int32)
        rewards = np.array([x[2] for x in minibatch], dtype=np.float32)
        next_states = tf.convert_to_tensor([
            x[3] if x[3] is not None
            else tf.zeros(self.state_size, dtype=tf.float32)
            for x in minibatch
        ])

        dones = np.array([x[4] for x in minibatch], dtype=np.bool_)

        # Double DQN logic
        online_next_q = self.online_net.predict(next_states, verbose=1)
        target_next_q = self.target_net.predict(next_states, verbose=1)

        target_q = rewards + (1 - dones) * self.gamma * target_next_q[
            np.arange(self.batch_size), np.argmax(online_next_q, axis=1)
        ]

        # Train online network
        with tf.GradientTape() as tape:
            q_values = self.online_net(states)
            current_q = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_size), axis=1)
            loss = tf.keras.losses.MSE(target_q, current_q)

        gradients = tape.gradient(loss, self.online_net.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=0.00025).apply_gradients(
            zip(gradients, self.online_net.trainable_variables))

        # Update target network and decay epsilon
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        self.online_net.save_weights(path)

    def load_model(self, path):
        self.online_net.load_weights(path)
        self._update_target_network()
