import numpy as np
from collections import deque
import random
import tensorflow as tf

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100_000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.batch_size = 128
        self.update_target_every = 1000
        self.step_count = 0

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
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

    def _update_target_network(self, hard_copy=False, tau=0.01):
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
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.expand_dims(state, axis=0)
        q_values = self.online_net.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch], dtype=np.float32)
        actions = np.array([x[1] for x in minibatch], dtype=np.int32)
        rewards = np.array([x[2] for x in minibatch], dtype=np.float32)
        next_states = np.array([
            x[3] if x[3] is not None else np.zeros(self.state_size)
            for x in minibatch
        ], dtype=np.float32)
        dones = np.array([x[4] for x in minibatch], dtype=np.float32)

        online_next_q = self.online_net.predict(next_states, verbose=0)
        target_next_q = self.target_net.predict(next_states, verbose=0)
        best_actions = np.argmax(online_next_q, axis=1)
        target_q = rewards + (1 - dones) * self.gamma * target_next_q[
            np.arange(self.batch_size), best_actions
        ]

        with tf.GradientTape() as tape:
            q_values = self.online_net(states)
            current_q = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_size), axis=1)
            loss = tf.keras.losses.MSE(target_q, current_q)

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target_network(hard_copy=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"[Step {self.step_count}] Loss: {loss.numpy():.4f}, Epsilon: {self.epsilon:.4f}")

    def save_model(self, path):
        self.online_net.save(path)

    def load_model(self, path):
        self.online_net = tf.keras.models.load_model(path)
        self._update_target_network(hard_copy=True)
