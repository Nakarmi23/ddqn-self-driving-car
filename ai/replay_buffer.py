import numpy as np


class PrioritizedReplayBuffer:
    """Proportional prioritized replay (Schaul et al. 2015) on numpy ring
    buffers plus two segment trees, so storing, sampling and priority updates
    are O(log n) instead of O(n).

    Priorities are stored raw (|td_error| + eps). A sum-tree over
    priority ** alpha drives sampling; a max-tree over raw priorities gives
    the priority for new transitions: the max currently in the buffer (1.0
    when empty), so it falls again once outlier TD errors are evicted or
    re-prioritised.
    """

    def __init__(self, capacity, state_size, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0  # next slot to write
        self.size = 0

        # Implicit binary trees: node i has children 2i and 2i+1, the root is
        # node 1 and leaf j is node _leaf_base + j. Unused leaves stay 0.
        self._leaf_base = 1
        while self._leaf_base < capacity:
            self._leaf_base *= 2
        self._sum_tree = np.zeros(2 * self._leaf_base, dtype=np.float64)
        self._max_tree = np.zeros(2 * self._leaf_base, dtype=np.float64)

    def __len__(self):
        return self.size

    @property
    def total_priority(self):
        return self._sum_tree[1]

    @property
    def max_priority(self):
        return self._max_tree[1] if self.size else 1.0

    def add(self, state, action, reward, next_state, done):
        i = self.pos
        priority = self.max_priority
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        # Single leaf: a scalar walk to the root beats numpy's per-call
        # overhead in update_priorities().
        self.priorities[i] = priority
        node = i + self._leaf_base
        self._sum_tree[node] = priority ** self.alpha
        self._max_tree[node] = priority
        while node > 1:
            node //= 2
            left, right = 2 * node, 2 * node + 1
            self._sum_tree[node] = self._sum_tree[left] + self._sum_tree[right]
            self._max_tree[node] = max(
                self._max_tree[left], self._max_tree[right])

    def sample(self, batch_size):
        """Stratified proportional sampling: one draw from each of
        batch_size equal slices of the total priority mass. Returns
        (indices, probabilities) where probabilities are P(i)."""
        total = self.total_priority
        targets = (np.arange(batch_size) + np.random.rand(batch_size)) \
            * (total / batch_size)
        nodes = np.ones(batch_size, dtype=np.int64)
        while nodes[0] < self._leaf_base:
            left = 2 * nodes
            left_sum = self._sum_tree[left]
            go_right = targets > left_sum
            targets = np.where(go_right, targets - left_sum, targets)
            nodes = np.where(go_right, left + 1, left)
        # Float rounding can walk onto an empty leaf past the filled range.
        indices = np.minimum(nodes - self._leaf_base, self.size - 1)
        probs = self._sum_tree[indices + self._leaf_base] / total
        return indices, probs

    def batch(self, indices):
        return (self.states[indices], self.actions[indices],
                self.rewards[indices], self.next_states[indices],
                self.dones[indices])

    def update_priorities(self, indices, priorities):
        indices = np.asarray(indices)
        priorities = np.asarray(priorities, dtype=np.float64)
        self.priorities[indices] = priorities
        nodes = indices + self._leaf_base
        self._sum_tree[nodes] = priorities ** self.alpha
        self._max_tree[nodes] = priorities
        # Recompute parents from their children, level by level. Duplicate
        # indices are harmless since each parent is recomputed, not adjusted.
        while nodes[0] > 1:
            nodes = np.unique(nodes // 2)
            left, right = 2 * nodes, 2 * nodes + 1
            self._sum_tree[nodes] = self._sum_tree[left] + self._sum_tree[right]
            self._max_tree[nodes] = np.maximum(
                self._max_tree[left], self._max_tree[right])

    def export_state(self):
        """Transitions oldest-first, so import_state can rebuild the ring."""
        start = self.pos if self.size == self.capacity else 0
        order = (start + np.arange(self.size)) % self.capacity
        return {
            "states": self.states[order],
            "actions": self.actions[order],
            "rewards": self.rewards[order],
            "next_states": self.next_states[order],
            "dones": self.dones[order],
            "priorities": self.priorities[order],
        }

    def import_state(self, data):
        """Replace the contents with export_state() output, keeping the
        newest `capacity` transitions."""
        n = min(len(data["priorities"]), self.capacity)
        newest = slice(len(data["priorities"]) - n, None)
        self._sum_tree[:] = 0.0
        self._max_tree[:] = 0.0
        self.priorities[:] = 0.0
        self.states[:n] = data["states"][newest]
        self.actions[:n] = data["actions"][newest]
        self.rewards[:n] = data["rewards"][newest]
        self.next_states[:n] = data["next_states"][newest]
        self.dones[:n] = data["dones"][newest]
        self.size = n
        self.pos = n % self.capacity
        if n:
            self.update_priorities(np.arange(n), data["priorities"][newest])

    @staticmethod
    def from_transition_list(memory, priorities):
        """Convert the old checkpoint format (list of (s, a, r, s', done)
        tuples plus a parallel priority list) to export_state() form."""
        if not memory:
            return None
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*memory))
        return {
            "states": states.astype(np.float32),
            "actions": actions.astype(np.int64),
            "rewards": rewards.astype(np.float32),
            "next_states": next_states.astype(np.float32),
            "dones": dones.astype(np.float32),
            "priorities": np.asarray(priorities, dtype=np.float64),
        }
