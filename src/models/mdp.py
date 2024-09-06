import numpy as np

class CarFollowingMDP:
    def __init__(self, v_max, g_max, v_steps, g_steps, a_min, a_max, a_steps, delta_t):
        self.v_max = v_max
        self.g_max = g_max
        self.v_steps = v_steps
        self.g_steps = g_steps
        self.n_states = v_steps * g_steps
        self.v_values = np.linspace(0, v_max, v_steps)
        self.g_values = np.linspace(0, g_max, g_steps)
        self.actions = np.linspace(a_min, a_max, a_steps)
        self.n_actions = len(self.actions)
        self.delta_t = delta_t

        # Create transition dictionary
        self.T = {}
        self._build_transition_matrix()

    def _state_to_index(self, v, g):
        v_idx = np.digitize(v, self.v_values) - 1
        g_idx = np.digitize(g, self.g_values) - 1
        return v_idx * self.g_steps + g_idx

    def _index_to_state(self, index):
        v_idx = index // self.g_steps
        g_idx = index % self.g_steps
        return self.v_values[v_idx], self.g_values[g_idx]

    def _transition(self, v, g, a):
        v_next = min(max(v + a * self.delta_t, 0), self.v_max)
        g_next = min(max(g + 0.5 * a * self.delta_t**2, 0), self.g_max)
        return v_next, g_next

    def _build_transition_matrix(self):
        for s in range(self.n_states):
            v, g = self._index_to_state(s)
            for a_idx, a in enumerate(self.actions):
                v_next, g_next = self._transition(v, g, a)
                s_next = self._state_to_index(v_next, g_next)
                self.T[(s, s_next, a_idx)] = 1.0

    def get_transition_prob(self, s, s_next, a):
        return self.T.get((s, s_next, a), 0.0)