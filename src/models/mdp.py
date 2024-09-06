import numpy as np

class CarFollowingMDP:
    def __init__(
            self,
            v_max: int = 20,
            g_max: int = 40,
            v_steps = 0.5,
            g_steps = 0.5,
            a_min: int = -3,
            a_max: int = 3,
            a_steps = 0.1,
            delta_t: float = 0.1,
    ) -> None:
        self.v_max = v_max
        self.g_max = g_max
        self.v_steps = v_steps
        self.g_steps = g_steps
        self.v_space = np.arange(0, v_max, v_steps)
        self.g_space = np.arange(0, g_max, g_steps)
        self.action_space = np.arange(a_min, a_max, a_steps)
        self.n_actions = len(self.action_space)
        self.n_states = len(self.v_space)*len(self.g_space)
        self.delta_t = delta_t
        self.T = {}
        self._build_transition_matrix()

    def _state_to_index(self, v, g):
        v_idx = np.digitize(v, self.v_space, right=False)
        g_idx = np.digitize(g, self.g_space, right=False)
        return v_idx * len(self.g_space) + g_idx

    def _index_to_state(self, index):
        v_idx = index // len(self.g_space)
        g_idx = index % len(self.g_space)
        return (self.v_space[v_idx], self.g_space[g_idx])

    def _action_to_index(self, a):
        return np.digitize(a, self.action_space) - 1

    def _index_to_action(self, index):
        return self.action_space[index]

    def _transition(self, v, g, a):
        v_next = min(max(v + a * self.delta_t, 0), self.v_max)
        g_next = min(max(g + 0.5 * a * self.delta_t**2, 0), self.g_max)
        return v_next, g_next

    def _build_transition_matrix(self):
        for s in range(self.n_states):
            v, g = self._index_to_state(s)
            for a_idx, a in enumerate(self.action_space):
                v_next, g_next = self._transition(v, g, a)
                s_next = self._state_to_index(v_next, g_next)
                self.T[(s, s_next, a_idx)] = 1.0

    def get_transition_prob(self, s, s_next, a):
        return self.T.get((s, s_next, a), 0.0)

class State:
    def __init__(self, mdp, state):
        self.mdp = mdp
        self.index = self.mdp._state_to_index(*state)
        self.state = self.mdp._index_to_state(self.index)
        

class Action:
    def __init__(self, mdp, action):
        self.mdp = mdp
        self.index = self.mdp._action_to_index(action)
        self.action = self.mdp._index_to_action(self.index)

class StateActionPair:
    def __init__(self, mdp, state, action):
        self.mdp = mdp
        self.state = State(mdp, state)
        self.action = Action(mdp, action)

    def get_state(self):
        return self.state.state

    def get_action(self):
        return self.action.action

    def get_state_index(self):
        return self.state.index

    def get_action_index(self):
        return self.action.index