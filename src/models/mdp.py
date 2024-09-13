import numpy as np

class CarFollowingMDP:
    def __init__(
            self,
            v_max: int = 20,
            g_max: int = 40,
            v_steps: float = 0.2,
            g_steps: float = 0.2,
            a_min: int = -1,
            a_max: int = 1.25,
            a_steps = 0.25,
            delta_t: float = 0.3,
    ) -> None:
        self.v_max = v_max
        self.g_max = g_max
        self.v_space = np.arange(0, v_max, v_steps)
        self.g_space = np.arange(0, g_max, g_steps)
        self.action_space = np.arange(a_min, a_max, a_steps)
        self.V, self.G = np.meshgrid(self.v_space, self.g_space)
        self.n_actions = len(self.action_space)
        self.n_states = len(self.v_space)*len(self.g_space)
        self.delta_t = delta_t
        self.state_space = self._create_statespace()
        self.T = {}
        self._build_transition_matrix()

    def _state_to_index(self, state: tuple):
        v, g = state
        v_index = np.abs(self.v_space - v).argmin()
        g_index = np.abs(self.g_space - g).argmin()
        return np.ravel_multi_index((g_index, v_index), self.V.shape)
    
    def _index_to_state(self, index):
        g_index, v_index = np.unravel_index(index, self.V.shape)
        v_value = self.v_space[v_index]
        g_value = self.g_space[g_index]
        return (v_value, g_value)
    
    def _create_statespace(self):
        F = np.zeros((self.n_states, 2))
        for state_idx in range(self.n_states):
            F[state_idx] = self._index_to_state(state_idx)
        return F

    def _action_to_index(self, a):
        return np.digitize(a, self.action_space) - 1

    def _index_to_action(self, index):
        return self.action_space[index]

    def _transition(self, state_idx, a_idx):
        v, g = self._index_to_state(state_idx)
        a = self._index_to_action(a_idx)
        v_next = min(v + a * self.delta_t, self.v_max)
        g_next = min(g + 0.5 * a * self.delta_t**2, self.g_max)
        s_next = self._state_to_index((v_next, g_next))
        return s_next

    def _build_transition_matrix(self):
        for s_from in range(self.n_states):
            for a in range(self.n_actions):
                s_next = self._transition(s_from, a)
                self.T[(s_from, a)] = s_next

class State:
    def __init__(self, mdp, state):
        self.mdp = mdp
        self.index = self.mdp._state_to_index(state)
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