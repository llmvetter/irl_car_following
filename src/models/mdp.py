import logging
import numpy as np
from scipy.stats import norm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CarFollowingMDP:
    def __init__(
            self,
            sigma_g: float = 0.4777,
            v_max: int = 20,
            g_max: int = 40,
            v_steps: float = 0.2,
            g_steps: float = 0.2,
            a_min: int = -1,
            a_max: int = 1.25,
            a_steps = 0.25,
            delta_t: float = 0.1, #TODO do not downsample
    ) -> None:
        self.v_max = v_max
        self.g_max = g_max
        self.g_sigma = sigma_g
        self.v_space = np.arange(0, v_max, v_steps)
        self.g_space = np.arange(0, g_max, g_steps)
        self.action_space = np.arange(a_min, a_max, a_steps)
        self.V, self.G = np.meshgrid(self.v_space, self.g_space)
        self.n_actions = len(self.action_space)
        self.n_states = len(self.v_space)*len(self.g_space)
        self.delta_t = delta_t
        self.state_space = self._create_statespace()

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

    def _gaussian_prob(self, x, mean, sigma):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    
    def get_transitions(
            self, 
            s_idx_from: int,
            a_idx: int,
            proba_threshold: float = 0.001,
    ) -> np.ndarray:
        """
        Compute the transition probabilities and next states for
         a given state-action pair.

        Args:
            s_from: The state index in which the transition originates.
            a: The action index via which the target state should be reached.

        Returns:
            A 2d numpy array with the indices of next states and their tran-
            sition probabilities.
        """
        
        v_from, g_from = self._index_to_state(s_idx_from)
        action = self._index_to_action(a_idx)
        v_next = max(min(v_from + action * self.delta_t, self.v_max), 0)

        if v_next != 0:
            g_next_mean = g_from-(v_from*self.delta_t)+(0.5*action*self.delta_t**2)
        else:
            g_next_mean = g_from

        # discard very unlikely transitions
        lower_bound = max(0, g_next_mean - 3*self.g_sigma)
        upper_bound = min(self.g_max, g_next_mean + 3*self.g_sigma)

        significant_bins = []
        probabilities = []

        for i in range(len(self.g_space) - 1):
            bin_start = self.g_space[i]
            bin_end = self.g_space[i + 1]

            if bin_start > upper_bound or bin_end < lower_bound:
                continue
            
            prob = norm.cdf(
                bin_end, g_next_mean, self.g_sigma
                )-norm.cdf(bin_start, g_next_mean, self.g_sigma)
            
            if prob > proba_threshold:
                probabilities.append(prob)
                significant_bins.append((v_next, (bin_start + bin_end) / 2))
                
        # normalize
        probabilities = np.array(probabilities)
        probabilities /= np.sum(probabilities)

        next_state_indices = [self._state_to_index(state) for state in significant_bins]

        # Combine probabilities for duplicate indices
        unique_indices, unique_probs = np.unique(next_state_indices, return_inverse=True)
        combined_probs = np.bincount(unique_probs, weights=probabilities)

        return np.column_stack((unique_indices, combined_probs))

    def step(self, state, action):
        transitions = self.get_transitions(state, action)
        next_states = transitions[:, 0].astype(int)
        probabilities = transitions[:, 1]
        if next_states.size == 0: #TODO: Handle thi scenerio as a crash?
            return state
        next_state = np.random.choice(next_states, p=probabilities)
        
        return next_state

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