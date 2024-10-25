import numpy as np

from src.models.mdp import CarFollowingMDP

class LinearRewardFunction:
    def __init__(
            self, 
            mdp: CarFollowingMDP,
            num_features: int = 2,
    ) -> None:  
        self.mdp = mdp
        self.num_features = num_features
        self.std_features = np.array(np.std(self.mdp.v_space), np.std(self.mdp.g_space))
        self.mean_features = np.array(np.mean(self.mdp.v_space), np.mean(self.mdp.g_space))
        self.weights: np.ndarray = None

    def set_weights(self, weights) -> None:
        self.weights = weights

    def normalize_state_features(self, state: tuple) -> np.ndarray:
        return (np.array(state) - self.mean_features) / self.std_features

    def get_reward(
            self, 
            state: int,
    ) -> np.ndarray:
        state = self.mdp._index_to_state(state)
        state_nomalized = self.normalize_state_features(state)
        raw_reward = np.dot(self.weights, np.array(state_nomalized))
        return raw_reward