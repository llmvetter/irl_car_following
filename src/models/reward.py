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
        self.weights: np.ndarray = None

    def set_weights(self, weights) -> None:
        self.weights = weights

    def get_reward(
            self, 
            state: int,
    ) -> np.ndarray:
        state = self.mdp._index_to_state(state)
        return np.dot(self.weights, state)