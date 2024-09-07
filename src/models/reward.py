import numpy as np

from car_following.src.models.mdp import State

class LinearRewardFunction:
    def __init__(
            self, 
            num_features: int = 2
    ):
        self.weights = np.random.randn(num_features)  # Initialize weights randomly

    def get_reward(
            self, 
            state: tuple,
    ) -> np.ndarray:
        return np.dot(self.weights, np.array(state))

    def update_weights(self, gradient, learning_rate):
        self.weights += learning_rate * gradient

def compute_gradient(reward_function, mdp, expert_feature_expectations, learner_feature_expectations):
    return expert_feature_expectations - learner_feature_expectations

