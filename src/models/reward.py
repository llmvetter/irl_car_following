import numpy as np

from car_following.src.models.mdp import CarFollowingMDP

class LinearRewardFunction:
    def __init__(
            self, 
            mdp: CarFollowingMDP,
            num_features: int = 2,
    ):  
        self.mdp = mdp
        self.weights = self.weights = np.random.uniform(0, 1, num_features)
        self.std_features = np.array(np.std(self.mdp.v_space), np.std(self.mdp.g_space))
        self.mean_features = np.array(np.mean(self.mdp.v_space), np.mean(self.mdp.g_space))


    def normalize_state_features(self, state: tuple) -> np.ndarray:
        return (np.array(state) - self.mean_features) / self.std_features

    def get_reward(
            self, 
            state: int,
    ) -> np.ndarray:
        state = self.mdp._index_to_state(state)
        state_nomalized = self.normalize_state_features(state)
        raw_reward = np.dot(self.weights, np.array(state_nomalized))
        return np.tanh(raw_reward)

    def update_weights(self, gradient, learning_rate):
        current_rewards = self.get_reward(np.arange(self.mdp.n_states))
        tanh_derivative = 1 - np.tanh(current_rewards)**2
        adjusted_gradient = gradient * tanh_derivative
        self.weights += learning_rate * adjusted_gradient

def compute_gradient(reward_function, mdp, expert_feature_expectations, learner_feature_expectations):
    return expert_feature_expectations - learner_feature_expectations

