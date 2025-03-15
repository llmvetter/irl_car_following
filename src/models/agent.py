import torch
import numpy as np

from src.models.env import CarFollowingEnv

class Agent:
    def __init__(
        self,
        policy: torch.Tensor,
        env: CarFollowingEnv,
    ):
        self.policy = policy
        self.env = env

    def choose_action(self, state: tuple[float]) -> int:
        state_discretized = self.env._discretize_state(np.array(state))
        state_idx = self.env._state_to_index(state_discretized)
        action_idx = torch.argmax(self.policy[state_idx]).item()
        return action_idx
