import torch
import numpy as np
import torch.optim as optim

from src.models.reward import RewardNetwork
from src.models.env import CarFollowingEnv

class GradientAscentOptimizer:
    
    def __init__(
            self,
            reward_network: RewardNetwork,
            mdp: CarFollowingEnv,
            lr: float = 1e-3
    ):
        self.reward_network = reward_network
        self.mdp = mdp
        self.optimizer = optim.Adam(self.reward_network.net.parameters(), lr=lr)

    def step(
            self,
            gradient: torch.tensor,
    ) -> float:

        states_tensor = torch.tensor(self.mdp.state_grid, dtype=torch.float32)
        rewards = self.reward_network(states_tensor).flatten()

        self.optimizer.zero_grad()
        rewards.backward(-gradient)
        self.optimizer.step()
        return (-gradient*rewards).sum().item()