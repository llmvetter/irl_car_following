import torch
import torch.nn as nn

from src.models.mdp import CarFollowingMDP

class RewardNetwork(nn.Module):
    def __init__(
            self,
            mdp: CarFollowingMDP,
            layers: tuple = (8, 16)
    ):
        super(RewardNetwork, self).__init__()
        self.mdp = mdp
        self.net = nn.Sequential(
            nn.Linear(2, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], 1),
            nn.Tanh(),
        )

    def forward(
            self,
            state_tensor: torch.tensor,
    ) -> torch.tensor:
        return self.net(state_tensor)
    

