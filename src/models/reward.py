import torch
import torch.nn as nn

from src.models.mdp import CarFollowingMDP

class RewardNetwork(nn.Module):
    def __init__(
            self,
            mdp: CarFollowingMDP,
            layers: list,
    ):
        super(RewardNetwork, self).__init__()
        self.mdp = mdp
        self.net = nn.Sequential(
            nn.Linear(3, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], 1),
            nn.Softplus(),
        )

    def forward(
            self,
            state_tensor: torch.tensor,
            grad: bool = True,
    ) -> torch.tensor:
        if grad:
            return self.net(state_tensor)
        else:
            with torch.no_grad():
                return self.net(state_tensor)

