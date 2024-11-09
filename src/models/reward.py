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

    def get_reward(self, state_idx):
        features = self.mdp._index_to_state(state_idx)
        feature_tensor = torch.tensor(features ,dtype=torch.float32)
        with torch.no_grad():
            return self.net(feature_tensor)
    

