import torch.nn as nn

class RewardNetwork(nn.Module):
    def __init__(self, state_dim=8):
        super(RewardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.network(state)