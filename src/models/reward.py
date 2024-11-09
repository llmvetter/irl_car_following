import torch
import torch.nn as nn

class RewardNetwork(nn.Module):
    def __init__(self, mdp, hidden_size=8):
        super(RewardNetwork, self).__init__()
        self.mdp = mdp
        self.input_size = 2
        self.hidden_size = hidden_size
        self.output_size = 1
        
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Softplus()
        )

    def forward(self, x):
        return self.model(x)

    def get_reward(self, state):
        state_vec = torch.tensor(self.mdp._index_to_state(state), dtype=torch.float32)
        with torch.no_grad():
            return self.forward(state_vec).item()

    def set_weights(self, weights):
        with torch.no_grad():
            for param, weight in zip(self.parameters(), weights):
                param.copy_(torch.tensor(weight))

    @property
    def weights(self):
        return [param.data.numpy() for param in self.parameters()]
