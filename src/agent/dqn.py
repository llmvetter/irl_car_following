import numpy as np
import torch
import torch.nn as nn

from lunar_lander.src.ml_models.dqn import DQN

  
class DQNAgent:
    def __init__(
              self,
              state_dim=8,
              learning_rate=0.001,
              gamma=0.99,
              epsilon=1.0,
              epsilon_decay=0.996,
              epsilon_min=0.01
    ):
            self.q_network = DQN(state_dim, 4)
            self.target_network = DQN(state_dim, 4)
            self.target_network.load_state_dict(
                 self.q_network.state_dict(),
            )
            self.optimizer = torch.optim.Adam(
                 self.q_network.parameters(),
                 lr=learning_rate,
            )
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.action_dim = 4

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)
        
        q_value = q_values.gather(1, torch.LongTensor([[action]]))
        next_q_value = next_q_values.max(1)[0].unsqueeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())