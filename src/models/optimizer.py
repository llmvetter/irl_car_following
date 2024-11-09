import torch
import torch.optim as optim

from src.models.reward import RewardNetwork

class GradientAscentOptimizer:
    def __init__(
            self,
            model: RewardNetwork, 
            learning_rate: float = 0.01
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def step(self, gradient: torch.Tensor):
        self.optimizer.zero_grad()
        
        # Manually set the gradients (negative for ascent)
        for param, grad in zip(self.model.parameters(), gradient):
            if param.grad is None:
                param.grad = -grad.clone().detach()
            else:
                param.grad.data = -grad.clone().detach()
        
        self.optimizer.step()
        return [param.data.clone().detach().numpy() for param in self.model.parameters()]

    @property
    def omega(self):
        return [param.data.clone().detach().numpy() for param in self.model.parameters()]