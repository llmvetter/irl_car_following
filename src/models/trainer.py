import numpy as np
import logging
import torch

from src.utils import (
    forward_pass,
    backward_pass,
    svf_from_trajectories,
)
from src.models.trajectory import Trajectories
from src.models.reward import RewardNetwork 
from src.models.mdp import CarFollowingMDP
from src.models.optimizer import GradientAscentOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    def __init__(
            self,
            trajectories: Trajectories,
            optimizer: GradientAscentOptimizer,
            reward_function: RewardNetwork,
            mdp: CarFollowingMDP,
            eps=1e-4,
    ) -> None:

        self.trajectories = trajectories
        self.optimizer = optimizer
        self.reward_function = reward_function
        self.mdp = mdp
        self.eps= eps
    
    def train(
            self,
            epochs: int = 20,
    ) -> np.ndarray:

        expert_svf = svf_from_trajectories(
            trajectories=self.trajectories,
            mdp=self.mdp,
        )

        for _ in range(epochs):

            logging.info("Entering Backwardpass")
            policy: torch.tensor = backward_pass(
                mdp=self.mdp,
                reward_func=self.reward_function,
                gamma=0.98,
                max_iterations=50,
                temperature=2,

            )
            logging.info("Entering Forward Pass")
            expected_svf: np.ndarray = forward_pass(
                mdp=self.mdp,
                policy=policy,
                iterations=100,
            )
            #calculate feature expectation from svf
            grad = np.dot((expert_svf - expected_svf), self.mdp.state_space)
            logging.info(
                f'Expected feature expectation: {np.dot(expected_svf, self.mdp.state_space)}'
            )

            # perform optimization step and compute delta for convergence
            loss = self.optimizer.step(torch.tensor(grad, dtype=torch.float32))
            
            logging.info(f'gradient computation complete: {loss}')

        return self.reward_function
