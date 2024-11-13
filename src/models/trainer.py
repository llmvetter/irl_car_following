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
    ) -> None:

        self.trajectories = trajectories
        self.optimizer = optimizer
        self.reward_function = reward_function
        self.mdp = mdp
    
    def train(
            self,
            epochs: int = 20,
    ) -> RewardNetwork:

        expert_svf = svf_from_trajectories(
            trajectories=self.trajectories,
            mdp=self.mdp,
        )

        for _ in range(epochs):

            logging.info("Entering Backwardpass")
            policy: torch.tensor = backward_pass(
                mdp=self.mdp,
                reward=self.reward_function,
                epsilon=0.1,
                max_iterations=20,
            )
            logging.info("Entering Forward Pass")
            expected_svf: np.ndarray = forward_pass(
                mdp=self.mdp,
                policy=policy,
                iterations=100,
            )
            #calculate feature expectation from svf
            grad = np.dot(
                (expert_svf - expected_svf), self.mdp.state_space
            )
            logging.info(
                f'Expected feature expectation: {np.dot(expected_svf, self.mdp.state_space)}'
            )

            # perform optimization step
            loss = self.optimizer.step(
                torch.tensor(grad, dtype=torch.float32)
            )
            
            logging.info(f'Epoch loss: {loss}')

        return self.reward_function
