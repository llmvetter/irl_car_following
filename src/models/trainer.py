import numpy as np
from typing import Tuple
import logging
import torch

from src.utils import (
    rollout,
    value_iteration,
    svf_from_trajectories,
)
from src.models.trajectory import Trajectories
from src.models.reward import RewardNetwork 
from src.models.env import CarFollowingEnv
from src.models.optimizer import GradientAscentOptimizer
from src.config import Config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    def __init__(
            self,
            config: Config,
            trajectories: Trajectories,
            optimizer: GradientAscentOptimizer,
            reward_function: RewardNetwork,
            mdp: CarFollowingEnv,
    ) -> None:
        self.config = config
        self.trajectories = trajectories
        self.optimizer = optimizer
        self.reward_function = reward_function
        self.mdp = mdp
    
    def train(
            self,
            epochs: int,
    ) -> Tuple[RewardNetwork, torch.tensor]:

        logging.info("Computing transition probability matrix")
        self.mdp.compute_transitions()

        expert_svf = svf_from_trajectories(
            trajectories=self.trajectories,
            mdp=self.mdp,
        )

        for epoch in range(epochs):
            
            logging.info("Entering Backward Pass")
            policy: torch.tensor = value_iteration(
                mdp=self.mdp,
                reward=self.reward_function,
                epsilon=self.config.backward_pass['epsilon'],
                discount=self.config.backward_pass['discount'],
                temperature=self.config.backward_pass['temperature'],
                max_iterations=self.config.backward_pass['iterations'],
            )
            logging.info("Entering Forward Pass")
            expected_svf: np.ndarray = rollout(
                mdp=self.mdp,
                policy=policy,
                steps=self.config.forward_pass['steps'],
                iterations=self.config.forward_pass['iterations'],
            )
            #calculate feature expectation from svf
            grad = expert_svf - expected_svf
            logging.info(
                f'Expected feature counts: {np.dot(expected_svf, self.mdp.state_grid)}'
            )

            # perform optimization step
            loss = self.optimizer.step(
                torch.tensor(grad, dtype=torch.float32)
            )
            
            logging.info(f'Epoch loss: {loss}')

        return self.reward_function, policy
