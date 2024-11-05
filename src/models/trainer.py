import numpy as np
import logging

from src.utils import (
    forward_pass,
    backward_pass,
    svf_from_trajectories,
)
from src.models.trajectory import Trajectories
from src.models.reward import LinearRewardFunction 
from src.models.mdp import CarFollowingMDP
from src.models.optimizer import GradientDescentOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    def __init__(
            self,
            trajectories: Trajectories,
            optimizer: GradientDescentOptimizer,
            reward_function: LinearRewardFunction,
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
    ) -> np.ndarray:

        expert_svf = svf_from_trajectories(
            trajectories=self.trajectories,
            mdp=self.mdp,
        )

        delta = np.inf

        while delta > self.eps:
            #retain old omega value
            omega_old = self.optimizer.omega.copy()

            # Set the current weights in the reward function
            self.reward_function.set_weights(self.optimizer.omega)

            logging.info("Backwardpass")
            policy = backward_pass(
                mdp=self.mdp,
                reward_func=self.reward_function,
            )
            logging.info("Forward Pass")
            expected_svf = forward_pass(
                mdp=self.mdp,
                policy=policy,
            )
            #calculate feature expectation from svf
            grad = np.dot((expert_svf - expected_svf), self.mdp.state_space)
            logging.info(f'Expected feature expectation: {np.dot(expected_svf, self.mdp.state_space)}')

            # perform optimization step and compute delta for convergence
            omega = self.optimizer.step(grad)
            
            # re-compute delta for convergence check
            delta = np.max(np.abs(omega_old - omega))
            logging.info(f'gradient computation complete: {delta}')

        # Set final weights and return the reward function
        self.reward_function.set_weights(omega)
        return self.reward_function.weights