import numpy as np 
from car_following.src.utils import (
    forward_pass,
    backwar_pass,
    feature_expectation_from_trajectories,
    initial_probabilities_from_trajectories,
)
from car_following.src.models.trajectory import Trajectories
from car_following.src.models.reward import LinearRewardFunction 
from car_following.src.models.mdp import CarFollowingMDP
from car_following.src.models.optimizer import GradientDescentOptimizer

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
            self
    ) -> LinearRewardFunction:

        expert_svf = feature_expectation_from_trajectories(
            trajectories=self.trajectories,
            mdp=self.mdp,
        )

        p_initial = initial_probabilities_from_trajectories(
            trajectories=self.trajectories,
            n_states=self.mdp.n_states,
        )
        # init weights randomly
        omega = np.random.uniform(0, 1, self.reward_function.num_features)
        delta = np.inf
        self.optimizer.reset(omega)

        while delta > self.eps:
            omega_old = omega.copy()

            # Set the current weights in the reward function
            self.reward_function.set_weights(omega)

            p_action = backwar_pass(
                mdp=self.mdp,
                reward_func=LinearRewardFunction,
            )

            expected_svf = forward_pass(
                mdp = self.mdp,
                p_initial = p_initial,
                p_action = p_action,
            )

            #calculate featurexx^x expectation from svf
            grad = np.dot((expert_svf - expected_svf), self.mdp.state_space)

            # perform optimization step and compute delta for convergence
            omega = self.optimizer.step(grad)
            
            # re-compute delta for convergence check
            delta = np.max(np.abs(omega_old - omega))
            print(f'gradient computation complete: {delta}')

        # Set final weights and return the reward function
        self.reward_function.set_weights(omega)
        return self.reward_function