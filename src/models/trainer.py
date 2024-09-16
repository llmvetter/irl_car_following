import numpy as np 
from car_following.src.utils import compute_expected_svf
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
        from car_following.src.utils import feature_expectation_from_trajectories

        expert_svf = feature_expectation_from_trajectories(
            trajectories=self.trajectories,
            mdp=self.mdp,
        )

        # init weights randomly
        omega = np.random.uniform(0, 1, self.reward_function.num_features)
        delta = np.inf
        self.optimizer.reset(omega)

        while delta > self.eps:
            omega_old = omega.copy()

            # Set the current weights in the reward function
            self.reward_function.set_weights(omega)

            # compute gradient of the log-likelihood
            expected_svf = compute_expected_svf(
                mdp=self.mdp,
                trajectories=self.trajectories,
                reward=self.reward_function,
            )

            grad = np.dot((expert_svf - expected_svf), self.mdp.state_space)

            # perform optimization step and compute delta for convergence
            omega = self.optimizer.step(grad)
            
            # re-compute delta for convergence check
            delta = np.max(np.abs(omega_old - omega))
            print(f'gradient computation complete: {delta}')

        # Set final weights and return the reward function
        self.reward_function.set_weights(omega)
        return self.reward_function