import numpy as np 
from car_following.src.utils import (
    feature_expectation_from_trajectories,
    compute_expected_svf,
)
from car_following.src.models.trajectories import Trajectories
from car_following.src.models.reward import LinearRewardFunction 
from car_following.src.models.mdp import CarFollowingMDP
from car_following.src.models.optimizer import GradientDescentOptimizer


def maxent_irl(
        trajectories: Trajectories,
        optimizer: GradientDescentOptimizer,
        reward_function: LinearRewardFunction,
        mdp: CarFollowingMDP,
        eps=1e-4,
):
    # compute feature expectation from trajectories
    expert_svf = feature_expectation_from_trajectories(
        trajectories=trajectories,
        mdp=mdp,
    )

    # init weights
    omega = np.random.uniform(0, 1, reward_function.num_features)
    delta = np.inf
    optimizer.reset(omega)

    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # Set the current weights in the reward function
        reward_function.set_weights(omega)

        # compute gradient of the log-likelihood
        expected_svf = compute_expected_svf(
            mdp=mdp,
            trajectories=trajectories,
            reward=reward_function,
        )

        grad = np.dot((expert_svf - expected_svf), mdp.state_space)

        # perform optimization step and compute delta for convergence
        omega = optimizer.step(grad)
        
        # re-compute delta for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # Set final weights and return the reward function
    reward_function.set_weights(omega)
    return reward_function