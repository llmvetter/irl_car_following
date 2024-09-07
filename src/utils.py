import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

from car_following.src.models.mdp import CarFollowingMDP
from lunar_lander.src.models.trajectories import Trajectories
from car_following.src.models.reward import LinearRewardFunction


def feature_expectation_from_trajectories(
        trajectories: Trajectories,
) -> np.ndarray:
    fe = np.zeros(3200)
    for trajectory in trajectories:
        for state_action_pair in trajectory:
            idx = state_action_pair.state.index
            fe[idx] += 1
    return fe/len(trajectories.trajectories)

def policy_state_visitation_frequency(
):
    pass

def initial_probabilities_from_trajectories(
        trajectories: Trajectories,
        n_states: int = 3200,
) -> np.ndarray:
    p = np.zeros(n_states)
    for trajectory in trajectories:
        initial_state = trajectory.trajectory[0].state.index
        p[initial_state] += 1.0

    return p/len(trajectories.trajectories)

def compute_expected_svf(
        mdp: CarFollowingMDP,
        trajectories: Trajectories, 
        reward: LinearRewardFunction,
        eps=1e-5
) -> np.ndarray:
    
    n_states = mdp.n_states
    n_actions = len(mdp.action_space)
    
    # Initialize zs based on frequency of end states in trajectories
    zs = initial_probabilities_from_trajectories(
        trajectories=trajectories,
        n_states = n_states,
    )

    # Backward Pass
    for _ in range(2 * n_states):  # longest trajectory: n_states
        za = np.zeros((n_states, n_actions))  # za: action partition function

        for s_from, a in product(range(n_states), range(n_actions)):
            for s_to in range(n_states):
                print(s_from/n_states)
                za[s_from, a] += mdp.get_transition_prob(
                    s=s_from,
                    s_next=s_to,
                    a=a
                ) * np.exp(reward.get_reward(mdp._index_to_state(s_from))) * zs[s_to]

        new_zs = za.sum(axis=1)

        if np.max(np.abs(new_zs - zs)) < eps:
            break
        
        zs = new_zs

    # Compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    d = np.zeros(n_states)  # d: state-visitation frequencies

    # Initialize with start states of all trajectories
    for trajectory in trajectories:
        d[trajectory[0]] += 1 / len(trajectories)

    for t in range(2 * n_states):  # longest trajectory: n_states
        new_d = np.zeros(n_states)
        for s_to in range(n_states):
            for s_from, a in product(range(n_states), range(n_actions)):
                new_d[s_to] += d[s_from] * p_action[s_from, a] * mdp.get_transition_prob(s=s_from, s_next=s_to, a=a)
        
        # Check for convergence
        if np.max(np.abs(new_d - d)) < eps:
            break
        
        d = new_d

    return d

def plot_heatmap(
        trajectories: Trajectories,
        mdp: CarFollowingMDP,
        dropout: int = 1000,
) -> None:
    heatmap = np.zeros((len(mdp.v_space), len(mdp.g_space)))
    visitation_vector = feature_expectation_from_trajectories(trajectories=trajectories)
    for index, frequency in enumerate(visitation_vector):
        if frequency < dropout:
            speed, distance = mdp._index_to_state(index)
            speed_index = np.digitize(speed, mdp.v_space, right=False)-1
            distance_gap_index = np.digitize(distance, mdp.g_space, right=False)-1
            
            # Increment the corresponding cell in the heatmap
            heatmap[speed_index, distance_gap_index] += frequency

    # Plot the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, xticklabels=mdp.g_space, yticklabels=mdp.v_space, cmap='viridis', cbar=True)
    plt.xlabel('Distance Gap')
    plt.ylabel('Speed')
    plt.title('State Visitations Heatmap')
    plt.show()
