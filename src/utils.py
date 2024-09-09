import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from car_following.src.models.mdp import CarFollowingMDP
from lunar_lander.src.models.trajectories import Trajectories
from car_following.src.models.reward import LinearRewardFunction


def feature_expectation_from_trajectories(
        trajectories: Trajectories,
        mdp: ConnectionAbortedError,
) -> np.ndarray:
    fe = np.zeros(mdp.n_states)
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
        n_states: int,
) -> np.ndarray:
    p = np.zeros(n_states)
    for trajectory in trajectories:
        initial_state = trajectory.trajectory[0].state.index
        p[initial_state] += 1.0

    return p/len(trajectories.trajectories)

def compute_action_probability(
        mdp: CarFollowingMDP,
        trajectories: Trajectories, 
        reward: LinearRewardFunction,
) -> np.ndarray:
    
    n_states = mdp.n_states
    n_actions = len(mdp.action_space)
    
    # Compute rewards in log space
    log_rewards = np.array([reward.get_reward(s) for s in range(n_states)])
    
    # Initialize log_zs based on frequency of end states in trajectories
    log_zs = initial_probabilities_from_trajectories(
        trajectories=trajectories,
        n_states = n_states,
    ) #thi should be end states not starting states

    # Perform backward pass
    for i in range(n_states):
        print(f"Iteration: {i+1}/{n_states}")
        log_za = np.full((n_states, n_actions), -np.inf)  # Initialize with log(0)
        
        for s_from in range(n_states):
            for a in range(n_actions):
                s_to = mdp.T[(s_from, a)]
                log_za[s_from, a] = log_rewards[s_from] + log_zs[s_to]
        
        log_zs = logsumexp(log_za, axis=1)
    
    log_p_action = log_za - logsumexp(log_za, axis=1)[:, None]
    p_action = np.exp(log_p_action)
    
    return p_action

def plot_heatmap(
        trajectories: Trajectories,
        mdp: CarFollowingMDP,
        dropout: int = 1000,
) -> None:
    heatmap = np.zeros((len(mdp.v_space), len(mdp.g_space)))
    visitation_vector = feature_expectation_from_trajectories(mdp=mdp, trajectories=trajectories)
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
