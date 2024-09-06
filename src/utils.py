import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from car_following.src.models.mdp import CarFollowingMDP
from lunar_lander.src.models.trajectories import Trajectories


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
) -> np.ndarray:
    p = np.zeros(3200)
    for trajectory in trajectories:
        initial_state = trajectory.trajectory[0].state.index
        p[initial_state] += 1.0

    return p/len(trajectories.trajectories)

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
