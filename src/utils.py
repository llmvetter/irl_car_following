import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from car_following.src.models.state_action import State
from lunar_lander.src.models.trajectories import Trajectories


def feature_expectation_from_trajectories(
        trajectories: Trajectories,
) -> np.ndarray:
    fe = np.zeros(3081)
    for trajectory in trajectories:
        for state_action_pair in trajectory:
            idx = state_action_pair.state.index
            fe[idx] += 1
    return fe/len(trajectories.trajectories)


def policy_state_visitation_frequency(
        policy,
        num_trajectories=1000,
        max_steps=1000,
):
    pass

def index_to_state(flattened_index: int) -> tuple:
        speed = [x/2 for x in range(1, 40, 1)]
        distance = [x/2 for x in range(1, 80, 1)]
        n_distance = len(distance)
        speed_index = flattened_index // n_distance
        distance_gap_index = flattened_index % n_distance
        speed_value = speed[speed_index]
        distance_gap_value = distance[distance_gap_index]
        return (speed_value, distance_gap_value)

def initial_probabilities_from_trajectories(
        trajectories: Trajectories,
) -> np.ndarray:
    p = np.zeros(3081)
    for trajectory in trajectories:
        initial_state = trajectory.trajectory[0].state.index
        p[initial_state] += 1.0

    return p/len(trajectories.trajectories)

def plot_heatmap(
        trajectories: Trajectories,
        dropout: int = 1000,
) -> None:
    dummy_state = State((0,0))
    speed_space = dummy_state.space['speed']
    distance_space = dummy_state.space['distance']
    heatmap = np.zeros((len(speed_space), len(distance_space)))
    visitation_vector = feature_expectation_from_trajectories(trajectories=trajectories)
    for flattened_index, frequency in enumerate(visitation_vector):
        if frequency < dropout:
            speed_value, distance_gap_value = index_to_state(flattened_index)
            speed_index = np.digitize(speed_value, speed_space, right=False) - 1
            distance_gap_index = np.digitize(distance_gap_value, distance_space, right=False) - 1
            
            # Increment the corresponding cell in the heatmap
            heatmap[speed_index, distance_gap_index] += frequency

    # Plot the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, xticklabels=distance_space, yticklabels=speed_space, cmap='viridis', cbar=True)
    plt.xlabel('Distance Gap')
    plt.ylabel('Speed')
    plt.title('State Visitations Heatmap')
    plt.show()
