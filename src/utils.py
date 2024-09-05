import numpy as np

from lunar_lander.src.models.trajectories import Trajectories


def feature_expectation_from_trajectories(
        trajectories: Trajectories
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