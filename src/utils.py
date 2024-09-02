import numpy as np
import torch

from lunar_lander.src.models.state_action import State


def expert_state_visitation_frequency(trajectories):
    state_dims = 393216
    n_states = np.prod(state_dims)
    svf_vector = np.zeros(n_states)
    # svf_matrix = []
    for trajectory in trajectories.trajectories:
        trajectory_visitation = []
        for state_visited in trajectory.trajectory:
            state_index = state_visited.index
            svf_vector[state_index] += 1.0
            trajectory_visitation.append(state_index)
        # svf_matrix.append(trajectory_visitation)
    
    total_visits = np.sum(svf_vector)
    if total_visits > 0:
        normalized_svf_vector = svf_vector / total_visits
    else:
        raise Exception("svf is 0")
    
    return normalized_svf_vector

def policy_state_visitation_frequency(
        policy_network,
        env,
        num_trajectories=1000,
        max_steps=1000,
):
    
    svf = np.zeros(393216)

    for _ in range(num_trajectories):
        state = State(state=env.reset()[0])
        for _ in range(max_steps):
            svf[state.index] += 1
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.state).unsqueeze(0)
                action = policy_network.get_action(state_tensor)
            state, _, done, _, _  = env.step(action)
            state = State(state)
            if done:
                break

    svf = svf / svf.sum()
    return svf