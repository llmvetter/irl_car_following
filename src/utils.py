import numpy as np
import logging
import torch

from src.models.mdp import CarFollowingMDP
from src.models.trajectory import Trajectories
from src.models.reward import RewardNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def svf_from_trajectories(
        trajectories: Trajectories,
        mdp: CarFollowingMDP,
) -> np.ndarray:
    svf = np.zeros(mdp.n_states)
    for trajectory in trajectories:
        for state_action_pair in trajectory:
            idx = state_action_pair.state.index
            svf[idx] += 1
    norm_svf = svf/sum(svf)
    return norm_svf

def terminal_probabilities_from_trajectories(
        trajectories: Trajectories,
        n_states: int,
) -> np.ndarray:
    p = np.zeros(n_states)
    for trajectory in trajectories:
        terminal_state = trajectory.trajectory[-1].state.index
        p[terminal_state] += 1.0

    return p/len(trajectories.trajectories)

def initial_probabilities_from_trajectories(
        trajectories: Trajectories,
        n_states: int,
) -> np.ndarray:
    p = np.zeros(n_states)
    for trajectory in trajectories:
        initial_state = trajectory.trajectory[0].state.index
        p[initial_state] += 1.0

    return p/len(trajectories.trajectories)

def backward_pass(
        mdp: CarFollowingMDP,
        reward: RewardNetwork,
        epsilon: float = 0.1,
        discount: float = 0.95,
        temperature: float = 0.7,
        max_iterations: int = 50,
) -> torch.tensor:
    iteration = 0
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    ValueFunction = torch.full((n_states,), float('-inf'))

    while iteration < max_iterations:
        iteration +=1
        logging.info(f'Backward pass {iteration/max_iterations*100:.2f}% complete')
        ValueFunction_t = ValueFunction.clone()
        Q_sa = torch.zeros((n_states, n_actions))
        for state in range(n_states):
            state_tensor = torch.tensor(mdp._index_to_state(state), dtype=torch.float32)
            state_reward = reward.forward(state_tensor, grad=False).item()
            Q_sa[state] = torch.full((n_actions,), state_reward)
            for action in range(n_actions):
                for next_state, proba in mdp.get_transitions(state, action):
                    Q_sa[state][action] += discount*proba*ValueFunction[int(next_state)]
            ValueFunction[state] = temperature * torch.logsumexp(Q_sa[state] / temperature, dim=0)

        if torch.max(torch.abs(ValueFunction - ValueFunction_t)) < epsilon:
            break
    
    policy = torch.zeros((n_states, n_actions))
    for state in range(n_states):
        for action in range(n_actions):
            policy[state][action] = torch.exp(Q_sa[state][action]-ValueFunction[state])
        policy[state] /= torch.sum(policy[state])
    return policy


def forward_pass(
        mdp: CarFollowingMDP,
        policy: torch.tensor,
        iterations: int = 100,
        steps: int = 2000,
) -> np.ndarray:
    state_visitations = torch.zeros(mdp.n_states)
    for i in range(iterations):
        state = torch.randint(0, mdp.n_states, (1,)).item() # TODO: maybe match trajectories
        for _ in range(steps):
            state_visitations[state] += 1
            action = torch.multinomial(policy[state], 1).item()
            next_state = mdp.step(state, action)
            state = next_state
    # Normalize
    state_visitations /= state_visitations.sum()

    return state_visitations.detach().numpy()