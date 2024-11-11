import numpy as np
import logging
from scipy.special import logsumexp
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
        reward_func: RewardNetwork,
        gamma: float=0.99,
        theta: float=1e-6, 
        max_iterations: int=50, 
        temperature: float=1.0,
) -> torch.tensor:
    """
    gamma: discount factor for future state reward
    """
    log_V = torch.zeros(mdp.n_states)

    for i in range(max_iterations):
        logging.info(f'Backward pass {i/max_iterations*100:.2f}% complete')
        delta = 0
        for s in range(mdp.n_states):
            old_v = log_V[s]
            log_Q_sa = torch.full((mdp.n_actions,), float('-inf'))
            for a in range(mdp.n_actions):
                feature_tensor = torch.tensor(mdp._index_to_state(s), dtype=torch.float32)
                log_Q_sa[a] = torch.log(reward_func.forward(feature_tensor, grad=False)) .squeeze()
                for next_s, prob in mdp.get_transitions(s, a):
                   log_Q_sa[a] = torch.logaddexp(log_Q_sa[a], torch.log(torch.tensor(prob)) + gamma * log_V[int(next_s)])
            log_V[s] = temperature * torch.logsumexp(log_Q_sa / temperature, dim=0)
            delta = max(delta, abs(torch.expm1(old_v - log_V[s])))
        if delta < theta:
            break

    # Compute the policy
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    for s in range(mdp.n_states):
        log_Q_sa = torch.full((mdp.n_actions,), float('-inf'))
        for a in range(mdp.n_actions):
            feature_tensor = torch.tensor(mdp._index_to_state(s), dtype=torch.float32)
            log_Q_sa[a] = torch.log(reward_func.forward(feature_tensor, grad=False))
            for next_s, prob in mdp.get_transitions(s, a):
                log_Q_sa[a] = torch.logaddexp(log_Q_sa[a], torch.log(torch.tensor(prob)) + gamma * log_V[int(next_s)])
        policy[s] = torch.exp((log_Q_sa - torch.logsumexp(log_Q_sa, dim=0)) / temperature)
    
    return policy

def forward_pass(
        mdp: CarFollowingMDP,
        policy: torch.tensor,
        iterations: int = 100,
) -> np.ndarray:
    state_visitations = torch.zeros(mdp.n_states)
    for i in range(iterations):
        state = torch.randint(0, mdp.n_states, (1,)).item() # TODO: maybe match trajectories
        for _ in range(2000):
            state_visitations[state] += 1
            action = torch.multinomial(policy[state], 1).item()
            next_state = mdp.step(state, action)
            state = next_state
    # Normalize
    state_visitations /= state_visitations.sum()

    return state_visitations.detach().numpy()