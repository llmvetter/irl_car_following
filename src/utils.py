import numpy as np
import logging
from scipy.special import logsumexp

from src.models.mdp import CarFollowingMDP
from src.models.trajectory import Trajectories
from src.models.reward import LinearRewardFunction

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
        reward_func: LinearRewardFunction,
        gamma: float=0.99,
        theta: float=1e-6, 
        max_iterations: int=50, 
        temperature: float=1.0,
) -> np.ndarray:
    """
    gamma: discount factor for future state reward
    """
    log_V = np.zeros(mdp.n_states)

    for i in range(max_iterations):
        logging.info(f'Backward pass {i/max_iterations*100:.2f}% complete')
        delta = 0
        for s in range(mdp.n_states):
            old_v = log_V[s]
            log_Q_sa = np.full(mdp.n_actions, -np.inf)
            for a in range(mdp.n_actions):
                log_Q_sa[a] = np.log(reward_func.get_reward(s) + 1e-300) 
                for next_s, prob in mdp.get_transitions(s, a):
                   log_Q_sa[a] = np.logaddexp(log_Q_sa[a], np.log(prob + 1e-300) + gamma * log_V[int(next_s)])
            log_V[s] = temperature * logsumexp(log_Q_sa / temperature)
            delta = max(delta, abs(np.exp(old_v) - np.exp(log_V[s])))
        if delta < theta:
            break

    # Compute the policy
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    for s in range(mdp.n_states):
        log_Q_sa = np.full(mdp.n_actions, -np.inf)
        for a in range(mdp.n_actions):
            log_Q_sa[a] = np.log(reward_func.get_reward(s) + 1e-300)
            for next_s, prob in mdp.get_transitions(s, a):
                log_Q_sa[a] = np.logaddexp(log_Q_sa[a], np.log(prob + 1e-300) + gamma * log_V[int(next_s)])
        policy[s] = np.exp((log_Q_sa - logsumexp(log_Q_sa)) / temperature)
    
    return policy

def forward_pass(
        mdp: CarFollowingMDP,
        policy: np.ndarray,
        iterations: int = 100,
) -> np.ndarray:
    state_visitations = np.zeros(mdp.n_states)
    for i in range(iterations):
        state = np.random.randint(0, mdp.n_states)  # Start from a randomly sampled state
        for _ in range(2000):
            state_visitations[state] += 1
            action = np.random.choice(mdp.n_actions, p=policy[state])
            next_state = mdp.step(state, action)
            state = next_state
    # Normalize
    state_visitations /= state_visitations.sum()

    return state_visitations