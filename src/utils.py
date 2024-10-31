import numpy as np
import logging

from src.models.mdp import CarFollowingMDP
from src.models.trajectory import Trajectories
from src.models.reward import LinearRewardFunction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def svf_from_trajectories(
        trajectories: Trajectories,
        mdp: CarFollowingMDP,
) -> np.ndarray:
    fe = np.zeros(mdp.n_states)
    for trajectory in trajectories:
        for state_action_pair in trajectory:
            idx = state_action_pair.state.index
            fe[idx] += 1
    return fe/len(trajectories.trajectories)

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
    V = np.random.uniform(low=0.0, high=0.1, size=mdp.n_states)
    for i in range(max_iterations):
        logging.info(f'Backwardpass {i/max_iterations}% complete')
        delta = 0
        for s in range(mdp.n_states):
            v = V[s]
            Q_sa = np.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                for next_s, prob in mdp.get_transitions(s, a):
                    Q_sa[a] += prob * (reward_func.get_reward(s) + gamma * V[int(next_s)])
            V[s] = temperature * np.log(np.sum(np.exp(Q_sa / temperature))+ 1e-8)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # Compute the policy
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    for s in range(mdp.n_states):
        Q_sa = np.zeros(mdp.n_actions)
        for a in range(mdp.n_actions):
            for next_s, prob in mdp.get_transitions(s, a):
                Q_sa[a] += prob * (reward_func.get_reward(s) + gamma * V[int(next_s)])
        policy[s] = np.exp((Q_sa - V[s]) / temperature)
        policy[s] /= np.sum(policy[s])
    
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