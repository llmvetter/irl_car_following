import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import sparse
from itertools import product
from scipy import sparse

from car_following.src.models.mdp import CarFollowingMDP
from car_following.src.models.trajectory import Trajectories
from car_following.src.models.reward import LinearRewardFunction


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
) -> np.ndarray:
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    
    # Precompute rewards
    reward = np.array([reward_func.get_reward(s) for s in range(n_states)])

    # Backward Pass
    # init zs (state partition function)
    log_zs = np.zeros(n_states)

    for _ in 2*n_states:
        log_za = np.full((n_states, n_actions), -np.inf)
        for s_from, a in product(range(n_states), range(n_actions)):
            #sum state value for all possible next state given current state-action pair
            log_za[s_from, a] = reward[s_from] + logsumexp(np.log(mdp.T[s_from, :, a] + 1e-300) + log_zs)
        log_zs = logsumexp(log_za, axis=1)
    p_action = np.exp(log_za - log_zs[:, None])
    return p_action

def create_sparse_transition_matrix(mdp):
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    T_reshaped = mdp.T.reshape(n_states * n_actions, n_states)
    T_sparse = sparse.csr_matrix(T_reshaped)
    
    return T_sparse

def forward_pass(
        mdp: CarFollowingMDP,
        p_action: np.ndarray,
) -> np.ndarray:

    p_initial = np.ones(mdp.n_states) / mdp.n_states
    T_sparse = create_sparse_transition_matrix(mdp)
    p_action_sparse = sparse.csr_matrix(p_action)
    
    # Initialize d with p_initial
    d = sparse.csr_matrix(p_initial).T
    
    # Small constant to prevent underflow
    epsilon = 1e-10
    
    for t in range(mdp.n_states):
        # Element-wise multiplication
        s_a_probs = d.multiply(p_action_sparse)
        s_a_probs_r = s_a_probs.reshape(1, -1)
        d_next = s_a_probs_r.dot(T_sparse).T
        
        # Convert to dense, add epsilon, and normalize
        d_dense = d_next.toarray() + epsilon
        d_dense /= d_dense.sum() + epsilon
        
        # Convert back to sparse
        d = sparse.csr_matrix(d_dense)
        
        # Break if d becomes all zeros
        if d.nnz == 0:
            print(f"Warning: d became all zeros at iteration {t}")
            break
    
    state_frequencies = d.sum(axis=1).A1
    return state_frequencies
