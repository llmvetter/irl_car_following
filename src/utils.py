import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import sparse
from tqdm import tqdm
from itertools import product
from scipy import sparse

from car_following.src.models.mdp import CarFollowingMDP
from car_following.src.models.trajectory import Trajectories
from car_following.src.models.reward import LinearRewardFunction


def feature_expectation_from_trajectories(
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

    for _ in tqdm(range(50)): #2 * n_states
        log_za = np.full((n_states, n_actions), -np.inf)
        for s_from, a in product(range(n_states), range(n_actions)):
            #sum state value for all possible next state given current state-action pair
            log_za[s_from, a] = reward[s_from] + logsumexp(np.log(mdp.T[s_from, :, a] + 1e-300) + log_zs)
        log_zs = logsumexp(log_za, axis=1)
    p_action = np.exp(log_za - log_zs[:, None])
    return p_action

def forward_pass(
        mdp: CarFollowingMDP,
        p_initial: np.ndarray,
        p_action: np.ndarray,
):

    n_states = mdp.n_states
    n_actions = mdp.n_actions

    # init with starting probability
    d = np.zeros((n_states, 2 * n_states))
    d[:, 0] = p_initial
    
    for t in range(1, 2 * n_states):
        for s_to in range(n_states):
            for s_from, a in product(range(n_states), range(n_actions)):
                d[s_to, t] += d[s_from, t-1] * p_action[s_from, a] * mdp.T[s_from, s_to, a]
    
    return np.sum(d, axis=1)

from scipy import sparse

def create_sparse_transition_matrix(mdp):
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    T_reshaped = mdp.T.reshape(n_states * n_actions, n_states)
    T_sparse = sparse.csr_matrix(T_reshaped)
    
    return T_sparse

def forward_pass_optimized(mdp, p_action, n_iterations=4000):

    p_initial = np.ones(mdp.n_states) / mdp.n_states
    T_sparse = create_sparse_transition_matrix(mdp)
    p_action_sparse = sparse.csr_matrix(p_action)
    
    # Initialize d with p_initial
    d = sparse.csr_matrix(p_initial).T
    
    # Small constant to prevent underflow
    epsilon = 1e-10
    
    for t in tqdm(range(1, n_iterations)):
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

# def compute_expected_svf(
#         mdp: CarFollowingMDP,
#         trajectories: Trajectories, 
#         reward: LinearRewardFunction,
# ) -> np.ndarray:
    
#     n_states = mdp.n_states
#     n_actions = mdp.n_actions
    
#     # Compute rewards in log space
#     log_rewards = np.array([reward.get_reward(s) for s in range(n_states)])
    
#     # Initialize log_zs based on frequency of end states in trajectories
#     log_zs = terminal_probabilities_from_trajectories(
#         trajectories=trajectories,
#         n_states = n_states,
#     )

#     # Perform backward pass
#     for _ in tqdm(range(n_states)):
    
#         log_za = np.full((n_states, n_actions), -np.inf)  # Initialize with log(0)
        
#         for s_from in range(n_states):
#             for a in range(n_actions):
#                 transition_probs = mdp.T[s_from, :, a]
#                 s_to = mdp.T[(s_from, a)]
#                 log_za[s_from, a] = log_rewards[s_from] + log_zs[s_to]
        
#         log_zs = logsumexp(log_za, axis=1)
    
#     log_p_action = log_za - logsumexp(log_za, axis=1)[:, None]
#     p_action = np.exp(log_p_action)

#     p_transition = np.zeros((mdp.n_states, mdp.n_states, mdp.n_actions))

#     for (s_from, a), s_to in mdp.T.items():
#         p_transition[s_from, s_to, a] = 1

#     p_initial = initial_probabilities_from_trajectories(
#         trajectories,
#         n_states,
#     )

#     p_transition_sparse = sparse.csr_matrix(p_transition.reshape(n_states * n_actions, n_states))
#     p_action_sparse = sparse.csr_matrix(p_action)
#     d = sparse.csr_matrix((n_states, n_states))
#     d[:, 0] = p_initial

#     for t in tqdm(range(1, 4000)):
#         s_a_probs = d[:, t-1].multiply(p_action_sparse)
#         s_a_probs_r = s_a_probs.reshape(1, -1)
#         d[:, t] = s_a_probs_r.dot(p_transition_sparse).T

#     state_frequencies = d.sum(axis=1).A1
#     return state_frequencies


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
