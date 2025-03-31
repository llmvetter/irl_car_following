from typing import Any
import numpy as np
import logging
import torch

from src.models.env import CarFollowingEnv
from src.models.trajectory import Trajectories
from src.models.reward import RewardNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def deep_merge(
        d1: dict[str, Any],
        d2: dict[str, Any],
) -> dict[str, Any]:

    merged = dict(d1)
    for key, value in d2.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def svf_from_trajectories(
        trajectories: Trajectories,
        mdp: CarFollowingEnv,
) -> np.ndarray:
    svf = np.zeros(mdp.n_states)
    for trajectory in trajectories:
        for state_action_pair in trajectory:
            idx = state_action_pair.state.index
            svf[idx] += 1
    norm_svf = svf/sum(svf)
    return norm_svf

def value_iteration(
        mdp: CarFollowingEnv,
        reward: RewardNetwork,
        epsilon: float = 0.1,
        discount: float = 0.95,
        temperature: float = 0.7,
        max_iterations: int = 50,
) -> torch.tensor:

    V = np.zeros(mdp.n_states, dtype=np.float32)
    Q = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float32)
    R = reward.forward(
        torch.from_numpy(mdp.state_grid).float(),
        grad=False,
    ).numpy().astype(dtype=np.float32)

    for iteration in range(max_iterations):
        # Compute Q-values: Q[s, a] = reward[s, a] + γ * V[next_state]
        Q = R + discount * V[mdp.T]
        Q_tensor = torch.tensor(Q, dtype=torch.float32)

        # Soft value update: V[s] = τ * logsumexp(Q(s, a) / τ)
        V_new = temperature * torch.logsumexp(Q_tensor / temperature, dim=1).numpy()

        # Check for convergence
        if np.max(np.abs(V_new - V)) < epsilon:
            print(f'Converged after {iteration} iterations')
            break

        V = V_new  # Update value function
    Q_tensor = torch.tensor(Q, dtype=torch.float32)
    policy = torch.zeros((mdp.n_states, mdp.n_actions))
    policy = torch.exp(Q_tensor - V[:, None])  # Broadcast subtraction
    policy /= policy.sum(dim=1, keepdim=True)  # Normalize across actions
    return policy

def rollout(
    mdp: CarFollowingEnv,
    policy: torch.tensor,
    iterations: int = 50,
    steps: int = 1000,
) -> np.ndarray:

    state_visitations = torch.zeros(mdp.n_states)
    for i in range(iterations):
        mdp.reset()
        state_idx = torch.randint(0, mdp.n_states, (1,)).item() # TODO: maybe match trajectories
        mdp.state = mdp._index_to_state(state_idx)

        for _ in range(steps):
            state_visitations[state_idx] += 1
            action = torch.multinomial(policy[state_idx], 1).item()
            next_state, reward, terminated, truncated, info = mdp.step(
                action=action,
                determinsitic=False,
            )
            next_state_index = info['index']
            state_idx = next_state_index

    # Normalize
    state_visitations /= state_visitations.sum()

    return state_visitations.detach().numpy()
