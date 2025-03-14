import logging

from typing import Optional, Any

import numpy as np
import gymnasium as gym

from src.models.simulator import Simulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CarFollowingEnv(gym.Env):
    def __init__(
            self,
            dataset_path: str,
            max_speed: int = 30,
            max_distance: int = 100,
            max_rel_speed: int = 75,
            granularity: float = 1.0,
            actions: list[float]= [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
            delta_t: float = 0.1,
    ) -> None:

        # Environment parameters
        super().__init__()
        self.max_speed = max_speed
        self.max_distance = max_distance
        self.max_rel_speed = max_rel_speed
        self.granularity = granularity
        self.delta_t = delta_t
        self.simulator = Simulator(path=dataset_path)

        # Dicretize action space
        self.actions = np.array(actions)
        self.n_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.action_mapping = {i: actions[i] for i in range(self.n_actions)}

        # Discretize state space
        self.v_space = np.arange(0, self.max_speed, granularity)
        self.g_space = np.arange(0, self.max_distance, granularity)
        self.v_rel_space = np.arange(-self.max_rel_speed, self.max_rel_speed, granularity)

        # State Grid
        self.state_grid = np.array(np.meshgrid(self.v_space, self.g_space, self.v_rel_space)).T.reshape(-1, 3)

        # Create mappings for fast lookup
        self.state_to_index = {tuple(state): idx for idx, state in enumerate(self.state_grid)}
        self.index_to_state = {idx: tuple(state) for idx, state in enumerate(self.state_grid)}

        # Define state and action spaces
        self.observation_space = gym.spaces.Discrete(len(self.state_grid))
        self.n_states = len(self.state_grid)
        self.n_actions = len(actions)

        # Define initial state
        self.state = None
        self.index = None

        # Init transition matrix
        self.T = np.zeros((self.n_states, self.n_actions), dtype=int)

    def reset(
            self, seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[np.array, dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        self.state = self._discretize_state(np.array([
            np.random.uniform(5, self.max_speed-5),  # ego speed
            np.random.uniform(5, self.max_distance/2),  # distance to lead vehicle
            np.random.uniform(-self.max_rel_speed/2, self.max_rel_speed)
        ], dtype=np.float32))

        self.index = self.state_to_index[tuple(self.state)]

        return self._get_obs(), self._get_info()

    def step(
            self,
            action: int,
            lead_speed: Optional[float] = None,
    ) -> None:
        """Take an action and return the next state, reward, done flag, and additional info."""
        ### RELATIVE SPEED = V_LEAD - V_FOLLOW ###

        ego_speed, distance_to_lead, relative_speed = self._get_obs()
        acceleration = self.action_mapping.get(action, 0)

        # velocity transition
        next_ego_speed = np.clip(ego_speed + acceleration * self.delta_t, 0, self.max_speed)

        # gap transition
        if lead_speed:
            relative_speed = lead_speed - ego_speed
            next_distance_gap = distance_to_lead + (relative_speed*self.delta_t) - (0.5*action*self.delta_t**2) #ego(v) - lead(v)
            next_relative_speed = lead_speed - next_ego_speed
        else:
            next_distance_gap = distance_to_lead + (relative_speed*self.delta_t) - (0.5*action*self.delta_t**2) #ego(v) - lead(v)
            # relative speed transition
            next_relative_speed = self.simulator.smooth_relative_speed(relative_speed)

        # Update state
        self.state = self._discretize_state(np.array([
            next_ego_speed,
            next_distance_gap,
            next_relative_speed,
        ], dtype=np.float32))

        self.index = self.state_to_index[self.state]

        terminated = False
        truncated = False
        reward = 0

        # Episode termination criterion -> crash
        if next_distance_gap < 0.5 or next_distance_gap > self.max_distance:
            terminated = True
            reward = -1

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def compute_transitions(self):
        for state_idx in range(self.n_states):
            for action_idx in range(self.n_actions):
                self.reset()
                state = self._index_to_state(state_idx)
                self.state = state
                next_state, reward, terminated, truncated, info = self.step(action_idx)
                self.T[state_idx, action_idx] = info['index']
    
    def _get_obs(self):
        """Return the current observation (state)."""
        return self.state

    def _get_info(self):
        """Return additional information about the current state."""
        return {
            "current_ego_speed": self.state[0],
            "distance_to_lead": self.state[1],
            "relative_speed": self.state[2],
            "index": self.index,
        }

    def _index_to_state(self, index: int) -> tuple[Any]:
        return self.index_to_state[index]

    def _state_to_index(self, state: tuple) -> int:
        return self.state_to_index[state]

    def _action_to_index(self, action: float) -> int:
        diffs = np.abs(self.actions - action)
        return np.argmin(diffs)


    def _index_to_action(self, index: int) -> float:
        return self.action_mapping.get(index)

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Returns discretized state."""
        v, g, v_rel = state
        v_discrete = min(self.v_space, key=lambda x: abs(x - v))
        g_discrete = min(self.g_space, key=lambda x: abs(x - g))
        v_rel_discrete = min(self.v_rel_space, key=lambda x: abs(x - v_rel))

        return (v_discrete, g_discrete, v_rel_discrete)

    def _discretize_action(self, action: float) -> int:
        """Returns dicsretized action."""
        diffs = np.abs(self.actions - action)
        action_index = np.argmin(diffs)
        return self.actions[action_index]


class State:
    def __init__(self, mdp:CarFollowingEnv, state: tuple[Any]):
        self.mdp = mdp
        self.state = self.mdp._discretize_state(state)
        self.index = self.mdp._state_to_index(self.state)

class Action:
    def __init__(self, mdp: CarFollowingEnv, action: float):
        self.mdp = mdp
        self.action = self.mdp._discretize_action(action)
        self.index = self.mdp._action_to_index(self.action)

class StateActionPair:
    def __init__(
            self,
            mdp: CarFollowingEnv, 
            state: tuple[float],
            action: float,
    ) -> None:
        self.mdp = mdp
        self.state = State(mdp, state)
        self.action = Action(mdp, action)

    def get_state(self):
        return self.state.state

    def get_action(self):
        return self.action.action

    def get_state_index(self):
        return self.state.index

    def get_action_index(self):
        return self.action.index