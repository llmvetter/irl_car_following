import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt

from src.models.mdp import CarFollowingMDP
from src.config import Config

class Evaluator:

    def __init__(
            self,
            policy: torch.Tensor,
            mdp: CarFollowingMDP,
    ):
        self.policy = policy
        self.mdp = mdp
        self.config = Config()
    
    def policy_infere(
        self,
        ego_vehicle_state:tuple,
        lead_vehicle_speed:float,
    ) -> tuple:
        v_ego, g_ego = ego_vehicle_state
        v_lead = lead_vehicle_speed
        ego_state_idx = self.mdp._state_to_index(ego_vehicle_state)
        action_idx = torch.argmax(self.policy[ego_state_idx])
        action = self.mdp._index_to_action(action_idx)
        v_next = max(min(v_ego + action * self.mdp.delta_t, self.mdp.v_max+5), 0)
        g_next = max(min(g_ego-(v_ego-v_lead)*self.mdp.delta_t-(0.5*action*self.mdp.delta_t**2), self.mdp.g_max+5), 0)
        return (v_next, g_next)

    def random_trajectory(
            self,
            initial_speed: float,
            num_steps: int = 1000,
            max_speed_change: float = 0.3,
            min_speed: int = 3,
            max_speed: int = 18,
            window_size: int = 50,
    ):
        speeds = [initial_speed]
        smoothed_speeds = [initial_speed]
        speed_window = deque([initial_speed], maxlen=window_size)
        
        for _ in range(num_steps - 1):
            speed_change = np.random.uniform(-1, 1) * max_speed_change
            new_speed = speeds[-1] + speed_change
            new_speed = max(min_speed, min(new_speed, max_speed))
            speeds.append(new_speed)
            speed_window.append(new_speed)
            smoothed_speed = sum(speed_window) / len(speed_window)
            smoothed_speeds.append(smoothed_speed)

        return smoothed_speeds
    
    def evaluate(
            self,
            v_lead_init,
            v_ego_init,
            d_ego_init,
    ) -> None:
        follower_trajectory = []
        ego_vehicle_state = tuple([v_ego_init, d_ego_init])
        leader_trajectory = self.random_trajectory(
            initial_speed = v_lead_init,
        )
        for step in range(len(leader_trajectory) - 1):
            next_state = self.policy_infere(
                ego_vehicle_state=ego_vehicle_state,
                lead_vehicle_speed=leader_trajectory[step+1]
            )
            follower_trajectory.append(next_state)
            ego_vehicle_state = next_state

        dummy_variable = follower_trajectory[-1]
        follower_trajectory.append(dummy_variable)

        follower_velocity = [item[0] for item in follower_trajectory]
        distance_gap = [item[1] for item in follower_trajectory]
        time_steps = range(len(leader_trajectory))


        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, follower_velocity, label='follower velocity', marker='o')
        plt.plot(time_steps, leader_trajectory, label='leader velocity', marker='o')
        plt.xlabel('Time Steps in 0.1s')
        plt.ylabel('Velocity in m/s')
        plt.title('Velocity over time')
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, distance_gap, label='distance gap', marker='o')
        plt.xlabel('Time steps in 0.1s')
        plt.ylabel('Distance gap in m')
        plt.title('Distance gap to lead vehicle')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
                

