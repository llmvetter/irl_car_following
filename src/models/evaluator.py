import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.models.env import CarFollowingEnv
from src.models.agent import Agent


class Evaluator():
    def __init__(
            self,
            environment: CarFollowingEnv,
            agent: Agent,
    ) -> None:
        self.env = environment
        self.agent = agent

    @classmethod
    def sample_trajectory(
        cls,
        steps: int=500,
        min_v: int=10,
        max_v: int=30,
        smoothness: int=50,
    ) -> np.ndarray:

        changes = np.random.normal(0, 0.1, steps)
        walk = np.cumsum(changes)
        walk = (walk - walk.min()) / (walk.max() - walk.min())
        walk = walk * (max_v - min_v) + min_v
        smooth_walk = gaussian_filter1d(walk, sigma=smoothness)
        smooth_walk = np.clip(smooth_walk, min_v, max_v)
        return smooth_walk

    def follow_trajectory(
        self,
        leader_trajectory: np.ndarray | None = None,
        v_ego_init: int = 20,
        d_ego_init: int = 30,
        visualize: bool = True,
     ) -> None:
        crash = False
        if leader_trajectory is None:
            leader_trajectory = self.sample_trajectory()

        follower_trajectory = []
        v_rel_init = leader_trajectory[0]-v_ego_init
        ego_vehicle_state = np.array([
            v_ego_init,
            d_ego_init,
            v_rel_init,
        ])
        for step in range(len(leader_trajectory) - 1):
            self.env.reset()
            self.env.state = ego_vehicle_state
            action = self.agent.choose_action(ego_vehicle_state)
            next_state, reward, terminated, truncated, _  = self.env.step(
                action=action,
                lead_speed=leader_trajectory[step+1]
            )
            if terminated:
                break
            follower_trajectory.append(next_state)
            ego_vehicle_state = next_state

        if terminated:
            missing_steps = len(leader_trajectory) - len(follower_trajectory)
            dummy_variable = np.zeros_like(ego_vehicle_state)
            follower_trajectory.extend([dummy_variable] * missing_steps)
            crash = True

        else:
            dummy_variable = follower_trajectory[-1]
            follower_trajectory.append(dummy_variable)

        follower_velocity = np.array([item[0] for item in follower_trajectory])
        distance_gap = np.array([item[1] for item in follower_trajectory])
        relative_speed = np.array([item[2] for item in follower_trajectory])
        time_steps = range(len(leader_trajectory))
        
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, follower_velocity, label='follower velocity', marker='o', markersize=5)
            plt.plot(time_steps, leader_trajectory, label='leader velocity', marker='o', markersize=5)
            plt.xlabel('Time Steps in 0.1s')
            plt.ylabel('Velocity in m/s')
            plt.title('Velocity over time')
            plt.legend()
            plt.grid(True)

            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, distance_gap, label='distance gap', marker='o', markersize=5)
            plt.xlabel('Time steps in 0.1s')
            plt.ylabel('Distance gap in m')
            plt.title('Distance gap to lead vehicle')
            plt.legend()
            plt.grid(True)

            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, relative_speed, label='distance gap/velocity', marker='o', markersize=5)
            plt.xlabel('Time steps in 0.1s')
            plt.ylabel('relative velocity in m/s')
            plt.title('Relative velocity (v_lead - v_ego)')
            plt.legend()
            plt.grid(True)

            # Show the plot
            plt.show()
        
        return follower_trajectory, crash


    def evaluate(
        self,
        num_trajectories: int = 50,
        v_ego_init: int = 20,
        d_ego_init: int = 30,
    ) -> dict:
        """Evaluates the agent over multiple random leader trajectories and computes averaged metrics, including crashes."""

        all_distance_gaps = []
        all_follower_speeds = []
        all_leader_speeds = []
        all_relative_speeds = []
        crash_counter = 0

        for _ in range(num_trajectories):
            leader_trajectory = self.sample_trajectory(
                steps=1000,
                smoothness=20,
            )
            follower_trajectory, crash = self.follow_trajectory(
                leader_trajectory,
                v_ego_init,
                d_ego_init,
                visualize=False,
            )
            if crash:
                crash_counter += 1
                continue

            all_leader_speeds.append(leader_trajectory)
            all_follower_speeds.append([item[0] for item in follower_trajectory])
            all_distance_gaps.append([item[1] for item in follower_trajectory])
            all_relative_speeds.append([item[2] for item in follower_trajectory])

        if len(all_distance_gaps) > 0:
            all_distance_gaps = np.concatenate(all_distance_gaps)
            all_follower_speeds = np.concatenate(all_follower_speeds)
            all_leader_speeds = np.concatenate(all_leader_speeds)
            all_relative_speeds = np.concatenate(all_relative_speeds)

            min_gap = np.min(all_distance_gaps)
            max_gap = np.max(all_distance_gaps)
            if max_gap > min_gap:  # min-max normalization
                normalized_distance_gaps = (all_distance_gaps - min_gap) / (max_gap - min_gap)
            else:
                normalized_distance_gaps = np.zeros_like(all_distance_gaps)

            normalized_distance_gap_variance = np.var(normalized_distance_gaps)

            metrics = {
                "distance_gap_variance": normalized_distance_gap_variance,
                "speed_mse": np.mean((all_follower_speeds - all_leader_speeds) ** 2),
                "relative_speed_mean": np.mean(all_relative_speeds),
                "relative_speed_variance": np.var(all_relative_speeds),
                "crash_ratio": crash_counter / num_trajectories
            }

            all_values = {
                "speed_mse": (all_follower_speeds - all_leader_speeds) ** 2,
                "relative_speed_mean": all_relative_speeds,
                "relative_speed_variance": all_relative_speeds,
            }

            normalized_metrics = {"distance_gap_variance": normalized_distance_gap_variance}
            for key, value in metrics.items():
                if key in ["crash_ratio", "distance_gap_variance"]:
                    normalized_metrics[key] = value
                    continue
                min_val = np.min(all_values[key])
                max_val = np.max(all_values[key])
                normalized_metrics[key] = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5

            weights = {
                "crash_ratio": 0.5, #lower is better
                "distance_gap_variance": 0.2, #lower is better
                "speed_mse": 0.1, #lower is better
                "relative_speed_variance": 0.2, #lower is better
            }

            final_score = sum(weights[key] * normalized_metrics[key] for key in weights)

        else:
            metrics = {"crash_ratio": 1.0}
            normalized_metrics = {"crash_ratio": 1.0}
            final_score = 1 # Worst case

        return {
            "metrics": metrics,
            "normalized_metrics": normalized_metrics,
            "final_score": final_score
        }
