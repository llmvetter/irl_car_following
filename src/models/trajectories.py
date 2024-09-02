import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lunar_lander.src.models.trajectory import Trajectory
from lunar_lander.src.models.trajectory import StateActionPair


class Trajectories():
    def __init__(self, trajectories):
        all_trajectories = []
        for trajectory in trajectories:
            single_trajectory = Trajectory(trajectory=[])
            for state_action_pair in trajectory:
                state, action = state_action_pair
                single_trajectory.trajectory.append(StateActionPair(state=state[0], action=action))
            # single_trajectory.discretize()
            all_trajectories.append(single_trajectory)
        self.trajectories = self.validate(all_trajectories)

    def validate(self, feature):
        if isinstance(feature, list):
            return feature
        else:
            raise ValueError('not of type list[Trajectory]')

    def plot(
            self,
            summary:bool=False
    ) -> None:
        feature_names = [
            "x-coordinate",
            "y-coordinate",
            "x-velocity",
            "y-velocity",
            "angle",
            "angular velocity",
            "leg contact (left)",
            "leg contact (right)"
        ]

        all_feature_values = {name: [] for name in feature_names}

        for traj in self.trajectories:
            for state in traj.trajectory:
                for i, name in enumerate(feature_names):
                    all_feature_values[name].append(state.state_clipped[i])

        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle('Feature Distributions Across All Expert Trajectories')

        for i, (name, values) in enumerate(all_feature_values.items()):
            row = i // 2
            col = i % 2
            sns.histplot(values, kde=True, ax=axes[row, col])
            axes[row, col].set_title(name)
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        if summary is True:
            print("Summary Statistics:")
            for name, values in all_feature_values.items():
                print(f"\n{name}:")
                print(f"  Mean: {np.mean(values):.4f}")
                print(f"  Std Dev: {np.std(values):.4f}")
                print(f"  Min: {np.min(values):.4f}")
                print(f"  Max: {np.max(values):.4f}")
                print(f"  25th Percentile: {np.percentile(values, 25):.4f}")
                print(f"  Median: {np.median(values):.4f}")
                print(f"  75th Percentile: {np.percentile(values, 75):.4f}")

