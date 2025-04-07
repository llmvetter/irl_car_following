import pandas as pd
from omegaconf import OmegaConf

from src.models.trajectory import Trajectory, Trajectories
from src.models.env import CarFollowingEnv


class MilanoPreprocessor:
    def __init__(
            self,
            mdp: CarFollowingEnv ,
            config: OmegaConf,
    ) -> None:
        self.kmh_to_ms = 0.27778
        self.mdp = mdp
        self.min_speed = config.data.speed_treshold

    def _filter_leader_follower_pairs(self, df: pd.DataFrame, min_entries: int = 800) -> pd.DataFrame:
        """
        Filters leader-follower pairs that have at least `min_entries` data points.
        """
        pair_counts = df.groupby(['Leader', 'Follower']).size()
        valid_pairs = pair_counts[pair_counts >= min_entries].index
        return df[df.set_index(['Leader', 'Follower']).index.isin(valid_pairs)].copy()

    def create_filtered_trajectory(
            self,
            df: pd.DataFrame,
            leader: int,
            follower: int,
    ) -> Trajectory:
        """
        Filters and formats the trajectory data for a specific leader-follower pair.
        """
        subset = df[(df['Leader'] == leader) & (df['Follower'] == follower)]
        subset = subset.sort_values(by="Time [s]").reset_index(drop=True)

        # Extract states
        speed = subset["Follower Speed"].to_numpy()
        distance = subset["gap[m]"].to_numpy()
        relative_speed = subset["Relative speed"].to_numpy()
        acceleration = subset["Follower Tan. Acc."].to_numpy()

        return Trajectory(
            speed=speed,
            distance=distance,
            rel_speed=relative_speed,
            acceleration=acceleration,
            mdp=self.mdp
        )

    def load(self, path: str) -> tuple[Trajectories]:
        """
        Loads and processes the dataset to extract leader-follower trajectories.
        """
        df = pd.read_csv(path)
        # df = df_init.iloc[::3].reset_index(drop=True)  # downsample to 10hz
        df["Follower Speed"] *= self.kmh_to_ms
        df["Relative speed"] *= self.kmh_to_ms

        # Keep only necessary columns
        df_reduced = df[[
            'Time [s]',
            'Leader',
            'Follower',
            'Follower Speed',
            'Leader Tan. Acc.',
            'Follower Tan. Acc.',
            'Relative speed',
            'gap[m]',
        ]].copy()

        df_filtered = self._filter_leader_follower_pairs(df_reduced)
        unique_pairs = df_filtered[['Leader', 'Follower']].drop_duplicates()

        # Generate train trajectories
        train_trajectories = []
        for _, row in unique_pairs.iterrows():
            leader, follower = row["Leader"], row["Follower"]
            traj = self.create_filtered_trajectory(df_filtered, leader, follower)
            if len(traj) > 0:  # Ensure trajectory is not empty
                train_trajectories.append(traj)
        
        return Trajectories(train_trajectories)
