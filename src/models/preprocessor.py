import pandas as pd
from omegaconf import OmegaConf

from src.models.trajectory import Trajectory, Trajectories
from src.models.mdp import CarFollowingMDP


class Preprocessor():
    def __init__(
            self,
            mdp: CarFollowingMDP,
            config: OmegaConf,
    ):
        self.mdp = mdp
        self.min_speed = config.data.speed_treshold

    def create_filtered_trajectory(
            self,
            df: pd.DataFrame,
            expert_num: int,
            min_speed: int,
    ) -> Trajectory:
        filtered_df = df[df[f'expert{expert_num}_speed'] > min_speed].copy()
        filtered_df = filtered_df.reset_index(drop=True)
        
        return Trajectory(
            speed=filtered_df[f'expert{expert_num}_speed'],
            distance=filtered_df[f'expert{expert_num}_distance'],
            acceleration=filtered_df[f'expert{expert_num}_acceleration'],
            mdp=self.mdp,
        )

    def load(
            self,
            path: str,

    ) -> Trajectories:
        df = pd.read_csv(path, sep='\t', header=None)
        df['expert1_acceleration'] = df[0].diff().shift(-1)
        df['expert2_acceleration'] = df[1].diff().shift(-1)
        df['expert3_acceleration'] = df[2].diff().shift(-1)
        df['expert4_acceleration'] = df[3].diff().shift(-1)
        df = df.dropna()
        df = df.reset_index(drop=True)
        df.rename(columns={
            0: 'expert1_speed',
            1: 'expert2_speed',
            2: 'expert3_speed',
            3:'expert4_speed',
            4:'expert1_distance',
            5:'expert2_distance',
            6:'expert3_distance'
            }, 
            inplace=True,
        )
        
        trajectory1 = self.create_filtered_trajectory(df, 1, self.min_speed)
        trajectory2 = self.create_filtered_trajectory(df, 2, self.min_speed)
        trajectory3 = self.create_filtered_trajectory(df, 3, self.min_speed)
        trajectories = Trajectories([trajectory1, trajectory2, trajectory3])
        return trajectories


class MilanoPreprocessor:
    def __init__(
            self,
            mdp: CarFollowingMDP,
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

    def load(self, path: str) -> Trajectories:
        """
        Loads and processes the dataset to extract leader-follower trajectories.
        """
        df = pd.read_csv(path)
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
        unique_pairs = df_filtered.groupby(["Leader", "Follower"]).size().reset_index()

        trajectories = []
        for _, row in unique_pairs.iterrows():
            leader, follower = row["Leader"], row["Follower"]
            traj = self.create_filtered_trajectory(df_filtered, leader, follower)
            if len(traj) > 0:  # Ensure trajectory is not empty
                trajectories.append(traj)

        return Trajectories(trajectories)
