import pandas as pd

from src.models.trajectory import Trajectory, Trajectories
from src.models.mdp import CarFollowingMDP
from src.config import Config

config = Config()

class Preprocessor():
    def __init__(
            self,
            mdp: CarFollowingMDP,
    ):
        self.mdp = mdp
        self.min_speed = config.preprocessor['speed_treshold']

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

