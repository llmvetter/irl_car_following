import pandas as pd

from car_following.src.models.trajectory import Trajectory, Trajectories
from car_following.src.models.mdp import CarFollowingMDP

class Preprocessor():
    def __init__(
            self,
            mdp: CarFollowingMDP,
    ):
        self.mdp = mdp

    def load(
            self,
            path: str,
            hertz: int=3,
    ) -> Trajectories:
        df = pd.read_csv(path, sep='\t', header=None)
        df = df.iloc[::hertz]
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
        trajectory1 = Trajectory(
        speed=df['expert1_speed'],
        distance=df['expert1_distance'],
        acceleration=df['expert1_acceleration'],
        mdp=self.mdp,
        )
        trajectory2 = Trajectory(
            speed=df['expert2_speed'],
            distance=df['expert2_distance'],
            acceleration=df['expert2_acceleration'],
            mdp=self.mdp,
        )
        trajectory3 = Trajectory(
            speed=df['expert3_speed'],
            distance=df['expert3_distance'],
            acceleration=df['expert3_acceleration'],
            mdp=self.mdp,
        ) 
        trajectories = Trajectories([trajectory1, trajectory2, trajectory3])
        return trajectories