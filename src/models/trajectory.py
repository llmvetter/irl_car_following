from src.models.env import CarFollowingEnv, StateActionPair

class Trajectory:
    def __init__(
            self, 
            speed: list[float],
            distance: list[float],
            rel_speed: list[float],
            acceleration: list[float],
            mdp: CarFollowingEnv,
    ) -> None:
        if not (len(speed) == len(distance) == len(rel_speed) == len(acceleration)):
            print(len(speed), len(distance), len(rel_speed), len(acceleration))
            raise ValueError("Speed, distance, relative speed and acceleration must have the same length")

        self.trajectory = []
        for s, d, s_rel, a in zip(speed, distance, rel_speed, acceleration):
            state = (s, d, s_rel)
            pair = StateActionPair(
                state=state,
                action=a,
                mdp=mdp,
            )
            self.trajectory.append(pair)

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, index):
        return self.trajectory[index]

    def __iter__(self):
        return iter(self.trajectory)
    
class Trajectories():
    def __init__(self, trajectories):
        if not isinstance(trajectories, list):
            raise ValueError('Expected object of type list[Trajectory]')

        self.trajectories = []
        for trajectory in trajectories:
            self.trajectories.append(trajectory)
    
    def __add__(self, other):
        if not isinstance(other, Trajectories):
            raise ValueError('Can only add another Trajectories object')
        return Trajectories(self.trajectories + other.trajectories)

    def __iter__(self):
        return iter(self.trajectories)