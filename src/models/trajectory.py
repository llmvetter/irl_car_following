from car_following.src.models.mdp import StateActionPair, CarFollowingMDP

class Trajectory:
    def __init__(
            self, 
            speed: float,
            distance: float,
            acceleration: float,
            mdp: CarFollowingMDP,
    ) -> None:
        if not (len(speed) == len(distance) == len(acceleration)):
            print(len(speed), len(distance), len(acceleration))
            raise ValueError("Speed, distance, and acceleration must have the same length")

        self.trajectory = []
        for s, d, a in zip(speed, distance, acceleration):
            pair = StateActionPair(
                state=(s, d),
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