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