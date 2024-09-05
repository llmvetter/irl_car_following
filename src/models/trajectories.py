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