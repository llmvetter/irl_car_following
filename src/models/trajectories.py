class Trajectories():
    def __init__(self, trajectories):
        if not isinstance(trajectories, list):
            raise ValueError('Expected object of type list[Trajectory]')

        self.trajectories = []
        for trajectory in trajectories:
            self.trajectories.append(trajectory)