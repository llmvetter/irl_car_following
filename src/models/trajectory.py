from lunar_lander.src.models.state_action import StateActionPair


class Trajectory():
    def __init__(self, trajectory):
        self.trajectory = self.validate(trajectory)
    def validate(self, feature):
        if isinstance(feature, list):
            return feature
        else:
            return ValueError('not of type list[StateActionPair]')
        
    def discretize(self):
        for state_action_pair in self.trajectory:
            state_action_pair.state = StateActionPair.clip_state(
                state_action_pair.state
            )