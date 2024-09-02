import numpy as np

class State():
    def __init__(self, state):
        self.state = self.validate(state)
        self.bins = {
            'x_bins': [-np.inf, -0.15, -0.05, -0.01, 0.01, 0.05, 0.15, np.inf],
            'y_bins' : [-np.inf, 0.1, 0.3, 0.6, 1.0, 1.3, np.inf],
            'x_vel_bins' : [-np.inf, -0.3, -0.1, -0.01, 0.01, 0.1, 0.3, np.inf],
            'y_vel_bins' : [-np.inf, -0.8, -0.6, -0.4, -0.2, 0, np.inf],
            'angle_bins' : [-np.inf, -0.3, -0.1, -0.01, 0.01, 0.1, 0.3, np.inf],
            'ang_vel_bins' : [-np.inf, -0.3, -0.1, -0.01, 0.01, 0.1, 0.3, np.inf]
        }
        self.state_clipped = self.clip_state()
        self.index = self.state_to_index(self.state_clipped)

    def validate(self, feature):
        if isinstance(feature, np.ndarray):
            return feature
        else:
            raise ValueError('not numpy array')

    def clip_state(self):
        x, y, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = self.state
        discretized_state = [
            np.digitize(x, self.bins['x_bins']) - 1,
            np.digitize(y, self.bins['y_bins']) - 1,
            np.digitize(x_vel, self.bins['x_vel_bins']) - 1,
            np.digitize(y_vel, self.bins['y_vel_bins']) - 1,
            np.digitize(angle, self.bins['angle_bins']) - 1,
            np.digitize(ang_vel, self.bins['ang_vel_bins']) - 1,
            int(left_leg),
            int(right_leg)
        ]
    
        return np.array(discretized_state)
    
    @staticmethod
    def state_to_index(state):
        x, y, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = state
        return (
            x * 6 * 7 * 6 * 7 * 7 * 2 * 2 +
            y * 7 * 6 * 7 * 7 * 2 * 2 +
            x_vel * 6 * 7 * 7 * 2 * 2 +
            y_vel * 7 * 7 * 2 * 2 +
            angle * 7 * 2 * 2 +
            ang_vel * 2 * 2 +
            left_leg * 2 +
            right_leg
        )

class StateActionPair(State):
    def __init__(self, state, action):
        super().__init__(state)
        self.action = self.validate(action)