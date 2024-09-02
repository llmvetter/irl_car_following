import numpy as np

class State:
    def __init__(self, state:tuple):
        self.state = state
        self.space = {
            'speed': [x/2 for x in range(1, 40, 1)],
            'distance_gap': [x/2 for x in range(1, 80, 1)],
        }
        self.coord = self.state_to_index()
        self.state_clipped = self.clip_state()
        self.index = self.flatten_index()

    def clip_state(self) -> np.array:
        s_index, d_index = self.coord
        discretized_state = [
           self.space['speed'][s_index], 
           self.space['distance_gap'][d_index],
        ]
        return np.array(discretized_state)
    
    def state_to_index(self) -> tuple:
        speed, acceleration = self.state
        speed_index = np.digitize(speed, self.space['speed'], right=False)
        distance_gap_index = np.digitize(acceleration, self.space['distance_gap'], right=False)
        return (speed_index, distance_gap_index)
    
    def flatten_index(self) -> int:
        s_index, d_index = self.coord
        try:
            flattened_index = s_index * len(self.space['distance_gap']) + d_index
            return int(flattened_index)
        except ValueError:
            return ('State index out of bounds')
            

class Action():
    def __init__(self, action:float):
        self.action = action #TODO add normalization step
        self.space = {
            'acceleration' : [x/20 for x in range(-10, 11, 1)]
        }
        self.action_clipped = self.clip_action()
        self.index = self.action_to_index()

    def clip_action(self):
        a_index = np.digitize(self.action, self.space['acceleration'])
        discretized_action = self.space['acceleration'][a_index]
        return np.array(discretized_action)
    
    def action_to_index(self):
        try:
            a_index = np.digitize(self.action, self.space['acceleration'])
            return np.array(a_index)
        except ValueError:
            return ('Action outside of discrete actionspace.')

class StateActionPair:
    def __init__(self, state: tuple, action: float):
        self.state = State(state)
        self.action = Action(action)

    def get_state(self):
        return self.state.state_clipped

    def get_action(self):
        return self.action.action_clipped

    def get_state_index(self):
        return self.state.index

    def get_action_index(self):
        return self.action.index
