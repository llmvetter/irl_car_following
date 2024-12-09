class Config:
    def __init__(
            self,
    ) -> None:
        self.preprocessor = {
            "speed_treshold":3
        }
        self.mdp = {
            "a_min":-3,
            "a_max":3,
            "a_steps":0.75,
            "v_steps":0.2,
            "g_steps":0.1
        }
        self.backward_pass = {
            "epsilon": 0.5,
            "discount": 0.97,
            "temperature": 0.7,
            "iterations": 80
        }
        self.forward_pass ={
            "iterations": 100,
            "steps": 1000
        }
        self.reward_network = {
            "learning_rate":0.001,
            "layers":[100,100,100]
        }
        self.epochs = 6
