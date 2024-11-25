class Config:
    def __init__(
            self,
    ) -> None:
        self.preprocessor = {
            "speed_treshold":4
        }
        self.mdp = {
            "a_min":-1.5,
            "a_max":2,
            "a_steps":0.75,
            "v_steps":0.25,
            "g_steps":0.25
        }
        self.backward_pass = {
            "epsilon": 0.5,
            "discount": 0.97,
            "temperature": 1,
            "iterations": 80
        }
        self.forward_pass ={
            "iterations": 100,
            "steps": 1000
        }
        self.reward_network = {
            "learning_rate":0.01,
            "layers":[100,100,100]
        }
        self.epochs = 5