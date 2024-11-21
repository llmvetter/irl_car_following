class Config:
    def __init__(
            self,
    ) -> None:
        self.preprocessor = {
            "speed_treshold":5
        }
        self.mdp = {
            "a_min":-1.5,
            "a_max":2,
            "a_steps":0.75,
            "v_steps":0.1,
            "g_steps":0.25
        }
        self.backward_pass = {
            "epsilon": 0.5,
            "discount": 0.95,
            "temperature": 0.7,
            "iterations": 80
        }
        self.forward_pass ={
            "iterations": 200,
            "steps": 2000
        }
        self.reward_network = {
            "learning_rate":0.0005,
            "layers":[30,90,20]
        }
        self.epochs = 8