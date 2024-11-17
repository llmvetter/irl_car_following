class Config:
    def __init__(
            self,
    ) -> None:
        self.backward_pass = {
            "epsilon": 0.5,
            "discount": 0.95,
            "temperature": 0.6,
            "iterations": 100,
        }
        self.forward_pass ={
            "iterations": 1000,
            "steps": 2000
        }
        self.reward_network = {
            "learning_rate":0.0005,
            "layers":[30,90,20]
        }
        self.epochs = 10