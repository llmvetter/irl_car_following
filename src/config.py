class Config:
    def __init__(
            self,
    ) -> None:
        self.backwardpass = {
            "epsilon": 0.5,
            "iterations": 50
        }
        self.forward_pass ={
            "iterations": 100,
            "steps": 2000
        }
        self.reward_network = {
            "learning_rate":0.05,
            "layers":[30,90,20]
        }
        self.epochs = 30