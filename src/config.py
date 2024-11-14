class Config:
    def __init__(
            self,
    ) -> None:
        self.backwardpass = {
            "epsilon": 0.1,
            "iterations": 1
        }
        self.forward_pass ={
            "iterations": 10,
            "steps": 2000
        }
        self.reward_network = {
            "learning_rate":0.05,
            "layers":[30,90,20]
        }
        self.epochs = 1