class Config:
    def __init__(
            self,
    ) -> None:
        self.preprocessor = {
            "speed_treshold":2
        }
        self.dataset_path = '/home/h6/leve469a/IQ-Learn/data/LF_data-2e5.csv'
        self.mdp = {
            "actions": [-5, -3, -1.6, -0.8, -0.4, -0.2, 0, 0.2, 0.4, 0.8, 1.6, 3, 5],
            "max_speed": 30,
            "max_distance": 100,
            "max_rel_speed": 30,
            "granularity": 0.5,
            "delta_t": 0.3
        }
        self.backward_pass = {
            "epsilon": 0.5,
            "discount": 0.98,
            "temperature": 0.7,
            "iterations": 80
        }
        self.forward_pass ={
            "iterations": 100,
            "steps": 500
        }
        self.reward_network = {
            "learning_rate":0.002,
            "layers":[50,50,50]
        }
        self.epochs = 5