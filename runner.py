import torch
import logging
import pickle

from src.models.trainer import Trainer
from src.models.trajectory import Trajectories
from src.models.optimizer import GradientAscentOptimizer
from src.models.reward import RewardNetwork
from src.models.preprocessor import Preprocessor
from src.models.mdp import CarFollowingMDP
from src.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = Config()
logging.info(f"loaded config with params: {config.__dict__}")

logging.info("Init MDP")
mdp = CarFollowingMDP(
    a_min= -1,
    a_max= 1.5,
    a_steps=0.5,
    v_steps=0.25,
    g_steps=0.25,
)
logging.info("Init Reward Function")
reward_function = RewardNetwork(
    mdp=mdp,
    layers=config.reward_network['layers'])

logging.info("Init Optimizer")
optimizer = GradientAscentOptimizer(
    mdp=mdp,
    reward_network=reward_function,
    lr=config.reward_network['learning_rate'],
)

logging.info("Loading Trajectories")
expert_trajectories = Trajectories([])
for i in range(1,6):
    path = f"/home/h6/leve469a/data/TrajData_Punzo_Napoli/drivetest{i}.FCdata"
    trajs = Preprocessor(mdp=mdp).load(path=path)
    expert_trajectories += trajs

logging.info("Init Trainer")
trainer = Trainer(
        expert_trajectories,
        optimizer,
        reward_function,
        mdp,
)

logging.info("Init IRL Loop")
extracted_reward_function, expected_svf = trainer.train(
    epochs=config.epochs,
    epsilon=config.backwardpass['epsilon'],
    backward_it=config.backwardpass['iterations'],
    forward_it=config.forward_pass['iterations'],
)

torch.save(extracted_reward_function.state_dict(), '/home/h6/leve469a/results/reward_function.pth')
logging.info("RewardNetwork has been saved.")
with open('/home/h6/leve469a/results/expected_svf.pkl', 'wb') as file:
    pickle.dump(expected_svf, file)
logging.info("Expected state visitation frequency has been saved.")
