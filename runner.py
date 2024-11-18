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

mdp = CarFollowingMDP(
    a_min= config.mdp['a_min'],
    a_max= config.mdp['a_max'],
    a_steps=config.mdp['a_steps'],
    v_steps=config.mdp['v_steps'],
    g_steps=config.mdp['g_steps'],
)
logging.info("Mdp initialized: "
             f"n_states = {mdp.n_states},"
             f"n_action = {mdp.n_actions}.")

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
        config,
        expert_trajectories,
        optimizer,
        reward_function,
        mdp,
)

logging.info("Init IRL Loop")
extracted_reward_function, policy = trainer.train(
    epochs=config.epochs,
)

torch.save(extracted_reward_function.state_dict(), '/home/h6/leve469a/results/reward_function.pth')
logging.info("RewardNetwork has been saved.")
with open('/home/h6/leve469a/results/policy.pkl', 'wb') as file:
    pickle.dump(policy, file)
logging.info("Policy has been saved.")
