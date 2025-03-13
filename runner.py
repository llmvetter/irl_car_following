import torch
import logging
import pickle

import gymnasium as gym

from src.models.trainer import Trainer
from src.models.optimizer import GradientAscentOptimizer
from src.models.reward import RewardNetwork
from src.models.preprocessor import MilanoPreprocessor
from src.models.env import CarFollowingEnv
from src.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = Config()
logging.info(f"loaded config with params: {config.__dict__}")

gym.register(
    id="CarFollowing",
    entry_point=CarFollowingEnv,
)

mdp = gym.make(
    id="CarFollowing",
    dataset_path=config.dataset_path,
    granularity=0.5,
    actions= [-5, -3, -1.5, -0.8, -0.4, -0.2, 0, 0.2, 0.4, 0.8, 1.5, 3, 5],
    )
mdp = mdp.unwrapped
mdp.reset()

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
expert_trajectories = MilanoPreprocessor(mdp=mdp).load(path=config['dataset_path'])

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

torch.save(extracted_reward_function.state_dict(), '/home/h6/leve469a/results/final_reward_function.pth')
logging.info("RewardNetwork has been saved.")
with open('/home/h6/leve469a/results/policy.pkl', 'wb') as file:
    pickle.dump(policy, file)
logging.info("Policy has been saved.")
