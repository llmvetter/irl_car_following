import torch
import logging
import pickle

import gymnasium as gym
from omegaconf import OmegaConf

from src.models.trainer import Trainer
from src.models.optimizer import GradientAscentOptimizer
from src.models.reward import RewardNetwork
from src.models.preprocessor import MilanoPreprocessor
from src.models.env import CarFollowingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = OmegaConf.load('/home/h6/leve469a/car_following/config.yaml')
logging.info(f"loaded config with params: {config.__dict__}")

gym.register(
    id="CarFollowing",
    entry_point=CarFollowingEnv,
)

mdp = gym.make(
    id="CarFollowing",
    dataset_path=config.data.exp_path,
    granularity=config.env.granularity,
    actions=config.env.actions,
    max_speed=config.env.max_speed,
    max_distance=config.env.max_distance,
    max_rel_speed=config.env.max_rel_speed,
    delta_t=config.env.delta_t,
)
mdp = mdp.unwrapped
mdp.reset()

logging.info("Mdp initialized: "
             f"n_states = {mdp.n_states},"
             f"n_action = {mdp.n_actions}.")

# logging.info("Computing transition probability matrix")
# mdp.compute_transitions()

logging.info("Loading transition probability matrix")
mdp.load_transitions(config.data.trans_path)

logging.info("Init Reward Function")
reward_function = RewardNetwork(
    mdp=mdp,
    layers=config.reward_network.layers)

logging.info("Init Optimizer")
optimizer = GradientAscentOptimizer(
    mdp=mdp,
    reward_network=reward_function,
    lr=config.reward_network.learning_rate,
)

logging.info("Loading Trajectories")
expert_trajectories = MilanoPreprocessor(
    mdp=mdp,
    config=config,
).load(path=config.data.exp_path)

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
