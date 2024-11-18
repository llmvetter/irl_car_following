import torch
import logging
import pickle

from src.models.reward import RewardNetwork
from src.models.mdp import CarFollowingMDP
from src.config import Config
from src.utils import backward_pass


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

reward_function.load_state_dict(torch.load('/home/h6/leve469a/results/reward_function2.pth', weights_only=True))
reward_function.net.eval()
logging.info("RewardNetwork initialized.")

policy = backward_pass(
    mdp=mdp,
    reward=reward_function,
    max_iterations=config.backward_pass['iterations'],
)

with open('/home/h6/leve469a/results/policy.pkl', 'wb') as file:
    pickle.dump(policy, file)
logging.info("Policy has been saved.")
