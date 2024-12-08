import torch
import logging
import pickle

from src.utils import backward_pass
from src.models.reward import RewardNetwork
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

reward_function.load_state_dict(torch.load('/home/h6/leve469a/data/reward_function.pth', weights_only=True))
reward_function.net.eval()

logging.info("Extracting policy...")
policy = backward_pass(
    mdp=mdp,
    reward=reward_function,
    temperature=config.backward_pass['temperature'],
    discount=config.backward_pass['discount'],
    epsilon=config.backward_pass['epsilon'],
    max_iterations=config.backward_pass['iterations'],
)

with open('/home/h6/leve469a/results/policy.pkl', 'wb') as file:
    pickle.dump(policy, file)
logging.info("Policy has been saved.")
