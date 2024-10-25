import pickle
import numpy as np
import logging

from src.models.trainer import Trainer
from src.models.trajectory import Trajectories
from src.models.optimizer import GradientDescentOptimizer
from src.models.reward import LinearRewardFunction
from src.models.preprocessor import Preprocessor
from src.models.mdp import CarFollowingMDP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Init MDP")
mdp = CarFollowingMDP(
    a_min= -1,
    a_max= 1.5,
    a_steps=0.5,
    v_steps=0.25,
    g_steps=0.25,
)
logging.info("Init Reward Function")
reward_function = LinearRewardFunction(mdp=mdp)

omega = np.random.uniform(0, 1, reward_function.num_features)

logging.info("Init Optimizer")
optimizer = GradientDescentOptimizer(omega=omega)

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
        eps=1e-4,
)

logging.info("Init IRL Loop")
extracted_reward = trainer.train()

with open('home/h6/leve469a/results/reward_function.pickle', 'wb') as file:
    pickle.dump(extracted_reward, file)