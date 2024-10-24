import pickle
import numpy as np

from car_following.src.models.trainer import Trainer
from car_following.src.models.trajectory import Trajectories
from car_following.src.models.optimizer import GradientDescentOptimizer
from car_following.src.models.reward import LinearRewardFunction
from car_following.src.models.preprocessor import Preprocessor
from car_following.src.models.mdp import CarFollowingMDP

#init mdp
mdp = CarFollowingMDP(
    a_min= -1,
    a_max= 1.5,
    a_steps=0.5,
    v_steps=0.25,
    g_steps=0.25,
)
#init reward function
reward_function = LinearRewardFunction(mdp=mdp)

omega = np.random.uniform(0, 1, reward_function.num_features)
#init optimizer
optimizer = GradientDescentOptimizer(omega=omega)

#change directory for remote directory
expert_trajectories = Trajectories([])
for i in range(1,6):
    path = f'home/leve469a/data/TrajData_Punzo_Napoli/drivetest{i}.FCdata'
    trajs = Preprocessor(mdp=mdp).load(path=path)
    expert_trajectories += trajs

trainer = Trainer(
        expert_trajectories,
        optimizer,
        reward_function,
        mdp,
        eps=1e-4,
)

extracted_reward = trainer.train()

with open('home/leve469a/results/reward_function.pickle', 'wb') as file:
    pickle.dump(extracted_reward, file)