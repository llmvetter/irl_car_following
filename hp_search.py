import logging
import sys
import numpy as np
import pandas as pd

import ray
from ray import tune
import gymnasium as gym
from omegaconf import OmegaConf

from src.utils import deep_merge
from src.models.env import CarFollowingEnv
from training.objective import objective

# Init config
config = OmegaConf.load('/home/h6/leve469a/car_following/config.yaml')
config_dict = OmegaConf.to_container(config, resolve=True)

# Hyperparameter search space
search_space = {
    'backward_pass': {
        'epsilon': tune.uniform(0.1, 0.9),
        'discount': tune.uniform(0.95, 0.99),
        'temperature': tune.uniform(0.5, 1.0),
    },
    'reward_network': {
        'learning_rate': tune.loguniform(1e-4, 5e-2),
    },
}

task_id = int(sys.argv[1]) - 1

def create_name(trial):
    return f"trial_{trial.trial_id}"

# precompute and save transition matrix
gym.register(
    id="CarFollowing",
    entry_point=CarFollowingEnv,
)
env = gym.make(
        id="CarFollowing",
        dataset_path=config.data.exp_path,
        granularity=config.env.granularity,
        actions=config.env.actions,
        max_speed=config.env.max_speed,
        max_distance=config.env.max_distance,
        max_rel_speed=config.env.max_rel_speed,
        delta_t=config.env.delta_t,
    )
env = env.unwrapped
env.reset()

# logging.info(
#     f"Computing transition probability matrix with n_states: {env.n_states}")
# env.compute_transitions()
# np.save(config.data.trans_path, env.T)

logging.info("Loading transition probability matrix")
env.load_transitions(config.data.trans_path)

# initialize tuning job
logging.info("Initializing tuner")
ray.init(logging_level=logging.WARN)

tuner = tune.Tuner(
    tune.with_parameters(objective),
    param_space=deep_merge(config_dict, search_space),
    tune_config=tune.TuneConfig(
        metric="score",
        mode="max",
        num_samples=100,
        trial_dirname_creator=create_name,
    ),
)

# Run tuning
results = tuner.fit()

# Get best config
best_result = results.get_best_result(metric='score', mode='min')
df: pd.DataFrame = results.get_dataframe(filter_metric="score", filter_mode="min")
df_sorted = df.sort_values(by="score", ascending=True)
best_config = best_result.config if best_result else None

print(f'Best config found: {best_config}')
print(df_sorted.head(5))
