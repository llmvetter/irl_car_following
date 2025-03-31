import logging
from typing import Any

import gymnasium as gym
from ray import tune

from src.models.trainer import Trainer
from src.models.agent import Agent
from src.models.evaluator import Evaluator
from src.models.optimizer import GradientAscentOptimizer
from src.models.reward import RewardNetwork
from src.models.preprocessor import MilanoPreprocessor
from src.models.env import CarFollowingEnv
from src.config import DotDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


gym.register(
    id="CarFollowing",
    entry_point=CarFollowingEnv,
)

def objective(config: dict[str, Any]):

    config = DotDict(config)
    logging.info(f"loaded config with params: {config.__dict__}")


    logging.info("Initializing Environment")
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

    logging.info("Loading transition probability matrix")
    env.load_transitions(config.data.trans_path)

    logging.info("Init Reward Function")
    reward_function = RewardNetwork(
        layers=config.reward_network.layers)

    logging.info("Init Optimizer")
    optimizer = GradientAscentOptimizer(
        mdp=env,
        reward_network=reward_function,
        lr=config.reward_network.learning_rate,
    )

    logging.info("Loading Trajectories")
    expert_trajectories, _ = MilanoPreprocessor(
        mdp=env,
        config=config,
    ).load(path=config.data.exp_path)

    logging.info("Init Trainer")
    trainer = Trainer(
            config,
            expert_trajectories,
            optimizer,
            reward_function,
            env,
    )

    logging.info("Init IRL Loop")
    _, policy = trainer.train(
        epochs=config.epochs,
    )

    logging.info("Initializing Evaluator")
    agent = Agent(
        policy=policy,
        env=env,
    )

    evaluator = Evaluator(
        environment=env,
        agent=agent,
    )

    logging.info("Initializing evaluation process")
    metrics = evaluator.evaluate(
        num_trajectories=100,
    )
    tune.report({'score': metrics['final_score']})

    return {'score': metrics['final_score']}