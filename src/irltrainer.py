import torch

from lunar_lander.src.agent.dqn import DQNAgent #TODO exchange for value iteration
from lunar_lander.src.models.trajectories import Trajectories
from lunar_lander.src.ml_models.reward import RewardNetwork #TODO exchange for gauss kernel
from lunar_lander.src.utils import (
    policy_state_visitation_frequency,
    expert_state_visitation_frequency,
)

class IRLTrainer:
    def __init__(self, env, learning_rate=0.001):
        self.env = env
        self.reward_network = RewardNetwork()
        self.dqn_agent = DQNAgent()
        self.optimizer = torch.optim.Adam(
            self.reward_network.parameters(), 
            lr=learning_rate,
        )

    def train(
            self,
            expert_trajectories: Trajectories,
            num_iterations:  int,
            num_episodes: int,
    ) -> tuple[RewardNetwork, DQNAgent]:
        expert_svf = expert_state_visitation_frequency(
            expert_trajectories,
        )

        for iteration in range(num_iterations):
            self._train_dqn(num_episodes)

            policy_svf = policy_state_visitation_frequency(
                policy_network=self.dqn_agent,
                env=self.env,
            )
            states = self._sample_states(num_samples=1000)
            loss = self._compute_irl_loss(
                expert_svf,
                policy_svf,
                states,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print(f"Iteration {iteration}, Loss: {loss.item()}")

        return self.reward_network, self.dqn_agent

    def _train_dqn(self, num_episodes):
        for _ in range(num_episodes):
            state = torch.from_numpy(self.env.reset()[0])
            terminated = False
            while not terminated:
                action = self.dqn_agent.get_action(state)
                next_state, _, terminated, _, _ = self.env.step(action)
                reward = self.reward_network(state)
                self.dqn_agent.train(state, action, reward, next_state, done=terminated)
                state = torch.from_numpy(next_state)

    def _compute_irl_loss(self, expert_svf, policy_svf, states):
        expert_svf = torch.FloatTensor(expert_svf)
        policy_svf = torch.FloatTensor(policy_svf)
        expert_nonzero = self.count_nonzero(expert_svf)
        policy_nonzero = self.count_nonzero(policy_svf)

        print(
            f"Expert SVF non-zero values: "
            f"{expert_nonzero} out of {expert_svf.numel()}"
        )
        print(
            f"Policy SVF non-zero values: "
            f"{policy_nonzero} out of {policy_svf.numel()}"
        )
        states = torch.FloatTensor(states)
        rewards = self.reward_network(states)
        loss = torch.mean(rewards * (expert_svf - policy_svf))
        return loss

    def _sample_states(self, num_samples=1000):
        states = []
        for _ in range(num_samples):
            state = self.env.reset()[0]
            states.append(state)
        return torch.FloatTensor(states)
    
    def count_nonzero(self, tensor):
        return torch.count_nonzero(tensor).item()