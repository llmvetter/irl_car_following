import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.reward import RewardNetwork
from src.models.mdp import CarFollowingMDP


class Analyzer:

    def __init__(
            self,
            reward: RewardNetwork,
            mdp: CarFollowingMDP,
    ):
        self.reward_network = reward
        self.mdp = mdp
    
    def plot_heatmap(
            self,
            data: np.ndarray,
    ):
        if len(data)==12800:
            rows, cols = 80, 160
        if len(data)==32000:
            rows, cols = 160, 200
        if len(data)==20000:
            rows, cols = 100, 200

        data_2d = data.reshape(rows, cols)
        _, ax = plt.subplots(figsize=(12, 8))
        amplified_data = np.power(data_2d, 0.3)
        im = ax.imshow(amplified_data, cmap='coolwarm', aspect='auto')
        cbar = plt.colorbar(im, label='Normalized SVF')
        cbar.set_ticks([amplified_data.min(), amplified_data.max()])
        cbar.set_ticklabels([f'{data_2d.min():.2e}', f'{data_2d.max():.2e}'])

        plt.title('State visitation heatmap')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.tight_layout()
        plt.show()

    def plot_reward_function(self):
        state_tensor = torch.tensor(self.mdp.state_space, dtype=torch.float32)
        reward_values = self.reward_network.forward(state_tensor, grad=False)
        velocities = np.array([state[0] for state in self.mdp.state_space])
        distance_gaps = np.array([state[1] for state in self.mdp.state_space])
        qt = distance_gaps / velocities
        qt_clipped = np.clip(qt, a_min=0, a_max=10)

        plt.figure(figsize=(12, 6))
        plt.scatter(qt_clipped, reward_values, s=1, alpha=0.1, color='gray')

        fixed_velocities = [2, 4, 6, 8, 10, 12, 14, 16, 18]
        colors = [
            '#FF1493', '#00CED1', '#FF8C00', '#4B0082', '#32CD32',
            '#FF69B4', '#1E90FF', '#FFD700', '#8A2BE2'
        ]

        max_rewards = []
        max_quotients = []

        for v, color in zip(fixed_velocities, colors):
            mask = np.isclose(velocities, v, atol=0.01)
            v_rewards = reward_values[mask]
            v_quotients = qt_clipped[mask]
            
            plt.scatter(v_quotients, v_rewards, s=5, c=color, label=f'v = {v}')
            
            if len(v_rewards) > 0:
                max_reward_index = np.argmax(v_rewards)
                max_reward = v_rewards[max_reward_index]
                max_quotient = v_quotients[max_reward_index]
                max_rewards.append(max_reward)
                max_quotients.append(max_quotient)
                
                plt.scatter(max_quotient, max_reward, s=50, c=color, edgecolors='black', linewidth=2)

        plt.xlabel('Quotient (distance_gap/velocity)')
        plt.ylabel('Reward Value')
        plt.title('Reward Value vs Quotient with Fixed Velocities')
        plt.grid(True)
        plt.legend()
        plt.savefig('C:/Users/lenna/Documents/IRL/car_following/results/07/plots/reward_qt.pdf')
        plt.show()

        for v, reward, quotient in zip(fixed_velocities, max_rewards, max_quotients):
            print(f"Velocity {v}: Max Reward = {reward}, Quotient = {quotient}")
    
    def plot_reward_heatmap(
            self,
    ):

        state_space = self.mdp.state_space
        state_space_tensor = torch.tensor(state_space, dtype=torch.float32)
        state_rewards = self.reward_network.forward(state_space_tensor, grad=False).detach().cpu().numpy()

        _, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(state_space[:, 0], state_space[:, 1], c=state_rewards, cmap='viridis')

        plt.colorbar(scatter, label='Reward')
        ax.set_xlabel('Velocity in m/s')
        ax.set_ylabel('Distance Gap in m')
        ax.set_title('State Space with Color-coded Rewards')
        plt.savefig('C:/Users/lenna/Documents/IRL/car_following/results/07/plots/reward_heatmap.pdf')
        plt.show()
    
    def plot_trajectory(
        self,
        policy: torch.tensor,
        steps: int = 2000,
    ) -> np.ndarray:

        velocity = np.zeros(steps)
        distance_gap = np.zeros(steps)
        timesteps = np.arange(steps)

        state = torch.randint(0, self.mdp.n_states, (1,)).item()
        print(f'initial state: {state}, with features: {self.mdp._index_to_state(state)}')
        for t in range(steps):
            state_features = self.mdp._index_to_state(state)
            velocity[t] = state_features[0]
            distance_gap[t] = state_features[1]
            action = torch.multinomial(policy[state], 1).item()
            next_state = self.mdp.step(state, action)
            state = next_state

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(timesteps, velocity, label='Velocity')
        plt.xlabel('Timestep')
        plt.ylabel('Velocity in m/s')
        plt.title('Velocity over trajectory')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(timesteps, distance_gap, label='Distance Gap')
        plt.xlabel('Timestep')
        plt.ylabel('Distance Gap in m')
        plt.title('Distance Gap over trajectory')
        plt.legend()

        plt.tight_layout()
        plt.show()