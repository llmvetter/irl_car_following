class IRLTrainer():
    def compute_gradient(reward_function, mdp, expert_feature_expectations, learner_feature_expectations):
        return expert_feature_expectations - learner_feature_expectations

    def train_irl(mdp, expert_trajectories, num_iterations, learning_rate):
    num_features = len(expert_trajectories[0][0])  # Assuming first state of first trajectory is representative
    reward_function = LinearRewardFunction(num_features)

    for iteration in range(num_iterations):
        # Compute expert feature expectations
        expert_feature_expectations = compute_expert_feature_expectations(expert_trajectories)

        # Compute learner's policy and feature expectations
        learner_policy = compute_policy(mdp, reward_function)
        learner_feature_expectations = compute_learner_feature_expectations(mdp, learner_policy)

        # Compute gradient
        gradient = compute_gradient(reward_function, mdp, expert_feature_expectations, learner_feature_expectations)

        # Update weights
        reward_function.update_weights(gradient, learning_rate)

        # Optionally: Print loss or other metrics
        loss = np.linalg.norm(expert_feature_expectations - learner_feature_expectations)
        print(f"Iteration {iteration}, Loss: {loss}")

    return reward_function

# Helper functions (to be implemented based on your MDP structure)
def compute_expert_feature_expectations(expert_trajectories):
    # Compute average feature counts from expert trajectories
    pass

def compute_policy(mdp, reward_function):
    # Compute optimal policy for the MDP given the current reward function
    pass

def compute_learner_feature_expectations(mdp, policy):
    # Compute expected feature counts for the learner's policy
    pass

# Example usage
mdp = YourMDPClass()  # Your MDP implementation
expert_trajectories = load_expert_trajectories()  # Load your expert data
num_iterations = 100
learning_rate = 0.01

trained_reward_function = train_irl(mdp, expert_trajectories, num_iterations, learning_rate)