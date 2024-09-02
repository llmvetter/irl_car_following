import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class Reward:
    def __init__(self, state_dim=2):
        self.state_dim = state_dim
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * state_dim, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10,
            random_state=0
        )

    def fit(self, X, y):
        """
        Fit the Gaussian Process model to the data.
        
        Args:
        X (array-like): State-action pairs, shape (n_samples, state_dim)
        y (array-like): Corresponding rewards, shape (n_samples,)
        """
        self.gp.fit(X, y)

    def forward(self, state):
        """
        Predict the reward for given state(s).
        
        Args:
        state (array-like): State(s) to predict reward for, shape (n_samples, state_dim)
        
        Returns:
        tuple: (mean_reward, std_reward)
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        mean, std = self.gp.predict(state, return_std=True)
        return mean, std

    def __call__(self, state):
        """
        Allow the class to be called like a function.
        """
        return self.forward(state)