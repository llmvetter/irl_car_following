import numpy as np

class GradientDescentOptimizer:
#have to use ascent optimizer i think?
    def __init__(
            self,
            omega: np.ndarray,
            learning_rate: float = 0.01,
    ) -> None:
        self.learning_rate = learning_rate
        self.omega = omega

    def step(
            self,
            gradient: float,
    ) -> None:
        self.omega += self.learning_rate * gradient
        return self.omega