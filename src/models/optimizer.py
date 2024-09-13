class GradientDescentOptimizer:
#have to use ascent optimizer i think?
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.omega = None

    def reset(self, omega):
        self.omega = omega

    def step(self, gradient):
        self.omega += self.learning_rate * gradient
        return self.omega