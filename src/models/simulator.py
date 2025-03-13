import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

class Simulator:
    def __init__(self, path: str):
        
        df = pd.read_csv(path)
        historic_speeds = np.array(df['Relative speed'])*0.27778
        self.kde = gaussian_kde(historic_speeds)

    def sample_relative_speed(self) -> float:
        """Samples a new relative speed based on the historical distribution."""
        return self.kde.resample(size=1)[0, 0]  # Sample a single speed

    def smooth_relative_speed(self, prev_speed: float, alpha=0.05) -> float:
        """Generates a smoothed relative speed using historical distribution sampling."""
        sampled_speed = self.sample_relative_speed()
        return (1 - alpha) * prev_speed + alpha * sampled_speed
