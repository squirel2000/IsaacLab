# manager_policy.py
import numpy as np

class RuleBasedManager:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def select_policy(self, height_scan_obs):
        # Use std or max-min as a simple terrain roughness metric
        if np.std(height_scan_obs) < self.threshold:
            return 0  # Flat policy
        else:
            return 1  # Rough policy