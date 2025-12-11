import gymnasium as gym

class BaseAgent:
    def __init__(self, env: gym.Env):
        """Initialize your agent here."""
        self.env = env

    def load(self, path):
        """Load model weights (Used in Phase 1)."""
        pass

    def act(self, observation):
        """Return an action given an observation (Used in Phase 1 & 2)."""
        raise NotImplementedError

    def train(self):
        """
        Train the agent (Used in Phase 2).
        You MUST interact with 'env' to register steps.
        The env will raise a StopIteration or set a flag when converged.
        """
        raise NotImplementedError