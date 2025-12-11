"""
Random Agent Solution - Sample Submission for ReinforceLab Competitions

This agent takes random actions from the environment's action space.
It serves as a baseline and demonstrates the expected interface.

Your submission should have an agent.py file with an Agent class that implements:
- act(observation): Returns an action given an observation
- train(env): Trains the agent using the provided environment
- load(path): Loads model weights from a file (optional for Phase 1)
- save(path): Saves model weights to a file (optional)
"""
import numpy as np
import gymnasium as gym


class Agent:
    """
    Random Agent - A baseline agent that takes random actions.
    
    This agent samples random actions from the environment's action space.
    It does not learn anything, making it a useful baseline for comparison.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize the agent."""
        self.env = env

    def act(self, observation):
        """
        Return a random action.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            A random action sampled from the action space.
        """
        return self.env.action_space.sample()

    def train(self, env: gym.Env):
        """
        Train the agent on the environment.
        
        For a random agent, we don't actually learn anything.
        We just run episodes until convergence is detected or max steps reached.
        
        Args:
            env: The gymnasium environment to train on.
                 The environment is wrapped to track convergence.
        """
        # Run episodes until the environment signals to stop
        # The wrapped environment will raise StopIteration when converged
        try:
            while True:
                obs, _ = self.env.reset()
                done = False
                while not done:
                    action = self.act(obs)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
        except StopIteration:
            # Training complete - convergence reached or max steps hit
            pass
    
    def load(self, path: str):
        """
        Load model weights from a file.
        
        For a random agent, there's nothing to load.
        
        Args:
            path: Path to the model file.
        """
        # Random agent has no weights to load
        pass
    
    def save(self, path: str):
        """
        Save model weights to a file.
        
        For a random agent, there's nothing to save.
        
        Args:
            path: Path where the model should be saved.
        """
        # Random agent has no weights to save
        pass

