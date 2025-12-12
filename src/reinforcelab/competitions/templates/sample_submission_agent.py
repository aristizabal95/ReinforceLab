"""
Sample Submission Agent - Demonstrates loading from checkpoint files

This agent extends the random agent to show how to load additional files
from your submission. In a real submission, you would load your trained
model weights instead of a text file.
"""
import os
import gymnasium as gym


class Agent:
    """
    Sample Agent that demonstrates loading from a checkpoint file.
    
    This agent takes random actions but shows the pattern for loading
    saved model weights or other data from your submission.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize the agent with the environment."""
        self.env = env
        self.checkpoint_data = None
    
    def act(self, observation):
        """
        Return a random action.
        
        In your implementation, this would use your trained model
        to select actions based on the observation.
        """
        return self.env.action_space.sample()
    
    def load(self, path: str):
        """
        Load model weights or data from a file.
        
        This demo loads a text file and prints its contents.
        In your implementation, you would load your trained model:
        
        Example for PyTorch:
            self.model = torch.load(path)
            
        Example for custom data:
            with open(path, 'rb') as f:
                self.weights = pickle.load(f)
        """
        # Demo: Load and print the checkpoint file
        checkpoint_dir = os.path.dirname(path)
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.txt")
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                self.checkpoint_data = f.read()
            print(f"Loaded checkpoint from: {checkpoint_file}")
            print(f"Checkpoint contents:\n{self.checkpoint_data}")
        else:
            print(f"No checkpoint.txt found at {checkpoint_file}")
    
    def save(self, path: str):
        """
        Save model weights to a file.
        
        In your implementation, you would save your trained model:
        
        Example for PyTorch:
            torch.save(self.model.state_dict(), path)
        """
        # Demo: Nothing to save for random agent
        print(f"Save called with path: {path}")
        pass
    
    def train(self):
        """
        Train the agent on the environment.
        
        In your implementation, this would contain your training loop.
        """
        # Demo: Just run random episodes
        try:
            while True:
                obs, _ = self.env.reset()
                done = False
                while not done:
                    action = self.act(obs)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
        except StopIteration:
            pass

