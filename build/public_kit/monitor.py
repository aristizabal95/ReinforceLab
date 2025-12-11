import gymnasium as gym
import numpy as np

class ConvergenceMonitor(gym.Wrapper):
    def __init__(self, env, goal_reward, stability_window, max_steps):
        super().__init__(env)
        self.goal = goal_reward
        self.window = stability_window
        self.max_steps = max_steps
        
        self.episode_rewards = []
        self.current_ep_reward = 0.0
        self.total_steps = 0
        self.converged = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_ep_reward += reward
        self.total_steps += 1
        
        if self.total_steps >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.current_ep_reward != 0 or len(self.episode_rewards) > 0:
            self.episode_rewards.append(self.current_ep_reward)
            
            if len(self.episode_rewards) >= self.window:
                recent = self.episode_rewards[-self.window:]
                avg = np.mean(recent)
                if avg >= self.goal:
                    self.converged = True
                    print(f"DEBUG: Converged! Avg: {avg} >= {self.goal}")
        
        self.current_ep_reward = 0.0
        return self.env.reset(**kwargs)