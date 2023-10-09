import torch
import numpy as np
from typing import List
from .experience import Experience

class BatchExperience(Experience):
    def __init__(self, experiences: List[List[Experience]]):
        self.experiences = np.array(experiences)

        # Concatenate contents of all experiences
        self.state = torch.tensor(np.array([[exp.state for exp in traj] for traj in self.experiences])).float()
        self.action = torch.tensor(np.array([[exp.action for exp in traj] for traj in self.experiences])).float()
        self.reward = torch.tensor(np.array([[exp.reward for exp in traj] for traj in self.experiences])).float()
        self.next_state = torch.tensor(np.array([[exp.next_state for exp in traj] for traj in self.experiences])).float()
        self.done = torch.tensor(np.array([[exp.done for exp in traj] for traj in self.experiences])).float()
        self.truncated = torch.tensor(np.array([[exp.truncated for exp in traj] for traj in self.experiences])).float()

    def __last_experience_mask(self):
        done = torch.clamp(self.done + self.truncated, 0, 1)
        batch_size, n_steps = self.experiences.shape
        relevance_matrix = np.tile(np.arange(n_steps), (batch_size, 1))
        relevance_matrix += (n_steps * done).int().numpy()
        mask = relevance_matrix.max(axis=1, keepdims=1) == relevance_matrix
        return mask

    def __relevant_experiences_mask(self):
        done = torch.clamp(self.done + self.truncated, 0, 1)
        row_indices, col_indices = np.where(done == 1)
        comparison_matrix = np.ones_like(done) * float("inf")
        comparison_matrix[row_indices, :] = col_indices[:, np.newaxis]
        mask = (np.arange(done.shape[1]) <= comparison_matrix)
        return mask

    def to_td0(self, gamma: float) -> float:
        """Transforms the experience from td-n to td-0 and
        for bootstrapping. This is done by accumulating the rewards
        with the discount factor, and gathering the relevant elements
        of the trajectory to compute the bootstrap estimation normally.
        Returns the value of the discount factor that must be used for
        bootstrapping the next_state


        Args:
            discount (float): original discount value

        Returns:
            float: modified discount value accounting for td-n experience
        """
        le_mask = self.__last_experience_mask()
        _, n_steps = self.experiences.shape
        gammas = gamma ** np.arange(n_steps)
        relevant_mask = self.__relevant_experiences_mask()

        self.state = self.state[:, 0]
        self.action = self.action[:, 0]
        self.reward = (self.reward * gammas * relevant_mask).sum(axis=-1).unsqueeze(-1).float()
        self.next_state = self.next_state[le_mask]
        self.done = self.done.max(-1)[0].unsqueeze(-1)

        bootstrap_gamma = torch.tensor(gamma ** (le_mask.argmax(-1) + 1))

        return bootstrap_gamma.unsqueeze(-1).float()

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done