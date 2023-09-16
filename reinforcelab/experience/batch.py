import torch
import numpy as np
from typing import List
from .experience import Experience

class BatchExperience(Experience):
    def __init__(self, experiences: List[List[Experience]]):
        self.experiences = np.array(experiences)

        self.vstate_getter = np.vectorize(lambda x: x.state)
        self.vaction_getter = np.vectorize(lambda x: x.action)
        self.vreward_getter = np.vectorize(lambda x: x.reward)
        self.vnstate_getter = np.vectorize(lambda x: x.next_state)
        self.vdone_getter = np.vectorize(lambda x: x.done)

        self.le_mask = self.__last_experience_idx()

        # Concatenate contents of all experiences
        self.state = torch.vstack([n_steps[0].state for n_steps in self.experiences])
        self.done = torch.tensor(self.vdone_getter(self.experiences).max(1))
        self.action = torch.vstack([n_steps[0].action for n_steps in self.experiences])
        self.reward = torch.tensor([exp.reward for exp in self.experiences]).float()
        self.next_state = torch.vstack([exp.next_state for exp in self.experiences]).float()
        self.done = torch.vstack([exp.done for exp in self.experiences]).float()

    def __last_experience_idx(self):
        batch_size, n_steps = self.experiences.shape
        relevance_matrix = np.tile(np.arange(n_steps), (batch_size, 1))
        dones = self.vdone_getter(self.experiences)
        relevance_matrix += (n_steps * dones).astype(int)
        mask = relevance_matrix.max(axis=1, keepdims=1) == relevance_matrix
        return mask

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done