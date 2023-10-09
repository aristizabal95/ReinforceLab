import random
import numpy as np
from collections import deque

from reinforcelab.memory_buffers.memory_buffer import MemoryBuffer
from reinforcelab.experience import Experience, BatchExperience


class ExperienceReplay(MemoryBuffer):
    def __init__(self, config):
        self.experience_buffer = deque(maxlen=config['max_size'])
        self.batch_size = config["batch_size"]
        self.n_steps = config["n_steps"]
        self.transform = None
        if "transform" in config:
            self.transform = config["transform"]

    def add(self, experience: Experience):
        """Adds an experience tuple to the memory buffer

        Args:
            experience (Experience): An experience instance
        """
        self.experience_buffer.append(experience)

    def sample(self) -> BatchExperience:
        """Retrieves a random set of experience tuples from the memory buffer

        Returns:
            (BatchExperience): A batch of experiences
        """
        if len(self) < self.batch_size + self.n_steps:
            raise RuntimeError(
                "There's not enough experience to create a batch")

        batch_idxs = np.random.choice(range(len(self.experience_buffer) - self.n_steps), size=self.batch_size, replace=False)
        if self.n_steps:
            experiences = [[self.experience_buffer[idx+offset] for offset in range(self.n_steps)] for idx in batch_idxs]
        else:
            experiences = [[self.experience_buffer[idx]] for idx in batch_idxs]
        batch = BatchExperience(experiences)
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def all(self) -> BatchExperience:
        """Retrieves all experiences from the memory buffer

        Returns:
            BatchExperience: All experiences as a BatchExperience
        """
        return BatchExperience(list(self.experience_buffer))

    def __len__(self):
        return len(self.experience_buffer)
