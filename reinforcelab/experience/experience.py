import torch

class Experience:
    def __init__(self, state, action, reward, next_state, done, truncated):
        self.state = torch.tensor(state)
        self.action = torch.tensor(action)
        self.reward = torch.tensor(reward)
        self.next_state = torch.tensor(next_state)
        self.done = torch.tensor(done).float()
        self.truncated = torch.tensor(truncated).float()

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done

    def __repr__(self):
        return f"Experience({self.state}, {self.action}, {self.reward}, {self.next_state}, {self.done}, {self.truncated})"