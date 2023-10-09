from gymnasium import Env

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QTable
from reinforcelab.estimators import MaxQEstimator
from reinforcelab.action_selectors import EpsilonGreedy
from reinforcelab.memory_buffers import OrderedBuffer


class QLearning(Agent):
    """A Q Learning Agent uses a Q Table to store the values
    of each action for each state. It updates the value estimates
    by assuming that the agents will act optimistically after the first
    action (This is implemented by the MaxQEstimator). To enable exploration,
    the agent uses an epsilon-greedy policy. Q Learning agents don't need
    experience replay, so they learn from the immediate experience.
    """

    def __init__(self, env: Env, discount_factor: float = 0.999, alpha=0.01, n_steps=0):
        action_selector = EpsilonGreedy(env)
        estimator = MaxQEstimator(env, discount_factor)
        brain = QTable(env, estimator, alpha=alpha)
        buffer = OrderedBuffer({"batch_size": 1, "max_size": 1, "n_steps": n_steps})

        super().__init__(brain, action_selector, buffer)
