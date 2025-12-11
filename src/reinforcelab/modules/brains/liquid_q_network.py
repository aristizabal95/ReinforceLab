from .brain import Brain
from reinforcelab.estimators import Estimator
from copy import deepcopy
from ncps.torch.cfc import CfC
import torch

class LiquidQNetwork(Brain):
    def __init__(self, model: CfC, estimator: Estimator, learning_rate=0.01, alpha=0.001):
        super(LiquidQNetwork, self).__init__()
        self.local_model = model
        self.target_model = deepcopy(model)
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.hidden = None

    def __call__(self, state):
        if self.hidden is None:
            batch_size = state.shape[0]
            hidden_size = self.local_model.state_size
            self.hidden = torch.zeros(batch_size, hidden_size)
        out, hidden = self.local_model(state)
        self.hidden = hidden
        return out
        