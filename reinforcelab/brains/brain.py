from torch import Tensor
from abc import ABCMeta, abstractmethod

from reinforcelab.experience import Experience


class Brain(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, state: Tensor) -> Tensor:
        """Performs a computation over a state to determine the next actions.
        This could be a state value (V), state action value (Q), or a distribution
        over actions (pi)

        Args:
            state (Tensor): a tensor description of the state

        Returns:
            Tensor: Result of the computation over the state
        """

    @abstractmethod
    def target(self, state: Tensor) -> Tensor:
        """Performs a computation over a state to determine the next actions
        according to the target function.

        Args:
            state (Tensor): a tensor description of the state

        Returns:
            Tensor: Result of the computation over the state
        """


    @abstractmethod
    def action_value(self, state: Tensor, action: Tensor, target: bool = False) -> Tensor:
        """Obtains the State-Action value for a given state and action

        Args:
            state (Tensor): state or observation
            action (Tensor): action performed on that state
            target (bool): Wether to obtain the value from the target network. Defaults to False

        Returns:
            Tensor: Value for the given state and action
        """

    @abstractmethod
    def update(self, experience: Experience):
        """Updates the brain estimation given the passed experience

        Args:
            experience (Experience): An experience instance
        """