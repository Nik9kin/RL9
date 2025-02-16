from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .misc import Const


class BaseAgent(ABC):
    """
    Abstract base class for multi-armed bandit agents.

    Subclasses must implement the `do_action` method to define action selection strategy.

    Attributes:
        n_actions (int): Number of possible actions.
        exploring_cycles (int): Number of cycles to explore all actions at the beginning
            of the episode.
        value_estimates (np.ndarray): Array storing the estimated value of each action.
    """

    def __init__(self, n_actions: int, *, exploring_cycles: int = 1):
        """
        Initialize the agent.

        Args:
            n_actions (int): Number of possible actions.
            exploring_cycles (int, optional): Number of initial exploration cycles.
                Each cycle tries all actions once. Defaults to 1.
        """

        self.n_actions = n_actions
        self.exploring_cycles = exploring_cycles

        self._value_estimates = np.zeros(self.n_actions, dtype=np.float64)
        self._action_attempts = np.zeros(self.n_actions, dtype=np.int64)
        self._last_action = None
        self._t = 0

    def reset(self):
        """Reset the agent's internal state, including value estimates and timestep."""
        self._value_estimates = np.zeros(self.n_actions, dtype=np.float64)
        self._action_attempts = np.zeros(self.n_actions, dtype=np.int64)
        self._last_action = None
        self._t = 0

    @abstractmethod
    def do_action(self) -> int:
        """
        Select and return an action to perform.

        Returns:
            int: Index of the selected action.
        """
        pass

    def observe(self, reward):
        """
        Update the value estimate for the last taken action using the observed reward.

        Args:
            reward (float): Reward received after the last action.
        """

        a = self._last_action
        step_size = 1 / self._action_attempts[a]
        self._value_estimates[a] += step_size * (reward - self._value_estimates[a])

    @property
    def value_estimates(self):
        """np.ndarray: Current estimates of the value of each action."""
        return self._value_estimates


class EpsilonGreedy(BaseAgent):
    """
    Epsilon-greedy bandit agent with exploration-exploitation trade-off.

    During each timestep, with probability epsilon, explores (random action),
    otherwise exploits the best-known action. Includes an initial exploration phase.

    Attributes:
        epsilon (float or Callable[[int], float]): Exploration rate, can be dynamic.
        seed (int, optional): Seed for reproducible exploration.
    """

    def __init__(
            self,
            n_actions: int,
            epsilon: float | Callable[[int], float],
            *,
            exploring_cycles: int = 1,
            seed: int | None = None
    ):
        """
        Initialize epsilon-greedy agent.

        Args:
            n_actions (int): Number of possible actions.
            epsilon: Exploration rate. Constant float or function of timestep (1-based).
            exploring_cycles (int, optional): Initial cycles of forced exploration.
                Defaults to 1.
            seed (int, optional): Random seed for exploration. Defaults to None.
        """

        super().__init__(n_actions, exploring_cycles=exploring_cycles)
        if callable(epsilon):
            self._epsilon = epsilon
        else:
            self._epsilon = Const(epsilon)
        self.seed = seed

        self._rng = None

        self.reset(self.seed)

    def reset(self, seed: int | None = None):
        """
        Reset agent state and random generator.

        Args:
            seed (int, optional): New random seed. Defaults to None.
        """
        super().reset()
        self._rng = np.random.default_rng(seed)

    def do_action(self) -> int:
        """
        Select action using epsilon-greedy strategy.

        Returns:
            int: Selected action index.
        """

        if self._t < self.exploring_cycles * self.n_actions:
            # Exploring start phase: cycle through actions
            self._last_action = self._t % self.n_actions
        elif self._rng.random() < self.epsilon:
            # Exploration: random action
            self._last_action = self._rng.integers(self.n_actions)
        else:
            # Exploitation: best-known action
            self._last_action = np.argmax(self._value_estimates)

        self._action_attempts[self._last_action] += 1
        self._t += 1
        return self._last_action

    @property
    def epsilon(self):
        """float: Current exploration rate, dynamically computed if callable."""
        return self._epsilon(self._t + 1)


class Greedy(EpsilonGreedy):
    """Greedy bandit agent that always exploits the best-known action (epsilon=0)."""

    def __init__(self, n_actions: int, *, exploring_cycles: int = 1):
        """
        Initialize greedy agent.

        Args:
            n_actions (int): Number of actions.
            exploring_cycles (int, optional): Initial exploration cycles. Defaults to 1.
        """
        super().__init__(n_actions, 0.0, exploring_cycles=exploring_cycles)


class Random(EpsilonGreedy):
    """Random bandit agent that always explores (epsilon=1)."""

    def __init__(self, n_actions: int, *, seed: int | None = None):
        """
        Initialize random agent.

        Args:
            n_actions (int): Number of actions.
            seed (int, optional): Random seed. Defaults to None.
        """
        super().__init__(n_actions, 1.0, exploring_cycles=0, seed=seed)


class UCB1(BaseAgent):
    """
    Upper Confidence Bound (UCB1) bandit agent.

    Selects actions based on value estimates plus exploration bonus term.
    """

    def __init__(self, n_actions: int, c: float, *, exploring_cycles: int = 1):
        """
        Initialize UCB1 agent.

        Args:
            n_actions (int): Number of actions.
            c (float): Exploration parameter controlling confidence width.
            exploring_cycles (int, optional): Initial exploration cycles. Defaults to 1.
        """

        super().__init__(n_actions, exploring_cycles=exploring_cycles)

        self._c = c

        self.reset()

    def do_action(self) -> int:
        """
        Select action using UCB1 algorithm.

        Returns:
            int: Selected action index.
        """

        if self._t < self.exploring_cycles * self.n_actions:
            # Exploring start phase: cycle through actions
            self._last_action = self._t % self.n_actions
        else:
            # UCB1 selection: value + confidence bound
            exploration_bonus = self._c * np.sqrt(np.log(self._t) / self._action_attempts)
            self._last_action = np.argmax(self._value_estimates + exploration_bonus)

        self._action_attempts[self._last_action] += 1
        self._t += 1
        return self._last_action

    @property
    def c(self):
        """float: Exploration parameter c."""
        return self._c
