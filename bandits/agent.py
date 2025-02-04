from collections.abc import Callable

import numpy as np

from misc import Const


class BaseAgent:
    def __init__(self, n_actions: int, *, exploring_start: int = 1):
        self.n_actions = n_actions
        self.exploring_start = exploring_start

        self._value_estimates = None
        self._action_attempts = None
        self._last_action = None
        self._t = 0

    def reset(self):
        self._value_estimates = np.zeros(self.n_actions)
        self._action_attempts = np.zeros(self.n_actions)
        self._last_action = None
        self._t = 0

    def do_action(self) -> int:
        raise NotImplementedError()

    def observe(self, reward):
        a = self._last_action
        step_size = 1 / self._action_attempts[a]
        self._value_estimates[a] += step_size * (reward - self._value_estimates[a])

    @property
    def value_estimates(self):
        return self._value_estimates


class EpsilonGreedy(BaseAgent):
    def __init__(
            self,
            n_actions: int,
            epsilon: float | Callable[[int], float],
            *,
            exploring_start: int = 1,
            seed: int | None = None
    ):
        super().__init__(n_actions, exploring_start=exploring_start)
        if callable(epsilon):
            self._epsilon = epsilon
        else:
            self._epsilon = Const(epsilon)
        self.seed = seed

        self._rng = None

        self.reset(seed=self.seed)

    def reset(self, *, seed: int | None = None):
        super().reset()
        self._rng = np.random.default_rng(seed)

    def do_action(self) -> int:
        if self._t < self.exploring_start * self.n_actions:
            self._last_action = self._t % self.n_actions
        elif self._rng.random() < self.epsilon:
            self._last_action = self._rng.integers(self.n_actions)
        else:
            self._last_action = np.argmax(self._value_estimates)

        self._action_attempts[self._last_action] += 1
        self._t += 1
        return self._last_action

    @property
    def epsilon(self):
        return self._epsilon(self._t + 1)


class Greedy(EpsilonGreedy):
    def __init__(self, n_actions: int, *, exploring_start: int = 1):
        super().__init__(n_actions, 0.0, exploring_start=exploring_start)


class Random(EpsilonGreedy):
    def __init__(self, n_actions: int, *, seed: int | None = None):
        super().__init__(n_actions, 1.0, exploring_start=0, seed=seed)


class UCB1(BaseAgent):
    def __init__(self, n_actions: int, c: float, *, exploring_start: int = 1):
        super().__init__(n_actions, exploring_start=exploring_start)
        self.c = c

        self.reset()

    def do_action(self) -> int:
        if self._t < self.exploring_start * self.n_actions:
            self._last_action = self._t % self.n_actions
        else:
            self._last_action = np.argmax(
                self._value_estimates + self.c * np.sqrt(np.log(self._t) / self._action_attempts)
            )

        self._action_attempts[self._last_action] += 1
        self._t += 1
        return self._last_action
