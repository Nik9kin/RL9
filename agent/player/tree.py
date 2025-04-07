from copy import deepcopy
from typing import Any

import numpy as np

from game.core.base import BasePlayer, BaseGameState, BaseGame


def _choose_random_optimal_action(actions: list[Any], values: np.ndarray):
    return actions[np.random.choice(np.argwhere(values == values.max()).flatten())]


class DFSPlayer(BasePlayer):
    """Optimal player using depth-first search to precompute state values."""

    def __init__(self, *, verbose: bool = False):
        self._state_value: dict[BaseGameState, int] = {}
        self._verbose = verbose
        self._verbosity_next_size = 10

    def __str__(self):
        return "DFSPlayer"

    def fit(self, game: BaseGame):
        """Precompute all possible state values through DFS."""

        state = deepcopy(game.start())
        self._dfs(state)
        if self._verbose:
            print(f"Found {len(self._state_value)} states in total.")

    def do_action(self, state: BaseGameState) -> Any:
        """Choose action with highest precomputed value."""

        if not self._state_value:
            raise NotFittedError("Model is not fitted yet")

        if state not in self._state_value:
            raise ValueError("Unknown state")

        actions = state.actions
        values = np.array([-self._state_value[state.next(a)] for a in actions])
        return _choose_random_optimal_action(actions, values)

    def _dfs(self, state: BaseGameState) -> int:
        if state not in self._state_value:
            if state.is_terminal:
                winner = state.winner
                if winner is None:
                    value = 0
                else:
                    value = 2 * int(winner == state.turn) - 1
                self._state_value[state] = value
            else:
                values = [-self._dfs(state.next(action)) for action in state.actions]
                self._state_value[state] = max(values)
            if self._verbose and len(self._state_value) == self._verbosity_next_size:
                print(f"Found {self._verbosity_next_size} states.")
                self._verbosity_next_size *= 10
        return self._state_value[state]


class BFSPlayer(BasePlayer):
    """Player using breadth-first search with limited depth for move selection."""

    def __init__(self, depth: int):
        self.depth = depth

    def __str__(self):
        return f"BFS-{self.depth} Player"

    def do_action(self, state: BaseGameState) -> Any:
        """Choose action maximizing BFS evaluation up to given depth."""

        actions = state.actions
        values = np.array([-self._bfs(state, a, self.depth - 1) for a in actions])
        return _choose_random_optimal_action(actions, values)

    def _bfs(self, state: BaseGameState, action: Any, depth: int):
        state = state.next(action)
        if state.is_terminal:
            winner = state.winner
            if winner is None:
                return 0
            else:
                return 2 * int(winner == state.turn) - 1
        elif depth == 0:
            return 0
        return max(-self._bfs(state, action, depth - 1) for action in state.actions)


class NotFittedError(AttributeError):
    pass
