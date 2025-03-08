from copy import deepcopy
from typing import Any

import numpy as np

from game.core.base import BasePlayer, BaseGameState, BaseGame


class DFSPlayer(BasePlayer):
    def __init__(self, *, verbose: bool = False):
        self._state_value: dict[BaseGameState, int] = {}
        self._verbose = verbose
        self._verbosity_next_size = 10

    def fit(self, game: BaseGame):
        state = deepcopy(game.start())
        self._dfs(state)
        if self._verbose:
            print(f"Found {len(self._state_value)} states in total.")

    def do_action(self, state: BaseGameState) -> Any:
        if not self._state_value:
            raise NotFittedError("Model is not fitted yet")

        if state not in self._state_value:
            raise ValueError("Unknown state")

        _, action = max((-self._state_value[state.next(a)], a) for a in state.actions)
        return action

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
    def __init__(self, depth: int):
        self.depth = depth

    def do_action(self, state: BaseGameState) -> Any:
        actions = state.actions
        values = np.array([-self._bfs(state, a, self.depth - 1) for a in actions])
        return actions[np.random.choice(np.argwhere(values == values.max()).flatten())]

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
