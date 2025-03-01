from copy import deepcopy
from typing import Any

from games.base import BasePlayer, BaseState, BaseGame


class TreeSearch(BasePlayer):
    def __init__(self, *, verbose: bool = False):
        self._state_value: dict[BaseState, int] = {}
        self._verbose = verbose
        self._verbosity_next_size = 10

    def fit(self, game: BaseGame):
        state = deepcopy(game.start())
        self._dfs(state)
        if self._verbose:
            print(f"Found {len(self._state_value)} states in total.")

    def do_action(self, state: BaseState) -> Any:
        if not self._state_value:
            raise NotFittedError("Model is not fitted yet")

        if state not in self._state_value:
            raise ValueError("Unknown state")

        _, action = max((-self._state_value[state.next(a)], a) for a in state.actions)
        return action

    def _dfs(self, state: BaseState) -> int:
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


class NotFittedError(AttributeError):
    pass
