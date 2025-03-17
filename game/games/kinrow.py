from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import scipy.signal as sig

from ..core.base import BaseGame
from ..core.exception import GameProcessError
from ..core.grid import RectangularGridTwoPlayerState, IntPair


@dataclass(eq=False)
class KInARowGameState(RectangularGridTwoPlayerState):
    """
    Game state for k-in-a-row games.

    Attributes:
        k: Number of consecutive marks required to win
    """

    k: int

    def is_winner(self, player: int) -> bool:
        """Check if player has k consecutive marks in any direction."""

        if self.last_action is None:
            bool_grid = (self.grid == player)
            return any(
                np.any(sig.convolve(bool_grid, kernel, mode="valid", method="direct") == self.k)
                for kernel in [
                    np.ones((1, self.k)),
                    np.ones((self.k, 1)),
                    np.eye(self.k),
                    np.fliplr(np.eye(self.k)),
                ]
            )

        if self.grid[*self.last_action] != player:
            return False

        row, col = self.last_action
        bool_grid = (self.grid == player)
        return any(
            np.any(np.convolve(slice_, np.ones(self.k), mode="valid") == self.k)
            for slice_ in [
                bool_grid[row, :],
                bool_grid[:, col],
                np.diagonal(bool_grid, col - row),
                np.diagonal(np.fliplr(bool_grid), self.grid.shape[1] - col - row - 1)
            ]
        )


class KInARowGame(BaseGame):
    """Generalized k-in-a-row game implementation."""

    def __init__(self, rows: int, cols: int, k: int, *, state_class: type = KInARowGameState):
        super().__init__(state_class=state_class)
        self.rows = rows
        self.cols = cols
        self.k = k
        self._state: KInARowGameState = self._state_class(
            grid=np.zeros((rows, cols), dtype=np.int_),
            k=k,
        )
        self._n_players = 2
        self._is_terminated = False
        self._winner = None

    def reset(self) -> None:
        """Reset game to initial empty state."""

        self._state = self._state_class(
            grid=np.zeros((self.rows, self.cols), dtype=np.int_),
            k=self.k,
        )
        self._is_terminated = False
        self._winner = None

    def start(self) -> KInARowGameState:
        if np.any(self._state.grid != 0):
            raise GameProcessError("game already started")
        return deepcopy(self._state)

    def step(self, action: IntPair) -> KInARowGameState:
        if self._is_terminated:
            raise GameProcessError("game already finished")

        self._state.next(action, inplace=True)
        self._is_terminated = self._state.is_terminal
        if self._is_terminated:
            self._winner = self._state.winner
        return deepcopy(self._state)
